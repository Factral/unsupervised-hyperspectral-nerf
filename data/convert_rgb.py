import pandas as pd
from re import S
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import OpenEXR
import Imath

import numpy as np


def g(x, alpha, mu, sigma1, sigma2):
    sigma = (x < mu)*sigma1 + (x >= mu)*sigma2
    return alpha*np.exp((x-mu)**2 / (-2*(sigma**2)))


def component_x(x): return g(x, 1.056, 5998, 379, 310) + \
    g(x, 0.362, 4420, 160, 267) + g(x, -0.065, 5011, 204, 262)


def component_y(x): return g(x, 0.821, 5688, 469, 405) + \
    g(x, 0.286, 5309, 163, 311)


def component_z(x): return g(x, 1.217, 4370, 118, 360) + \
    g(x, 0.681, 4590, 260, 138)


def xyz_from_xy(x, y):
    """Return the vector (x, y, 1-x-y)."""
    return np.array((x, y, 1-x-y))


ILUMINANT = {
    'D65': xyz_from_xy(0.3127, 0.3291),
    'E':  xyz_from_xy(1/3, 1/3),
}

COLOR_SPACE = {
    'sRGB': (xyz_from_xy(0.64, 0.33),
             xyz_from_xy(0.30, 0.60),
             xyz_from_xy(0.15, 0.06),
             ILUMINANT['D65']),

    'AdobeRGB': (xyz_from_xy(0.64, 0.33),
                 xyz_from_xy(0.21, 0.71),
                 xyz_from_xy(0.15, 0.06),
                 ILUMINANT['D65']),

    'AppleRGB': (xyz_from_xy(0.625, 0.34),
                 xyz_from_xy(0.28, 0.595),
                 xyz_from_xy(0.155, 0.07),
                 ILUMINANT['D65']),

    'UHDTV': (xyz_from_xy(0.708, 0.292),
              xyz_from_xy(0.170, 0.797),
              xyz_from_xy(0.131, 0.046),
              ILUMINANT['D65']),

    'CIERGB': (xyz_from_xy(0.7347, 0.2653),
               xyz_from_xy(0.2738, 0.7174),
               xyz_from_xy(0.1666, 0.0089),
               ILUMINANT['E']),
}


class ColourSystem:

    def __init__(self, start=450, end=640, num=21, cs='sRGB'):

        # Chromaticities
        bands = np.linspace(start=start, stop=end, num=num)*10

        self.cmf = np.array([component_x(bands),
                             component_y(bands),
                             component_z(bands)])

        self.red, self.green, self.blue, self.white = COLOR_SPACE[cs]

        # The chromaticity matrix (rgb -> xyz) and its inverse
        self.M = np.vstack((self.red, self.green, self.blue)).T
        self.MI = np.linalg.inv(self.M)

        # White scaling array
        self.wscale = self.MI.dot(self.white)

        # xyz -> rgb transformation matrix
        self.A = self.MI / self.wscale[:, np.newaxis]

  
    def get_transform_matrix(self):

        XYZ = self.cmf
        RGB = XYZ.T @ self.A.T
        RGB = RGB / np.sum(RGB, axis=0, keepdims=True)
        return RGB

    def spec_to_rgb(self, spec):
        """Convert a spectrum to an rgb value."""
        M = self.get_transform_matrix()
        print(M.shape)
        rgb = spec @ M
        return rgb



def read_exr_as_np(fn):
    f = OpenEXR.InputFile(fn)
    channels = f.header()['channels']
    print(channels)
    dw = f.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    ch_names = []

    image = np.zeros((size[1], size[0], len(channels)))
    for i, ch_name in enumerate(channels):
        ch_names.append(ch_name)
        ch_dtype = channels[ch_name].type
        ch_str = f.channel(ch_name, ch_dtype)
        
        if ch_dtype == Imath.PixelType(Imath.PixelType.FLOAT):
            np_dtype = np.float32
        elif ch_dtype == Imath.PixelType(Imath.PixelType.HALF):
            np_dtype = np.half
            
        image_ch = np.fromstring(ch_str, dtype=np_dtype)
        image_ch.shape = (size[1], size[0])
        image[:,:,i] = image_ch
    
    print(image.shape,"a")
    return image, ch_names

def read_exr_as_stokes(fn):
    image, _ = read_exr_as_np(fn)   # image shape = [512, 512, 16]
    stokes = np.clip(image, -1, 1)     # stokes shape = [512, 512, 4]
    
    return stokes

wave_to_rgb = pd.read_csv("spec_to_rgb.csv")

def color_mapping(PATH, FIG_OUTPATH, OBJECT, wavelength, FILE, scalar=1, s123_min=-1, s123_max=1):
    img_path = f"{PATH}/{OBJECT}/{wavelength}/train/{FILE}"
    
    stokes = read_exr_as_stokes(img_path)    
    H, W = stokes.shape[:2]
    rgb = np.array(wave_to_rgb[wave_to_rgb["wavelength"] == wavelength][["R", "G", "B"]])/255

    plt.figure()
    s = stokes[..., 0]       # s shape = [512, 512, 3]
    #s = s * np.expand_dims(rgb,(0)) * scalar
    plt.imshow(s)
    FILE_OUT = f"{FILE[:-4]}_rgb_vis_{wavelength}.png"
    out_img_path = f"{FIG_OUTPATH}/{FILE_OUT}"
    #cv2.imwrite(out_img_path, s[:,:]*255)
    plt.show()
    return s

FIG_OUTPATH='./ajar_adapted'
PATH = './nespof'    # directly accessing remote notework seems to take long time. better to use a local path 
wvls = np.arange(450, 650+1, 10)


idxs = list(range(54))
remove_idx =  [0,8,16,24,32,40,48] # remove_idx
idxs = [i for i in idxs if i not in remove_idx]

for k in idxs:
    cube = np.zeros((512, 512, len(wvls)))
    for i, wvl in enumerate(wvls):
        print(f'processing wavelength {wvl}')
        s = color_mapping(PATH, FIG_OUTPATH, "ajar/ajar", wvl, f"r_{k}.exr")
        cube[..., i] = s

    print(cube.shape)

    filtered_df = wave_to_rgb[wave_to_rgb['wavelength'].isin(wvls)]


    rgb_array = filtered_df[['R', 'G', 'B']].to_numpy()

    print(rgb_array)
    rgb_array_normalized = rgb_array / 255.0

    print(f"Shape of the normalized RGB array: {rgb_array_normalized.shape}")
    print(cube.shape)

    cube=cube.clip(0, 1)

    np.save(f"{FIG_OUTPATH}/train/r_{k}.npy", cube)

    print(cube.max(), cube.min())
    print(rgb_array_normalized.max(), rgb_array_normalized.min())

    spectorgb = ColourSystem(cs='sRGB')
    try1= spectorgb.spec_to_rgb(cube)
    try1= try1.clip(0, 1)

    plt.figure()
    plt.imshow(try1)
    plt.show()
    cv2.imwrite(f"{FIG_OUTPATH}/train/r_{k}.png", try1[:,:,::-1]*255)


    rgb = cube @ rgb_array_normalized
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

    plt.figure()
    plt.imshow(rgb)
    plt.show()
    cv2.imwrite(f"{FIG_OUTPATH}/train/r_{k}_bad.png", rgb[:,:,::-1]*255)