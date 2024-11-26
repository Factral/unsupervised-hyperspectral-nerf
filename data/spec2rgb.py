import os
import numpy as np
import pandas as pd
import torch
import cv2
import OpenEXR
import Imath

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
    
    return image, ch_names


def color_mapping(PATH, OBJECT, wavelength, FILE, split, scalar=1, s123_min=-1, s123_max=1):
    img_path = f"{PATH}/{OBJECT}/{wavelength}/{split}/{FILE}"
    
    image, _ = read_exr_as_np(img_path)   # image shape = [512, 512, 4]
    stokes = np.clip(image, -1, 1)     # stokes shape = [512, 512, 4] 
    s = stokes[..., 0]       # s shape = [512, 512, 3]

    return s

def spec_to_rgb(img_path, save_path, view, split):

    wvls = np.arange(450, 650+1, 10)
    cube = np.zeros((512, 512, len(wvls)))
    for i, wvl in enumerate(wvls):
        print(f'processing wavelength {wvl}')
        s = color_mapping("./nespof", img_path, wvl, f"{view}.exr", split)
        cube[..., i] = s

    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, f"{view}.npy"), cube)
    
    imgs = cube
    H, W, c = imgs.shape
    imgs = imgs.reshape(H*W, c)
    
    cmf_table = pd.read_csv('../datasheet/spec_to_XYZ.csv')  
    cmf_array = np.array(cmf_table[(cmf_table["wavelength"]%10 == 0) & (cmf_table["wavelength"] >= 450) & (cmf_table["wavelength"] <= 650)])
    cmf = cmf_array[:, 1:]
    
    img_rgb = []
    
    for i in range(H*W):
        xyz = np.sum(imgs[i][:, np.newaxis] * cmf, axis=0)
        
        # https://en.wikipedia.org/wiki/SRGB
        srgb = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, +1.8758, +0.0415], [+0.0557, -0.2040, +1.0570]]) @ xyz
        gamma_correct = np.vectorize(lambda x: 12.92*x if x < 0.0031308 else 1.055*(x**(1.0/2.4))-0.055)
        rgb = gamma_correct(srgb)
        
        img_rgb.append(rgb)
            
    imgs_rgb = np.array(img_rgb).reshape(H, W, 3)[...,::-1] * 100    # BGR -> RGB  +  Intensity 
    
    cv2.imwrite(os.path.join(save_path, f"{view}.png"), imgs_rgb)


idxs = list(range(54))
split = "train"

if split == "train":
    remove_idx =  [0,8,16,24,32,40,48] # remove_idx
    idxs = [i for i in idxs if i not in remove_idx]
elif split == "val":
    idxs = [0,8,16,24,32,40,48]


for k in idxs:
    spec_to_rgb("ajar/ajar", f"./processed/ajar/{split}/", "r_{}".format(k), split)