import os
import matplotlib.pyplot as plt
import numpy as np
import OpenEXR
import Imath

def read_exr_as_np(fn):
    f = OpenEXR.InputFile(fn)
    print(f.header())
    channels = f.header()['channels']
    print(f"Channels: {channels}")
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

def plot_exr_files(root_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            if file == 'r_1.exr':
                filepath = os.path.join(subdir, file)
                print(filepath)
                img, ch_names = read_exr_as_np(filepath)
                
                plt.figure(figsize=(10, 10))
                
                if img.shape[2] >= 3:
                    plt.imshow(np.clip(img[:,:,:3], 0, 1))
                else:
                    plt.imshow(np.clip(img[:,:,0], 0, 1), cmap='viridis')
                
                plt.title(f'EXR Image: {os.path.relpath(filepath, root_folder)}')
                plt.axis('off')
                
                output_path = os.path.join(output_folder, f"{os.path.relpath(filepath, root_folder).replace('/', '_').replace('.exr', '.png')}")
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
                plt.close()
                
                print(f"Saved plot to {output_path}")

root_folder = './nespof/cbox_dragon'
output_folder = './'
plot_exr_files(root_folder, output_folder)