import json
import numpy as np
import argparse
import os

def generate_camera_params(json_path, H, W):
   with open(json_path, 'r') as f:
       meta = json.load(f)
   camera_angle_x = float(meta['camera_angle_x'])
   focal = .5 * W / np.tan(.5 * camera_angle_x)
  
   return {
       "fl_x": focal,
       "fl_y": focal,
       "cx": W / 2,
       "cy": H / 2,
       "w": W,
       "h": H,
       "camera_model": "OPENCV",
       "k1": 0.0,
       "k2": 0.0,
       "p1": 0.0,
       "p2": 0.0,
   }


def update_json_with_camera_params(json_path, camera_params):
   with open(json_path, 'r') as f:
       data = json.load(f)
  
   data.update(camera_params)
  
   for frame in data.get('frames', []):
       file_path = frame.get('file_path', "")
       if not file_path.endswith(".png"):
           file_path += ".png"
       frame['file_path'] = file_path
       frame['hyperspectral_file_path'] = os.path.splitext(file_path)[0] + ".npy"
  
   with open(json_path, 'w') as f:
       json.dump(data, f, indent=4)


def process_folder(folder_path):
   transform_files = ['transforms_train.json', 'transforms_test.json', 'transforms_val.json']
   
   for transform_file in transform_files:
       json_path = os.path.join(folder_path, transform_file)
       if os.path.exists(json_path):
           camera_params = generate_camera_params(json_path, 512, 512)
           update_json_with_camera_params(json_path, camera_params)


if __name__ == "__main__":
   parser = argparse.ArgumentParser(description='Add camera parameters to transforms.json files in folder')
   parser.add_argument('--folder_path', type=str, required=True, help='Path to folder containing transforms json files')
   
   args = parser.parse_args()
   process_folder(args.folder_path)