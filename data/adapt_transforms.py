import json
import numpy as np
import argparse
import os

def generate_camera_params(json_path, H, W):
    """
    Generate camera parameters from a JSON file and specified dimensions
    Args:
        json_path: Path to the transforms.json file
        H: Image height
        W: Image width
    Returns:
        Dict of camera parameters
    """
    with open(json_path, 'r') as f:
        meta = json.load(f)
    camera_angle_x = float(meta['camera_angle_x'])

    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    camera_params = {
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
    
    return camera_params

def update_json_with_camera_params(json_path, camera_params):
    """
    Update existing JSON file with camera parameters for each frame
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    data.update(camera_params)
    
    print(data)
    for frame in data.get('frames', []):
        print(frame)
        file_path = frame.get('file_path', "")
        
        if not file_path.endswith(".png"):
            file_path += ".png"
        frame['file_path'] = file_path
        
        hyperspectral_file_path = os.path.splitext(file_path)[0] + ".npy"
        frame['hyperspectral_file_path'] = hyperspectral_file_path
    
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    return json_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add camera parameters to transforms.json')
    parser.add_argument('--json_path', type=str, required=True, help='Path to transforms.json')

    
    args = parser.parse_args()
    
    camera_params = generate_camera_params(args.json_path, 512, 512)
    
    output_path = update_json_with_camera_params(args.json_path, camera_params)