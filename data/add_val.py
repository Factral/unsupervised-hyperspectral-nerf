import json

def merge_json_frames(json_path1, json_path2):
    """
    Merge frames from two JSON files that have the same structure.
    
    Args:
        json_path1 (str): Path to the first JSON file
        json_path2 (str): Path to the second JSON file
        
    Returns:
        dict: Merged JSON data with combined frames
    """
    with open(json_path1, 'r') as f:
        data1 = json.load(f)
        
    with open(json_path2, 'r') as f:
        data2 = json.load(f)
    
    merged_data = data1.copy()
    
    merged_data['frames'].extend(data2['frames'])
    
    return merged_data

def save_merged_json(merged_data, output_path):
    """
    Save the merged JSON data to a file.
    
    Args:
        merged_data (dict): The merged JSON data
        output_path (str): Path where to save the merged JSON
    """
    with open(output_path, 'w') as f:
        json.dump(merged_data, f, indent=4)

if __name__ == "__main__":
    json_path1 = "./ajar/transforms_test.json"
    json_path2 = "./ajar/transforms_train.json"
    output_path = "./transforms.json"
    
    try:
        merged_data = merge_json_frames(json_path1, json_path2)
        
        save_merged_json(merged_data, output_path)
        print(f"Successfully merged JSON files. Result saved to {output_path}")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find one of the input files - {e}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in one of the files - {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")