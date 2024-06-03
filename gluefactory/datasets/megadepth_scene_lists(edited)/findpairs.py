from pathlib import Path

def filter_valid_pairs(input_path, output_path, keep_scene):
    # Read the original file
    with open(input_path, 'r') as file:
        lines = file.readlines()
    
    # Filter lines that contain the specified scene
    filtered_lines = [line for line in lines if keep_scene in line]
    
    # Write the filtered lines to a new file
    with open(output_path, 'w') as file:
        file.writelines(filtered_lines)

# Define paths
input_path = '/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/datasets/megadepth_scene_lists/valid_pairs.txt'
output_path = '/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/datasets/megadepth_scene_lists/valid_scenes1.txt'

# Call the function to filter the pairs
filter_valid_pairs(input_path, output_path, keep_scene='0022')
