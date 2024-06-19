# import os

# def generate_image_pairs(base_dir, scenes_file, output_file):
#     # Read the list of scene directories from the provided file
#     with open(scenes_file, 'r') as file:
#         scenes = file.read().strip().split()

#     # Initialize a list to hold the pairs
#     pairs = []

#     # Loop through each scene directory
#     for scene in scenes:
#         # Construct the full path to the scene directory
#         scene_dir = os.path.join(base_dir, scene)
#         # Get all image files in the directory, sorted to ensure correct pairing
#         files = sorted(f for f in os.listdir(scene_dir) if f.endswith('.png'))
        
#         # Generate pairs of consecutive images
#         for i in range(len(files) - 1):
#             first_file = os.path.join(scene, files[i])
#             second_file = os.path.join(scene, files[i+1])
#             pairs.append(f"{first_file} {second_file}")

#     # Write the pairs to the output file
#     with open(output_file, 'w') as file:
#         file.write('\n'.join(pairs))

# # Define paths
# base_dir = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/data/syntheticForestData/imageData"
# scenes_file = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/datasets/tartanSceneLists/valid_scenes_clean.txt"
# output_file = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/datasets/tartanSceneLists/valid_pairs.txt"

# # Generate image pairs
# generate_image_pairs(base_dir, scenes_file, output_file)

import os
import logging

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_image_pairs(base_dir, scenes_file, output_file):
    # Read the list of scene directories from the provided file
    with open(scenes_file, 'r') as file:
        scenes = file.read().strip().split()

    # Initialize a list to hold the pairs
    pairs = []

    # Loop through each scene directory
    for scene in scenes:
        # Construct the full path to the scene directory
        if scene == "APartial":
            print("Processing scene APartial")
            scene_dir = os.path.join(base_dir, scene)
            
            # Check if the directory exists
            if not os.path.exists(scene_dir):
                logging.warning(f"Directory not found: {scene_dir}")
                continue
            
            # Get all image files in the directory, sorted to ensure correct pairing
            files = sorted(f for f in os.listdir(scene_dir) if f.endswith('.png'))
            
            # Log the number of images found
            logging.info(f"Processing directory '{scene}' with {len(files)} images")
            
            # Generate pairs of consecutive images
            for i in range(len(files) - 1):
                first_file = os.path.join(scene, files[i])
                second_file = os.path.join(scene, files[i+1])
                pairs.append(f"{first_file} {second_file}")
        else:
            print(f"Skipping scene {scene}")

    # Log the number of pairs created
    logging.info(f"Total pairs generated: {len(pairs)}")

    # Write the pairs to the output file
    with open(output_file, 'w') as file:
        file.write('\n'.join(pairs))

# Define paths
base_dir = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/data/syntheticForestData/imageData/"
scenes_file = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/datasets/tartanSceneLists(Partial)/valid_scenes_clean.txt"
output_file = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/datasets/tartanSceneLists(Partial)/valid_pairs.txt"

# Generate image pairs
generate_image_pairs(base_dir, scenes_file, output_file)


