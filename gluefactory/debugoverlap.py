import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
import shutil
import tarfile
from collections.abc import Iterable
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch

import os 


import cv2
import numpy as np
from multiprocessing import Pool
import os
import h5py


# filtered_depth_paths = ['phoenix/S6/zl548/MegaDepth_v1/0204/dense0/depths/6090224351_e47c7a7841_b.h5',
#  'phoenix/S6/zl548/MegaDepth_v1/0204/dense0/depths/2311566555_0c2995407f_o.h5',
#  'phoenix/S6/zl548/MegaDepth_v1/0204/dense0/depths/4500482425_e7f608c5de_o.h5']


# filtered_poses = [np.array([[-0.99981896,  0.00518729,  0.01830706,  0.0913207 ],
#                          [-0.00689591,  0.79793348, -0.60270607, -0.857202  ],
#                          [-0.01773423, -0.6027232 , -0.79775324,  0.996966  ],
#                          [ 0.        ,  0.        ,  0.        ,  1.        ]]), 
#                  np.array([[ 0.99881031, -0.0174577 ,  0.04553234, -0.350281  ],
#                         [-0.01570648,  0.76879143,  0.63930669, -0.0445705 ],
#                         [-0.04616569, -0.63926127,  0.76760261, -0.659616  ],
#                         [ 0.        ,  0.        ,  0.        ,  1.        ]]), 
#                 np.array([[ 9.99895122e-01,  3.73813649e-04, -1.44777531e-02, -8.72253000e-02],
#                        [ 2.89577373e-03,  9.74319525e-01,  2.25151232e-01,1.51859000e+00],
#                        [ 1.41901222e-02, -2.25169543e-01,  9.74216258e-01,6.27333000e+00], 
#                         [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,1.00000000e+00]])]

# filtered_intrinsics = [np.array([[3.38324e+03, 0.00000e+00, 3.39500e+02],[0.00000e+00, 3.38324e+03, 5.10000e+02], [0.00000e+00, 0.00000e+00, 1.00000e+00]]),
#                        np.array([[1.72107e+03, 0.00000e+00, 5.91500e+02], [0.00000e+00, 1.72072e+03, 8.00000e+02], [0.00000e+00, 0.00000e+00, 1.00000e+00]]), 
#                        np.array([[1.29853e+03, 0.00000e+00, 5.38500e+02], [0.00000e+00, 1.29876e+03, 8.00000e+02], [0.00000e+00, 0.00000e+00, 1.00000e+00]])]

# truth_matrix =  [[-1. -1. -1.],  [-1. -1. -1.], [-1. -1. -1.]]


"""
The plan:
1) check input for pose, depth, and intrinsics is same format for both the treedepth and for megadepth (espeically pose quaternion etc + if intrinsics hardcoded in case not overwritten)
2) if need convert pose use the lightglue utils to convert
3) check that calling and loading the correct matrices for files + try on other matrices that isnt just the 46 x 46 one
3) check the projectr 2D_to_3D etc keys in the info dictionary and see if that helps / can be used
4) Check the -1 values are in the positions for all the None values or whether that is just the problem
5) Check if different way to calc overlap matrix / check the github repo to see if helps:https://github.com/mzjb/overlap-only-OpenMX/blob/main/README.md
6) see if dataset online with using depth, intrinsics and pose to see if can get the same overlap matrix
7) check the overlap matrix is correct by visualizing it?? or checking the values
8) see if correct way to calc overlap matrix, should we be using flow info to verify it on treedepth?
9) for tree depth check the difference between pose left and right and see if the file repeats itself on the original download. 
=10) snip the overlap matrix to 3x3 and compare to the ground truth overlap matrix
"""

def calculate_overlap_matrix(depth_paths, poses, intrinsics):
    """Calculates the overlap matrix between frames in parallel.    
    Args:
        depth_paths: List of paths to depth maps.
        poses: List of camera poses.
        intrinsics: List of camera intrinsics.
        print_interval: Number of iterations after which to print an overlap value.
    """
    from multiprocessing import Pool

    print("!!!calculating overlalping matrix for", depth_paths.shape, poses.shape, intrinsics.shape)
    num_frames = len(depth_paths)
    # overlap_matrix = np.zeros((num_frames, num_frames))
    overlap_matrix = np.full((num_frames, num_frames), -1.0)

    # Prepare arguments for parallel processing
    pairs = [(i, j, depth_paths, poses, intrinsics) 
            for i in range(num_frames) for j in range(i + 1, num_frames)]
    
    print("about to claculate the overlap pairs")
    with Pool() as p:  # Use all available CPU cores
        # print("result calc")
        results = p.map(calculate_overlap_for_pair, pairs)

    # Fill overlap matrix
    for i, j, overlap in results:
        overlap_matrix[i, j] = overlap
        overlap_matrix[j, i] = overlap  # Make symmetric
        # if i % 500 == 0:
        #     print(f"Overlap between frame {i} and {j}: {overlap}")  # Print at intervals

    return overlap_matrix

# def project_points(depth, intrinsics, pose):
#     h, w = depth.shape
#     x, y = np.meshgrid(np.arange(w), np.arange(h))
#     z = depth.flatten()

#     x = (x.flatten() - intrinsics[0, 2]) * z / intrinsics[0, 0]
#     y = (y.flatten() - intrinsics[1, 2]) * z / intrinsics[1, 1]
#     points = np.vstack((x, y, z, np.ones_like(z)))

#     transformed_points = pose @ points
#     transformed_points /= transformed_points[2, :]

#     x_proj = intrinsics[0, 0] * transformed_points[0, :] / transformed_points[2, :] + intrinsics[0, 2]
#     y_proj = intrinsics[1, 1] * transformed_points[1, :] / transformed_points[2, :] + intrinsics[1, 2]

#     valid = (x_proj >= 0) & (x_proj < w) & (y_proj >= 0) & (y_proj < h)
#     return np.count_nonzero(valid), valid.size

def project_points(depth, intrinsics, pose):
    """Project points from 3D space to 2D image plane using camera intrinsics and pose."""
    # h, w = depth.shape
    # y, x = np.indices((h, w))
    # z = depth
    # x = (x - intrinsics[0, 2]) * z / intrinsics[0, 0]
    # y = (y - intrinsics[1, 2]) * z / intrinsics[1, 1]

    # points = np.vstack((x, y, z)).T
    # points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])  # make points homogeneous
    # transformed_points = pose @ points_homogeneous.T  # apply transformation

    # x_proj = transformed_points[0, :] * intrinsics[0, 0] / transformed_points[2, :] + intrinsics[0, 2]
    # y_proj = transformed_points[1, :] * intrinsics[1, 1] / transformed_points[2, :] + intrinsics[1, 2]

    # valid = (x_proj >= 0) & (x_proj < w) & (y_proj >= 0) & (y_proj < h)
    # return np.count_nonzero(valid), len(points)
    # Transform points to the camera frame
    
    # points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    # points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    # points_transformed = pose @ points_homogeneous.T

    # # Project points onto the image plane
    # x_proj = points_transformed[0, :] * intrinsics[0, 0] / points_transformed[2, :] + intrinsics[0, 2]
    # y_proj = points_transformed[1, :] * intrinsics[1, 1] / points_transformed[2, :] + intrinsics[1, 2]
    # valid = (x_proj >= 0) & (x_proj < w) & (y_proj >= 0) & (y_proj < h)

    # return np.count_nonzero(valid), valid.size 

    """Project points from 3D space to 2D image plane using camera intrinsics and pose."""
    h, w = depth.shape
    y, x = np.indices((h, w))
    z = depth.flatten()

    x = (x.flatten() - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (y.flatten() - intrinsics[1, 2]) * z / intrinsics[1, 1]

    points = np.vstack((x, y, z)).T
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])  # make points homogeneous
    transformed_points = pose @ points_homogeneous.T  # apply transformation

    x_proj = transformed_points[0, :] * intrinsics[0, 0] / transformed_points[2, :] + intrinsics[0, 2]
    y_proj = transformed_points[1, :] * intrinsics[1, 1] / transformed_points[2, :] + intrinsics[1, 2]

    valid = (x_proj >= 0) & (x_proj < w) & (y_proj >= 0) & (y_proj < h)
    return np.count_nonzero(valid), len(points)


    

# def load_npy_file(partial_file_path):
#     import os
#     base_directory = DATA_PATH
#     # base_directory = Path("/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/data/")
#     # base_directory = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/data/syntheticForestData"
#     print("BASE DIRECTORY: ", base_directory, "Partial file path ", partial_file_path)
#     file_path = os.path.join(base_directory, partial_file_path)
#     print("FILE PATH: ", file_path)

#     if os.path.exists(file_path):
#         return np.load(file_path)
#     else:
#         print(f"File not found: {file_path}")
#         raise Exception
#         return None

def load_npy_file(partial_file_path):
    import os
    # Define the correct base directory
    base_directory = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/data/megadepth/depth_undistorted"

    # Print initial path details for debugging
    # print("BASE DIRECTORY: ", base_directory, "Partial file path ", partial_file_path)

    # /homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/data/megadepth/depth_undistorted/0001
    # /homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/data/megadepth/depth_undistorted/0001/dense0/depths/2750838037_06ac72a948_o.h5

    # print(partial_file_path)

    # Adjust file path as needed
    partial_path_elements = partial_file_path.split('/')
    if len(partial_path_elements) > 4:
        modified_path = os.path.join(partial_path_elements[-4], partial_path_elements[-1]) 
        file_path = os.path.join(base_directory, modified_path)
    else:
        file_path = os.path.join(base_directory, partial_file_path)

    # for idx, partial_path in enumerate(partial_path_elements):
    #     print(f"Index: {idx}, Partial Path: {partial_path}")        
        # Do something with the data

    # Print final path for verification
    # print("FINAL FILE PATH: ", file_path)

    # Check if the file exists and load it
    if os.path.exists(file_path):
        with h5py.File(file_path, 'r') as f:
            # print("Available keys in the HDF5 file:", list(f.keys()))
            # Assuming the data you need is stored under a specific dataset name
            data = depth_data = f['depth'][:]    
            ### Temp plotting code for debugging, depth values loaded correctly
            #  plt.figure(figsize=(10, 8))
            # plt.imshow(depth_data, cmap='gray')
            # plt.colorbar()
            # plt.title(f'Depth Map Visualization {partial_path_elements[-1]}')

            # save_dir = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/datasets"

            # # Ensure the save directory exists
            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir)

            # save_path = os.path.join(save_dir, partial_path_elements[-1] + '.png')
            # plt.savefig(save_path)
            # plt.close()
            # print(f"Plot saved to: {save_path}")
        
            # print(f"{file_path} data shape: {data.shape}")
            # print(f"{file_path} data: {data}")
            # raise Exception
        return data
        # return np.load(file_path, allow_pickle=True)
    else:
        print(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

def verify_pose_format(pose):
    """Check if the pose matrix is in the correct format."""
    if pose.shape != (4, 4):
        print("Pose matrix format error: Expected 4x4, got", pose.shape)
        return False
    if not np.isclose(pose[3, :], [0, 0, 0, 1]).all():
        print("Pose matrix bottom row error: Expected [0, 0, 0, 1], got", pose[3, :])
        return False
    return True

def calculate_relative_transformation(pose_i, pose_j):
    """Calculate the relative transformation matrix between two poses."""
    # Check pose format correctness
    if not (verify_pose_format(pose_i) and verify_pose_format(pose_j)):
        return None
    # Compute the relative transformation from frame j to frame i
    relative_transform = np.linalg.inv(pose_j) @ pose_i
    return relative_transform


def calculate_overlap_for_pair(args):
    defaultVal = -1.0
    # print("calculating overlap for pairs!!!")
    i, j, depth_paths, poses, intrinsics = args
    # print(f"Calculating overlap for pair {i}-{j}")
    # print(f"Depth Path i: {depth_paths[i]}")
    # print(f"Depth Path j: {depth_paths[j]}")

        # Handle None values directly within the function
    if depth_paths[i] is None or depth_paths[j] is None:
        # print(f"Depth data missing for pair {i}-{j}")
        return i, j, defaultVal

    depth_i = load_npy_file(depth_paths[i])
    pose_i = poses[i]
    depth_j = load_npy_file(depth_paths[j])
    pose_j = poses[j]
    
    if depth_i is None or depth_j is None or pose_i is None or pose_j is None:
        return i, j, defaultVal  # Return -1 overlap if data is missing
    
    """
    Potential Issue: If the poses are defined differently (e.g., camera-to-world instead of world-to-camera), you would need to adjust this calculation.
    """
    if not verify_pose_format(pose_i) or not verify_pose_format(pose_j):
        print(f"Invalid pose format for frames {i} or {j}")
        return i, j, defaultVal
    
    relative_pose = calculate_relative_transformation(pose_i, pose_j)
    count, total = project_points(depth_i, intrinsics[i], relative_pose)

    overlap = count / total if total > 0 else defaultVal
    return i, j, overlap
    
    # pose = np.linalg.inv(pose_j) @ pose_i
    # count, total = project_points(depth_i, intrinsics[i], pose)
    # overlap = count / total
    # # print("returning overlap values")
    # overlap = count / total if total > 0 else defaultVal
    # return i, j, overlap



depth_path = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory0204_depth.npy"
ground_overlap_path = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory0204_groundOverlap.npy"
intrinsics_path = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory0204_intrinsics.npy"
poses_path = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory0204_poses.npy"

depth = np.load(depth_path, allow_pickle=True)
ground_overlap = np.load(ground_overlap_path, allow_pickle=True)
intrinsics = np.load(intrinsics_path, allow_pickle=True)
poses = np.load(poses_path, allow_pickle=True)

print(f"Depth Shape: {depth.shape}")
print(f"Poses Shape: {poses.shape}")
print(f"Intrinsics Shape: {intrinsics.shape}")
print(f"Ground Overlap Shape: {ground_overlap.shape}")

base_dir = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/overlapTesting"
var = "0204_depth"

# "['image_paths', 'depth_paths', 'intrinsics', 'poses', 'overlap_matrix', 'scale_ratio_matrix', 'angles', 'n_points3D', 'points3D_id_to_2D', 'points3D_id_to_ndepth']"

# i = 5
# for i in range(depth.shape[0]):
#     if depth[i] is not None:
#         print(f"{i} and depth {depth[i]}")


def save_depth_data(base_dir, var, index, depth_path):
    import os
    # Ensure the base directory exists
    os.makedirs(base_dir, exist_ok=True)
    
    # Load data from file
    data = load_npy_file(depth_path)
    if data is not None:
        # Save the numpy array
        save_path = f"{base_dir}/{var}_{index}.npy"
        np.save(save_path, data, allow_pickle=True)
        print(f"Data saved to {save_path}")
    else:
        print(f"Failed to load data from {depth_path}")

"""
i want to for loop from 0 to 300 and then i wil ceheck in this path: /homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/overlapTesting .. for a file in theformat of f"{var}_{i}" (these are defined ebfore and the i comes form the for loop. if no file exists then 
"""

###### Save shapes of all the depth arrays
# directory = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/overlapTesting"

# # List to store the shapes
# shapes = []

# # Loop through all files in the directory
# for file in os.listdir(directory):
#     if file.endswith(".npy"):
#         # Load the .npy file
#         data = np.load(os.path.join(directory, file))
#         # Append the shape of the numpy array to the list
#         shapes.append(data.shape)
#         print(file, data.shape)

# # Convert list of shapes to a numpy array
# shapes_array = np.array(shapes)

# # Save the numpy array with shapes to a .npy file
# save_path = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/depthDims"
# np.save(save_path, shapes_array)
# var = np.load("/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/overlapTesting/0204_depth_15.npy", allow_pickle=True)
# print(var)
# print(var.shape)
#####

#### save depth data for al the paths in depth
# for i in range(depth.shape[0]):
#     if depth[i] is not None:
#         save_depth_data(base_dir, var, i, depth[i])
#     else:
#         print(f"Depth at index {i} is None")

####### Finds taht the None values for depth are -1 in ground truth overlap matrix
# for i in range(depth.shape[0]):
#     for j in range(depth.shape[0]):
#         if ground_overlap[i][j] != -1:
#             print(f"Depth value: {depth[i]}, Overlap matrix value: {ground_overlap[i][j]}, Index: ({i}, {j})")
#         # if depth[i] is None and ground_overlap[i][j] != -1:
        #     print(f"Depth value: {depth[i]}, Overlap matrix value: {ground_overlap[i][j]}, Index: ({i}, {j})")


#### Calculate overlap matrix
# print(depth, poses, intrinsics)
# print(ground_overlap)
expNum = "changedOverlapPose"
path = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/overlapTesting/0204_calculated_overlap"
full_path = path + expNum
# overlap = calculate_overlap_matrix(depth, poses, intrinsics)
# print(overlap)
# print(ground_overlap)
# np.save(f"{full_path}", overlap, allow_pickle=True)
# print(f"saved {expNum} to {full_path}.npy")

# overlap = calculate_overlap_matrix(np.array(filtered_depth_paths), np.array(filtered_poses), np.array(filtered_intrinsics))
# print(overlap)


# # and ground_overlap

# for i in range(calc.shape[0]):
#     for j in range(calc.shape[0]):
#         if calc[i][j] != ground_overlap[i][j]:
#             if (calc[i][j] != -1 or calc[i][j] != 0) and (ground_overlap[i][j] != -1 or ground_overlap[i][j] != 0):
#                 print(f"Calculated overlap: {calc[i][j]}, Ground truth overlap: {ground_overlap[i][j]}, Index: ({i}, {j})")
def count_matching_values(calculated_matrix, ground_truth_matrix):
    """Count the number of matching values in two matrices, excluding -1 and 0."""
    # Ensure the matrices have the same shape
    if calculated_matrix.shape != ground_truth_matrix.shape:
        raise ValueError("Matrices must have the same dimensions.")
    
    count = 0
    # Iterate over all elements in the matrices
    for i in range(calculated_matrix.shape[0]):
        for j in range(calculated_matrix.shape[1]):
            if calculated_matrix[i, j] == ground_truth_matrix[i, j] and calculated_matrix[i, j] not in [-1, 0]:
                count += 1
    
    return count

import numpy as np
import matplotlib.pyplot as plt

path = full_path + ".npy"
calc = np.load(path, allow_pickle=True)

calculated_overlap_matrix = calc


path2 = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/overlapTesting/0204_calculated_overlapNew1.npy"
usingGround = True

if not usingGround:
    ground_truth_overlap_matrix = np.load(path2, allow_pickle=True)
    secondTitle = path2.split("/")[-1].split(".")[0]
else:
    secondTitle = "ground truth"
    ground_truth_overlap_matrix = ground_overlap

titleStr = path.split("/")[-1].split(".")[0] + " vs " + secondTitle

# Assume 'calculated_matrix' and 'ground_truth_matrix' are your numpy arrays for the matrices
calculated_matrix = np.array(calculated_overlap_matrix)
ground_truth_matrix = np.array(ground_truth_overlap_matrix)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
cmap = "viridis"  # Color map for visualization

# Plot calculated overlap matrix
cax = ax[0].imshow(calculated_matrix, cmap=cmap, interpolation='nearest')
ax[0].set_title('Calculated Overlap Matrix')
fig.colorbar(cax, ax=ax[0])

# Plot ground truth overlap matrix
gax = ax[1].imshow(ground_truth_matrix, cmap=cmap, interpolation='nearest')
ax[1].set_title('Ground Truth Overlap Matrix')
fig.colorbar(gax, ax=ax[1])

# Save the figure
plt.title(titleStr)
plt.savefig(f'/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/overlapTesting/overlap_matrices{expNum}.png')
plt.close()  # Close the plot to free up memory

comparison = np.isclose(calculated_matrix, ground_truth_matrix, atol=0.01)

# Find indices where comparison is False and print corresponding values
false_indices = np.where(comparison == False)
print("Indices and Values where comparison is False:")
for idx in zip(false_indices[0], false_indices[1]):
    calc_value = calculated_matrix[idx]
    truth_value = ground_truth_matrix[idx]
    print(f"Index: {idx}, Calculated Value: {calc_value}, Ground Truth Value: {truth_value}")

print("Comparison Result (True means close enough):")
print()

count = count_matching_values(calculated_matrix, ground_truth_matrix)
print("Number of matching values (excluding -1 and 0):", count)

print(f"Plot saved to: /homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/overlapTesting/overlap_matrices{expNum}.png")


#######################

# def load_and_compare_matrices():
#     # Paths to the saved matrices
#     """
#     Calculated overlap matrix saved to: /homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/datasets/0204_calculated_overlap.npy
#     Full overlap matrix saved to: /homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/datasets/0204_full_overlap.npy
#     Ground truth overlap matrix saved to: /homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/datasets/0204_ground_truth_overlap.npy
#     """
#     # raw_path = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/datasets/0204_calculated_overlap.npy"
#     calculated_path = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/datasets/0204_full_overlap.npy"
#     ground_truth_path = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/datasets/0204_ground_truth_overlap.npy"

#     # Load matrices
#     # calculated_overlap_matrix = np.load(raw_path)
#     calculated_matrix = np.load(calculated_path)
#     ground_truth_matrix = np.load(ground_truth_path)

#     # Print shapes and comparison result
#     # print(f"calculated_overlap_matrix Matrix Shape: {calculated_overlap_matrix.shape}")
#     print(f"Calculated Matrix Shape: {calculated_matrix.shape}")
#     print(f"Ground Truth Matrix Shape: {ground_truth_matrix.shape}")
#     # Default val should ahve been -1?
   
#     print()
#     # print(calculated_overlap_matrix)
#     print()
#     print(calculated_matrix)
#     print()
#     print(ground_truth_matrix)
#     print()
#     comparison = np.isclose(calculated_matrix, ground_truth_matrix, atol=0.01)
#     print("Comparison Result (True means close enough):")
#     print()

#     # Find indices where comparison is False and print corresponding values
#     false_indices = np.where(comparison == False)
#     print("Indices and Values where comparison is False:")
#     for idx in zip(false_indices[0], false_indices[1]):
#         calc_value = calculated_matrix[idx]
#         truth_value = ground_truth_matrix[idx]
#         print(f"Index: {idx}, Calculated Value: {calc_value}, Ground Truth Value: {truth_value}")

        
#     # # Visualization
#     # plt.figure(figsize=(12, 6))

#     # plt.subplot(1, 2, 1)
#     # plt.imshow(calculated_matrix, cmap='viridis', interpolation='nearest')
#     # plt.colorbar()
#     # plt.title('Calculated Overlap Matrix')

#     # plt.subplot(1, 2, 2)
#     # plt.imshow(ground_truth_matrix, cmap='viridis', interpolation='nearest')
#     # plt.colorbar()
#     # plt.title('Ground Truth Overlap Matrix')

#     # plt.show()

# if __name__ == "__main__":
#     load_and_compare_matrices()


# def depth_to_pointcloud(depth, intrinsics):
#     """Convert a depth map to a 3D point cloud using the camera intrinsics."""
#     height, width = depth.shape
#     fx, fy = intrinsics[0, 0], intrinsics[1, 1]
#     cx, cy = intrinsics[0, 2], intrinsics[1, 2]
#     i, j = np.indices((height, width))
#     x = (j - cx) / fx * depth
#     y = (i - cy) / fy * depth
#     z = depth
#     points = np.stack((x, y, z), axis=-1)
#     return points.reshape(-1, 3)

# def transform_points(points, pose):
#     """Transform 3D points using a given pose matrix (camera extrinsics)."""
#     points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])  # Convert to homogeneous coordinates
#     transformed_points = pose @ points_homogeneous.T  # Apply transformation
#     return transformed_points.T[:, :3]

# def project_to_image_plane(points, intrinsics):
#     """Project 3D points onto an image plane using the camera intrinsics."""
#     fx, fy = intrinsics[0, 0], intrinsics[1, 1]
#     cx, cy = intrinsics[0, 2], intrinsics[1, 2]
#     x = points[:, 0] * fx / points[:, 2] + cx
#     y = points[:, 1] * fy / points[:, 2] + cy
#     return np.vstack((x, y)).T

# def calculate_overlap_for_pair(args):
#     """Calculate the overlap of projected points from one frame to another."""
#     i, j, depth_paths, poses, intrinsics = args
#     defaultVal = -1.0

#     if depth_paths[i] is None or depth_paths[j] is None:
#         return i, j, defaultVal

#     depth_i = load_npy_file(depth_paths[i])
#     pose_i = poses[i]
#     intrinsic_i = intrinsics[i]
#     depth_j = load_npy_file(depth_paths[j])
#     pose_j = poses[j]
#     intrinsic_j = intrinsics[j]

#     if any(map(lambda x: x is None, [depth_i, depth_j, pose_i, pose_j])):
#         return i, j, defaultVal

#     # Process first frame
#     points_i = depth_to_pointcloud(depth_i, intrinsic_i)
#     points_i_transformed = transform_points(points_i, np.linalg.inv(pose_i) @ pose_j)
#     projected_points_i = project_to_image_plane(points_i_transformed, intrinsic_j)

#     # Check if points are within the image bounds
#     valid_i = (projected_points_i[:, 0] >= 0) & (projected_points_i[:, 0] < depth_j.shape[1]) & \
#               (projected_points_i[:, 1] >= 0) & (projected_points_i[:, 1] < depth_j.shape[0])

#     overlap = np.sum(valid_i) / len(projected_points_i) if len(projected_points_i) > 0 else defaultVal
#     return i, j, overlap

# def calculate_overlap_matrix(depth_paths, poses, intrinsics):
#     """Calculate the overlap matrix for all pairs of frames."""
#     num_frames = len(depth_paths)
#     overlap_matrix = np.full((num_frames, num_frames), -1.0)
#     pairs = [(i, j, depth_paths, poses, intrinsics) for i in range(num_frames) for j in range(i + 1, num_frames)]

#     with Pool() as pool:
#         results = pool.map(calculate_overlap_for_pair, pairs)

#     for i, j, overlap in results:
#         overlap_matrix[i, j] = overlap
#         overlap_matrix[j, i] = overlap  # Make symmetric

#     return overlap_matrix

# overlap = calculate_overlap_matrix(depth, poses, intrinsics)
# print(overlap)
# np.save("/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/overlapTesting/0204_calculated_overlapNew1", overlap, allow_pickle=True)