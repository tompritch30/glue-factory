import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from torchvision import transforms as T
from torchvision.transforms import Compose, ToTensor

from scipy.spatial.transform import Rotation as R

from gluefactory.geometry.wrappers import Camera, Pose

# Import Camera class from wrappers.py
# from .wrappers import Camera
# def image2cam(calibration_matrix, points):
#     """
#     Transform image coordinates to camera coordinates.
#     points: [N, 2] image coordinates
#     Returns: [N, 3] camera coordinates
#     """
#     # Convert points to homogeneous coordinates by adding a third column of ones
#     ones = torch.ones((points.shape[0], 1), device=points.device)
#     points_homogeneous = torch.cat([points, ones], dim=1)
#
#     # Invert the calibration matrix to get camera coordinates
#     calibration_matrix_inv = torch.inverse(calibration_matrix)
#     points_cam = torch.matmul(calibration_matrix_inv, points_homogeneous.T).T
#
#     return points_cam
#
#
# def cam2image(calibration_matrix, points):
#     """
#     Transform camera coordinates to image coordinates.
#     points: [N, 3] camera coordinates
#     Returns: [N, 2] image coordinates
#     """
#     # Project points from 3D to 2D by dividing by the z coordinate
#     points = points / points[:, 2].unsqueeze(-1)
#
#     # Apply the calibration matrix to get image coordinates
#     points_image = torch.matmul(calibration_matrix, points.T).T
#
#     # Return the 2D part of the coordinates
#     return points_image[:, :2]

from .base_dataset import BaseDataset

class ForestImagesDataset(BaseDataset):
    # def __init__(self, root_dir, pose_file, depth_dir, transform=None):
    #     self.root_dir = root_dir
    #     self.pose_file = pose_file
    #     self.depth_dir = depth_dir
    #     self.transform = transform or T.Compose([T.ToTensor()])
    #     # self.transform = transform
    #     self.image_files = sorted([f for f in os.listdir(root_dir) if f.endswith('_left.png')])
    #     self.poses = self._load_poses(pose_file)

    default_conf = {
        "root_dir": "???",
        "pose_file": "???",
        "depth_dir": "???",
        "transform": None,
        "preprocessing": {
            "resize": "???",
            "side": "???",
            "square_pad": "???"
        },
        "train_split": "???",
        "train_num_per_scene": "???",
        "val_split": "???",
        "val_pairs": "???",
        "min_overlap": 0.1,
        "max_overlap": 0.7,
        "num_overlap_bins": 3,
        "read_depth": True,
        "read_image": True,
        "batch_size": 8,
        "num_workers": 2,
        "load_features": {
            "do": False,
            "path": "forest_features/{scene}.h5",
            "padding_length": 2048,
            "padding_fn": "pad_local_features"
        }
    }

    def _init(self, conf):
        self.root_dir = conf.root_dir
        self.pose_file = conf.pose_file
        self.depth_dir = conf.depth_dir
        self.transform = T.Compose([T.ToTensor()]) if conf.transform is None else conf.transform
        self.image_files = sorted([f for f in os.listdir(self.root_dir) if f.endswith('_left.png')])
        self.poses = self._load_poses(self.pose_file)


    def load_data(self):
        # Load your image files, poses, and other necessary data here
        self.image_files = sorted(os.listdir(self.root_dir))
        self.poses = np.loadtxt(self.pose_file)
        # Make sure the data lengths match
        assert len(self.image_files) == len(self.poses), "Mismatch between images and poses."

    def _load_poses(self, pose_file):
        poses = []
        with open(pose_file, 'r') as f:
            for line in f:
                pose = [float(val) for val in line.strip().split()]
                poses.append(pose)
        return poses

    def _load_depth(self, depth_file):
        depth = np.load(depth_file)
        # Ensure depth is of shape [1, H, W]
        # depth = depth[np.newaxis, ...]
        return depth
        #
        # # Ensure that we can access idx and idx+1
        # return len(self.image_files) - 1
        # # return np.load(depth_file)

    def pose_to_matrix(self, pose):
        """Convert a pose (tx, ty, tz, qx, qy, qz, qw) to a 4x4 transformation matrix."""
        tx, ty, tz, qx, qy, qz, qw = pose
        rotation = R.from_quat([qx, qy, qz, qw])
        rotation_matrix = rotation.as_matrix()

        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = [tx, ty, tz]

        return transformation_matrix

    def compute_transform_two_matrix(self, pose1, pose2):
        # Convert poses to transformation matrices
        cam0_T_w = self.pose_to_matrix(pose1)
        cam1_T_w = self.pose_to_matrix(pose2)

        # Calculate the adjusted transformation matrix from camera 1 to camera 0
        cam1_T_cam0 = cam1_T_w @ np.linalg.inv(cam0_T_w)
        # T_0to1 = cam1_T_cam0
        #
        # # Calculate the reverse transformation matrix from camera 0 to camera 1
        # T_1to0 = np.linalg.inv(T_0to1)
        #
        # return T_0to1, T_1to0
        T_0to1 = Pose.from_4x4mat(torch.tensor(cam1_T_cam0, dtype=torch.float32))

        # Calculate the reverse transformation matrix from camera 0 to camera 1
        T_1to0 = Pose.from_4x4mat(torch.tensor(np.linalg.inv(cam1_T_cam0), dtype=torch.float32))

        return T_0to1, T_1to0

    def get_dataset(self, split):
        # Load and return the appropriate dataset split
        return self

    def __len__(self):
        return len(self.image_files) - 1

    def __getitem__(self, idx):
        img_name_0 = os.path.join(self.root_dir, self.image_files[idx])
        img_name_1 = os.path.join(self.root_dir, self.image_files[idx + 1])
        depth_name_0 = os.path.join(self.depth_dir, self.image_files[idx].replace('_left.png', '_left_depth.npy'))
        depth_name_1 = os.path.join(self.depth_dir, self.image_files[idx + 1].replace('_left.png', '_left_depth.npy'))

        image_0 = Image.open(img_name_0).convert('RGB')
        image_1 = Image.open(img_name_1).convert('RGB')
        depth_0 = self._load_depth(depth_name_0)
        depth_1 = self._load_depth(depth_name_1)
        pose_0 = np.array(self.poses[idx])
        pose_1 = np.array(self.poses[idx + 1])

        if self.transform:
            image_0 = self.transform(image_0)
            image_1 = self.transform(image_1)

        T_0to1, T_1to0 = self.compute_transform_two_matrix(pose_0, pose_1)

        # Assuming a simple pinhole camera model for intrinsics
        camera_intrinsics = torch.tensor([
            [320.0, 0, 320.0],
            [0, 320.0, 240.0],
            [0, 0, 1.0]
        ], dtype=torch.float32)

        # Ensure it's in the right shape
        camera_intrinsics = camera_intrinsics.unsqueeze(0)  # Shape should be (1, 3, 3)

        # Ensure T_0to1 and T_1to0 are converted properly to tensors
        # T_0to1_tensor = torch.tensor(T_0to1, dtype=torch.float32)
        # T_1to0_tensor = torch.tensor(T_1to0, dtype=torch.float32)

        # camera0 = Camera(camera_intrinsics)
        # camera1 = Camera(camera_intrinsics)
        # # camera0_dict = {'intrinsics': camera0.intrinsics, 'other_attr': camera0.other_attr}
        # # camera1_dict = {'intrinsics': camera1.intrinsics, 'other_attr': camera1.other_attr}
        # print(type(camera0), type(camera1))

        # Convert Camera objects to dictionaries
        camera0 = Camera.from_calibration_matrix(camera_intrinsics)
        camera1 = Camera.from_calibration_matrix(camera_intrinsics)
        # camera0_dict = {'intrinsics': camera0.intrinsics, 'other_attr': camera0.other_attr}
        # camera1_dict = {'intrinsics': camera1.intrinsics, 'other_attr': camera1.other_attr}
        # print(type(camera0), type(camera1))

        sample = {
            'view0': {
                'image': image_0,
                'depth': torch.tensor(depth_0, dtype=torch.float32),
                'pose': torch.tensor(pose_0, dtype=torch.float32),
                'camera': camera0  #Camera object
            },
            'view1': {
                'image': image_1,
                'depth': torch.tensor(depth_1, dtype=torch.float32),
                'pose': torch.tensor(pose_1, dtype=torch.float32),
                'camera': camera1  #Camera object
            },
            'T_0to1': T_0to1, # torch.tensor(T_0to1, dtype=torch.float32),
            'T_1to0': T_1to0 # torch.tensor(T_1to0, dtype=torch.float32)
        }

        print("t tranform types", type(T_0to1), type(T_1to0))

        # Debugging statements
        # print(f"Sample at idx {idx}: {sample}")
        print(f"Type of camera0: {type(camera0)}")
        print(f"Type of camera1: {type(camera1)}")

        return sample

    # def __getitem__(self, idx):
    #     img_name_0 = os.path.join(self.root_dir, self.image_files[idx])
    #     img_name_1 = os.path.join(self.root_dir, self.image_files[idx + 1])
    #     depth_name_0 = os.path.join(self.depth_dir, self.image_files[idx].replace('_left.png', '_left_depth.npy'))
    #     depth_name_1 = os.path.join(self.depth_dir, self.image_files[idx + 1].replace('_left.png', '_left_depth.npy'))
    #
    #     image_0 = Image.open(img_name_0).convert('RGB')
    #     image_1 = Image.open(img_name_1).convert('RGB')
    #     depth_0 = self._load_depth(depth_name_0)
    #     depth_1 = self._load_depth(depth_name_1)
    #     pose_0 = np.array(self.poses[idx])
    #     pose_1 = np.array(self.poses[idx + 1])
    #
    #     if self.transform:
    #         image_0 = self.transform(image_0)
    #         image_1 = self.transform(image_1)
    #
    #     T_0to1, T_1to0 = self.compute_transform_two_matrix(pose_0, pose_1)
    #
    #     # Assuming a simple pinhole camera model for intrinsics
    #     # K1 = np.array([[320.0, 0, 320.0], [0, 320.0, 240.0], [0, 0, 1.0]])
    #     # Correctly define the calibration matrix
    #     camera_intrinsics = torch.tensor([
    #         [320.0, 0, 320.0],
    #         [0, 320.0, 240.0],
    #         [0, 0, 1.0]
    #     ], dtype=torch.float32)
    #
    #     # Ensure it's in the right shape
    #     camera_intrinsics = camera_intrinsics.unsqueeze(0)  # Shape should be (1, 3, 3)
    #
    #     camera0 = Camera.from_calibration_matrix(camera_intrinsics)
    #     camera1 = Camera.from_calibration_matrix(camera_intrinsics)
    #
    #     sample = {
    #         'view0': {
    #             'image': image_0,
    #             'depth': torch.tensor(depth_0, dtype=torch.float32),
    #             'pose': torch.tensor(pose_0, dtype=torch.float32),
    #             'camera': camera0
    #         },
    #         'view1': {
    #             'image': image_1,
    #             'depth': torch.tensor(depth_1, dtype=torch.float32),
    #             'pose': torch.tensor(pose_1, dtype=torch.float32),
    #             'camera': camera1
    #         },
    #         'T_0to1': torch.tensor(T_0to1, dtype=torch.float32),
    #         'T_1to0': torch.tensor(T_1to0, dtype=torch.float32)
    #     }
    #
    #     return sample

    # def __getitem__(self, idx):
    #     img_name = os.path.join(self.root_dir, self.image_files[idx])
    #     depth_name = os.path.join(self.depth_dir, self.image_files[idx].replace('_left.png', '_left_depth.npy'))
    #     image = Image.open(img_name).convert('RGB')
    #     depth = self._load_depth(depth_name)
    #     pose = np.array(self.poses[idx])
    #
    #     if self.transform:
    #         image = self.transform(image)
    #
    #     sample = {'image': image, 'pose': torch.tensor(pose, dtype=torch.float32),
    #               'depth': torch.tensor(depth, dtype=torch.float32)}
    #
    #     # sample = {'image': image, 'pose': pose, 'depth': depth}
    #
    #     return sample

# class ForestDataset(Dataset):
#     def __init__(self, root_dir, pose_file, transform=None):
#         self.root_dir = root_dir
#         self.pose_file = pose_file
#         self.transform = transform
#         self.image_files = sorted([f for f in os.listdir(root_dir) if f.endswith('_left.png')])
#         self.poses = self._load_poses(pose_file)
#
#     def _load_poses(self, pose_file):
#         poses = []
#         with open(pose_file, 'r') as f:
#             for line in f:
#                 pose = [float(val) for val in line.strip().split()]
#                 poses.append(pose)
#         return poses
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir, self.image_files[idx])
#         image = Image.open(img_name).convert('RGB')
#         pose = np.array(self.poses[idx])
#
#         if self.transform:
#             image = self.transform(image)
#
#         sample = {'image': image, 'pose': pose}
#
#         return sample


# ####### HOMOGRAPHIES #######
#
# import os
# from PIL import Image
# import torch
# from torchvision import datasets, transforms
# from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import Compose, Resize, ToTensor
# import numpy as np
# import cv2
#
# class ForestImagesDataset(Dataset):
#     def __init__(self, root, transform=None):
#         self.root = root
#         self.transform = transform if transform else transforms.Compose([
#             transforms.Resize((256, 256)),
#             transforms.ToTensor(),
#         ])
#         self.image_files = []
#         for subdir, _, files in os.walk(root):
#             self.image_files.extend([os.path.join(subdir, file) for file in files if file.endswith(('.png', '.jpg', '.jpeg'))])
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def generate_random_homography(self):
#         pts1 = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
#         pts2 = pts1 + np.random.normal(0, 0.1, pts1.shape).astype(np.float32)
#         H, _ = cv2.findHomography(pts1, pts2)
#         return H
#
#     def warp_image(self, image, H):
#         image_np = np.array(image)
#         h, w = image_np.shape[:2]
#         warped_image_np = cv2.warpPerspective(image_np, H, (w, h))
#         warped_image = Image.fromarray(warped_image_np)
#         return warped_image
#
#     def __getitem__(self, idx):
#         img_path = self.image_files[idx]
#         image = Image.open(img_path).convert('RGB')
#
#         # Generate a random homography matrix
#         H_0to1 = self.generate_random_homography()
#
#         # Apply the homography to create view1
#         warped_image = self.warp_image(image, H_0to1)
#
#         if self.transform:
#             image = self.transform(image)
#             warped_image = self.transform(warped_image)
#
#         # Return image with keys 'view0' and 'view1'
#         return {'view0': {'image': image}, 'view1': {'image': warped_image}, 'H_0to1': torch.tensor(H_0to1, dtype=torch.float64)}
#         # return {'view0': image, 'view1': warped_image, 'H_0to1': torch.tensor(H_0to1, dtype=torch.float32)}
