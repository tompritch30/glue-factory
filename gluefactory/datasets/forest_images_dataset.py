import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class ForestDataset(Dataset):
    def __init__(self, root_dir, pose_file, transform=None):
        self.root_dir = root_dir
        self.pose_file = pose_file
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.endswith('_left.png')])
        self.poses = self._load_poses(pose_file)

    def _load_poses(self, pose_file):
        poses = []
        with open(pose_file, 'r') as f:
            for line in f:
                pose = [float(val) for val in line.strip().split()]
                poses.append(pose)
        return poses

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        pose = np.array(self.poses[idx])

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'pose': pose}

        return sample



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
