# import os
# from torchvision import transforms
# from torchvision.datasets import ImageFolder
#

# class ForestImagesDataset:
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform or transforms.Compose([
#             transforms.Resize((256, 256)),
#             transforms.ToTensor(),
#         ])
#         self.dataset = ImageFolder(root=self.root_dir, transform=self.transform)

# import os
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
#
# class ForestImagesDataset(datasets.ImageFolder):
#     def __init__(self, root, transform=None):  # Changed parameter name to 'root'
#         if transform is None:
#             transform = transforms.Compose([
#                 transforms.Resize((256, 256)),
#                 transforms.ToTensor(),
#             ])
#         super().__init__(root=root, transform=transform)
#
#     # def __len__(self):
#     #     return len(self.dataset)
#     #
#     # def __getitem__(self, idx):
#     #     return self.dataset[idx]

import os
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
import numpy as np
import cv2

class ForestImagesDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        self.image_files = []
        for subdir, _, files in os.walk(root):
            self.image_files.extend([os.path.join(subdir, file) for file in files if file.endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.image_files)

    def generate_random_homography(self):
        pts1 = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        pts2 = pts1 + np.random.normal(0, 0.1, pts1.shape).astype(np.float32)
        H, _ = cv2.findHomography(pts1, pts2)
        return H

    def warp_image(self, image, H):
        image_np = np.array(image)
        h, w = image_np.shape[:2]
        warped_image_np = cv2.warpPerspective(image_np, H, (w, h))
        warped_image = Image.fromarray(warped_image_np)
        return warped_image

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')

        # Generate a random homography matrix
        H_0to1 = self.generate_random_homography()

        # Apply the homography to create view1
        warped_image = self.warp_image(image, H_0to1)

        if self.transform:
            image = self.transform(image)
            warped_image = self.transform(warped_image)

        # Return image with keys 'view0' and 'view1'
        return {'view0': {'image': image}, 'view1': {'image': warped_image}, 'H_0to1': torch.tensor(H_0to1, dtype=torch.float64)}
        # return {'view0': image, 'view1': warped_image, 'H_0to1': torch.tensor(H_0to1, dtype=torch.float32)}


# class ForestImagesDataset(Dataset):
#     def __init__(self, root, transform=None):
#         self.root = root
#         self.transform = transform if transform else Compose([
#             Resize((256, 256)),
#             ToTensor(),
#         ])
#         self.image_files = []
#         for subdir, _, files in os.walk(root):
#             self.image_files.extend([os.path.join(subdir, file) for file in files if file.endswith(('.png', '.jpg', '.jpeg'))])
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         img_path = self.image_files[idx]
#         image = Image.open(img_path).convert('RGB')
#
#         # Generate a random homography matrix
#         H_0to1 = generate_random_homography()
#
#         # Apply the homography to create view1
#         warped_image = warp_image(image, H_0to1)
#
#         if self.transform:
#             image = self.transform(image)
#             arped_image = self.transform(warped_image)
#         # Return image with keys 'view0' and 'view1'
#         H_0to1 = np.eye(3, dtype=np.float32)
#         #return {'view0': {'image': image}, 'view1': {'image': warped_image}, 'H_0to1': H_0to1}
#         return {'view0': image, 'view1': warped_image, 'H_0to1': H_0to1}
#
#     def generate_random_homography():
#         """Generate a random homography matrix for synthetic data generation."""
#         pts1 = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])
#         pts2 = pts1 + np.random.normal(0, 20, pts1.shape).astype(np.float32)
#         H, _ = cv2.findHomography(pts1, pts2)
#         return H
#
#     def warp_image(image, H):
#         """Apply the homography transformation to the image."""
#         h, w = image.size
#         warped_image = cv2.warpPerspective(np.array(image), H, (w, h))
#         return Image.fromarray(warped_image)

# class ForestImagesDataset(Dataset):
#     def __init__(self, root, transform=None):
#         self.root = root
#         self.transform = transform if transform else Compose([
#             Resize((256, 256)),
#             ToTensor(),
#         ])
#         self.image_files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(('.png', '.jpg', '.jpeg'))]
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         img_path = self.image_files[idx]
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image
#
# #
# class ForestImagesDataset(Dataset):
#     def __init__(self, root, transform=None):
#         self.root = root
#         if transform is None:
#             self.transform = transforms.Compose([
#                 transforms.Resize((256, 256)),
#                 transforms.ToTensor(),
#             ])
#         else:
#             self.transform = transform
#
#         # List all image files in the root directory
#         self.image_files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(('.png', '.jpg', '.jpeg'))]
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         img_path = self.image_files[idx]
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image

# class ForestImagesDataset(datasets.ImageFolder):
#     def __init__(self, root, transform=None):
#         if transform is None:
#             transform = transforms.Compose([
#                 transforms.Resize((256, 256)),
#                 transforms.ToTensor(),
#             ])
#         super().__init__(root=root, transform=transform)
#         self.image_files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(('.png', '.jpg', '.jpeg'))]
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         img_path = self.image_files[idx]
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image

# class ForestImagesDataset(datasets.ImageFolder):
#     # def __init__(self, root_dir, transform=None):
#     #     self.root_dir = root_dir
#     #     self.transform = transform or transforms.Compose([
#     #         transforms.Resize((256, 256)),
#     #         transforms.ToTensor(),
#     #     ])
#     #     super().__init__(root=root, transform=transform)
#     #     self.image_files = [os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
#     def __init__(self, root, transform=None):
#         if transform is None:
#             transform = transforms.Compose([
#                 transforms.Resize((256, 256)),
#                 transforms.ToTensor(),
#             ])
#         super().__init__(root=root, transform=transform)
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         img_path = self.image_files[idx]
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image