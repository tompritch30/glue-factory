import logging
from pathlib import Path

import numpy as np
import torch

from ..geometry.homography import compute_homography, warp_points
from ..settings import DATA_PATH
from ..utils.image import load_image, ImagePreprocessor
from .base_dataset import BaseDataset

from ..geometry.wrappers import Camera, Pose

logger = logging.getLogger(__name__)

class HomogTreePairsDataset(BaseDataset, torch.utils.data.Dataset):
    # default_conf = {
    #     "pairs": "syntheticForestData/pairs_info_calibrated.txt",  # Path to the pairs file
    #     "root": "syntheticForestData/imageData/SF_E_L_P007",       # Root directory for images
    #      "preprocessing": {                                          # Preprocessing configuration
    #         "resize": 256,
    #         "normalize": True,
    #     },
    #     "extra_data": "homography",                                 # Type of extra data
    # }
  
    default_conf = {
        "pairs": "???",  # ToDo: add image folder interface
        "root": "???",
        "preprocessing": ImagePreprocessor.default_conf,
        "extra_data": None,  # relative_pose, homography
    }

    def _init(self, conf):
        # Ensure the pairs file exists
        pair_f = (
            Path(conf.pairs) if Path(conf.pairs).exists() else DATA_PATH / conf.pairs
        )
        with open(str(pair_f), "r") as f:
            self.items = [line.rstrip() for line in f]
        self.preprocessor = ImagePreprocessor(conf.preprocessing)
        # self.root = Path(conf.root)

    def get_dataset(self, split):
        return self

    def _read_view(self, name):
        # Load and preprocess the image
        # path = self.root / name
        path = DATA_PATH / self.conf.root / name
        img = load_image(path)
        return self.preprocessor(img)

    def __getitem__(self, idx):
        # Get the line corresponding to the index
        line = self.items[idx]
        pair_data = line.split(" ")
        name0, name1 = pair_data[:2]
        data0 = self._read_view(name0)
        data1 = self._read_view(name1)

        # Prepare the data dictionary
        data = {
            "view0": data0,
            "view1": data1,
        }

        # Handle extra data based on configuration
        if self.conf.extra_data == "relative_pose":
            data["view0"]["camera"] = parse_camera(pair_data[2:11]).scale(data0["scales"])
            data["view1"]["camera"] = parse_camera(pair_data[11:20]).scale(data1["scales"])
            data["T_0to1"] = parse_relative_pose(pair_data[20:32])
        elif self.conf.extra_data == "homography":
            data["H_0to1"] = (
                data1["transform"]
                @ parse_homography(pair_data[2:11])
                @ np.linalg.inv(data0["transform"])
            )
            # data["H_0to1"] = np.array([float(x) for x in pair_data[20:29]]).reshape(3, 3)
        else:
            assert self.conf.extra_data is None, f"Unknown extra data format {self.conf.extra_data}"

        # Add the name pair to the data
        data["name"] = names_to_pair(name0, name1)

        return data

    def __len__(self):
        return len(self.items)

# Utility functions
def names_to_pair(name0, name1, separator="/"):
    return separator.join((name0.replace("/", "-"), name1.replace("/", "-")))

def parse_camera(calib_elems):
    K = np.array([float(x) for x in calib_elems[:9]]).reshape(3, 3).astype(np.float32)
    return Camera.from_calibration_matrix(K)

def parse_relative_pose(pose_elems):
    R, t = pose_elems[:9], pose_elems[9:12]
    R = np.array([float(x) for x in R]).reshape(3, 3).astype(np.float32)
    t = np.array([float(x) for x in t]).astype(np.float32)
    return Pose.from_Rt(R, t)

def parse_homography(homography_elems) -> Camera:
    return (
        np.array([float(x) for x in homography_elems[:9]])
        .reshape(3, 3)
        .astype(np.float32)
    )

# """
# Simply load images from a folder or nested folders (does not have any split),
# and apply homographic adaptations to it. Yields an image pair without border
# artifacts.
# """

# import argparse
# import logging
# import shutil
# import tarfile
# from pathlib import Path

# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# import omegaconf
# import torch
# from omegaconf import OmegaConf
# from tqdm import tqdm

# from ..geometry.homography import (
#     compute_homography,
#     sample_homography_corners,
#     warp_points,
# )
# from ..models.cache_loader import CacheLoader, pad_local_features
# from ..settings import DATA_PATH
# from ..utils.image import read_image
# from ..utils.tools import fork_rng
# from ..visualization.viz2d import plot_image_grid
# from .augmentations import IdentityAugmentation, augmentations
# from .base_dataset import BaseDataset

# logger = logging.getLogger(__name__)

# # python -m gluefactory.train sp+lg_homography \  --conf gluefactory/configs/superpoint+lightglue_homography_synthTree43K.yaml

# def sample_homography(img, conf: dict, size: list):
#     data = {}
#     H, _, coords, _ = sample_homography_corners(img.shape[:2][::-1], **conf)
#     ### I added
#     logger.info(f"Input image shape: {img.shape}")
#     ###
#     data["image"] = cv2.warpPerspective(img, H, tuple(size))
#     ### I added
#     logger.info(f"Output image shape: {data['image'].shape}")
#     ###
#     data["H_"] = H.astype(np.float32)
#     data["coords"] = coords.astype(np.float32)
#     data["image_size"] = np.array(size, dtype=np.float32)
#     return data


# class HomogTreePairsDataset(BaseDataset):
#     default_conf = {
#         # image search
#         "data_dir": "SynthTree43/test_set",  # the top-level directory
#         "image_dir": "",  # no subdirectory in this case
#         "image_list": "image_list.txt",  # the generated image list
#         "glob": ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"],
#         # splits
#         "train_size": 100,  # these values will be dynamically adjusted
#         "val_size": 10,     # based on your dataset
#         "shuffle_seed": 0,  # or None to skip
#         # image loading
#         "grayscale": False,
#         "triplet": False,
#         "right_only": False,  # image0 is orig (rescaled), image1 is right
#         "reseed": False,
#         "homography": {
#             "difficulty": 0.8,
#             "translation": 1.0,
#             "max_angle": 60,
#             "n_angles": 10,
#             "patch_shape": [640, 480],
#             "min_convexity": 0.05,
#         },
#         "photometric": {
#             "name": "dark",
#             "p": 0.75,
#         },
#         # feature loading
#         "load_features": {
#             "do": False,
#             **CacheLoader.default_conf,
#             "collate": False,
#             "thresh": 0.0,
#             "max_num_keypoints": -1,
#             "force_num_keypoints": False,
#         },
#         ### I added
#         "log_level": "ERROR"  # INFO = will log, ERROR = no logs, DEBUG, WARNING, CRITICAL
#     }

#     def _init(self, conf):
#         # Set the logging level based on the configuration
#         logging_level = getattr(logging, conf.log_level.upper(), logging.INFO)
#         logger.setLevel(logging_level)

#         data_dir = DATA_PATH / conf.data_dir
#         if not data_dir.exists():
#             raise FileNotFoundError(data_dir)

#         image_dir = data_dir / conf.image_dir
#         images = []
        
#         logger.info(f"Data directory: {data_dir}")
#         logger.info(f"Image directory: {image_dir}")
#         logger.info(f"Image list: {conf.image_list}")
        
#         if conf.image_list is None:
#             glob = [conf.glob] if isinstance(conf.glob, str) else conf.glob
#             for g in glob:
#                 images += list(image_dir.glob("**/" + g))
#             if len(images) == 0:
#                 raise ValueError(f"Cannot find any image in folder: {image_dir}.")
#             images = [i.relative_to(image_dir).as_posix() for i in images]
#             images = sorted(images)  # for deterministic behavior
#             logger.info("Found %d images in folder.", len(images))
#             logger.info(f"Sample images: {images[:5]}")
#         elif isinstance(conf.image_list, (str, Path)):
#             image_list = data_dir / conf.image_list
#             if not image_list.exists():
#                 raise FileNotFoundError(f"Cannot find image list {image_list}.")
#             images = image_list.read_text().rstrip("\n").split("\n")
#             for image in images:
#                 if not (image_dir / image).exists():
#                     raise FileNotFoundError(image_dir / image)
#             logger.info("Found %d images in list file.", len(images))
#             logger.info(f"Sample images: {images[:5]}")
#         elif isinstance(conf.image_list, omegaconf.listconfig.ListConfig):
#             images = conf.image_list.to_container()
#             for image in images:
#                 if not (image_dir / image).exists():
#                     raise FileNotFoundError(image_dir / image)
#             logger.info(f"Sample images: {images[:5]}")
#         else:
#             raise ValueError(conf.image_list)

#         if conf.shuffle_seed is not None:
#             np.random.RandomState(conf.shuffle_seed).shuffle(images)

#         images = sorted(images)
#         logger.info(f"Total images found: {len(images)}")

#         # Dynamically determine the split based on an 80-20 split
#         train_size = int(len(images) * 0.8)

#         # Split images into training and validation sets
#         train_images = images[:train_size]
#         val_images = images[train_size:]
#         self.images = {"train": train_images, "val": val_images}

#         logger.info(f"Loaded {len(train_images)} training and {len(val_images)} validation images.")

#     def get_dataset(self, split):
#         return _Dataset(self.conf, self.images[split], split)


# class _Dataset(torch.utils.data.Dataset):
#     def __init__(self, conf, image_names, split):
#         self.conf = conf
#         self.split = split
#         self.image_names = np.array(image_names)
#         self.image_dir = DATA_PATH / conf.data_dir / conf.image_dir

#         aug_conf = conf.photometric
#         aug_name = aug_conf.name
#         assert (
#             aug_name in augmentations.keys()
#         ), f'{aug_name} not in {" ".join(augmentations.keys())}'
#         self.photo_augment = augmentations[aug_name](aug_conf)
#         self.left_augment = (
#             IdentityAugmentation() if conf.right_only else self.photo_augment
#         )
#         self.img_to_tensor = IdentityAugmentation()

#         if conf.load_features.do:
#             self.feature_loader = CacheLoader(conf.load_features)
        
#         ### I added
#         logger.info(f"Photometric augmentation: {aug_name}")
#         logger.info(f"Grayscale: {conf.grayscale}")
#         logger.info(f"Loading features: {conf.load_features.do}")

#     def _transform_keypoints(self, features, data):
#         """Transform keypoints by a homography, threshold them,
#         and potentially keep only the best ones."""
#         # Warp points
#         features["keypoints"] = warp_points(
#             features["keypoints"], data["H_"], inverse=False
#         )
#         h, w = data["image"].shape[1:3]
#         valid = (
#             (features["keypoints"][:, 0] >= 0)
#             & (features["keypoints"][:, 0] <= w - 1)
#             & (features["keypoints"][:, 1] >= 0)
#             & (features["keypoints"][:, 1] <= h - 1)
#         )
#         features["keypoints"] = features["keypoints"][valid]

#         # Threshold
#         if self.conf.load_features.thresh > 0:
#             valid = features["keypoint_scores"] >= self.conf.load_features.thresh
#             features = {k: v[valid] for k, v in features.items()}

#         # Get the top keypoints and pad
#         n = self.conf.load_features.max_num_keypoints
#         if n > -1:
#             inds = np.argsort(-features["keypoint_scores"])
#             features = {k: v[inds[:n]] for k, v in features.items()}

#             if self.conf.load_features.force_num_keypoints:
#                 features = pad_local_features(
#                     features, self.conf.load_features.max_num_keypoints
#                 )

#         return features

#     def __getitem__(self, idx):
#         if self.conf.reseed:
#             with fork_rng(self.conf.seed + idx, False):
#                 return self.getitem(idx)
#         else:
#             return self.getitem(idx)

#     def _read_view(self, img, H_conf, ps, left=False):
#         data = sample_homography(img, H_conf, ps)
#         if left:
#             data["image"] = self.left_augment(data["image"], return_tensor=True)
#         else:
#             data["image"] = self.photo_augment(data["image"], return_tensor=True)

#         gs = data["image"].new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
#         if self.conf.grayscale:
#             data["image"] = (data["image"] * gs).sum(0, keepdim=True)

#         if self.conf.load_features.do:
#             features = self.feature_loader({k: [v] for k, v in data.items()})
#             features = self._transform_keypoints(features, data)
#             data["cache"] = features

#         return data

#     def getitem(self, idx):
#         name = self.image_names[idx]
#         img = read_image(self.image_dir / name, False)
        

#         if img is None:
#             logging.warning("Image %s could not be read.", name)
#             img = np.zeros((1024, 1024) + (() if self.conf.grayscale else (3,)))
#         img = img.astype(np.float32) / 255.0
#         size = img.shape[:2][::-1]
#         ps = self.conf.homography.patch_shape

#         ### I added
#         logger.info(f"Loading image {name}")
#         logger.info(f"Original image shape: {img.shape}")

#         left_conf = omegaconf.OmegaConf.to_container(self.conf.homography)
#         if self.conf.right_only:
#             left_conf["difficulty"] = 0.0

#         data0 = self._read_view(img, left_conf, ps, left=True)
#         data1 = self._read_view(img, self.conf.homography, ps, left=False)

#         H = compute_homography(data0["coords"], data1["coords"], [1, 1])

#         ### I added
#         logger.info(f"Homography matrix: {H}")
#         logger.info(f"View0 shape: {data0['image'].shape}")
#         logger.info(f"View1 shape: {data1['image'].shape}")

#         data = {
#             "name": name,
#             "original_image_size": np.array(size),
#             "H_0to1": H.astype(np.float32),
#             "idx": idx,
#             "view0": data0,
#             "view1": data1,
#         }

#         if self.conf.triplet:
#             # Generate third image
#             data2 = self._read_view(img, self.conf.homography, ps, left=False)
#             H02 = compute_homography(data0["coords"], data2["coords"], [1, 1])
#             H12 = compute_homography(data1["coords"], data2["coords"], [1, 1])

#             data = {
#                 "H_0to2": H02.astype(np.float32),
#                 "H_1to2": H12.astype(np.float32),
#                 "view2": data2,
#                 **data,
#             }

#         return data

#     def __len__(self):
#         return len(self.image_names)


# def visualize(args):
#     conf = {
#         "batch_size": 1,
#         "num_workers": 1,
#         "prefetch_factor": 1,
#     }
#     conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.dotlist))
#     dataset = HomographySynthTreeDataset(conf)
#     loader = dataset.get_data_loader("train")
#     logger.info("The dataset has %d elements.", len(loader))

#     with fork_rng(seed=dataset.conf.seed):
#         images = []
#         for _, data in zip(range(args.num_items), loader):
#             images.append(
#                 [data[f"view{i}"]["image"][0].permute(1, 2, 0) for i in range(2)]
#             )
#     plot_image_grid(images, dpi=args.dpi)
#     plt.tight_layout()
#     plt.show()



# if __name__ == "__main__":
#     from .. import logger  # overwrite the logger

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--num_items", type=int, default=8)
#     parser.add_argument("--dpi", type=int, default=100)
#     parser.add_argument("dotlist", nargs="*")
#     args = parser.parse_intermixed_args()
#     visualize(args)
