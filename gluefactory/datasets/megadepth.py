# import argparse
# import logging
# import shutil
# import tarfile
# from collections.abc import Iterable
# from pathlib import Path

# import h5py
# import matplotlib.pyplot as plt
# import numpy as np
# import PIL.Image
# import torch
# from omegaconf import OmegaConf

# from ..geometry.wrappers import Camera, Pose
# from ..models.cache_loader import CacheLoader
# from ..settings import DATA_PATH
# from ..utils.image import ImagePreprocessor, load_image
# from ..utils.tools import fork_rng
# from ..visualization.viz2d import plot_heatmaps, plot_image_grid
# from .base_dataset import BaseDataset
# from .utils import rotate_intrinsics, rotate_pose_inplane, scale_intrinsics

# logger = logging.getLogger(__name__)
# scene_lists_path = Path(__file__).parent / "megadepth_scene_lists"

# """python -m gluefactory.train sp+lg_megadepth \
#     --conf gluefactory/configs/superpoint+lightglue_megadepth.yaml \
#     train.load_experiment=sp+lg_homography
# """
    
# def sample_n(data, num, seed=None):
#     if len(data) > num:
#         selected = np.random.RandomState(seed).choice(len(data), num, replace=False)
#         return data[selected]
#     else:
#         return data


# class MegaDepth(BaseDataset):
#     default_conf = {
#         # paths
#         "data_dir": "megadepth/",
#         "depth_subpath": "depth_undistorted/",
#         "image_subpath": "Undistorted_SfM/",
#         "info_dir": "scene_info/",  # @TODO: intrinsics problem?
#         # Training
#         "train_split": "train_scenes_clean.txt",
#         "train_num_per_scene": 500,
#         # Validation
#         "val_split": "valid_scenes_clean.txt",
#         "val_num_per_scene": None,
#         "val_pairs": None,
#         # Test
#         "test_split": "test_scenes_clean.txt",
#         "test_num_per_scene": None,
#         "test_pairs": None,
#         # data sampling
#         "views": 2,
#         "min_overlap": 0.3,  # only with D2-Net format
#         "max_overlap": 1.0,  # only with D2-Net format
#         "num_overlap_bins": 1,
#         "sort_by_overlap": False,
#         "triplet_enforce_overlap": False,  # only with views==3
#         # image options
#         "read_depth": True,
#         "read_image": True,
#         "grayscale": False,
#         "preprocessing": ImagePreprocessor.default_conf,
#         "p_rotate": 0.0,  # probability to rotate image by +/- 90°
#         "reseed": False,
#         "seed": 0,
#         # features from cache
#         "load_features": {
#             "do": False,
#             **CacheLoader.default_conf,
#             "collate": False,
#         },
#     }

#     def _init(self, conf):
#         if not (DATA_PATH / conf.data_dir).exists():
#             logger.info("Downloading the MegaDepth dataset.")
#             self.download()

#     def download(self):
#         data_dir = DATA_PATH / self.conf.data_dir
#         tmp_dir = data_dir.parent / "megadepth_tmp"
#         if tmp_dir.exists():  # The previous download failed.
#             shutil.rmtree(tmp_dir)
#         tmp_dir.mkdir(exist_ok=True, parents=True)
#         url_base = "https://cvg-data.inf.ethz.ch/megadepth/"
#         for tar_name, out_name in (
#             ("Undistorted_SfM.tar.gz", self.conf.image_subpath),
#             ("depth_undistorted.tar.gz", self.conf.depth_subpath),
#             ("scene_info.tar.gz", self.conf.info_dir),
#         ):
#             tar_path = tmp_dir / tar_name
#             torch.hub.download_url_to_file(url_base + tar_name, tar_path)
#             with tarfile.open(tar_path) as tar:
#                 tar.extractall(path=tmp_dir)
#             tar_path.unlink()
#             shutil.move(tmp_dir / tar_name.split(".")[0], tmp_dir / out_name)
#         shutil.move(tmp_dir, data_dir)

#     def get_dataset(self, split):
#         assert self.conf.views in [1, 2, 3]
#         if self.conf.views == 3:
#             return _TripletDataset(self.conf, split)
#         else:
#             return _PairDataset(self.conf, split)


# class _PairDataset(torch.utils.data.Dataset):
#     def __init__(self, conf, split, load_sample=True):
#         self.root = DATA_PATH / conf.data_dir
#         assert self.root.exists(), self.root
#         self.split = split
#         self.conf = conf

#         split_conf = conf[split + "_split"]
#         if isinstance(split_conf, (str, Path)):
#             scenes_path = scene_lists_path / split_conf
#             scenes = scenes_path.read_text().rstrip("\n").split("\n")
#         elif isinstance(split_conf, Iterable):
#             scenes = list(split_conf)
#         else:
#             raise ValueError(f"Unknown split configuration: {split_conf}.")
#         scenes = sorted(set(scenes))

#         if conf.load_features.do:
#             self.feature_loader = CacheLoader(conf.load_features)

#         self.preprocessor = ImagePreprocessor(conf.preprocessing)

#         self.images = {}
#         self.depths = {}
#         self.poses = {}
#         self.intrinsics = {}
#         self.valid = {}

#         # load metadata
#         self.info_dir = self.root / self.conf.info_dir
#         self.scenes = []
#         for scene in scenes:
#             path = self.info_dir / (scene + ".npz")
#             try:
#                 info = np.load(str(path), allow_pickle=True)
#             except Exception:
#                 logger.warning(
#                     "Cannot load scene info for scene %s at %s.", scene, path
#                 )
#                 continue
#             self.images[scene] = info["image_paths"]
#             self.depths[scene] = info["depth_paths"]
#             self.poses[scene] = info["poses"]
#             self.intrinsics[scene] = info["intrinsics"]
#             self.scenes.append(scene)

#         if load_sample:
#             self.sample_new_items(conf.seed)
#             assert len(self.items) > 0

#     def sample_new_items(self, seed):
#         logger.info("Sampling new %s data with seed %d.", self.split, seed)
#         self.items = []
#         split = self.split
#         num_per_scene = self.conf[self.split + "_num_per_scene"]
#         if isinstance(num_per_scene, Iterable):
#             num_pos, num_neg = num_per_scene
#         else:
#             num_pos = num_per_scene
#             num_neg = None
#         if split != "train" and self.conf[split + "_pairs"] is not None:
#             # Fixed validation or test pairs
#             assert num_pos is None
#             assert num_neg is None
#             assert self.conf.views == 2
#             pairs_path = scene_lists_path / self.conf[split + "_pairs"]
#             for line in pairs_path.read_text().rstrip("\n").split("\n"):
#                 im0, im1 = line.split(" ")
#                 scene = im0.split("/")[0]
#                 assert im1.split("/")[0] == scene
#                 im0, im1 = [self.conf.image_subpath + im for im in [im0, im1]]
#                 assert im0 in self.images[scene]
#                 assert im1 in self.images[scene]
#                 idx0 = np.where(self.images[scene] == im0)[0][0]
#                 idx1 = np.where(self.images[scene] == im1)[0][0]
#                 self.items.append((scene, idx0, idx1, 1.0))
#         elif self.conf.views == 1:
#             for scene in self.scenes:
#                 if scene not in self.images:
#                     continue
#                 valid = (self.images[scene] != None) | (  # noqa: E711
#                     self.depths[scene] != None  # noqa: E711
#                 )
#                 ids = np.where(valid)[0]
#                 if num_pos and len(ids) > num_pos:
#                     ids = np.random.RandomState(seed).choice(
#                         ids, num_pos, replace=False
#                     )
#                 ids = [(scene, i) for i in ids]
#                 self.items.extend(ids)
#         else:
#             for scene in self.scenes:
#                 path = self.info_dir / (scene + ".npz")
#                 assert path.exists(), path
#                 info = np.load(str(path), allow_pickle=True)
#                 valid = (self.images[scene] != None) & (  # noqa: E711
#                     self.depths[scene] != None  # noqa: E711
#                 )
#                 ind = np.where(valid)[0]
#                 mat = info["overlap_matrix"][valid][:, valid]

#                 if num_pos is not None:
#                     # Sample a subset of pairs, binned by overlap.
#                     num_bins = self.conf.num_overlap_bins
#                     assert num_bins > 0
#                     bin_width = (
#                         self.conf.max_overlap - self.conf.min_overlap
#                     ) / num_bins
#                     num_per_bin = num_pos // num_bins
#                     pairs_all = []
#                     for k in range(num_bins):
#                         bin_min = self.conf.min_overlap + k * bin_width
#                         bin_max = bin_min + bin_width
#                         pairs_bin = (mat > bin_min) & (mat <= bin_max)
#                         pairs_bin = np.stack(np.where(pairs_bin), -1)
#                         pairs_all.append(pairs_bin)
#                     # Skip bins with too few samples
#                     has_enough_samples = [len(p) >= num_per_bin * 2 for p in pairs_all]
#                     num_per_bin_2 = num_pos // max(1, sum(has_enough_samples))
#                     pairs = []
#                     for pairs_bin, keep in zip(pairs_all, has_enough_samples):
#                         if keep:
#                             pairs.append(sample_n(pairs_bin, num_per_bin_2, seed))
#                     pairs = np.concatenate(pairs, 0)
#                 else:
#                     pairs = (mat > self.conf.min_overlap) & (
#                         mat <= self.conf.max_overlap
#                     )
#                     pairs = np.stack(np.where(pairs), -1)

#                 pairs = [(scene, ind[i], ind[j], mat[i, j]) for i, j in pairs]
#                 if num_neg is not None:
#                     neg_pairs = np.stack(np.where(mat <= 0.0), -1)
#                     neg_pairs = sample_n(neg_pairs, num_neg, seed)
#                     pairs += [(scene, ind[i], ind[j], mat[i, j]) for i, j in neg_pairs]
#                 self.items.extend(pairs)
#         if self.conf.views == 2 and self.conf.sort_by_overlap:
#             self.items.sort(key=lambda i: i[-1], reverse=True)
#         else:
#             np.random.RandomState(seed).shuffle(self.items)

#     def _read_view(self, scene, idx):
#         path = self.root / self.images[scene][idx]

#         # read pose data
#         K = self.intrinsics[scene][idx].astype(np.float32, copy=False)
#         T = self.poses[scene][idx].astype(np.float32, copy=False)

#         # read image
#         if self.conf.read_image:
#             img = load_image(self.root / self.images[scene][idx], self.conf.grayscale)
#         else:
#             size = PIL.Image.open(path).size[::-1]
#             img = torch.zeros(
#                 [3 - 2 * int(self.conf.grayscale), size[0], size[1]]
#             ).float()

#         # read depth
#         if self.conf.read_depth:
#             depth_path = (
#                 self.root / self.conf.depth_subpath / scene / (path.stem + ".h5")
#             )
#             with h5py.File(str(depth_path), "r") as f:
#                 depth = f["/depth"].__array__().astype(np.float32, copy=False)
#                 depth = torch.Tensor(depth)[None]
#             assert depth.shape[-2:] == img.shape[-2:]
#         else:
#             depth = None

#         # add random rotations
#         do_rotate = self.conf.p_rotate > 0.0 and self.split == "train"
#         if do_rotate:
#             p = self.conf.p_rotate
#             k = 0
#             if np.random.rand() < p:
#                 k = np.random.choice(2, 1, replace=False)[0] * 2 - 1
#                 img = np.rot90(img, k=-k, axes=(-2, -1))
#                 if self.conf.read_depth:
#                     depth = np.rot90(depth, k=-k, axes=(-2, -1)).copy()
#                 K = rotate_intrinsics(K, img.shape, k + 2)
#                 T = rotate_pose_inplane(T, k + 2)

#         name = path.name

#         data = self.preprocessor(img)
#         if depth is not None:
#             data["depth"] = self.preprocessor(depth, interpolation="nearest")["image"][
#                 0
#             ]
#         K = scale_intrinsics(K, data["scales"])

#         data = {
#             "name": name,
#             "scene": scene,
#             "T_w2cam": Pose.from_4x4mat(T),
#             "depth": depth,
#             "camera": Camera.from_calibration_matrix(K).float(),
#             **data,
#         }

#         if self.conf.load_features.do:
#             features = self.feature_loader({k: [v] for k, v in data.items()})
#             if do_rotate and k != 0:
#                 # ang = np.deg2rad(k * 90.)
#                 kpts = features["keypoints"].copy()
#                 x, y = kpts[:, 0].copy(), kpts[:, 1].copy()
#                 w, h = data["image_size"]
#                 if k == 1:
#                     kpts[:, 0] = w - y
#                     kpts[:, 1] = x
#                 elif k == -1:
#                     kpts[:, 0] = y
#                     kpts[:, 1] = h - x

#                 else:
#                     raise ValueError
#                 features["keypoints"] = kpts

#             data = {"cache": features, **data}
#         return data

#     def __getitem__(self, idx):
#         if self.conf.reseed:
#             with fork_rng(self.conf.seed + idx, False):
#                 return self.getitem(idx)
#         else:
#             return self.getitem(idx)

#     def getitem(self, idx):
#         if self.conf.views == 2:
#             if isinstance(idx, list):
#                 scene, idx0, idx1, overlap = idx
#             else:
#                 scene, idx0, idx1, overlap = self.items[idx]
#             data0 = self._read_view(scene, idx0)
#             data1 = self._read_view(scene, idx1)
#             data = {
#                 "view0": data0,
#                 "view1": data1,
#             }
#             data["T_0to1"] = data1["T_w2cam"] @ data0["T_w2cam"].inv()
#             data["T_1to0"] = data0["T_w2cam"] @ data1["T_w2cam"].inv()
#             data["overlap_0to1"] = overlap
#             data["name"] = f"{scene}/{data0['name']}_{data1['name']}"
#         else:
#             assert self.conf.views == 1
#             scene, idx0 = self.items[idx]
#             data = self._read_view(scene, idx0)
#         data["scene"] = scene
#         data["idx"] = idx
#         return data

#     def __len__(self):
#         return len(self.items)


# class _TripletDataset(_PairDataset):
#     def sample_new_items(self, seed):
#         logging.info("Sampling new triplets with seed %d", seed)
#         self.items = []
#         split = self.split
#         num = self.conf[self.split + "_num_per_scene"]
#         if split != "train" and self.conf[split + "_pairs"] is not None:
#             if Path(self.conf[split + "_pairs"]).exists():
#                 pairs_path = Path(self.conf[split + "_pairs"])
#             else:
#                 pairs_path = DATA_PATH / "configs" / self.conf[split + "_pairs"]
#             for line in pairs_path.read_text().rstrip("\n").split("\n"):
#                 im0, im1, im2 = line.split(" ")
#                 assert im0[:4] == im1[:4]
#                 scene = im1[:4]
#                 idx0 = np.where(self.images[scene] == im0)
#                 idx1 = np.where(self.images[scene] == im1)
#                 idx2 = np.where(self.images[scene] == im2)
#                 self.items.append((scene, idx0, idx1, idx2, 1.0, 1.0, 1.0))
#         else:
#             for scene in self.scenes:
#                 path = self.info_dir / (scene + ".npz")
#                 assert path.exists(), path
#                 info = np.load(str(path), allow_pickle=True)
#                 if self.conf.num_overlap_bins > 1:
#                     raise NotImplementedError("TODO")
#                 valid = (self.images[scene] != None) & (  # noqa: E711
#                     self.depth[scene] != None  # noqa: E711
#                 )
#                 ind = np.where(valid)[0]
#                 mat = info["overlap_matrix"][valid][:, valid]
#                 good = (mat > self.conf.min_overlap) & (mat <= self.conf.max_overlap)
#                 triplets = []
#                 if self.conf.triplet_enforce_overlap:
#                     pairs = np.stack(np.where(good), -1)
#                     for i0, i1 in pairs:
#                         for i2 in pairs[pairs[:, 0] == i0, 1]:
#                             if good[i1, i2]:
#                                 triplets.append((i0, i1, i2))
#                     if len(triplets) > num:
#                         selected = np.random.RandomState(seed).choice(
#                             len(triplets), num, replace=False
#                         )
#                         selected = range(num)
#                         triplets = np.array(triplets)[selected]
#                 else:
#                     # we first enforce that each row has >1 pairs
#                     non_unique = good.sum(-1) > 1
#                     ind_r = np.where(non_unique)[0]
#                     good = good[non_unique]
#                     pairs = np.stack(np.where(good), -1)
#                     if len(pairs) > num:
#                         selected = np.random.RandomState(seed).choice(
#                             len(pairs), num, replace=False
#                         )
#                         pairs = pairs[selected]
#                     for idx, (k, i) in enumerate(pairs):
#                         # We now sample a j from row k s.t. i != j
#                         possible_j = np.where(good[k])[0]
#                         possible_j = possible_j[possible_j != i]
#                         selected = np.random.RandomState(seed + idx).choice(
#                             len(possible_j), 1, replace=False
#                         )[0]
#                         triplets.append((ind_r[k], i, possible_j[selected]))
#                     triplets = [
#                         (scene, ind[k], ind[i], ind[j], mat[k, i], mat[k, j], mat[i, j])
#                         for k, i, j in triplets
#                     ]
#                     self.items.extend(triplets)
#         np.random.RandomState(seed).shuffle(self.items)

#     def __getitem__(self, idx):
#         scene, idx0, idx1, idx2, overlap01, overlap02, overlap12 = self.items[idx]
#         data0 = self._read_view(scene, idx0)
#         data1 = self._read_view(scene, idx1)
#         data2 = self._read_view(scene, idx2)
#         data = {
#             "view0": data0,
#             "view1": data1,
#             "view2": data2,
#         }
#         data["T_0to1"] = data1["T_w2cam"] @ data0["T_w2cam"].inv()
#         data["T_0to2"] = data2["T_w2cam"] @ data0["T_w2cam"].inv()
#         data["T_1to2"] = data2["T_w2cam"] @ data1["T_w2cam"].inv()
#         data["T_1to0"] = data0["T_w2cam"] @ data1["T_w2cam"].inv()
#         data["T_2to0"] = data0["T_w2cam"] @ data2["T_w2cam"].inv()
#         data["T_2to1"] = data1["T_w2cam"] @ data2["T_w2cam"].inv()

#         data["overlap_0to1"] = overlap01
#         data["overlap_0to2"] = overlap02
#         data["overlap_1to2"] = overlap12
#         data["scene"] = scene
#         data["name"] = f"{scene}/{data0['name']}_{data1['name']}_{data2['name']}"
#         return data

#     def __len__(self):
#         return len(self.items)


# def visualize(args):
#     conf = {
#         "min_overlap": 0.1,
#         "max_overlap": 0.7,
#         "num_overlap_bins": 3,
#         "sort_by_overlap": False,
#         "train_num_per_scene": 5,
#         "batch_size": 1,
#         "num_workers": 0,
#         "prefetch_factor": None,
#         "val_num_per_scene": None,
#     }
#     conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.dotlist))
#     dataset = MegaDepth(conf)
#     loader = dataset.get_data_loader(args.split)
#     logger.info("The dataset has elements.", len(loader))

#     with fork_rng(seed=dataset.conf.seed):
#         images, depths = [], []
#         for _, data in zip(range(args.num_items), loader):
#             images.append(
#                 [
#                     data[f"view{i}"]["image"][0].permute(1, 2, 0)
#                     for i in range(dataset.conf.views)
#                 ]
#             )
#             depths.append(
#                 [data[f"view{i}"]["depth"][0] for i in range(dataset.conf.views)]
#             )

#     axes = plot_image_grid(images, dpi=args.dpi)
#     for i in range(len(images)):
#         plot_heatmaps(depths[i], axes=axes[i])
#     plt.show()


# if __name__ == "__main__":
#     from .. import logger  # overwrite the logger

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--split", type=str, default="val")
#     parser.add_argument("--num_items", type=int, default=4)
#     parser.add_argument("--dpi", type=int, default=100)
#     parser.add_argument("dotlist", nargs="*")
#     args = parser.parse_intermixed_args()
#     visualize(args)


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
from omegaconf import OmegaConf

from ..geometry.wrappers import Camera, Pose
from ..models.cache_loader import CacheLoader
from ..settings import DATA_PATH
from ..utils.image import ImagePreprocessor, load_image
from ..utils.tools import fork_rng
from ..visualization.viz2d import plot_heatmaps, plot_image_grid
from .base_dataset import BaseDataset
from .utils import rotate_intrinsics, rotate_pose_inplane, scale_intrinsics

logger = logging.getLogger(__name__)
scene_lists_path = Path(__file__).parent / "megadepth_scene_lists"

"""
python -m gluefactory.train sp+lg_megadepth     --conf gluefactory/configs/superpoint+lightglue_megadepth.yaml     train.load_experiment=sp+lg_homography
python -m gluefactory.train sp+lg_debug     --conf gluefactory/configs/superpoint+lightglue_megadepth.yaml     train.load_experiment=sp+lg_homography
"""

import logging, threading
from collections import defaultdict

class LimitedLogger:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, max_count=1):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LimitedLogger, cls).__new__(cls)
                cls._instance.logger = logging.getLogger(__name__)
                cls._instance.max_count = max_count
                cls._instance.msg_count = defaultdict(int)
        return cls._instance

    def log(self, *args):
        # Join all arguments into a single message string, handling multiple arguments.
        message = ' '.join(str(arg) for arg in args)
        with self._lock:
            if self.msg_count[message] < self.max_count:
                self.logger.info(message)
                self.msg_count[message] += 1


def sample_n(data, num, seed=None):
    if len(data) > num:
        selected = np.random.RandomState(seed).choice(len(data), num, replace=False)
        return data[selected]
    else:
        return data

# limited_logger = LimitedLogger()

class MegaDepth(BaseDataset):
    default_conf = {
        # paths
        "data_dir": "megadepth/",
        "depth_subpath": "depth_undistorted/",
        "image_subpath": "Undistorted_SfM/",
        "info_dir": "scene_info/",  # @TODO: intrinsics problem?
        # Training
        "train_split": "train_scenes_clean.txt",
        "train_num_per_scene": 500,
        # Validation
        "val_split": "valid_scenes_clean.txt",
        "val_num_per_scene": None,
        "val_pairs": None,
        # Test
        "test_split": "test_scenes_clean.txt",
        "test_num_per_scene": None,
        "test_pairs": None,
        # data sampling
        "views": 2,
        "min_overlap": 0.3,  # only with D2-Net format
        "max_overlap": 1.0,  # only with D2-Net format
        "num_overlap_bins": 1,
        "sort_by_overlap": False,
        "triplet_enforce_overlap": False,  # only with views==3
        # image options
        "read_depth": True,
        "read_image": True,
        "grayscale": False,
        "preprocessing": ImagePreprocessor.default_conf,
        "p_rotate": 0.0,  # probability to rotate image by +/- 90°
        "reseed": False,
        "seed": 0,
        # features from cache
        "load_features": {
            "do": False,
            **CacheLoader.default_conf,
            "collate": False,
        },
    }

    def _init(self, conf):
        if not (DATA_PATH / conf.data_dir).exists():
           #  # logger.info("Downloading the MegaDepth dataset.")
            self.download()
        ### I added
        # logger.info(f"Initialized MegaDepth dataset with configuration: {conf}")
          

    def download(self):
        data_dir = DATA_PATH / self.conf.data_dir
        tmp_dir = data_dir.parent / "megadepth_tmp"
        if tmp_dir.exists():  # The previous download failed.
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(exist_ok=True, parents=True)
        url_base = "https://cvg-data.inf.ethz.ch/megadepth/"
        for tar_name, out_name in (
            ("Undistorted_SfM.tar.gz", self.conf.image_subpath),
            ("depth_undistorted.tar.gz", self.conf.depth_subpath),
            ("scene_info.tar.gz", self.conf.info_dir),
        ):
            tar_path = tmp_dir / tar_name
            torch.hub.download_url_to_file(url_base + tar_name, tar_path)
            with tarfile.open(tar_path) as tar:
                tar.extractall(path=tmp_dir)
            tar_path.unlink()
            shutil.move(tmp_dir / tar_name.split(".")[0], tmp_dir / out_name)
        shutil.move(tmp_dir, data_dir)
       #  # logger.info("Completed downloading and extracting MegaDepth dataset.")


    def get_dataset(self, split):
        assert self.conf.views in [1, 2, 3]
        if self.conf.views == 3:
            return _TripletDataset(self.conf, split)
        else:
            return _PairDataset(self.conf, split)

def project_points(depth, intrinsics, pose):
    h, w = depth.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.flatten()

    x = (x.flatten() - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (y.flatten() - intrinsics[1, 2]) * z / intrinsics[1, 1]
    points = np.vstack((x, y, z, np.ones_like(z)))

    transformed_points = pose @ points
    transformed_points /= transformed_points[2, :]

    x_proj = intrinsics[0, 0] * transformed_points[0, :] / transformed_points[2, :] + intrinsics[0, 2]
    y_proj = intrinsics[1, 1] * transformed_points[1, :] / transformed_points[2, :] + intrinsics[1, 2]

    valid = (x_proj >= 0) & (x_proj < w) & (y_proj >= 0) & (y_proj < h)
    return np.count_nonzero(valid), valid.size

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

def calculate_overlap_for_pair(args):
    # print("calculating overlap for pairs!!!")
    i, j, depth_paths, poses, intrinsics = args
    # print(f"Calculating overlap for pair {i}-{j}")
    # print(f"Depth Path i: {depth_paths[i]}")
    # print(f"Depth Path j: {depth_paths[j]}")

    depth_i = load_npy_file(depth_paths[i])
    pose_i = poses[i]
    depth_j = load_npy_file(depth_paths[j])
    pose_j = poses[j]
    
    if depth_i is None or depth_j is None or pose_i is None or pose_j is None:
        return i, j, 0.0  # Return 0 overlap if data is missing

    pose = np.linalg.inv(pose_j) @ pose_i
    count, total = project_points(depth_i, intrinsics[i], pose)
    overlap = count / total
    # print("returning overlap values")
    return i, j, overlap

class _PairDataset(torch.utils.data.Dataset):
    def __init__(self, conf, split, load_sample=True):
        self.root = DATA_PATH / conf.data_dir
        assert self.root.exists(), self.root
        self.split = split
        self.conf = conf

        # # limited_logger.log("IN PAIR DATASET INIT\n\n")

        # # limited_logger.log("pair dataset")
        # # limited_logger.log("self.root, self.split, self.conf", self.root, self.split, self.conf)

        split_conf = conf[split + "_split"]
        # # limited_logger.log("split_conf", split_conf)

        if isinstance(split_conf, (str, Path)):
            scenes_path = scene_lists_path / split_conf
            scenes = scenes_path.read_text().rstrip("\n").split("\n")
        elif isinstance(split_conf, Iterable):
            scenes = list(split_conf)
        else:
            raise ValueError(f"Unknown split configuration: {split_conf}.")
        scenes = sorted(set(scenes))
        # limited_logger.log(len(scenes))
        # limited_logger.log("scenes", scenes)


        if conf.load_features.do:
            self.feature_loader = CacheLoader(conf.load_features)

        self.preprocessor = ImagePreprocessor(conf.preprocessing)

        self.images = {}
        self.depths = {}
        self.poses = {}
        self.intrinsics = {}
        self.valid = {}

        # load metadata
        self.info_dir = self.root / self.conf.info_dir
        # limited_logger.log("self.info_dir" , self.info_dir)
        self.scenes = []

        count = 0
        for scene in scenes:
            # print("scenes[:10]", scenes[:10])
            # if count < 1:
            #     # limited_logger.log(scene)
            # count +=1
            path = self.info_dir / (scene + ".npz")
            # print("path", path)
            
            try:
                info = np.load(str(path), allow_pickle=True)
                

                # ### I added
                # if count == 6:
                #     # logger.info(f"Loaded scene info for {scene}: {info.keys()}")
                #     # limited_logger.log(f"Loaded info for scene {scene}. Keys: {list(info.keys())}")
                #     # ['image_paths', 'depth_paths', 'intrinsics', 'poses', 'overlap_matrix', 'scale_ratio_matrix', 'angles', 'n_points3D', 'points3D_id_to_2D', 'points3D_id_to_ndepth']
                #     """
                #     image_paths: Paths to image files. [Undistorted_SfM/0012/images/2303158722_39e1c8d583_o.jpg, ...] for each image in dir P001 etc
                #     depth_paths: Paths to corresponding depth data files. H5!!? [phoenix/S6/zl548/MegaDepth_v1/0012/dense0/depths/2303158722_39e1c8d583_o.h5, ...] for each image in dir P001 etc
                #     intrinsics: Camera intrinsic parameters. for each image [array([[1.31926e+03, 0.00000e+00, 5.21000e+02], [0.00000e+00, 1.31965e+03, 8.00000e+02], [0.00000e+00, 0.00000e+00, 1.00000e+00]])] ..]
                #     poses: Camera poses or transformations. for each iamge: [None array([[ 0.94299385,  0.06705927, -0.32598414,  3.20741   ], [ 0.13619231,  0.81596702,  0.56182691, -1.21774   ],
                #                                                             [ 0.30366801, -0.57419585,  0.76031892, -1.44514   ], [ 0.        ,  0.        ,  0.        ,  1.        ]])
                #     overlap_matrix: A matrix showing overlap metrics between different images.
                #     scale_ratio_matrix, angles, n_points3D, points3D_id_to_2D, points3D_id_to_ndepth: These likely pertain to geometric transformations, alignment metrics, and 3D-to-2D point correspondences used in depth and image processing.
                #     """
                #     # limited_logger.log("info['image_paths']", info['image_paths'].shape, info['image_paths'])
                #     # limited_logger.log("info['depth_paths']", info['depth_paths'].shape, info['depth_paths'])
                #     # limited_logger.log("info['intrinsics']", info['intrinsics'].shape,info['intrinsics'])
                #     # limited_logger.log("info['poses']", info['poses'].shape, info['poses'])
                #     ## this stuff likely not needed. 
                #     # # limited_logger.log("info['overlap_matrix']", info['overlap_matrix'].shape, info['overlap_matrix'])
                #     # # limited_logger.log("info['scale_ratio_matrix']", info['scale_ratio_matrix'].shape, info['scale_ratio_matrix'])
                #     # # limited_logger.log("info['angles']", info['angles'].shape, info['angles'])
                #     # # limited_logger.log("info['n_points3D']", info['n_points3D'].shape, info['n_points3D'])

                #     # for key in info.keys():
                #     #     data = info[key]
                #     #     if data is not None:
                #     #         # limited_logger.log(f"{key}: type={type(data)}, shape={data.shape}")
                #     #         # limited_logger.log(f"Sample data from {key} (first 10 entries): {data[:1]}")  
                #     #         # limited_logger.log(f"^^^^ KEY {key}")  
                            
                #     #         # # limited_logger.log(f"{key}: type={type(data)}, shape={getattr(data, 'shape', 'N/A')}")
                #     #     else:
                #     #         # limited_logger.log(f"{key} is None")
                # print(f"Depth Paths: {info['depth_paths']}\n\n")
                # print(f"Intrinsics: {info['intrinsics']}\n\n")
                # print(f"Poses: {info['poses']}\n\n")
                
                
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
                    overlap_matrix = np.zeros((num_frames, num_frames))
                    
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
                        if i % 500 == 0:
                            print(f"Overlap between frame {i} and {j}: {overlap}")  # Print at intervals

                    return overlap_matrix

                # print(info["overlap_matrix"])

                depth_paths = info["depth_paths"]
                poses = info["poses"]
                intrinsics = info["intrinsics"]
                ground_truth_overlap_matrix = info["overlap_matrix"]

                """
                1) CHeck this in same format as expected/is for treedepth:
                                    Depth Paths: ['phoenix/S6/zl548/MegaDepth_v1/0189/dense0/depths/17873346101_bd320a0869_o.h5'
                    None None ... None None None]


                    Intrinsics: [array([[1.57790e+03, 0.00000e+00, 5.33000e+02],
                            [0.00000e+00, 1.57801e+03, 8.00000e+02],
                            [0.00000e+00, 0.00000e+00, 1.00000e+00]]) None None ... None None
                    None]


                    Poses: [array([[-0.98438464, -0.07556228, -0.15898812, -0.713126  ],
                            [ 0.07904087,  0.61728158, -0.78276177, -1.86648   ],
                            [ 0.1572877 , -0.78310522, -0.60167   , -2.36769   ],
                            [ 0.        ,  0.        ,  0.        ,  1.        ]]) None None
                    ... None None None]
                2) Check if depth path are all none, have acount
                """
                # # Count the number of non-None values in depth_paths and poses
                # depth_count = sum(1 for depth in depth_paths if depth is not None)
                # pose_count = sum(1 for pose in poses if pose is not None)

                # # Print the counts
                # print("Number of non-None depth paths:", depth_count)
                # print("Number of non-None poses:", pose_count)
                # # Number of non-None depth paths: 496
                # # Number of non-None poses: 496

                # Check if any value is None in depth_paths, poses, or intrinsics
                if all(x is None for x in depth_paths) or all(x is None for x in poses) or all(x is None for x in intrinsics):
                    print(f"Skipping overlap calculation due to ALL NONE data in {scene}.")
                else:
                    depth_count, depth_none_count = sum(1 for depth in depth_paths if depth is not None), sum(1 for depth in depth_paths if depth is None)
                    pose_count, pose_none_count = sum(1 for pose in poses if pose is not None), sum(1 for pose in poses if pose is None)

                    maxVal = 50
                    if depth_count < maxVal: 
                        # Print the counts
                        print("Number of non-None depth paths:", depth_count)
                        print("Number of non-None poses:", pose_count)
                        print("Number of None depth paths:", depth_none_count)
                        print("Number of None poses:", pose_none_count)
                        print(f"All data present, and less than {maxVal}!. Proceeding with overlap calculation.")                        

                        # Create a boolean mask where None values are marked as False
                        # Create masks for non-None entries
                        mask_depth = np.array([dp is not None for dp in depth_paths])
                        mask_poses = np.array([ps is not None for ps in poses])
                        mask_intrinsics = np.array([intrinsic is not None for intrinsic in intrinsics])

                        # Combine masks to keep only entries where all corresponding elements are not None
                        mask = mask_depth & mask_poses & mask_intrinsics

                        # Filter arrays using the combined mask
                        filtered_depth_paths = depth_paths[mask]
                        filtered_poses = poses[mask]
                        filtered_intrinsics = intrinsics[mask]


                        assert(len(filtered_depth_paths) == depth_count and len(filtered_poses) == pose_count)
                        
                        # Calculate the overlap matrix
                        calculated_overlap_matrix = calculate_overlap_matrix(filtered_depth_paths, filtered_poses, filtered_intrinsics)

                        full_overlap_matrix = np.zeros((len(depth_paths), len(depth_paths)))
                        # Populate the full matrix using the mask to place calculated overlaps correctly
                        indices = np.where(mask)[0]
                        for i, idx_i in enumerate(indices):
                            for j, idx_j in enumerate(indices):
                                full_overlap_matrix[idx_i, idx_j] = calculated_overlap_matrix[i, j]
                                full_overlap_matrix[idx_j, idx_i] = calculated_overlap_matrix[i, j]  # Symmetric


                        # Compare the calculated matrix with the ground truth
                        try:
                            print(f"Shape of full_overlap_matrix matrix: {full_overlap_matrix.shape}")
                            print(f"Shape of ground truth overlap matrix: {ground_truth_overlap_matrix.shape}")
                            print()
                            comparison_result = np.isclose(full_overlap_matrix, ground_truth_overlap_matrix)
                            print("---------------------Comparison result:\n", comparison_result, "---------------------------------")

                            # Check if all values are close enough
                            if np.all(comparison_result):
                                print("\n\nCalculated overlap matrix matches the ground truth.\n\n")
                            else:
                                print("\n\nDiscrepancy found in the calculated overlap matrix.\n\n")
                        except Exception as e:
                            print(f"Error comparing calculated and ground truth overlap matrices: {str(e)} for {scene}")
                            
                        # Save directories and file paths
                        import os
                        save_dir = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/datasets"
                        calculated_path = os.path.join(save_dir, f"{scene}_calculated_overlap.npy")
                        ground_truth_path = os.path.join(save_dir, f"{scene}_ground_truth_overlap.npy")

                        # Ensure the save directory exists
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)

                        # Save the calculated overlap matrix
                        np.save(calculated_path, calculated_overlap_matrix)
                        print(f"Calculated overlap matrix saved to: {calculated_path}")

                        # Save the ground truth overlap matrix
                        np.save(ground_truth_path, ground_truth_overlap_matrix)
                        print(f"Ground truth overlap matrix saved to: {ground_truth_path}")
                
            except Exception as e:
                
                if count < 2:
                    print(f"Failed to load data for scene {scene}: {str(e)}")
                    logger.warning(
                        "Cannot load scene info for scene %s at %s.", scene, path
                    )
                count +=1
                continue
                # raise Exception
                # continue
            
            # if count < 2:
            #     # limited_logger.log("info loaded from str(path)", str(path))
            #     # limited_logger.log("info type", type(info))          
            #     # limited_logger.log("\n\ninfo:\n !!!!", info, "\n\n")
            #     ## Might fail 
            #     # # # limited_logger.log("info.shape", info.shape)
            #     # limited_logger.log("info.keys():", info.keys())
            #     # limited_logger.log("\nIMAGE: ", info["image_paths"],  "\nDEPTH: ",  info["depth_paths"], "\nPOSES: ", info["poses"], "\nINTRINSRICS: ", info["intrinsics"])
            
            # print(info["poses"].shape)
            # print(info["poses"])

            # print(info["depth_paths"].shape)
            # print(info["depth_paths"])

            self.images[scene] = info["image_paths"]
            self.depths[scene] = info["depth_paths"]
            self.poses[scene] = info["poses"]
            self.intrinsics[scene] = info["intrinsics"] # = np.array([[320.0, 0, 320.0], [0, 320.0, 240.0], [0, 0, 1.0]]) 
            self.scenes.append(scene)
        
        # print("ASCENS: ", self.scenes)
        # # Jusyt  alist of 0000, 0001 etc for both images keys and self.scenes
        # print(self.images.keys())
        # print(self.images)

        if load_sample:
            self.sample_new_items(conf.seed)
            # assert len(self.items) > 0
            assert len(self.items) > 0, "No items sampled; check configuration."

    

    def sample_new_items(self, seed):
        ##  # logger.info("Sampling new %s data with seed %d.", self.split, seed)
        # # limited_logger.log("IN SAMPLE_NEW_ITEMS\n\n")
       #  # logger.info(f"Sampling new items for {self.split} with seed {seed}.")
   
        self.items = []
        split = self.split
        num_per_scene = self.conf[self.split + "_num_per_scene"]
        # # limited_logger.log("num_per_scene", num_per_scene)
        if isinstance(num_per_scene, Iterable):
            num_pos, num_neg = num_per_scene
        else:
            num_pos = num_per_scene
            num_neg = None
        # # limited_logger.log("split", split)
        # # limited_logger.log("num_pos, num_neg", num_pos, num_neg)
        # # # limited_logger.log("self.conf[split + _pairs] ", self.conf[split + "_pairs"] )

        if split != "train" and self.conf[split + "_pairs"] is not None:
            # Fixed validation or test pairs
            assert num_pos is None and num_neg is None, "Pairs are pre-defined, no sampling needed."
            # assert num_pos is None
            # assert num_neg is None
            # # limited_logger.log("self.conf.views is 2!: ", self.conf.views)
            assert self.conf.views == 2
            pairs_path = scene_lists_path / self.conf[split + "_pairs"]
            # # limited_logger.log("pairs_path",  pairs_path)
            # # limited_logger.log("pairs_path.read_text.restip...", pairs_path.read_text().rstrip("\n").split("\n"))
            count = 0
            for line in pairs_path.read_text().rstrip("\n").split("\n"):
                # if count < 5 :
                # # limited_logger.log("looping lines - line:", line)
                # count += 1
                im0, im1 = line.split(" ")
                scene = im0.split("/")[0]
                # # limited_logger.log("im0, im1, scene", im0, im1, scene)
                assert im1.split("/")[0] == scene
                im0, im1 = [self.conf.image_subpath + im for im in [im0, im1]]
                # # limited_logger.log("after [self.conf.image_subpath + im for im in [im0, im1]]", im0, im1)
                assert im0 in self.images[scene]
                assert im1 in self.images[scene]
                idx0 = np.where(self.images[scene] == im0)[0][0]
                idx1 = np.where(self.images[scene] == im1)[0][0]
                # # limited_logger.log("idx0, idx1", idx0, idx1)
                self.items.append((scene, idx0, idx1, 1.0))
                ### I added
               # #  # logger.info(f"Added fixed pair: {scene}, {im0}, {im1}")
        elif self.conf.views == 1:
            # # limited_logger.log("in self.conf views 1... :/")
            for scene in self.scenes:
                if scene not in self.images:
                    continue
                valid = (self.images[scene] != None) | (  # noqa: E711
                    self.depths[scene] != None  # noqa: E711
                )
                ids = np.where(valid)[0]
                if num_pos and len(ids) > num_pos:
                    ids = np.random.RandomState(seed).choice(
                        ids, num_pos, replace=False
                    )
                ids = [(scene, i) for i in ids]
                self.items.extend(ids)
               #  # logger.info(f"Scene {scene}: {len(ids)} items added.")
        else:
            # # limited_logger.log("in the else block1")
            for scene in self.scenes:
                path = self.info_dir / (scene + ".npz")
                # # limited_logger.log("path", path)
                assert path.exists(), f"Info file missing: {path}"
                # assert path.exists(), path
                info = np.load(str(path), allow_pickle=True)
                # # limited_logger.log("str(path)", str(path))
                # print("self.images[scene].shape, self.depths[scene].shape", self.images[scene].shape, self.depths[scene].shape)
                valid = (self.images[scene] != None) & (  # noqa: E711
                    self.depths[scene] != None  # noqa: E711
                )
                # print("valid array:", valid)

                ind = np.where(valid)[0]
                # print( "info[overlap_matrix].shape", info["overlap_matrix"].shape)
                #print( "info[overlap_matrix]", info["overlap_matrix"])

                mat = info["overlap_matrix"][valid][:, valid]
                # print("mat", mat)
                # print("mat shape", mat.shape)
                # # limited_logger.log("info[overlap_matrix][valid][:, valid]", info["overlap_matrix"][valid][:, valid])
                

                if num_pos is not None:
                    # # limited_logger.log("num_pos", num_pos)
                    # Sample a subset of pairs, binned by overlap.
                    num_bins = self.conf.num_overlap_bins
                    assert num_bins > 0
                    bin_width = (
                        self.conf.max_overlap - self.conf.min_overlap
                    ) / num_bins
                    num_per_bin = num_pos // num_bins
                    pairs_all = []
                    for k in range(num_bins):
                        bin_min = self.conf.min_overlap + k * bin_width
                        bin_max = bin_min + bin_width
                        pairs_bin = (mat > bin_min) & (mat <= bin_max)
                        pairs_bin = np.stack(np.where(pairs_bin), -1)
                        pairs_all.append(pairs_bin)
                    # Skip bins with too few samples
                    has_enough_samples = [len(p) >= num_per_bin * 2 for p in pairs_all]
                    num_per_bin_2 = num_pos // max(1, sum(has_enough_samples))
                    pairs = []
                    for pairs_bin, keep in zip(pairs_all, has_enough_samples):
                        if keep:
                            pairs.append(sample_n(pairs_bin, num_per_bin_2, seed))
                    pairs = np.concatenate(pairs, 0)
                else:
                    pairs = (mat > self.conf.min_overlap) & (
                        mat <= self.conf.max_overlap
                    )
                    pairs = np.stack(np.where(pairs), -1)

                # print(f"\n\ninput to megadepth pairs is {pairs}")
                # print(pairs.shape)
                pairs = [(scene, ind[i], ind[j], mat[i, j]) for i, j in pairs]
                if num_neg is not None:
                    neg_pairs = np.stack(np.where(mat <= 0.0), -1)
                    neg_pairs = sample_n(neg_pairs, num_neg, seed)
                    pairs += [(scene, ind[i], ind[j], mat[i, j]) for i, j in neg_pairs]
                self.items.extend(pairs)
                ### I added
               #  # logger.info(f"Scene {scene}: {len(pairs)} pairs added.")
        if self.conf.views == 2 and self.conf.sort_by_overlap:
            self.items.sort(key=lambda i: i[-1], reverse=True)
        else:
            np.random.RandomState(seed).shuffle(self.items)

    def _read_view(self, scene, idx):
        path = self.root / self.images[scene][idx]
        ### I added
       #  # logger.info(f"Reading view from {path}")
        # print("path, self.root , self.images[scene][idx]", path, self.root , self.images[scene][idx])
        # read pose data
        # print("scene, idx", scene, idx)
        # print("self.intrinsics[scene][idx].shape", self.intrinsics[scene][idx].shape)
        # print("self.poses[scene][idx].shape", self.poses[scene][idx].shape)
        # read pose data
        K = self.intrinsics[scene][idx].astype(np.float32, copy=False)
        T = self.poses[scene][idx].astype(np.float32, copy=False)

        ### I added
        #  # logger.info(f"Pose data: {K.shape}, {T.shape}")

        # read image
        if self.conf.read_image:
            img = load_image(self.root / self.images[scene][idx], self.conf.grayscale)
            ### I added
           #  # logger.info(f"Loaded image {path.name} with shape {img.shape}")
        else:
            size = PIL.Image.open(path).size[::-1]
            img = torch.zeros(
                [3 - 2 * int(self.conf.grayscale), size[0], size[1]]
            ).float()
            ### I added
           #  # logger.info(f"Created placeholder image with shape {img.shape}")

        # read depth
        if self.conf.read_depth:
            depth_path = (
                self.root / self.conf.depth_subpath / scene / (path.stem + ".h5")
            )
            with h5py.File(str(depth_path), "r") as f:
                depth = f["/depth"].__array__().astype(np.float32, copy=False)
                depth = torch.Tensor(depth)[None]
            assert depth.shape[-2:] == img.shape[-2:]
        else:
            depth = None

        # add random rotations
        do_rotate = self.conf.p_rotate > 0.0 and self.split == "train"
        if do_rotate:
            p = self.conf.p_rotate
            k = 0
            if np.random.rand() < p:
                k = np.random.choice(2, 1, replace=False)[0] * 2 - 1
                img = np.rot90(img, k=-k, axes=(-2, -1))
                if self.conf.read_depth:
                    depth = np.rot90(depth, k=-k, axes=(-2, -1)).copy()
                K = rotate_intrinsics(K, img.shape, k + 2)
                T = rotate_pose_inplane(T, k + 2)
                ### I added
               #  # logger.info(f"Applied random rotation: {k * 90} degrees")


        name = path.name

        data = self.preprocessor(img)
        if depth is not None:
            data["depth"] = self.preprocessor(depth, interpolation="nearest")["image"][
                0
            ]
        K = scale_intrinsics(K, data["scales"])

        data = {
            "name": name,
            "scene": scene,
            "T_w2cam": Pose.from_4x4mat(T),
            "depth": depth,
            "camera": Camera.from_calibration_matrix(K).float(),
            **data,
        }

        if self.conf.load_features.do:
            features = self.feature_loader({k: [v] for k, v in data.items()})
            if do_rotate and k != 0:
                # ang = np.deg2rad(k * 90.)
                kpts = features["keypoints"].copy()
                x, y = kpts[:, 0].copy(), kpts[:, 1].copy()
                w, h = data["image_size"]
                if k == 1:
                    kpts[:, 0] = w - y
                    kpts[:, 1] = x
                elif k == -1:
                    kpts[:, 0] = y
                    kpts[:, 1] = h - x

                else:
                    raise ValueError
                features["keypoints"] = kpts

            data = {"cache": features, **data}
            ### I added
           #  # logger.info(f"Features loaded and processed for {name}")

        return data

    def __getitem__(self, idx):
        if self.conf.reseed:
            with fork_rng(self.conf.seed + idx, False):
                return self.getitem(idx)
        else:
            return self.getitem(idx)

    def getitem(self, idx):
        if self.conf.views == 2:
            if isinstance(idx, list):
                scene, idx0, idx1, overlap = idx
            else:
                scene, idx0, idx1, overlap = self.items[idx]
            data0 = self._read_view(scene, idx0)
            data1 = self._read_view(scene, idx1)
            data = {
                "view0": data0,
                "view1": data1,
            }
            data["T_0to1"] = data1["T_w2cam"] @ data0["T_w2cam"].inv()
            data["T_1to0"] = data0["T_w2cam"] @ data1["T_w2cam"].inv()
            data["overlap_0to1"] = overlap
            data["name"] = f"{scene}/{data0['name']}_{data1['name']}"
            ### I added
           #  # logger.info(f"Processed data pair: {data['name']} with overlap {overlap}")
        else:
            assert self.conf.views == 1
            scene, idx0 = self.items[idx]
            data = self._read_view(scene, idx0)
            ### I added
           #  # logger.info(f"Processed single view data for {scene}")
        data["scene"] = scene
        data["idx"] = idx
        return data

    def __len__(self):
        return len(self.items)


class _TripletDataset(_PairDataset):
    def sample_new_items(self, seed):
        logging.info("Sampling new triplets with seed %d", seed)
        self.items = []
        split = self.split
        num = self.conf[self.split + "_num_per_scene"]
        if split != "train" and self.conf[split + "_pairs"] is not None:
            if Path(self.conf[split + "_pairs"]).exists():
                pairs_path = Path(self.conf[split + "_pairs"])
            else:
                pairs_path = DATA_PATH / "configs" / self.conf[split + "_pairs"]
            for line in pairs_path.read_text().rstrip("\n").split("\n"):
                im0, im1, im2 = line.split(" ")
                assert im0[:4] == im1[:4]
                scene = im1[:4]
                idx0 = np.where(self.images[scene] == im0)
                idx1 = np.where(self.images[scene] == im1)
                idx2 = np.where(self.images[scene] == im2)
                self.items.append((scene, idx0, idx1, idx2, 1.0, 1.0, 1.0))
        else:
            for scene in self.scenes:
                path = self.info_dir / (scene + ".npz")
                assert path.exists(), path
                info = np.load(str(path), allow_pickle=True)
                if self.conf.num_overlap_bins > 1:
                    raise NotImplementedError("TODO")
                valid = (self.images[scene] != None) & (  # noqa: E711
                    self.depth[scene] != None  # noqa: E711
                )
                
                ind = np.where(valid)[0]
                mat = info["overlap_matrix"][valid][:, valid]
                good = (mat > self.conf.min_overlap) & (mat <= self.conf.max_overlap)
                triplets = []
                if self.conf.triplet_enforce_overlap:
                    pairs = np.stack(np.where(good), -1)
                    for i0, i1 in pairs:
                        for i2 in pairs[pairs[:, 0] == i0, 1]:
                            if good[i1, i2]:
                                triplets.append((i0, i1, i2))
                    if len(triplets) > num:
                        selected = np.random.RandomState(seed).choice(
                            len(triplets), num, replace=False
                        )
                        selected = range(num)
                        triplets = np.array(triplets)[selected]
                else:
                    # we first enforce that each row has >1 pairs
                    non_unique = good.sum(-1) > 1
                    ind_r = np.where(non_unique)[0]
                    good = good[non_unique]
                    pairs = np.stack(np.where(good), -1)
                    if len(pairs) > num:
                        selected = np.random.RandomState(seed).choice(
                            len(pairs), num, replace=False
                        )
                        pairs = pairs[selected]
                    for idx, (k, i) in enumerate(pairs):
                        # We now sample a j from row k s.t. i != j
                        possible_j = np.where(good[k])[0]
                        possible_j = possible_j[possible_j != i]
                        selected = np.random.RandomState(seed + idx).choice(
                            len(possible_j), 1, replace=False
                        )[0]
                        triplets.append((ind_r[k], i, possible_j[selected]))
                    triplets = [
                        (scene, ind[k], ind[i], ind[j], mat[k, i], mat[k, j], mat[i, j])
                        for k, i, j in triplets
                    ]
                    self.items.extend(triplets)
        np.random.RandomState(seed).shuffle(self.items)

    def __getitem__(self, idx):
        scene, idx0, idx1, idx2, overlap01, overlap02, overlap12 = self.items[idx]
        data0 = self._read_view(scene, idx0)
        data1 = self._read_view(scene, idx1)
        data2 = self._read_view(scene, idx2)
        data = {
            "view0": data0,
            "view1": data1,
            "view2": data2,
        }
        data["T_0to1"] = data1["T_w2cam"] @ data0["T_w2cam"].inv()
        data["T_0to2"] = data2["T_w2cam"] @ data0["T_w2cam"].inv()
        data["T_1to2"] = data2["T_w2cam"] @ data1["T_w2cam"].inv()
        data["T_1to0"] = data0["T_w2cam"] @ data1["T_w2cam"].inv()
        data["T_2to0"] = data0["T_w2cam"] @ data2["T_w2cam"].inv()
        data["T_2to1"] = data1["T_w2cam"] @ data2["T_w2cam"].inv()

        data["overlap_0to1"] = overlap01
        data["overlap_0to2"] = overlap02
        data["overlap_1to2"] = overlap12
        data["scene"] = scene
        data["name"] = f"{scene}/{data0['name']}_{data1['name']}_{data2['name']}"
        ### I added
       #  # logger.info(f"Processed triplet: {data['name']} with overlaps {overlap01}, {overlap02}, {overlap12}")
    
        return data

    def __len__(self):
        return len(self.items)


def visualize(args):
    conf = {
        "min_overlap": 0.1,
        "max_overlap": 0.7,
        "num_overlap_bins": 3,
        "sort_by_overlap": False,
        "train_num_per_scene": 5,
        "batch_size": 1,
        "num_workers": 0,
        "prefetch_factor": None,
        "val_num_per_scene": None,
    }
    conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.dotlist))
    dataset = MegaDepth(conf)
    loader = dataset.get_data_loader(args.split)
   #  # logger.info("The dataset has elements.", len(loader))

    with fork_rng(seed=dataset.conf.seed):
        images, depths = [], []
        for _, data in zip(range(args.num_items), loader):
            images.append(
                [
                    data[f"view{i}"]["image"][0].permute(1, 2, 0)
                    for i in range(dataset.conf.views)
                ]
            )
            depths.append(
                [data[f"view{i}"]["depth"][0] for i in range(dataset.conf.views)]
            )

    axes = plot_image_grid(images, dpi=args.dpi)
    for i in range(len(images)):
        plot_heatmaps(depths[i], axes=axes[i])
    plt.show()


if __name__ == "__main__":
    from .. import logger  # overwrite the logger
   
    # # # limited_logger.log("This message will appear only once.")
    # # # limited_logger.log("This message will appear only once.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num_items", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()
    visualize(args)
