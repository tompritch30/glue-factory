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


def sample_n(data, num, seed=None):
    if len(data) > num:
        selected = np.random.RandomState(seed).choice(len(data), num, replace=False)
        return data[selected]
    else:
        return data


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
            logger.info("Downloading the MegaDepth dataset.")
            self.download()

    def download(self):
        import os
        import shutil
        import zipfile
        import torch    
        data_dir = DATA_PATH / self.conf.data_dir
        tmp_dir = data_dir.parent / "megadepth_tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(exist_ok=True, parents=True)
        url_base = "https://cvg-data.inf.ethz.ch/megadepth/"
        
        tar_name = "megadepth1500.zip"
        out_name = self.conf.image_subpath  # Adjust this to the correct output directory for the extracted data
        
        tar_path = tmp_dir / tar_name
        torch.hub.download_url_to_file(url_base + tar_name, tar_path)
        
        # Since it's a zip file, use the appropriate library to extract it
        with zipfile.ZipFile(tar_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)
        
        tar_path.unlink()  # Remove the zip file after extraction

        # List all the contents of the extraction path
        print("Directory structure post-extraction:")
        for root, dirs, files in os.walk(tmp_dir):
            print(root)
            for d in dirs:
                print(f" - {d}/")
            for f in files:
                print(f" - {f}")

        # Move the extracted directories to their final locations
        if (tmp_dir / 'Undistorted_SfM').exists():
            shutil.move(str(tmp_dir / 'Undistorted_SfM'), str(data_dir / self.conf.image_subpath))

        if (tmp_dir / 'depth_undistorted').exists():
            shutil.move(str(tmp_dir / 'depth_undistorted'), str(data_dir / self.conf.depth_subpath))
        
        if (tmp_dir / 'scene_info').exists():
            shutil.move(str(tmp_dir / 'scene_info'), str(data_dir / self.conf.info_dir))

        # Clean up the temporary directory if it's empty
        if not any(tmp_dir.iterdir()):
            shutil.rmtree(tmp_dir)

    # def download(self):
    #     data_dir = DATA_PATH / self.conf.data_dir
    #     tmp_dir = data_dir.parent / "megadepth_tmp"
    #     if tmp_dir.exists():  # The previous download failed.
    #         shutil.rmtree(tmp_dir)
    #     tmp_dir.mkdir(exist_ok=True, parents=True)
    #     url_base = "https://cvg-data.inf.ethz.ch/megadepth/"
        
    #     # Specific file for the smaller dataset
    #     tar_name = "megadepth1500.zip"
    #     out_name = self.conf.image_subpath  # Adjust this to the correct output directory for the extracted data
        
    #     tar_path = tmp_dir / tar_name
    #     torch.hub.download_url_to_file(url_base + tar_name, tar_path)
        
    #     # Since it's a zip file, use the appropriate library to extract it
    #     import zipfile
    #     with zipfile.ZipFile(tar_path, 'r') as zip_ref:
    #         zip_ref.extractall(tmp_dir / out_name)
        
    #     tar_path.unlink()  # Remove the zip file after extraction
    #     # Move the extracted directories to their final locations
    #     if (tmp_dir / 'Undistorted_SfM').exists():
    #         print('Undistorted_SfM')
    #         shutil.move(str(tmp_dir / 'Undistorted_SfM'), str(data_dir / self.conf.image_subpath))

    #     # Assuming other potential directories are named as expected
    #     if (tmp_dir / 'depth_undistorted').exists():
    #         print('depth_undistorted')
    #         shutil.move(str(tmp_dir / 'depth_undistorted'), str(data_dir / self.conf.depth_subpath))
        
    #     if (tmp_dir / 'scene_info').exists():
    #         print('scene_info')
    #         shutil.move(str(tmp_dir / 'scene_info'), str(data_dir / self.conf.info_dir))

    #     # Clean up the temporary directory if it's empty
    #     if not any(tmp_dir.iterdir()):
    #         shutil.rmtree(tmp_dir)
        
        # for path in tmp_dir.iterdir():
        #     print(f"Contained directory or file: {path}")
        
        # shutil.move(tmp_dir / out_name, data_dir / out_name)


        # file_limit = 1  # Only download the first file for testing purposes
        # # Iterate over the dataset files you want to potentially download
        # for index, (tar_name, out_name) in enumerate([
        #     ("Undistorted_SfM.tar.gz", self.conf.image_subpath),
        #     ("depth_undistorted.tar.gz", self.conf.depth_subpath),
        #     ("scene_info.tar.gz", self.conf.info_dir),
        # ]):
        #     if index >= file_limit:
        #         break  # Stop downloading after reaching the file limit
        #     tar_path = tmp_dir / tar_name
        #     torch.hub.download_url_to_file(url_base + tar_name, tar_path)
            
        #     # Extract the tar file and clean up
        #     with tarfile.open(tar_path) as tar:
        #         tar.extractall(path=tmp_dir)
        #     tar_path.unlink()
            
        #     # Move extracted contents to their respective output directories
        #     shutil.move(tmp_dir / tar_name.split(".")[0], tmp_dir / out_name)
        
        # # Move the temporary directory with all its contents to the final data directory
        # shutil.move(tmp_dir, data_dir)

        ##### Original Code #####        
        # for tar_name, out_name in (
        #     ("Undistorted_SfM.tar.gz", self.conf.image_subpath),
        #     ("depth_undistorted.tar.gz", self.conf.depth_subpath),
        #     ("scene_info.tar.gz", self.conf.info_dir),
        # ):
        #     tar_path = tmp_dir / tar_name
        #     torch.hub.download_url_to_file(url_base + tar_name, tar_path)
        #     with tarfile.open(tar_path) as tar:
        #         tar.extractall(path=tmp_dir)
        #     tar_path.unlink()
        #     shutil.move(tmp_dir / tar_name.split(".")[0], tmp_dir / out_name)
        # shutil.move(tmp_dir, data_dir)

    def get_dataset(self, split):
        assert self.conf.views in [1, 2, 3]
        if self.conf.views == 3:
            return _TripletDataset(self.conf, split)
        else:
            return _PairDataset(self.conf, split)


class _PairDataset(torch.utils.data.Dataset):
    def __init__(self, conf, split, load_sample=True):
        self.root = DATA_PATH / conf.data_dir
        assert self.root.exists(), self.root
        self.split = split
        self.conf = conf

        split_conf = conf[split + "_split"]
        if isinstance(split_conf, (str, Path)):
            scenes_path = scene_lists_path / split_conf
            scenes = scenes_path.read_text().rstrip("\n").split("\n")
        elif isinstance(split_conf, Iterable):
            scenes = list(split_conf)
        else:
            raise ValueError(f"Unknown split configuration: {split_conf}.")
        scenes = sorted(set(scenes))

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
        self.scenes = []
        for scene in scenes:
            path = self.info_dir / (scene + ".npz")
            try:
                info = np.load(str(path), allow_pickle=True)
            except Exception:
                logger.warning(
                    "Cannot load scene info for scene %s at %s.", scene, path
                )
                continue
            self.images[scene] = info["image_paths"]
            self.depths[scene] = info["depth_paths"]
            self.poses[scene] = info["poses"]
            self.intrinsics[scene] = info["intrinsics"]
            self.scenes.append(scene)

        if load_sample:
            self.sample_new_items(conf.seed)
            assert len(self.items) > 0

    def sample_new_items(self, seed):
        # Logging the seed use
        logger.info("Sampling new %s data with seed %d.", self.split, seed)
        self.items = []

        # Iterate over each scene to sample items
        for scene in self.scenes:
            scene_images = self.images[scene]
            num_images = len(scene_images)

            # Calculate the number of samples to take from this scene
            sample_size = min(num_images, self.conf['train_num_per_scene'])
            sampled_indices = np.random.RandomState(seed).choice(num_images, sample_size, replace=False)

            # Append sampled items from the scene
            for idx in sampled_indices:
                self.items.append((scene, idx))

        # Dynamically adjust the split between training and validation
        if self.split == 'train':
            self.items = self.items[:int(0.8 * len(self.items))]
        elif self.split == 'val':
            self.items = self.items[int(0.8 * len(self.items)):]

    ### Original code
    # def sample_new_items(self, seed):
    #     logger.info("Sampling new %s data with seed %d.", self.split, seed)
    #     self.items = []
    #     split = self.split
        
    #     num_per_scene = self.conf[self.split + "_num_per_scene"]
    #     if isinstance(num_per_scene, Iterable):
    #         num_pos, num_neg = num_per_scene
    #     else:
    #         num_pos = num_per_scene
    #         num_neg = None
    #     if split != "train" and self.conf[split + "_pairs"] is not None:
    #         # Fixed validation or test pairs
    #         assert num_pos is None
    #         assert num_neg is None
    #         assert self.conf.views == 2
    #         pairs_path = scene_lists_path / self.conf[split + "_pairs"]
    #         for line in pairs_path.read_text().rstrip("\n").split("\n"):
    #             im0, im1 = line.split(" ")
    #             scene = im0.split("/")[0]
    #             assert im1.split("/")[0] == scene
    #             im0, im1 = [self.conf.image_subpath + im for im in [im0, im1]]
    #             assert im0 in self.images[scene]
    #             assert im1 in self.images[scene]
    #             idx0 = np.where(self.images[scene] == im0)[0][0]
    #             idx1 = np.where(self.images[scene] == im1)[0][0]
    #             self.items.append((scene, idx0, idx1, 1.0))
    #     elif self.conf.views == 1:
    #         for scene in self.scenes:
    #             if scene not in self.images:
    #                 continue
    #             valid = (self.images[scene] != None) | (  # noqa: E711
    #                 self.depths[scene] != None  # noqa: E711
    #             )
    #             ids = np.where(valid)[0]
    #             if num_pos and len(ids) > num_pos:
    #                 ids = np.random.RandomState(seed).choice(
    #                     ids, num_pos, replace=False
    #                 )
    #             ids = [(scene, i) for i in ids]
    #             self.items.extend(ids)
    #     else:
    #         for scene in self.scenes:
    #             path = self.info_dir / (scene + ".npz")
    #             assert path.exists(), path
    #             info = np.load(str(path), allow_pickle=True)
    #             valid = (self.images[scene] != None) & (  # noqa: E711
    #                 self.depths[scene] != None  # noqa: E711
    #             )
    #             ind = np.where(valid)[0]
    #             mat = info["overlap_matrix"][valid][:, valid]

    #             if num_pos is not None:
    #                 # Sample a subset of pairs, binned by overlap.
    #                 num_bins = self.conf.num_overlap_bins
    #                 assert num_bins > 0
    #                 bin_width = (
    #                     self.conf.max_overlap - self.conf.min_overlap
    #                 ) / num_bins
    #                 num_per_bin = num_pos // num_bins
    #                 pairs_all = []
    #                 for k in range(num_bins):
    #                     bin_min = self.conf.min_overlap + k * bin_width
    #                     bin_max = bin_min + bin_width
    #                     pairs_bin = (mat > bin_min) & (mat <= bin_max)
    #                     pairs_bin = np.stack(np.where(pairs_bin), -1)
    #                     pairs_all.append(pairs_bin)
    #                 # Skip bins with too few samples
    #                 has_enough_samples = [len(p) >= num_per_bin * 2 for p in pairs_all]
    #                 num_per_bin_2 = num_pos // max(1, sum(has_enough_samples))
    #                 pairs = []
    #                 for pairs_bin, keep in zip(pairs_all, has_enough_samples):
    #                     if keep:
    #                         pairs.append(sample_n(pairs_bin, num_per_bin_2, seed))
    #                 pairs = np.concatenate(pairs, 0)
    #             else:
    #                 pairs = (mat > self.conf.min_overlap) & (
    #                     mat <= self.conf.max_overlap
    #                 )
    #                 pairs = np.stack(np.where(pairs), -1)

    #             pairs = [(scene, ind[i], ind[j], mat[i, j]) for i, j in pairs]
    #             if num_neg is not None:
    #                 neg_pairs = np.stack(np.where(mat <= 0.0), -1)
    #                 neg_pairs = sample_n(neg_pairs, num_neg, seed)
    #                 pairs += [(scene, ind[i], ind[j], mat[i, j]) for i, j in neg_pairs]
    #             self.items.extend(pairs)
    #     if self.conf.views == 2 and self.conf.sort_by_overlap:
    #         self.items.sort(key=lambda i: i[-1], reverse=True)
    #     else:
    #         np.random.RandomState(seed).shuffle(self.items)

    def _read_view(self, scene, idx):
        path = self.root / self.images[scene][idx]

        # read pose data
        K = self.intrinsics[scene][idx].astype(np.float32, copy=False)
        T = self.poses[scene][idx].astype(np.float32, copy=False)

        # read image
        if self.conf.read_image:
            img = load_image(self.root / self.images[scene][idx], self.conf.grayscale)
        else:
            size = PIL.Image.open(path).size[::-1]
            img = torch.zeros(
                [3 - 2 * int(self.conf.grayscale), size[0], size[1]]
            ).float()

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
        else:
            assert self.conf.views == 1
            scene, idx0 = self.items[idx]
            data = self._read_view(scene, idx0)
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
    logger.info("The dataset has elements.", len(loader))

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num_items", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()
    visualize(args)
