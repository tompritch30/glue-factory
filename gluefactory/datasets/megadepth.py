originalCode = False


"""
        python -m gluefactory.train sp+lg_megadepthtest     --conf gluefactory/configs/superpoint+lightglue_megadepth.yaml     train.load_experiment=sp+lg_homography
    python -m gluefactory.train sp+lg_megadepth     --conf gluefactory/configs/superpoint+lightglue_megadepth.yaml     train.load_experiment=sp+lg_homography
    python -m gluefactory.train sp+lg_debug     --conf gluefactory/configs/superpoint+lightglue_megadepth.yaml     train.load_experiment=sp+lg_homography
    python -m gluefactory.train sp+lg_debug     --conf gluefactory/configs/superpoint+lightglue_megadepth.yaml     train.load_experiment=sp+lg_homography
"""

if originalCode:
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
            logger.info("Sampling new %s data with seed %d.", self.split, seed)
            self.items = []
            split = self.split
            num_per_scene = self.conf[self.split + "_num_per_scene"]
            if isinstance(num_per_scene, Iterable):
                num_pos, num_neg = num_per_scene
            else:
                num_pos = num_per_scene
                num_neg = None
            if split != "train" and self.conf[split + "_pairs"] is not None:
                # Fixed validation or test pairs
                assert num_pos is None
                assert num_neg is None
                assert self.conf.views == 2
                pairs_path = scene_lists_path / self.conf[split + "_pairs"]
                for line in pairs_path.read_text().rstrip("\n").split("\n"):
                    im0, im1 = line.split(" ")
                    scene = im0.split("/")[0]
                    assert im1.split("/")[0] == scene
                    im0, im1 = [self.conf.image_subpath + im for im in [im0, im1]]
                    assert im0 in self.images[scene]
                    assert im1 in self.images[scene]
                    idx0 = np.where(self.images[scene] == im0)[0][0]
                    idx1 = np.where(self.images[scene] == im1)[0][0]
                    self.items.append((scene, idx0, idx1, 1.0))
            elif self.conf.views == 1:
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
            else:
                for scene in self.scenes:
                    path = self.info_dir / (scene + ".npz")
                    assert path.exists(), path
                    info = np.load(str(path), allow_pickle=True)
                    valid = (self.images[scene] != None) & (  # noqa: E711
                        self.depths[scene] != None  # noqa: E711
                    )
                    ind = np.where(valid)[0]
                    mat = info["overlap_matrix"][valid][:, valid]

                    if num_pos is not None:
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

                    pairs = [(scene, ind[i], ind[j], mat[i, j]) for i, j in pairs]
                    if num_neg is not None:
                        neg_pairs = np.stack(np.where(mat <= 0.0), -1)
                        neg_pairs = sample_n(neg_pairs, num_neg, seed)
                        pairs += [(scene, ind[i], ind[j], mat[i, j]) for i, j in neg_pairs]
                    self.items.extend(pairs)
            if self.conf.views == 2 and self.conf.sort_by_overlap:
                self.items.sort(key=lambda i: i[-1], reverse=True)
            else:
                np.random.RandomState(seed).shuffle(self.items)

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
#### MY CODE STARTS HERE
else:
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
            return None
            raise FileNotFoundError(f"File not found: {file_path}")

   
    def calculate_overlap_for_pair(args):
        defaultVal = -1.0
        # print("calculating overlap for pairs!!!")
        i, j, depth_paths, poses, intrinsics = args
        # print(f"Calculating overlap for pair {i}-{j}")
        # print(f"Depth Path i: {depth_paths[i]}")
        # print(f"Depth Path j: {depth_paths[j]}")

            # Handle None values directly within the function
        if depth_paths[i] is None or depth_paths[j] is None:
            return i, j, defaultVal

        depth_i = load_npy_file(depth_paths[i])
        pose_i = poses[i]
        depth_j = load_npy_file(depth_paths[j])
        pose_j = poses[j]
        
        if depth_i is None or depth_j is None or pose_i is None or pose_j is None:
            return i, j, defaultVal  # Return -1 overlap if data is missing

        pose = np.linalg.inv(pose_j) @ pose_i
        count, total = project_points(depth_i, intrinsics[i], pose)
        overlap = count / total
        # print("returning overlap values")
        overlap = count / total if total > 0 else defaultVal
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
                            # if i % 500 == 0:
                            #     print(f"Overlap between frame {i} and {j}: {overlap}")  # Print at intervals

                        return overlap_matrix

                    # print(info["overlap_matrix"])

                    depth_paths = info["depth_paths"]
                    poses = info["poses"]
                    intrinsics = info["intrinsics"]
                    ground_truth_overlap_matrix = info["overlap_matrix"]

                    # print("--------------------------------------")
                    # print(f"Type of depth_paths: {type(depth_paths)}")
                    # print(f"Length of depth_paths: {len(depth_paths)}")
                    # print(f"Shape of depth_paths: {depth_paths.shape}")                

                    # print(f"Type of poses: {type(poses)}")
                    # print(f"Length of poses: {len(poses)}")
                    # print(f"Shape of poses: {poses.shape}")

                    # for pose in poses:
                    #     print(f"Type of poses[0]: {type(pose)}")
                    #     print(f"Shape of poses[0]: {pose.shape}")

                    # print() 
                    # print()
                    # print(f"Type of poses[1]: {type(poses[1])}")
                    # print(f"Shape of poses[1]: {poses.shape[1]}")

                    # try:
                    #     print(f"Type of poses[0][0]: {type(poses[0][0])}")
                    #     print(f"Shape of poses[0][0]: {poses.shape[0][0]}")
                    #     print()
                    # except:
                    #     print("could not [][] index into poses")

                    # print(f"Type of intrinsics: {type(intrinsics)}")
                    # print(f"Length of intrinsics: {len(intrinsics)}")
                    # print(f"Shape of intrinsics: {intrinsics.shape}")

                    # for intrinsic in intrinsics:
                    #     print(f"Type of intrinsic: {type(intrinsic)}")
                    #     print(f"Shape of intrinsic: {intrinsic.shape}")

                    # print(f"Type of image_paths: {type(info['image_paths'])}")
                    # print(f"Length of image_paths: {len(info['image_paths'])}")
                    # print(f"Shape of image_paths: {info['image_paths'].shape}")
                    # print("--------------------------------------")
                    # raise Exception("stop loading data in")

                    """
                    Type of depth_paths: <class 'numpy.ndarray'>
                    Length of depth_paths: 2508
                    Shape of depth_paths: (2508,)
                    Type of poses: <class 'numpy.ndarray'>
                    Length of poses: 2508
                    Shape of poses: (2508,)
                    Type of intrinsics: <class 'numpy.ndarray'>
                    Length of intrinsics: 2508
                    Shape of intrinsics: (2508,)
                    Type of image_paths: <class 'numpy.ndarray'>
                    Length of image_paths: 559
                    Shape of image_paths: (559,)

                    """
                    # print("\n\n\n\n\n\n")

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
                            # print("Number of non-None depth paths:", depth_count)
                            # print("Number of non-None poses:", pose_count)
                            # print("Number of None depth paths:", depth_none_count)
                            # print("Number of None poses:", pose_none_count)
                            # print(f"All data present, and less than {maxVal}!. Proceeding with overlap calculation.")                        

                            # # Create a boolean mask where None values are marked as False
                            # # Create masks for non-None entries
                            # mask_depth = np.array([dp is not None for dp in depth_paths])
                            # mask_poses = np.array([ps is not None for ps in poses])
                            # mask_intrinsics = np.array([intrinsic is not None for intrinsic in intrinsics])

                            # # Combine masks to keep only entries where all corresponding elements are not None
                            # mask = mask_depth & mask_poses & mask_intrinsics

                            # # Filter arrays using the combined mask
                            # filtered_depth_paths = depth_paths[mask]
                            # filtered_poses = poses[mask]
                            # filtered_intrinsics = intrinsics[mask]


                            # assert(len(filtered_depth_paths) == depth_count and len(filtered_poses) == pose_count)
                            
                            # filtered_depth_paths, filtered_poses, filtered_intrinsics
                            # print(f"depth_paths.shape: {filtered_depth_paths.shape}")
                            # print(f"poses.shape: {filtered_poses.shape}")
                            # print(f"intrinsics.shape: {filtered_intrinsics.shape}")
                            """
                            depth_paths.shape: (42,)
                            poses.shape: (42,)
                            intrinsics.shape: (42,)

                            depth_paths: ['phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/2731109953_c47f11e87f_o.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/4616346459_8152e1c925_o.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/2731109953_7bbd9a5c31_b.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/12814398515_2b73018687_b.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/15911191105_1940ee1102_b.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/14012291198_1f164ae81f_o.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/17642178242_4651342242_b.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/14633000666_9b721fe156_o.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/8623461282_327b770d4b_b.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/3183350276_c8fd3c4347_o.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/16183274100_52cd63f2cf_o.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/14173910244_baaeb39b1a_o.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/20475699074_1600ef0b08_o.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/8383133438_a8bebaa2eb_b.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/3230368659_a0d1b157f9_o.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/3611113589_757d6bb9db_o.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/5139113315_b5dd63a897_o.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/7502697280_dffe5941db_b.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/7502676158_98b8d80091_b.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/8982824776_aa2cc65141_b.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/6477593945_78d2439aac_b.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/7819868364_36870b427d_b.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/8072618597_587d0676f1_o.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/7819877530_875498003d_b.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/3742382023_053e818345_b.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/279149446_76a18cd0e7_o.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/3219949451_62a2f4763e_o.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/3051851921_9f8b38e01b_o.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/11496937126_2e20c0b079_o.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/5891836143_7b16a9ac78_b.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/23050300142_3b24a51024_o.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/4199141817_7684ae2443_o.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/3183350276_e0094330bf_b.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/9329347566_8d9308846e_o.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/8982839504_9b62652ff5_b.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/2932500028_67e22dd87a_b.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/2090306129_3bf2638650_o.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/8166343728_2c8f74f66e_o.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/2401137845_4e93a99821_o.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/31485380926_f164a801d2_o.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/582233175_708cac455c_o.h5'
                            'phoenix/S6/zl548/MegaDepth_v1/0294/dense0/depths/8981641641_144284ff6c_b.h5']
                            poses: [array([[-0.99972585,  0.00444084,  0.02298924, -0.582747  ],
                                    [-0.0025746 ,  0.95504571, -0.2964474 , -0.482966  ],
                                    [-0.02327225, -0.29642532, -0.95477245, -1.49953   ],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
                            array([[-0.86847403,  0.00488399, -0.49571062, -3.27473   ],
                                    [ 0.16254231,  0.94747838, -0.27543551, -0.877939  ],
                                    [ 0.46832987, -0.31978254, -0.82365421, -2.38428   ],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
                            array([[-0.99972948,  0.00437591,  0.02284325, -0.583693  ],
                                    [-0.00257747,  0.95524887, -0.29579206, -0.478772  ],
                                    [-0.02311535, -0.29577093, -0.95497918, -1.51625   ],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
                            array([[-9.99752476e-01,  2.77850660e-03,  2.20741369e-02,
                                    -4.77733000e-01],
                                    [ 8.12059633e-04,  9.96067149e-01, -8.85978251e-02,
                                    -1.68906000e-02],
                                    [-2.22334922e-02, -8.85579695e-02, -9.95822855e-01,
                                    -1.31883000e+00],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                    1.00000000e+00]])
                            array([[-9.99479298e-01,  6.76844363e-03,  3.15487132e-02,
                                    -5.44184000e-01],
                                    [ 4.23067663e-04,  9.80416454e-01, -1.96935010e-01,
                                    -6.03309000e-01],
                                    [-3.22638210e-02, -1.96819118e-01, -9.79908812e-01,
                                    -2.72059000e+00],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                    1.00000000e+00]])
                            array([[-9.99669587e-01,  4.06585571e-03,  2.53808294e-02,
                                    -5.74567000e-01],
                                    [-8.64920446e-04,  9.81530888e-01, -1.91301771e-01,
                                    -6.12403000e-01],
                                    [-2.56898734e-02, -1.91260515e-01, -9.81203060e-01,
                                    -2.93338000e+00],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                    1.00000000e+00]])
                            array([[-0.99998614,  0.00280347,  0.00445595, -0.667458  ],
                                    [ 0.00210453,  0.98871932, -0.14976539, -0.197613  ],
                                    [-0.00482554, -0.14975394, -0.98871152, -1.54306   ],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
                            array([[-0.99914609,  0.02118594,  0.03547191, -0.479979  ],
                                    [ 0.01540492,  0.98764247, -0.15596485, -0.483033  ],
                                    [-0.03833783, -0.15528523, -0.98712548, -2.55965   ],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
                            array([[-9.99544744e-01,  1.53393144e-03,  3.01322300e-02,
                                    -5.34193000e-01],
                                    [-1.08838609e-02,  9.13129964e-01, -4.07523264e-01,
                                    -1.92533000e+00],
                                    [-2.81397549e-02, -4.07665692e-01, -9.12697561e-01,
                                    -4.05533000e+00],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                    1.00000000e+00]])
                            array([[-0.99975451,  0.00295795,  0.02195826, -0.581064  ],
                                    [-0.00600631,  0.91775767, -0.39709543, -1.30448   ],
                                    [-0.02132695, -0.39712983, -0.91751461, -2.91693   ],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
                            array([[-0.9612945 , -0.00208003,  0.27551508,  0.54847   ],
                                    [-0.04788665,  0.98601373, -0.15963641, -0.175843  ],
                                    [-0.27132961, -0.1666511 , -0.94794918, -0.747372  ],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
                            array([[-0.97734259,  0.00347704, -0.21163502, -1.62605   ],
                                    [ 0.04954506,  0.97584553, -0.21276935, -0.0312505 ],
                                    [ 0.20578328, -0.21843401, -0.95390766, -0.253784  ],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
                            array([[-0.98688832,  0.00299025, -0.16137692, -1.70324   ],
                                    [ 0.02303675,  0.99220181, -0.12249441, -0.0108484 ],
                                    [ 0.15975218, -0.1246059 , -0.97926126, -0.365282  ],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
                            array([[-0.99906826,  0.01340628,  0.04102307, -0.564272  ],
                                    [-0.00718389,  0.88561129, -0.46437166, -1.90453   ],
                                    [-0.04255599, -0.46423369, -0.88468982, -3.46337   ],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
                            array([[-0.67645415,  0.01488896,  0.73633423,  1.1893    ],
                                    [-0.11229099,  0.98602123, -0.123097  , -0.107505  ],
                                    [-0.72787397, -0.16595317, -0.66532626, -0.51331   ],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
                            array([[-0.99873061, -0.01800377,  0.04704287, -0.450043  ],
                                    [-0.02201655,  0.9960324 , -0.08622492, -0.104315  ],
                                    [-0.04530385, -0.08715119, -0.99516442, -1.01253   ],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
                            array([[-0.82443179, -0.01179799, -0.56583835, -4.10259   ],
                                    [ 0.10975919,  0.97746843, -0.18030082,  0.141872  ],
                                    [ 0.55521631, -0.21075169, -0.80456111,  0.968638  ],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
                            array([[ 8.71523272e-01, -1.62466636e-03,  4.90351452e-01,
                                    -2.20463000e+00],
                                    [-9.61858729e-02,  9.80000919e-01,  1.74202403e-01,
                                    5.30074000e-01],
                                    [-4.80827894e-01, -1.98986331e-01,  8.53937338e-01,
                                    1.77062000e+00],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                    1.00000000e+00]])
                            array([[ 9.22098828e-01, -1.86607637e-03,  3.86949956e-01,
                                    -2.31614000e+00],
                                    [-2.23193160e-02,  9.98067058e-01,  5.79999666e-02,
                                    2.62241000e-01],
                                    [-3.86310236e-01, -6.21181596e-02,  9.20274815e-01,
                                    1.25927000e+00],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                    1.00000000e+00]])
                            array([[-0.40665592,  0.00693633,  0.91355506,  4.13675   ],
                                    [-0.32715921,  0.93254827, -0.15271076, -1.02274   ],
                                    [-0.85299344, -0.36097869, -0.376957  , -2.18289   ],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
                            array([[-0.9993626 ,  0.00411972,  0.03546026, -0.513235  ],
                                    [-0.0061844 ,  0.95832039, -0.28562876, -1.02807   ],
                                    [-0.035159  , -0.285666  , -0.95768407, -3.19998   ],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
                            array([[-0.05556001,  0.0122891 , -0.99837972,  2.48214   ],
                                    [ 0.10354447,  0.99460371,  0.00648035,  0.0226272 ],
                                    [ 0.99307181, -0.10301665, -0.05653267,  0.730242  ],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
                            array([[-0.06709196,  0.01705559, -0.99760101,  2.49648   ],
                                    [ 0.0982385 ,  0.99510849,  0.01040612, -0.00904792],
                                    [ 0.99289872, -0.09730466, -0.0684393 ,  0.732719  ],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
                            array([[ 0.99762216, -0.01162534,  0.06793293, -0.175219  ],
                                    [ 0.0019285 ,  0.98999396,  0.14109654,  0.00404   ],
                                    [-0.06889348, -0.14063002,  0.98766233,  1.37536   ],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
                            array([[-9.99787933e-01,  2.82895676e-03,  2.03981749e-02,
                                    -5.75806000e-01],
                                    [-1.88246115e-03,  9.73817867e-01, -2.27321837e-01,
                                    -7.74343000e-01],
                                    [-2.05071908e-02, -2.27312028e-01, -9.73606028e-01,
                                    -3.01446000e+00],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                    1.00000000e+00]])
                            array([[-0.99947993,  0.03158652,  0.0064924 , -0.746779  ],
                                    [ 0.02578047,  0.90363612, -0.42752441, -1.70059   ],
                                    [-0.01937078, -0.42713469, -0.90398049, -3.45401   ],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
                            array([[-0.97239146, -0.00280004, -0.23333881, -2.41866   ],
                                    [ 0.047822  ,  0.97631469, -0.211004  ,  0.0372795 ],
                                    [ 0.22840293, -0.21633722, -0.94922617,  0.265874  ],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
                            array([[-9.99409584e-01,  6.89522780e-03,  3.36591695e-02,
                                    -4.95002000e-01],
                                    [ 1.31189352e-03,  9.86599348e-01, -1.63156382e-01,
                                    -7.64536000e-01],
                                    [-3.43331151e-02, -1.63015894e-01, -9.86025890e-01,
                                    -2.73343000e+00],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                    1.00000000e+00]])
                            array([[-0.07812388,  0.00807909, -0.99691092, -1.96074   ],
                                    [ 0.27251422,  0.96205621, -0.01355921,  0.29343   ],
                                    [ 0.9589748 , -0.2727317 , -0.07736123,  0.130886  ],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
                            array([[ 0.99605172, -0.03081168,  0.08325626, -0.261141  ],
                                    [ 0.02389427,  0.9962763 ,  0.08284082,  0.246341  ],
                                    [-0.0854987 , -0.0805244 ,  0.99307895,  5.23756   ],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
                            array([[-9.97638085e-01,  2.64834892e-02,  6.33788348e-02,
                                    -3.82349000e-01],
                                    [ 9.15452424e-04,  9.27729825e-01, -3.73251302e-01,
                                    -1.79661000e+00],
                                    [-6.86834321e-02, -3.72311694e-01, -9.25562850e-01,
                                    -4.17631000e+00],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                    1.00000000e+00]])
                            array([[-9.99995738e-01, -3.50622682e-04,  2.89846465e-03,
                                    -6.40514000e-01],
                                    [-6.77694685e-04,  9.93523402e-01, -1.13625659e-01,
                                    8.64102000e-02],
                                    [-2.83985272e-03, -1.13627139e-01, -9.93519405e-01,
                                    4.00750000e+00],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                    1.00000000e+00]])
                            array([[-0.99974724,  0.00311226,  0.02226595, -0.579105  ],
                                    [-0.00598842,  0.91773021, -0.39715917, -1.30483   ],
                                    [-0.0216702 , -0.39719212, -0.9174796 , -2.91365   ],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
                            array([[ 0.9982526 ,  0.00649297,  0.05873315, -0.090143  ],
                                    [-0.01479755,  0.98974331,  0.14208873, -0.168247  ],
                                    [-0.05720817, -0.14270955,  0.98810992, -1.40088   ],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
                            array([[-9.97358195e-01,  1.27484147e-03,  7.26292278e-02,
                                    -6.40843000e-01],
                                    [-3.18366741e-02,  8.91029718e-01, -4.52827195e-01,
                                    -2.17449000e+00],
                                    [-6.52920833e-02, -4.53943187e-01, -8.88635205e-01,
                                    -3.96062000e+00],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                    1.00000000e+00]])
                            array([[-0.99931079,  0.01453928,  0.03415476, -0.499323  ],
                                    [ 0.00961131,  0.99006948, -0.14024996, -0.143399  ],
                                    [-0.03585472, -0.13982503, -0.98952686, -1.4437    ],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
                            array([[-9.96318213e-01, -5.86741218e-04,  8.57302384e-02,
                                    -2.18917000e-01],
                                    [-1.09767917e-02,  9.92619363e-01, -1.20773801e-01,
                                    -8.68449000e-02],
                                    [-8.50266316e-02, -1.21270181e-01, -9.88971190e-01,
                                    -1.08687000e+00],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                    1.00000000e+00]])
                            array([[-0.04651738,  0.01578293, -0.99879279,  2.47687   ],
                                    [ 0.09894153,  0.99503117,  0.01111542, -0.0180802 ],
                                    [ 0.99400539, -0.09830502, -0.04784783,  0.639943  ],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
                            array([[-0.80556545, -0.03039052,  0.5917269 ,  2.61333   ],
                                    [-0.11002661,  0.98898645, -0.0989947 ,  0.242532  ],
                                    [-0.58220139, -0.14485241, -0.80003708,  1.36182   ],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
                            array([[-9.99951638e-01, -9.56656647e-03, -2.28108938e-03,
                                    -6.99228000e-01],
                                    [-7.89687348e-03,  9.19266843e-01, -3.93555727e-01,
                                    -2.04564000e+00],
                                    [ 5.86190685e-03, -3.93518680e-01, -9.19297931e-01,
                                    -4.45089000e+00],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                    1.00000000e+00]])
                            array([[-0.9998722 , -0.00439875,  0.01536995, -0.583942  ],
                                    [-0.00732746,  0.98056691, -0.19604808, -0.482357  ],
                                    [-0.0142089 , -0.19613565, -0.98047382, -2.18479   ],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
                            array([[-9.99775900e-01, -2.13711688e-03, -2.10613872e-02,
                                    -8.16152000e-01],
                                    [ 1.08813125e-02,  8.01524397e-01, -5.97863059e-01,
                                    -2.90113000e+00],
                                    [ 1.81589189e-02, -5.97958254e-01, -8.01321521e-01,
                                    -3.67602000e+00],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                    1.00000000e+00]])                                ]
                            intrinsics: [array([[4.62013e+03, 0.00000e+00, 7.88000e+02],
                                    [0.00000e+00, 4.62013e+03, 5.86000e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[1.28993e+03, 0.00000e+00, 8.00000e+02],
                                    [0.00000e+00, 1.28938e+03, 5.18500e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[2.94571e+03, 0.00000e+00, 5.04000e+02],
                                    [0.00000e+00, 2.94571e+03, 3.74500e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[1.5529e+03, 0.0000e+00, 3.4100e+02],
                                    [0.0000e+00, 1.5529e+03, 5.1300e+02],
                                    [0.0000e+00, 0.0000e+00, 1.0000e+00]])
                            array([[1.22719e+03, 0.00000e+00, 8.00000e+02],
                                    [0.00000e+00, 1.22753e+03, 5.12000e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[1.22846e+03, 0.00000e+00, 8.00000e+02],
                                    [0.00000e+00, 1.22886e+03, 5.21000e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[1.59268e+03, 0.00000e+00, 8.00000e+02],
                                    [0.00000e+00, 1.59326e+03, 5.99000e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[2.00696e+03, 0.00000e+00, 6.13500e+02],
                                    [0.00000e+00, 2.00696e+03, 7.82000e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[684.294,   0.   , 512.5  ],
                                    [  0.   , 684.294, 341.   ],
                                    [  0.   ,   0.   ,   1.   ]])
                            array([[1.67201e+03, 0.00000e+00, 8.00000e+02],
                                    [0.00000e+00, 1.67131e+03, 5.91500e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[2.23709e+03, 0.00000e+00, 5.33000e+02],
                                    [0.00000e+00, 2.23785e+03, 8.00000e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[2.22849e+03, 0.00000e+00, 8.00000e+02],
                                    [0.00000e+00, 2.22897e+03, 5.33000e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[3.81466e+03, 0.00000e+00, 6.00000e+02],
                                    [0.00000e+00, 3.81395e+03, 8.00000e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[710.304,   0.   , 508.5  ],
                                    [  0.   , 710.304, 381.5  ],
                                    [  0.   ,   0.   ,   1.   ]])
                            array([[3.57674e+03, 0.00000e+00, 5.32000e+02],
                                    [0.00000e+00, 3.57674e+03, 7.95000e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[2.02324e+03, 0.00000e+00, 8.00000e+02],
                                    [0.00000e+00, 2.02362e+03, 5.89000e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[3.35335e+03, 0.00000e+00, 8.00000e+02],
                                    [0.00000e+00, 3.35482e+03, 5.25000e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[2.27753e+03, 0.00000e+00, 5.16500e+02],
                                    [0.00000e+00, 2.27753e+03, 3.41000e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[2.21041e+03, 0.00000e+00, 5.16000e+02],
                                    [0.00000e+00, 2.21041e+03, 3.40500e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[1.02282e+03, 0.00000e+00, 5.17500e+02],
                                    [0.00000e+00, 1.02282e+03, 3.86000e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[1.88963e+03, 0.00000e+00, 5.10000e+02],
                                    [0.00000e+00, 1.88963e+03, 3.40000e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[1.83534e+03, 0.00000e+00, 5.05500e+02],
                                    [0.00000e+00, 1.83534e+03, 3.37000e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[2.25145e+03, 0.00000e+00, 8.00000e+02],
                                    [0.00000e+00, 2.25048e+03, 5.01500e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[1.68738e+03, 0.00000e+00, 5.43000e+02],
                                    [0.00000e+00, 1.68738e+03, 3.49000e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[1.18992e+03, 0.00000e+00, 3.41500e+02],
                                    [0.00000e+00, 1.18992e+03, 5.13500e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[1.21945e+03, 0.00000e+00, 6.17000e+02],
                                    [0.00000e+00, 1.21945e+03, 4.56500e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[6.47013e+03, 0.00000e+00, 7.96500e+02],
                                    [0.00000e+00, 6.47013e+03, 5.97000e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[1.45767e+03, 0.00000e+00, 5.95500e+02],
                                    [0.00000e+00, 1.45717e+03, 8.00000e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[1.70532e+03, 0.00000e+00, 8.00000e+02],
                                    [0.00000e+00, 1.70586e+03, 5.27500e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[2.05074e+03, 0.00000e+00, 5.32500e+02],
                                    [0.00000e+00, 2.05074e+03, 3.46500e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[1.00574e+03, 0.00000e+00, 4.59500e+02],
                                    [0.00000e+00, 1.00574e+03, 6.87500e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[4.63872e+03, 0.00000e+00, 8.00000e+02],
                                    [0.00000e+00, 4.63883e+03, 5.65500e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[1.10333e+03, 0.00000e+00, 5.27000e+02],
                                    [0.00000e+00, 1.10333e+03, 3.89500e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[3.32801e+03, 0.00000e+00, 5.33000e+02],
                                    [0.00000e+00, 3.32916e+03, 8.00000e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[1.00108e+03, 0.00000e+00, 3.87000e+02],
                                    [0.00000e+00, 1.00108e+03, 5.21000e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[1.70536e+03, 0.00000e+00, 2.87500e+02],
                                    [0.00000e+00, 1.70536e+03, 5.11500e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[3.05297e+03, 0.00000e+00, 5.14500e+02],
                                    [0.00000e+00, 3.05297e+03, 7.72000e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[2.35364e+03, 0.00000e+00, 8.00000e+02],
                                    [0.00000e+00, 2.35282e+03, 5.33000e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[2.05895e+03, 0.00000e+00, 8.00000e+02],
                                    [0.00000e+00, 2.05852e+03, 5.95500e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[1.07688e+03, 0.00000e+00, 5.33000e+02],
                                    [0.00000e+00, 1.07726e+03, 8.00000e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[2.77356e+03, 0.00000e+00, 5.87000e+02],
                                    [0.00000e+00, 2.77449e+03, 8.00000e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])
                            array([[1.02557e+03, 0.00000e+00, 5.20500e+02],
                                    [0.00000e+00, 1.02557e+03, 3.87000e+02],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])]
                            """
                            # print()
                            # # print(f"depth_paths: {filtered_depth_paths}")
                            # # print(f"poses: {filtered_poses}")                    
                            # print(f"intrinsics: {filtered_intrinsics}")          

                            # # Take the first three elements from each list
                            # aafiltered_depth_paths = filtered_depth_paths[:3]
                            # aafiltered_poses = filtered_poses[:3]
                            # aafiltered_intrinsics = filtered_intrinsics[:3]

                            # # Print the first three elements
                            # print("-----------------------")
                            # print(f"filtered_depth_paths: {aafiltered_depth_paths}")
                            # print(f"filtered_poses: {aafiltered_poses}")
                            # print(f"filtered_intrinsics: {aafiltered_intrinsics}")
                            # print(f"ground truth overlap matrix: {ground_truth_overlap_matrix[:3, :3]}")
                            # print("-----------------------")
                            np.save(f"/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory{scene}_depth", info["depth_paths"], allow_pickle=True)
                            np.save(f"/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory{scene}_poses", info["poses"], allow_pickle=True)
                            np.save(f"/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory{scene}_intrinsics", info["intrinsics"], allow_pickle=True)
                            np.save(f"/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory{scene}_groundOverlap", info["overlap_matrix"], allow_pickle=True)
                            

                            # Calculate the overlap matrix
                            calculated_overlap_matrix = calculate_overlap_matrix(info["depth_paths"], info["poses"], info["intrinsics"])
                            
                            full_overlap_matrix = calculated_overlap_matrix

                            # full_overlap_matrix = np.full((len(depth_paths), len(depth_paths)), -1.0)
                            # # Populate the full matrix using the mask to place calculated overlaps correctly
                            # indices = np.where(mask)[0]  # Get indices where mask is True
                            # for i, idx_i in enumerate(indices):
                            #     for j, idx_j in enumerate(indices):
                            #         if i <= j:  # Fill upper triangle and diagonal
                            #             full_overlap_matrix[idx_i, idx_j] = calculated_overlap_matrix[i, j]
                            #             if idx_i != idx_j:  # Fill lower triangle if not on the diagonal
                            #                 full_overlap_matrix[idx_j, idx_i] = calculated_overlap_matrix[i, j]


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

                            # Save the full overlap matrix
                            full_overlap_path = os.path.join(save_dir, f"{scene}_full_overlap.npy")
                            np.save(full_overlap_path, full_overlap_matrix)
                            print(f"Full overlap matrix saved to: {full_overlap_path}")

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
                    
                    ## SKIP USING OVERLAP MATRIX AND JUST PASS IN ALL THE PAIRS!!!
                    # pairs = np.stack(np.where(np.triu(mat, 1)), -1)

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

