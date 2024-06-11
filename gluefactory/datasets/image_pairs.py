"""
Simply load images from a folder or nested folders (does not have any split).
"""

from pathlib import Path

import numpy as np
import torch

from ..geometry.wrappers import Camera, Pose
from ..settings import DATA_PATH
from ..utils.image import ImagePreprocessor, load_image
from .base_dataset import BaseDataset

import pickle

# TEMPORARY
DATA_PATH = Path("/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/data")


def names_to_pair(name0, name1, separator="/"):
    return separator.join((name0.replace("/", "-"), name1.replace("/", "-")))


def parse_homography(homography_elems) -> Camera:
    return (
        np.array([float(x) for x in homography_elems[:9]])
        .reshape(3, 3)
        .astype(np.float32)
    )


def parse_camera(calib_elems) -> Camera:
    # assert len(calib_list) == 9
    K = np.array([float(x) for x in calib_elems[:9]]).reshape(3, 3).astype(np.float32)
    return Camera.from_calibration_matrix(K)


def parse_relative_pose(pose_elems) -> Pose:
    # assert len(calib_list) == 9
    if len(pose_elems) != 12:
        raise ValueError(f"Expected 12 elements for pose, got {len(pose_elems)}: {pose_elems}")   
    # print(f"in parse_relative_pose the pose_elems is {pose_elems} and len is {len(pose_elems)}")

    R, t = pose_elems[:9], pose_elems[9:12]
    R = np.array([float(x) for x in R]).reshape(3, 3).astype(np.float32)
    t = np.array([float(x) for x in t]).astype(np.float32)
    return Pose.from_Rt(R, t)


class ImagePairs(BaseDataset, torch.utils.data.Dataset):
    default_conf = {
        "pairs": "???",  # ToDo: add image folder interface
        "root": "???",
        "preprocessing": ImagePreprocessor.default_conf,
        "extra_data": "relative_pose",  # relative_pose, homography
    }

    def _init(self, conf):
        pair_f = (
            Path(conf.pairs) if Path(conf.pairs).exists() else DATA_PATH / conf.pairs
        )
        with open(str(pair_f), "r") as f:
            self.items = [line.rstrip() for line in f]
        self.preprocessor = ImagePreprocessor(conf.preprocessing)

    def get_dataset(self, split):
        return self

    def _read_view(self, name):
        # error ahdningly for pose files in the iageData folder
        if "pose" in name:
            return None
        path = DATA_PATH / self.conf.root / name
        img = load_image(path)
        return self.preprocessor(img)
    
   
    def save_to_pickle(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Dataset saved to {file_path}")

    @classmethod
    def load_from_pickle(cls, file_path):
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
        print(f"Dataset loaded from {file_path}")
        return dataset

    @classmethod
    def from_pickle_or_create(cls, conf, pickle_file):
        print("pickle_file", pickle_file)
        # if Path(pickle_file).exists():
        #     print(f"Loading dataset from {pickle_file}")
        #     return cls.load_from_pickle(pickle_file)
        # else:
        print("Creating new dataset")
        dataset = cls(conf)
        dataset.save_to_pickle(pickle_file)
        return dataset
        
    def __getitem__(self, idx):
        line = self.items[idx]
        pair_data = line.split(" ")
        # print("\n\nnpair_data:\n", pair_data, "\n\n")
        name0, name1 = pair_data[:2]
        # print("name0, name1 passed into read_view", name0, name1)
        data0 = self._read_view(name0)
        data1 = self._read_view(name1)

        # print(f"Total elements in pair_data: {len(pair_data)}")
        # print("pair_data", pair_data)
        # print("\npair_data[2:11], pair_data[11:20], pair_data[20:32]", pair_data[2:11], pair_data[11:20], pair_data[20:32], sep="\n")

        """
        check the pair_Data is correct, the number of rows positioning of everyting etc
        check that data0 and data1 is okay, espeically for the poses data - maybe change code from root to ignore poses when looping through dir
        find source of key error
        """

        # print(data0, data1)
        # try:
        #     print("data0.keys, data1.keys",data0.keys, data1.keys)
        # except:
        #     print("data 0 and 1 was not a dict")

        # Skip pairs where either view is None
        if data0 is None or data1 is None:
            # print("SKIPPED: data0, data1", data0, data1)
            return self.__getitem__((idx + 1) % len(self))

        data = {
            "view0": data0,
            "view1": data1,
        }
        # if self.conf.extra_data == "relative_pose":
        # TEMPORARY TO ENSURE IT RUNS
        if self.conf.extra_data == "homography":
            # print("homography mode", data1["transform"]
            #     @ parse_homography(pair_data[2:11])
            #     @ np.linalg.inv(data0["transform"])
            # )

            print("data1[transform]", data1["transform"])
            print("data0[transform]", data0["transform"])
            data["H_0to1"] = (
                data1["transform"]
                @ parse_homography(pair_data[2:11])
                @ np.linalg.inv(data0["transform"])
            )
        # DEFAULY IS USING RELATIVE_POSE
        else: 
            # print("data.keys",data.keys)
            # print("data",data)
            # print("\n\nnpair_data:\n", pair_data, "\n\n")
            # print(pair_data[2:11], pair_data[11:20], pair_data[20:32])

            # print("\n\n\n parse_camera(pair_data[2:11]).scale(data0[scales])", parse_camera(pair_data[2:11]).scale(
            #     data0["scales"]
            # ))
            # print("parse_camera(pair_data[11:20]).scale(data0[scales])", parse_camera(pair_data[11:20]).scale(
            #     data0["scales"]
            # ))
            # print("assign t_0to1 as", parse_relative_pose(pair_data[20:32]))

            data["view0"]["camera"] = parse_camera(pair_data[2:11]).scale(
                data0["scales"]
            )
            data["view1"]["camera"] = parse_camera(pair_data[11:20]).scale(
                data1["scales"]
            )
            data["T_0to1"] = parse_relative_pose(pair_data[20:32])
        # else:
        #     assert (
        #         self.conf.extra_data is None
        #     ), f"Unknown extra data format {self.conf.extra_data}"

        data["name"] = names_to_pair(name0, name1)
        return data

    def __len__(self):
        return len(self.items)


# """
# Simply load images from a folder or nested folders (does not have any split).
# """

# from pathlib import Path

# import numpy as np
# import torch

# from ..geometry.wrappers import Camera, Pose
# from ..settings import DATA_PATH
# from ..utils.image import ImagePreprocessor, load_image
# from .base_dataset import BaseDataset


# def names_to_pair(name0, name1, separator="/"):
#     return separator.join((name0.replace("/", "-"), name1.replace("/", "-")))


# def parse_homography(homography_elems) -> Camera:
#     return (
#         np.array([float(x) for x in homography_elems[:9]])
#         .reshape(3, 3)
#         .astype(np.float32)
#     )


# def parse_camera(calib_elems) -> Camera:
#     # assert len(calib_list) == 9
#     K = np.array([float(x) for x in calib_elems[:9]]).reshape(3, 3).astype(np.float32)
#     return Camera.from_calibration_matrix(K)


# def parse_relative_pose(pose_elems) -> Pose:
#     # assert len(calib_list) == 9
#     R, t = pose_elems[:9], pose_elems[9:12]
#     R = np.array([float(x) for x in R]).reshape(3, 3).astype(np.float32)
#     t = np.array([float(x) for x in t]).astype(np.float32)
#     return Pose.from_Rt(R, t)


# class ImagePairs(BaseDataset, torch.utils.data.Dataset):
#     default_conf = {
#         "pairs": "???",  # ToDo: add image folder interface
#         "root": "???",
#         "preprocessing": ImagePreprocessor.default_conf,
#         "extra_data": None,  # relative_pose, homography
#     }

#     def _init(self, conf):
#         pair_f = (
#             Path(conf.pairs) if Path(conf.pairs).exists() else DATA_PATH / conf.pairs
#         )
#         with open(str(pair_f), "r") as f:
#             self.items = [line.rstrip() for line in f]
#         self.preprocessor = ImagePreprocessor(conf.preprocessing)

#     def get_dataset(self, split):
#         return self

#     def _read_view(self, name):
#         path = DATA_PATH / self.conf.root / name
#         img = load_image(path)
#         return self.preprocessor(img)

#     def __getitem__(self, idx):
#         line = self.items[idx]
#         pair_data = line.split(" ")
#         name0, name1 = pair_data[:2]
#         data0 = self._read_view(name0)
#         data1 = self._read_view(name1)

#         data = {
#             "view0": data0,
#             "view1": data1,
#         }
#         if self.conf.extra_data == "relative_pose":
#             data["view0"]["camera"] = parse_camera(pair_data[2:11]).scale(
#                 data0["scales"]
#             )
#             data["view1"]["camera"] = parse_camera(pair_data[11:20]).scale(
#                 data1["scales"]
#             )
#             data["T_0to1"] = parse_relative_pose(pair_data[20:32])
#         elif self.conf.extra_data == "homography":
#             data["H_0to1"] = (
#                 data1["transform"]
#                 @ parse_homography(pair_data[2:11])
#                 @ np.linalg.inv(data0["transform"])
#             )
#         else:
#             assert (
#                 self.conf.extra_data is None
#             ), f"Unknown extra data format {self.conf.extra_data}"

#         data["name"] = names_to_pair(name0, name1)
#         return data

#     def __len__(self):
#         return len(self.items)