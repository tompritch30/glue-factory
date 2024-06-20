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

import pickle

from ..geometry.wrappers import Camera, Pose
from ..models.cache_loader import CacheLoader
from ..settings import DATA_PATH
from ..utils.image import ImagePreprocessor, load_image
from ..utils.tools import fork_rng
from ..visualization.viz2d import plot_heatmaps, plot_image_grid
from .base_dataset import BaseDataset
from .utils import rotate_intrinsics, rotate_pose_inplane, scale_intrinsics

from multiprocessing import Pool

# DATA_PATH = Path("/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/data/")


logger = logging.getLogger(__name__)
partial_mode = False

if partial_mode:
    scene_lists_path = Path(__file__).parent / "tartanSceneLists(Partial)"
else:
    scene_lists_path = Path(__file__).parent / "tartanSceneLists(Full)"

"""
python -m gluefactory.train sp+lg_megadepth     --conf gluefactory/configs/superpoint+lightglue_treedepth.yaml     train.load_experiment=sp+lg_homography
python -m gluefactory.train sp+lg_treedepth     --conf gluefactory/configs/superpoint+lightglue_treedepth.yaml     train.load_experiment=sp+lg_homography
python -m gluefactory.train sp+lg_treedepth     --conf gluefactory/configs/superpoint+lightglue_treedepth.yaml     train.load_experiment=sp+lg_densehomography
python -m gluefactory.train sp+lg_treedepthPartial     --conf gluefactory/configs/superpoint+lightglue_treedepth.yaml     train.load_experiment=sp+lg_densehomography
python -m gluefactory.train sp+lg_treedepthPartial     --conf gluefactory/configs/superpoint+lightglue_treedepth.yaml     train.load_experiment=sp+lg_homography
python -m gluefactory.train sp+lg_treedepthdebug     --conf gluefactory/configs/superpoint+lightglue_treedepth.yaml     train.load_experiment=sp+lg_homography
"""


def sample_n(data, num, seed=None):
    if len(data) > num:
        selected = np.random.RandomState(seed).choice(len(data), num, replace=False)
        return data[selected]
    else:
        return data


class TreeDepth(BaseDataset):
    if not partial_mode:
        infoDir = "fileLists"
    else:
        infoDir = "fileListsPartial" 
    default_conf = {
        # paths
        "data_dir": "syntheticForestData/",
        "depth_subpath": "depthData/",
        "image_subpath": "imageData/",
        "info_dir": infoDir,  # @TODO: intrinsics problem? -- is this required/whta is it for??
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
        # if not (DATA_PATH / conf.data_dir).exists():
        #     logger.info("Downloading the MegaDepth dataset.")
        #     self.download()
        ### I added
        logger.info(f"Initialized TreeDepth dataset with configuration: {conf}")
          

    def get_dataset(self, split):
        assert self.conf.views in [1, 2, 3]
        if self.conf.views == 3:
            # print("tripletdataset")
            return _TripletDataset(self.conf, split)
        else:
            # print("pairdataset")
            return _PairDataset(self.conf, split)

import os

def load_scene_data(base_dir, scene):
    # fix this
    if partial_mode == False:
        fileListDir = base_dir + '/fileLists'
    else:
        fileListDir = base_dir + '/fileListsPartial'
    print(f"fileListDir: {fileListDir}")
    
    # Removing 'L_' or 'R_' for pose files naming
    flow_scene = scene.replace('_L', '').replace('_R', '')
    
    # Define file paths for depth, image, and pose data
    depth_file_path = os.path.join(fileListDir, f"depthData_{scene}.txt")
    image_file_path = os.path.join(fileListDir, f"imageData_{scene}.txt")
    # THIS IS FLOW DATA!! need to be poses!  
    # NEED TO REVIEW THIS FILE NAMING???  
    flow_file_path  = os.path.join(fileListDir, f"flowData_{flow_scene}.txt")

    # Function to read data from a text file and convert to numpy array
    def load_data_from_file(file_path):
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                return np.array([line.strip() for line in file.readlines()])
        else:
            print(f"{file_path} no data found")
            return np.array([])  # Return an empty array if file does not exist

    # Load data
    depth_data = load_data_from_file(depth_file_path)
    image_data = load_data_from_file(image_file_path)
    flow_data = load_data_from_file(flow_file_path)

    
    def pose_to_matrix(pose):
        import numpy as np
        from scipy.spatial.transform import Rotation as R
        """Convert a pose (tx, ty, tz, qx, qy, qz, qw) to a 4x4 transformation matrix."""

        tx, ty, tz, qx, qy, qz, qw = pose
        rotation = R.from_quat([qx, qy, qz, qw])
        rotation_matrix = rotation.as_matrix()

        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = [tx, ty, tz]

        return transformation_matrix

    def load_poses_from_file(file_path):
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                lines = file.readlines()
                # return as 4x4 matrix as expected
                return np.array([pose_to_matrix(np.fromstring(line, sep=' ')) for line in lines])

        else:
            return np.array([])

    # Handle pose data based on scene naming
    poses = []

    if partial_mode == False:
        # Full data loading
        if '_L_' in scene:
            pose_file_path = os.path.join(base_dir, "poseData", f"{scene}_pose_left.txt")
        elif '_R_' in scene:
            pose_file_path = os.path.join(base_dir, "poseData", f"{scene}_pose_right.txt")
        else:
            print("Neither L nor R in scene identifier. Check naming convention!")
    else: # For partial mode
        pose_file_path = os.path.join(base_dir, "poseDataPartial", f"{scene}_pose_left.txt")

    if os.path.exists(pose_file_path):
        poses = load_poses_from_file(pose_file_path)
    else:
        print("pose_file_path not exist", pose_file_path)

    length = max(len(depth_data), len(image_data), len(flow_data))
    K = np.array([[320.0, 0, 320.0], [0, 320.0, 240.0], [0, 0, 1.0]])
    camera_intrinsics = np.array([K] * length)

    # print(len(image_data), len(depth_data), len(flow_data), len(poses))

    # remove SFW_E_L_P001 from train list

    # give code to check that every image, depth, flow and pose has a corresponding value
    # temporary fix
    image_data = [i for i in image_data if ".txt" not in i]
    flow_data = [f for f in flow_data if "flow.npy" in f]
    # temp fix the off by one error
    # flow_data = flow_data[1:]
    # mask_paths = [os.path.join(fileListDir, f) for f in os.listdir(fileListDir) if f.endswith("_mask.npy")]

    # flow_data = [load_npy_file(path) for path in flow_paths]
    # mask_data = [load_npy_file(path) for path in mask_paths]


    # print(len(image_data), len(depth_data), len(flow_data), len(poses))

    image_data, depth_data  = sorted(image_data), sorted(depth_data)

    """
    Off by 1: 
    imageData/SF_H_L_P001/000549_left.png depthData/SF_H_L_P001/000549_left_depth.npy flowData/SF_H_P001/000548_000549_flow.npy
    BUT
    389 389 778 389
    """
    flow_data = sorted(flow_data, key=lambda x: int(str(x).split("/")[-1].split("_")[0]) if str(x).split("/")[-1].split("_")[0].isdigit() else 0)
    # print(len(image_data), len(depth_data), len(flow_data), len(poses))

    # mask_data = sorted(mask_data, key=lambda x: int(x.split("/")[-1].split("_")[0]))
    # print(image_data[-1], depth_data[-1], flow_data[-1])

    # for i in range(-1, -10):
    #     print(f"image_data[{i}]: {image_data[i]}, depth_data[{i}]: {depth_data[i]}, flow_data[{i}]: {flow_data[i]}, poses[{i}]: {poses[i]}", sep="\n")

    # Add the times two if not including the .npy filter
    ## will cause error with the off by one error with overlap matrices being too small - may have to check overlap matrix calcualtion ensure correct order of depth values etc
    # image_data = image_data[:-1]
    # depth_data = depth_data[:-1]
    # # flow_data = flow_data[:-1]
    # poses = poses[:-1]

    # for i in range(len(image_data)):
    #     print(f"Image: {image_data[i]}, Depth: {depth_data[i]}, Flow: {flow_data[i]}, Poses: {poses[i]}")

    # Add plus one unti review oervlap matrix
    try:
        assert len(image_data) == len(depth_data) == len(flow_data) + 1 == len(poses)
    except:
        print(f"LENGTH ASSERTION: Scene: {scene}, image_data: {len(image_data)}, depth_data: {len(depth_data)}, flow_data: {len(flow_data)}, poses: {len(poses)}")
        # print(len(image_data), len(depth_data), len(flow_data), len(poses))

    # print(f"image_paths shape: {len(image_data)}")
    # print(f"depth_paths shape: {len(depth_data)}")
    # print(f"flow_data shape: {len(flow_data)}")
    # print(f"intrinsics shape: {camera_intrinsics.shape}")
    # print(f"poses shape: {len(poses)}")

    return {
        "image_paths": image_data,
        "depth_paths": depth_data,
        "flow_data": flow_data,
        "intrinsics" : camera_intrinsics, 
        "poses" : poses,
        # NEED TO REVIEW IF NEED LEFT AND RIGHT POSES??
    }

def load_npy_file(partial_file_path):
    base_directory = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/data/syntheticForestData"
    file_path = os.path.join(base_directory, partial_file_path)

    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        print(f"File not found: {file_path}")
        return None

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

def project_points(depth, intrinsics, pose, test_mode=None):
    print(f"inputs to project points are: depth: {depth}, intrinsics: {intrinsics}, pose : {pose}")
    h, w = depth.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.flatten()

    x = (x.flatten() - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (y.flatten() - intrinsics[1, 2]) * z / intrinsics[1, 1]
    points = np.vstack((x, y, z, np.ones_like(z)))

    transformed_points = pose @ points
    # transformed_points /= transformed_points[2, :]
    ### UPDATED 19-06-24 for homogenous coordinates!
    transformed_points /= transformed_points[3, :]

    x_proj = intrinsics[0, 0] * transformed_points[0, :] / transformed_points[2, :] + intrinsics[0, 2]
    y_proj = intrinsics[1, 1] * transformed_points[1, :] / transformed_points[2, :] + intrinsics[1, 2]

   
    if test_mode == 'full_overlap':
        print(f"projection test mode: {test_mode}")
        valid = np.ones_like(x_proj, dtype=bool)  # All points are valid
    elif test_mode == 'no_overlap':
        print(f"projection test mode: {test_mode}")
        valid = np.zeros_like(x_proj, dtype=bool)  # No points are valid
    elif test_mode == 'partial_overlap':
        print(f"projection test mode: {test_mode}")
        valid = (x_proj >= 0) & (x_proj < w) & (y_proj >= 0) & (y_proj < h)
        # Force half of the points to be invalid regardless of the projection
        valid[:valid.size // 2] = False
    else:
        valid = (x_proj >= 0) & (x_proj < w) & (y_proj >= 0) & (y_proj < h)

    print(f"Valid Points Count: {np.count_nonzero(valid)} / {valid.size}")  # Debugging output

    return np.count_nonzero(valid), valid.size
                    
def save_overlap_matrix(data_root, scene_name, overlap_matrix):
    overlap_matrices_dir = os.path.join(data_root, 'overlappingMatrices')
    os.makedirs(overlap_matrices_dir, exist_ok=True)

    overlap_file_path = os.path.join(overlap_matrices_dir, f"{scene_name}.npz")
    print(f"Saving overlap matrix of shape {overlap_matrix.shape} to {overlap_file_path}")
    np.savez_compressed(overlap_file_path, overlap_matrix=overlap_matrix)


# def calculate_overlap_for_pair(args):
#     i, j, depth_paths, poses, intrinsics = args
#     depth_i = load_npy_file(depth_paths[i])
#     pose_i = poses[i]
#     depth_j = load_npy_file(depth_paths[j])
#     pose_j = poses[j]
    
#     if depth_i is None or depth_j is None or pose_i is None or pose_j is None:
#         return i, j, 0.0  # Return 0 overlap if data is missing

#     pose = np.linalg.inv(pose_j) @ pose_i
#     count, total = project_points(depth_i, intrinsics[i], pose)
#     overlap = count / total
#     return i, j, overlap


def calculate_overlap_for_pair(args, test_mode=None):
    i, j, depth_paths, poses, intrinsics = args[:5]

    # Extract test_mode if it's explicitly included in args; otherwise, it defaults to None
    # test_mode = False
    if len(args) > 5:
        test_mode = args[5]

    pose_i = poses[i]
    pose_j = poses[j]
    if not test_mode:
        print("loading depth data in calc overlap for pair")
        depth_i = load_npy_file(depth_paths[i])        
        depth_j = load_npy_file(depth_paths[j])
    else:
        # Create dummy depth maps for testing purposes
        depth_i = depth_paths[0]
        depth_j = depth_paths[1]
        print("depth_i, depth_j", depth_i, depth_j)
        # pose_i = np.eye(4)
        # pose_j = np.eye(4)
    
    if depth_i is None or depth_j is None or pose_i is None or pose_j is None:
        logging.warning(f"Missing data for frames {i} or {j}")
        return i, j, 0.0  # Return 0 overlap if data is missing
    
    try:
        pose = np.linalg.inv(pose_j) @ pose_i
        #  THIS WORKED BEFORE IN TESTS , is the same :( np.linalg.inv(poses[j]) @ poses[i]
        count, total = project_points(depth_i, intrinsics[i], pose)
        overlap = count / total
        return i, j, overlap
    except Exception as e:
        logging.error(f"Error calculating overlap between frames {i} and {j}: {e}")
        # raise Exception("calculate overlap fair)")
        return i, j, 0.0


# def calculate_overlap_matrix(depth_paths, poses, intrinsics):
#     """Calculates the overlap matrix between frames in parallel.    
#     Args:
#         depth_paths: List of paths to depth maps.
#         poses: List of camera poses.
#         intrinsics: List of camera intrinsics.
#         print_interval: Number of iterations after which to print an overlap value.
#     """
#     print("calculating overlalping matrix for", depth_paths.shape, poses.shape, intrinsics.shape)
#     num_frames = len(depth_paths)
#     overlap_matrix = np.zeros((num_frames, num_frames))
    
#     # Prepare arguments for parallel processing
#     pairs = [(i, j, depth_paths, poses, intrinsics) 
#             for i in range(num_frames) for j in range(i + 1, num_frames)]
    
#     with Pool() as p:  # Use all available CPU cores
#         results = p.map(calculate_overlap_for_pair, pairs)

#     # Fill overlap matrix
#     for i, j, overlap in results:
#         overlap_matrix[i, j] = overlap
#         overlap_matrix[j, i] = overlap  # Make symmetric
#         if i % 500 == 0:
#             print(f"Overlap between frame {i} and {j}: {overlap}")  # Print at intervals

#     return overlap_matrix


def calculate_overlap_matrix(depth_paths, poses, intrinsics):
    logging.info("Calculating overlap matrix")
    num_frames = len(depth_paths)
    overlap_matrix = np.zeros((num_frames, num_frames))
    
    pairs = [(i, j, depth_paths, poses, intrinsics) 
             for i in range(num_frames) for j in range(i + 1, num_frames)]
    
    # with Pool() as p:
    #     results = p.map(calculate_overlap_for_pair, pairs)

    results = []
    for pair in pairs:
        #### WILL NOT USE THE TEST MODE FOR CACL!!!
        results.append(calculate_overlap_for_pair(pair, test_mode=False))

    for i, j, overlap in results:
        overlap_matrix[i, j] = overlap
        overlap_matrix[j, i] = overlap  # Make symmetric
        logging.debug(f"Overlap between frame {i} and {j}: {overlap}")

    return overlap_matrix

class _PairDataset(torch.utils.data.Dataset):
    def __init__(self, conf, split, load_sample=True):
        self.root = DATA_PATH / conf.data_dir
        assert self.root.exists(), self.root
        self.split = split
        self.conf = conf

        # redefine it
        # bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/datasets/tartanSceneLists")
        if not partial_mode:
            scene_lists_path = Path("/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/datasets/tartanSceneLists(Full)")
        else:
            scene_lists_path = Path("/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/datasets/tartanSceneLists(Partial)")

        split_conf = conf[split + "_split"]
        if isinstance(split_conf, (str, Path)):            
            scenes_path = scene_lists_path / split_conf
            # print("scenes_path", scenes_path)
            scenes = scenes_path.read_text().rstrip("\n").split("\n")
            # print("scenes", scenes)
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
        print("root and info", self.root, self.conf.info_dir)
        # root /vol/bitbucket/tp4618/SuperGlueThesis/external/glue-factory/data/syntheticForestData
        #  self.conf.info_dir poseData/
        self.scenes = []
        count = 0
        
        # for every list in the sceneList file 
        for scene in scenes:
            path = self.info_dir / (scene) # + ".npz")            

            self.scenes.append(scene)
            # scene = SFW_E_L_P001
            # Base = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/data/syntheticForestData/fileLists"
            # convert into fileName
            # depthData_SFW_E_L_P001.txt
            # imageData_SFW_E_L_P001.txt
            # Note the lack of L and R
            # depthData_SFW_E_P001.txt
            # get full path e.g. "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/data/syntheticForestData/fileLists/depthData_SFW_E_P000.txt"
            # make the text file into a numpy array, each row into an item 
            # it is the equiaaletn value for info["image_paths"]

            """
            depthData/SF_E_L_P001/000191_left_depth.npy
            depthData/SF_E_L_P001/000032_left_depth.npy     
            ...
            imageData/SF_E_L_P005/000158_left.png
            imageData/SF_E_L_P005/000255_left.png  
            ...
            flowData/SFW_H_P017/000704_000705_flow.npy
            flowData/SFW_H_P017/001010_001011_flow.npy  

            """
            # np.array([[320.0, 0, 320.0], [0, 320.0, 240.0], [0, 0, 1.0]])           
            
            base_directory = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/data/syntheticForestData"
            info = load_scene_data(base_directory, scene)

            # print(f"Depth Paths type: {type(info['depth_paths'])}\n\n")
            # print(f"Intrinsics type: {type(info['intrinsics'])}\n\n")
            # print(f"Poses type: {type(info['poses'])}\n\n")
            # print(f"Depth Paths: {info['depth_paths']}\n\n")
            # print(f"Intrinsics: {info['intrinsics']}\n\n")
            # print(f"Poses: {info['poses']}\n\n")
                
            
            ## I added 16-06-24 for having each as shape attribute
            self.images[scene] = np.array(info["image_paths"])
            self.depths[scene] = np.array(info["depth_paths"])
            self.poses[scene] = np.array(info["poses"])            
            self.intrinsics[scene] = np.array(info["intrinsics"])
            # c = 0
            # if c < 1:
            #     print("image, depth, pose, intrinsrtic", info["image_paths"].shape, info["depth_paths"].shape, info["poses"].shape, info["intrinsics"].shape, sep="\n")
            #     # print(info["image_paths"], info["depth_paths"], info["poses"], info["intrinsics"], sep = "\n\n")
            # c+=1

        # data_root = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/data/syntheticForestData"
        # scene_list_file = os.path.join(data_root, "scene_list.pkl")  # Specify a filename for the pickled data
        # with open(scene_list_file, 'wb') as f:  # Use 'wb' for binary write mode
        #     pickle.dump(scenes, f)
        # print("finished the pickle!!!")

        if load_sample:
            self.sample_new_items(conf.seed)
            # assert len(self.items) > 0
            assert len(self.items) > 0, "No items sampled; check configuration."

        
       


    def sample_new_items(self, seed):
        # logger.info("Sampling new %s data with seed %d.", self.split, seed)
        logger.info(f"Sampling new items for {self.split} with seed {seed}.")
   
        self.items = []
        split = self.split
        num_per_scene = self.conf[self.split + "_num_per_scene"]
        # print(f"!!!!!sample_new_items: num_per_scene: {num_per_scene}")
        if isinstance(num_per_scene, Iterable):
            num_pos, num_neg = num_per_scene
        else:
            num_pos = num_per_scene
            num_neg = None
        if split != "train" and self.conf[split + "_pairs"] is not None:
            # Fixed validation or test pairs
            assert num_pos is None and num_neg is None, "Pairs are pre-defined, no sampling needed."
            # assert num_pos is None
            # assert num_neg is None
            assert self.conf.views == 2
            pairs_path = scene_lists_path / self.conf[split + "_pairs"]
            # print(f"the pairs path used is: {pairs_path}")
            for line in pairs_path.read_text().rstrip("\n").split("\n"):
                # print(f"the line used for im0 and im1 is {line}")
                im0, im1 = line.split(" ")
                scene = im0.split("/")[0]
                assert im1.split("/")[0] == scene
                im0, im1 = [self.conf.image_subpath + im for im in [im0, im1]]
                assert im0 in self.images[scene]
                assert im1 in self.images[scene]
                idx0 = np.where(self.images[scene] == im0)[0][0]
                idx1 = np.where(self.images[scene] == im1)[0][0]
                self.items.append((scene, idx0, idx1, 1.0))
                ### I added
                # logger.info(f"Added fixed pair: {scene}, {im0}, {im1}")
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
                logger.info(f"Scene {scene}: {len(ids)} items added.")
        else:
            for scene in self.scenes:
                ## Code to replace
                # path = self.info_dir / (scene + ".npz")
                # assert path.exists(), f"Info file missing: {path}"
                # # assert path.exists(), path
                # info = np.load(str(path), allow_pickle=True)                
            
                base_directory = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/data/syntheticForestData/"
                info = load_scene_data(base_directory, scene)
                # print(scene)
                # valid = (self.images[scene] != None) & (  # noqa: E711
                #         self.depths[scene] != None  # noqa: E711
                #     )
                # except:
                #     print("failed for: ", scene)
                # valid = (self.images[scene] != None) & (  # noqa: E711
                #     self.depths[scene] != None  # noqa: E711
                # )
                # # 1. Find the size difference
                # size_diff = len(self.images[scene]) - len(self.depths[scene])
                # self.images[scene] = sorted(self.images[scene])
                # self.depths[scene] = sorted(self.depths[scene])

                # # 2. Safely trim the end of self.images (avoiding errors if size_diff is negative)
                # print(f"Removing  images: {self.images[scene][-size_diff:]}")  # Print the removed image
                # self.images[scene] = self.images[scene][:-size_diff]
                
                # have psoe files in imageData folder
                self.images[scene] = [img for img in self.images[scene] if "pose" not in img]

                # 3. Perform your validation after trimming
                ### MYVALID
                # NEED TO FIND THE SORUCE OF THIS DATA where self.images is loaded
                self.images[scene] = np.array(self.images[scene])
                self.depths[scene] = np.array(self.depths[scene])
                # print("self.images[scene].shape, self.depths[scene].shape", self.images[scene].shape, self.depths[scene].shape)
                
                valid = (self.images[scene] != None) & (  # noqa: E711
                    self.depths[scene] != None  # noqa: E711
                )
                # print("valid array:", valid)


                # valid = (self.images.get(scene) is not None) and (self.depths.get(scene) is not None)
                # valid = np.logical_and(self.images[scene] != None, self.depths[scene] != None)

                #  valid = (self.images[scene] != None) | (self.depths[scene] != None)

                ind = np.where(valid)[0]
                # print(info.keys())
                # print(info["image_paths"], info["depth_paths"], info["poses"], info["intrinsics"], sep = "\n\n")                   
                
                ### ^Calculate Overlap matrix run time ###
                # info["image_paths"], info["depth_paths"], info["poses"], info["intrinsics"]
                # print(np.array(info["depth_paths"]), np.array(info["poses"]), np.array(info["intrinsics"]), sep="\n\n")
                b_dir = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/data/syntheticForestData/overlappingMatrices"
                def load_overlap_matrix(base_directory, scene):
                    # Construct the filename for the overlap matrix
                    filename = os.path.join(base_directory, scene + ".npz")
                    try:
                        data = np.load(filename)
                        overlap_matrix = data['overlap_matrix']
                        print(f"Loaded overlap matrix for {scene} from file.")
                        return overlap_matrix, True  # Return matrix and success flag
                    except FileNotFoundError:
                        print(f"Skipping {scene}: Overlap matrix file not found.")
                        return None, False 
                    # # Load the overlap matrix from the file
                    # data = np.load(filename)
                    # overlap_matrix = data['overlap_matrix']
                    # print(f"Loaded overlap matrix for {scene} from file.")
                    # return overlap_matrix
                # NEED TO RECALC THE OVERLAP MATRIX FOR THIS ONE: [1 2 3] 

                ##'######## overlapoooo
                # if False or artial_mode == False:
                #     overlap_matrix, success = load_overlap_matrix(b_dir, scene)
                
                ############ overlapoooo
                success = False
                if partial_mode:
                    success = False  # Force recalculation of overlap matrix

                """
                
                """

                if not success:
                    print("did not load overlap matrix for scene:", scene)                    
                    
                    depth_paths = np.array(info["depth_paths"])
                    poses = np.array(info["poses"])
                    intrinsics = np.array(info["intrinsics"])
                    
                    print(f"depth_paths.shape: {depth_paths.shape}")
                    print(f"poses.shape: {poses.shape}")
                    print(f"intrinsics.shape: {intrinsics.shape}")
                    """
                    depth_paths.shape: (3,)
                    poses.shape: (3, 4, 4)
                    intrinsics.shape: (6, 3, 3)

                    depth_paths: ['depthData/SF_E_L_APartial/000000_left_depth.npy'
                    'depthData/SF_E_L_APartial/000001_left_depth.npy'
                    'depthData/SF_E_L_APartial/000002_left_depth.npy']
                    poses: [[[-7.17845000e-01  6.96202956e-01  0.00000000e+00  5.53746414e+01]
                    [-6.96202956e-01 -7.17845000e-01  0.00000000e+00 -2.64403152e+01]
                    [ 0.00000000e+00  0.00000000e+00  1.00000000e+00 -9.90465045e-01]
                    [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]

                    [[-7.09192097e-01  7.04933809e-01  1.07188313e-02  5.54618149e+01]
                    [-7.05003892e-01 -7.09012179e-01 -1.64694201e-02 -2.63598995e+01]
                    [-4.01006915e-03 -1.92368004e-02  9.99806914e-01 -9.21572864e-01]
                    [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]

                    [[-6.99696462e-01  7.14067656e-01  2.30704096e-02  5.54505005e+01]
                    [-7.14388747e-01 -6.98893452e-01 -3.45927944e-02 -2.62395477e+01]
                    [-8.57783747e-03 -4.06856969e-02  9.99135173e-01 -8.79076540e-01]
                    [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]]
                    intrinsics: [[[320.   0. 320.]
                    [  0. 320. 240.]
                    [  0.   0.   1.]]

                    [[320.   0. 320.]
                    [  0. 320. 240.]
                    [  0.   0.   1.]]

                    [[320.   0. 320.]
                    [  0. 320. 240.]
                    [  0.   0.   1.]]

                    [[320.   0. 320.]
                    [  0. 320. 240.]
                    [  0.   0.   1.]]

                    [[320.   0. 320.]
                    [  0. 320. 240.]
                    [  0.   0.   1.]]

                    [[320.   0. 320.]
                    [  0. 320. 240.]
                    [  0.   0.   1.]]]
                    """
                    print()
                    print(f"depth_paths: {depth_paths}")
                    print(f"poses: {poses}")                    
                    print(f"intrinsics: {intrinsics}")
                    
                    # print(np.array(info["depth_paths"]), np.array(info["poses"]), np.array(info["intrinsics"]))
                    overlap_matrix = calculate_overlap_matrix(depth_paths, poses, intrinsics)
                    # # overlap_matrix = np.array([1, 2, 3])
                    save_overlap_matrix(base_directory, scene, overlap_matrix)
                ### ^Calculate Overlap matrix run time ###

                #### Load overlap matrix #####
                print("\n\n\n\n IN TESTING ZONE ")
                test_full_overlap()
                test_no_overlap()
                test_partial_overlap()
                test_overlap_matrix()
                print("\n\n\n\n END TESTING ZONE ")
                                
                # if scene == "SFW_E_L_P000":
                #     print("skipping SFW_E_L_P000")
                #     continue 
                # # print("loading overlap matrix for:scene:", scene)
                # overlap_matrix, success = load_overlap_matrix(b_dir, scene)
                # if not success:
                #     print("did not load overlap matrix for scene:", scene)
                #     continue 
                #### ^^Load overlap matrix #####

                info["overlap_matrix"] = overlap_matrix
                # print("overlap_matrix shape:", info["overlap_matrix"].shape)

                # print("overlap_matrix:", overlap_matrix, "\n\n")
                mat = info["overlap_matrix"][valid][:, valid]
                # print("Mat:", mat, "\n\n")
                # print("Mat shape:", mat.shape, "\n\n")

                if partial_mode:
                    if num_pos is not None:
                        num_images = len(self.depths[scene])
                        print(num_images)
                        pairs = np.stack(np.triu_indices(num_images, 1), -1)  # Create all unique pairs (0,1), (0,2), (1,2)
                        if pairs.size == 0:
                            print("Insufficient data to form any pairs.")
                    else:
                        print("No positioning number specified, skipping pair formation.")
                    pass
                elif num_pos is not None:
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
                    if pairs:
                        pairs = np.concatenate(pairs, 0)
                    else:
                        print("No valid pairs formed; check overlap criteria or add more data.")
                else:                    
                    pairs = (mat > self.conf.min_overlap) & (
                        mat <= self.conf.max_overlap
                    )
                    pairs = np.stack(np.where(pairs), -1)
                    if pairs.size == 0:
                        print("No valid pairs formed; check overlap criteria or add more data.")

                # print(f"\n\ninput to treepdepth pairs is {pairs}")
                # 0-1 317-318

                # pairs = pairs[:, 2:]
                # print(pairs.shape)
                # print(pairs)
                pairs = [(scene, ind[i], ind[j], mat[i, j]) for i, j in pairs]
                if num_neg is not None:
                    neg_pairs = np.stack(np.where(mat <= 0.0), -1)
                    neg_pairs = sample_n(neg_pairs, num_neg, seed)
                    pairs += [(scene, ind[i], ind[j], mat[i, j]) for i, j in neg_pairs]
                self.items.extend(pairs)
                # else:
                #     print("scene invalid:", scene)
                
                ### I added
                #
                # logger.info(f"Scene {scene}: {len(pairs)} pairs added.")
        if self.conf.views == 2 and self.conf.sort_by_overlap:
            self.items.sort(key=lambda i: i[-1], reverse=True)
        else:
            np.random.RandomState(seed).shuffle(self.items)

    def _read_view(self, scene, idx):
        path = self.root / self.images[scene][idx]
        ### I added
        #logger.info(f"Reading view from {path}")        
        # K = self.intrinsics[scene][idx].astype(np.float32, copy=False)
        # T = self.poses[scene][idx].astype(np.float32, copy=False)
        
        ## likely loading poses wrong?!
        if idx >= len(self.intrinsics[scene]):
            print("idx,len(self.intrinsics[scene])", idx,len(self.intrinsics[scene]))
            idx = len(self.intrinsics[scene]) - 1
            print(f"new INTRINSICS idx is: {idx}")
            K = self.intrinsics[scene][idx].astype(np.float32, copy=False)            
        else:
            K = self.intrinsics[scene][idx].astype(np.float32, copy=False)
            

        if idx >= len(self.poses[scene]):
            print("idx,len(self.poses[scene])", idx,len(self.poses[scene]))
            idx = len(self.poses[scene]) - 1
            print(f"new POSES idx is: {idx}")
            T = self.poses[scene][idx].astype(np.float32, copy=False)            
        else:            
            T = self.poses[scene][idx].astype(np.float32, copy=False)
        


        # try:
        #     # print("UPDATED VERSION")
        #     K = self.intrinsics[scene][idx].astype(np.float32, copy=False)
        #     T = self.poses[scene][idx].astype(np.float32, copy=False)
        # except Exception as e:  # Catching a general Exception
        #     print(f"Exception occurred: {type(e).__name__} - {e}")   
        #     print("path, self.root , self.images[scene][idx]", path, self.root , self.images[scene][idx])
        #     # read pose data
        #     print("scene, idx", scene, idx)
        #     print("self.intrinsics[scene][idx].shape", self.intrinsics[scene][idx].shape)
        #     print("self.poses[scene][idx].shape", self.poses[scene][idx].shape)
        #     max_idx = len(self.poses[scene]) - 1  # Get the last valid index

        #     MAX_BOUNDS = 5
        #     # Check if the requested index is close to the valid range
        #     if abs(idx - max_idx) <= MAX_BOUNDS:
        #         print(f"Adjusting out-of-bounds index {idx} to {max_idx} for scene {scene}")
        #         idx = max_idx  # Adjust index to the last valid one
        #         K = self.intrinsics[scene][idx].astype(np.float32, copy=False)
        #         T = self.poses[scene][idx].astype(np.float32, copy=False)
        #     else:
        #         raise IndexError(f"Skipping index {idx}: Index too far out of bounds for scene {scene}")


        ### I added
        #logger.info(f"Pose data: {K.shape}, {T.shape}")

        # read image
        if self.conf.read_image:
            img = load_image(self.root / self.images[scene][idx], self.conf.grayscale)
            ### I added
            #logger.info(f"Loaded image {path.name} with shape {img.shape}")
        else:
            size = PIL.Image.open(path).size[::-1]
            img = torch.zeros(
                [3 - 2 * int(self.conf.grayscale), size[0], size[1]]
            ).float()
            ### I added
            #logger.info(f"Created placeholder image with shape {img.shape}")

        # read depth
        if self.conf.read_depth:
            localRoot = Path("/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/data/syntheticForestData/")
            # if "_L_" in scene:
            #     name = "left"
            # /homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/data/syntheticForestData/depthData/SF_E_L_P005/000239_left_depth.npy'
            depth_path = (
                localRoot / self.conf.depth_subpath / scene / (path.stem + "_depth.npy")
            )
            # with h5py.File(str(depth_path), "r") as f:
            #     depth = f["/depth"].__array__().astype(np.float32, copy=False)
            #     depth = torch.Tensor(depth)[None]
            # depth_filename = info["depth_paths"][idx0]  # Get original depth filename
            # depth_filename = depth_filename.replace(".h5", "_depth.npy") # Replace .h5 with _depth.npy

            # # Update the path to your local directory
            # depth_path = os.path.join(
            #     "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/data/syntheticForestData/depthData",
            #     scene,
            #     depth_filename
            # )
            # print(depth_path)
            with open(depth_path, "rb") as f:  # Use open() in binary mode for .npy
                depth = np.load(f)
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
                #logger.info(f"Applied random rotation: {k * 90} degrees")


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
            #logger.info(f"Features loaded and processed for {name}")

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
            #logger.info(f"Processed data pair: {data['name']} with overlap {overlap}")
        else:
            assert self.conf.views == 1
            scene, idx0 = self.items[idx]
            data = self._read_view(scene, idx0)
            ### I added
            #logger.info(f"Processed single view data for {scene}")
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
        #logger.info(f"Processed triplet: {data['name']} with overlaps {overlap01}, {overlap02}, {overlap12}")
    
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
    dataset = TreeDepth(conf)
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

def test_overlap_matrix():
        # Define camera intrinsics (simplified)
    intrinsics = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]])

    # Create dummy depth maps
    depth_maps = [np.full((2, 2), 10), np.full((2, 2), 10)]

    #### Print testing full overlap
    # Create camera poses
    # Create dummy depth maps
    depth_maps = [np.full((2, 2), 10), np.full((2, 2), 10)]

    # Camera poses for different scenarios
    poses_full = [np.eye(4) for _ in range(2)]  # Full overlap
    poses_partial = [np.eye(4) for _ in range(2)]
    poses_partial[1][:3, 3] = [1, 0, 0]  # Move the second camera to the right, partial overlap
    poses_none = [np.eye(4) for _ in range(2)]
    poses_none[1][:3, 3] = [100, 0, 0]  # Move far away, no overlap

    # All pose scenarios
    all_poses = [poses_full, poses_partial, poses_none]

    # Expected results for each scenario
    expected_results = [
        np.array([[1, 1], [1, 1]]),  # Full overlap
        np.array([[1, 0.5], [0.5, 1]]),  # Partial overlap
        np.array([[1, 0], [0, 1]])  # No overlap
    ]

    test_mode = ['full_overlap', 'no_overlap', 'partial_overlap']
    # Test each scenario
    for poses, expected in zip(all_poses, expected_results):
        # Calculate overlap
       
        actual_overlap = np.zeros((2, 2))
        for i in range(2):
            print(f"testing mode {test_mode[i]}")
            for j in range(2):
                count, total = project_points(depth_maps[i], intrinsics, np.linalg.inv(poses[j]) @ poses[i], test_mode[i])
                actual_overlap[i, j] = count / total

        # Check if the calculated overlap matches expected results
        print(f"Testing scenario with expected results:\n{expected}\nCalculated overlap:\n{actual_overlap}\n")
        assert np.allclose(actual_overlap, expected), "Overlap test failed."

    print("Overlap Matrix Calculation Successful")
    # # Simulate camera poses and depth maps
    # poses = [np.eye(4), np.eye(4)]  # Identity matrix for simplicity in this example
    # intrinsics = [np.array([[500, 0, 256], [0, 500, 256], [0, 0, 1]]) for _ in poses]
    # depth_maps = [np.full((512, 512), 10), np.full((512, 512), 10)]  # Simple constant depth map

    # # Adjust poses for partial and no overlap
    # poses[1][:, 3] = [100, 0, 0, 1]  # Translate second camera to the right for partial overlap

    # # Calculate overlap matrix
    # overlap_matrix = calculate_overlap_matrix(depth_maps, poses, intrinsics)

    # # Define expected results based on manual calculation or known scenarios
    # expected_full_overlap = np.array([[1, 1], [1, 1]])  # Simplified; adjust for realistic tests
    # expected_partial_overlap = np.array([[1, 0.5], [0.5, 1]])  # Example values; needs accurate setup
    # expected_no_overlap = np.array([[1, 0], [0, 1]])  # Example values; needs accurate setup

    # # Print and check results
    # print("Calculated Overlap Matrix:", overlap_matrix)
    # assert np.allclose(overlap_matrix, expected_full_overlap), "Overlap matrix calculation error!"


def test_full_overlap():
    # Assume we have depth maps and intrinsics for simplicity
    depth = np.ones((2, 2))
    intrinsics = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]])
    pose = np.eye(4)

    # We expect all points to overlap perfectly
    count, total = project_points(depth, intrinsics, pose, test_mode='full_overlap')
    print("Full Overlap Test: ", "Pass" if count == total else "Fail")

def test_no_overlap():
    depth = np.ones((2, 2))
    intrinsics = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]])
    pose = np.eye(4)

    # We expect no points to overlap
    count, total = project_points(depth, intrinsics, pose, test_mode='no_overlap')
    print("No Overlap Test: ", "Pass" if count == 0 else "Fail")

def test_partial_overlap():
    depth = np.ones((2, 2))
    intrinsics = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]])
    pose = np.eye(4)

    # We expect half of the points to overlap
    count, total = project_points(depth, intrinsics, pose, test_mode='partial_overlap')
    print("Partial Overlap Test: ", "Pass" if count == total // 2 else "Fail")


if __name__ == "__main__":
    from .. import logger  # overwrite the logger
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num_items", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()
    visualize(args)
