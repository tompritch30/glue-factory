import logging
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from ..datasets.image_pairs import ImagePairs
from ..datasets import get_dataset
from ..models.cache_loader import CacheLoader
from ..settings import DATA_PATH, EVAL_PATH # datapath is data/ .. 
from ..utils.export_predictions import export_predictions
from ..visualization.viz2d import plot_cumulative
from .eval_pipeline import EvalPipeline
from .io import get_eval_parser, load_model, parse_eval_args
from .utils import eval_matches_epipolar, eval_poses, eval_relative_pose_robust

""" Test Scenes
SF_E_L_P007
SF_H_L_P002
SF_H_R_P004
SF_E_L_P008
SFW_E_L_P003
SFW_E_L_P009
SFW_E_R_P005
"""

"""
python -m gluefactory.eval.treeEval1 --checkpoint sp+lg_megadepth --overwrite
and --overwrite if new configs
"""

logger = logging.getLogger(__name__)

# TEMPORARY
DATA_PATH = Path("/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/data")

class ForestPipeline(EvalPipeline):
    default_conf = {
        "data": {
            "name": "image_pairs",    # this is a python file!!!  image_pairs likely will need to rewrite
            "pairs": "syntheticForestData/pairs_info_calibrated.txt", # is e.g. SF_E_R_P001/filename.jpg  SF_E_R_P001/filename.jpg intrinsic1 intrinsic2  poses: tx ty tz qx qy qz qw
            "root": "syntheticForestData/imageData/",
            "extra_data": "relative_pose", # "poseData/SF_E_P007",
            "preprocessing": {
                "side": "long",
            },
        },
        "model": {
            "ground_truth": {
                "name": None,  # remove gt matches -- HOW TO SPECITY MY MODEL in command line?
            } 
        },
        "eval": {
            "estimator": "poselib",
            "ransac_th": 1.0,
        },
    }

    export_keys = [
        "keypoints0",
        "keypoints1",
        "keypoint_scores0",
        "keypoint_scores1",
        "matches0",
        "matches1",
        "matching_scores0",
        "matching_scores1",
    ]
    optional_export_keys = []
    
    def _init(self, conf):
        # Ensure the dataset exists
        if not (DATA_PATH / "syntheticForestData").exists():
            logger.error("syntheticForestData dataset not found.")
            raise FileNotFoundError("syntheticForestData dataset directory is missing.")

    @classmethod
    def get_dataloader(self, data_conf=None):
        """Creates a dataloader to load forest dataset images with depth information."""
        data_conf = data_conf if data_conf else self.default_conf["data"]
        # dataset = get_dataset(data_conf["name"])(data_conf)
        # Fetch the correct dataset class based on the name
        DatasetClass = get_dataset(data_conf["name"])
        print("DatasetClass", DatasetClass)
        
        # Instantiate the dataset with its configuration
        # dataset = DatasetClass(data_conf)
        print(str(DATA_PATH) + "/image_pairs.pkl")
        dataset = ImagePairs.from_pickle_or_create(data_conf, DATA_PATH / "image_pairs.pkl")
        
        print("dataset instance created", dataset)

        # dataset = get_dataset(data_conf["name"])(data_conf)

        # dataset = get_dataset('HomographySynthTreeDataset')('gluefactory.datasets.homographiesTree')
        return dataset.get_data_loader("test")
        # make a class that inherits from baseDataSet and then has method get_data_loader where pass string "test"
        #     data_conf[name] image_pairs
        # data_conf {'name': 'image_pairs', 'pairs': 'megadepth1500/pairs_calibrated.txt', 'root': 'megadepth1500/images/', 'extra_data': 'relative_pose', 'preprocessing': {'side': 'long'}}

    def get_predictions(self, experiment_dir, model=None, overwrite=False):
        """Generates predictions for each evaluation data point in the forest dataset."""
        pred_file = experiment_dir / "predictions.h5"
        
        def backup_existing_file(file_path):
            if file_path.exists():
                backup_path = file_path.with_suffix('.bak')
                file_path.rename(backup_path)
                print(f"Existing file {file_path} renamed to {backup_path}")

        # Inside your `treeEval1.py` script before `export_predictions` is called
        pred_file = experiment_dir / "predictions.h5"
        if pred_file.exists() and overwrite:
            backup_existing_file(pred_file)
                
        # # I added
        # import os
        # if pred_file.exists() and overwrite:
        #     os.remove(pred_file)

        if not pred_file.exists() or overwrite:
            if model is None:
                model = load_model(self.conf.model, self.conf.checkpoint)
            export_predictions(
                self.get_dataloader(self.conf.data),
                model,
                pred_file,
                keys=self.export_keys,
                optional_keys=self.optional_export_keys,
            )
            print(f"in get prediction called get_dataloader with {self.conf.data}")
        return pred_file

    def run_eval(self, loader, pred_file):
        """Run the eval on cached predictions"""
        conf = self.conf.eval
        results = defaultdict(list)
        test_thresholds = (
            ([conf.ransac_th] if conf.ransac_th > 0 else [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
            if not isinstance(conf.ransac_th, Iterable)
            else conf.ransac_th
        )
        pose_results = defaultdict(lambda: defaultdict(list))
        cache_loader = CacheLoader({"path": str(pred_file), "collate": None}).eval()
        for i, data in enumerate(tqdm(loader)):
            pred = cache_loader(data)
            # add custom evaluations here
            results_i = eval_matches_epipolar(data, pred)
            for th in test_thresholds:
                pose_results_i = eval_relative_pose_robust(
                    data,
                    pred,
                    {"estimator": conf.estimator, "ransac_th": th},
                )
                [pose_results[th][k].append(v) for k, v in pose_results_i.items()]

            # we also store the names for later reference
            results_i["names"] = data["name"][0]
            if "scene" in data.keys():
                results_i["scenes"] = data["scene"][0]

            for k, v in results_i.items():
                results[k].append(v)

        # summarize results as a dict[str, float]
        # you can also add your custom evaluations here
        summaries = {}
        for k, v in results.items():
            arr = np.array(v)
            if not np.issubdtype(np.array(v).dtype, np.number):
                continue
            summaries[f"m{k}"] = round(np.mean(arr), 3)

        best_pose_results, best_th = eval_poses(
            pose_results, auc_ths=[5, 10, 20], key="rel_pose_error"
        )
        results = {**results, **pose_results[best_th]}
        summaries = {
            **summaries,
            **best_pose_results,
        }

        figures = {
            "pose_recall": plot_cumulative(
                {self.conf.eval.estimator: results["rel_pose_error"]},
                [0, 30],
                unit="°",
                title="Pose ",
            )
        }

        return summaries, figures, results

if __name__ == "__main__":
    from .. import logger  # overwrite the logger

    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(ForestPipeline.default_conf)

    # mingle paths
    output_dir = Path(EVAL_PATH, dataset_name)
    output_dir.mkdir(exist_ok=True, parents=True)

    name, conf = parse_eval_args(
        dataset_name,
        args,
        "configs/",
        default_conf,
    )

    experiment_dir = output_dir / name
    experiment_dir.mkdir(exist_ok=True)

    pipeline = ForestPipeline(conf)
    s, f, r = pipeline.run(
        experiment_dir,
        overwrite=args.overwrite,
        overwrite_eval=args.overwrite_eval,
    )

    pprint(s)

    if args.plot:
        for name, fig in f.items():
            fig.canvas.manager.set_window_title(name)
        plt.show()