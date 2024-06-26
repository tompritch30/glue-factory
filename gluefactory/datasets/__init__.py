# import importlib.util
#
# from ..utils.tools import get_class
# from .base_dataset import BaseDataset
#
# # def get_dataset(name):
# #     import_paths = [name, f"{__name__}.{name}"]
# #     for path in import_paths:
# #         try:
# #             spec = importlib.util.find_spec(path)
# #         except ModuleNotFoundError:
# #             spec = None
# #         if spec is not None:
# #             try:
# #                 return get_class(path, BaseDataset)
# #             except AssertionError:
# #                 mod = __import__(path, fromlist=[""])
# #                 try:
# #                     return mod.__main_dataset__
# #                 except AttributeError as exc:
# #                     print(exc)
# #                     continue
# #
# #     raise RuntimeError(f'Dataset {name} not found in any of [{" ".join(import_paths)}]')


# I added
# from .forest_images_dataset import ForestImagesDataset
#
# def get_dataset(name):
#     print(f"Getting dataset for name: {name}")
#     datasets = {
#         'forest_images': ForestImagesDataset,
#         # other datasets mappings
#     }
#     if name in datasets:
#         return datasets[name]
#     else:
#         raise ValueError(f"Dataset {name} not found.")
#

from .forest_images_dataset import ForestImagesDataset
from .base_dataset import BaseDataset

def get_dataset(name) -> BaseDataset:
    print(f"Getting dataset for name: {name}")
    datasets = {
        'forest_images': ForestImagesDataset,
        # other datasets mappings
    }
    if name in datasets:
        return datasets[name]
    else:
        raise ValueError(f"Dataset {name} not found.")