import importlib.util

from ..utils.tools import get_class
from .base_dataset import BaseDataset

from .homogTreePairs import HomogTreePairsDataset
 # HomogTreePairsDataset


def get_dataset(name):
    import_paths = [name, f"{__name__}.{name}"]
    for path in import_paths:
        try:
            spec = importlib.util.find_spec(path)
        except ModuleNotFoundError:
            spec = None
        if spec is not None:
            try:
                return get_class(path, BaseDataset)
            except AssertionError:
                mod = __import__(path, fromlist=[""])
                try:
                    return mod.__main_dataset__
                except AttributeError as exc:
                    print(exc)
                    continue

    ## Janky tempoary loading code for eval:
    print("name", name)
    dataset_classes = {
        'HomogTreePairs': HomogTreePairsDataset,
        # other dataset classes
    }
    if name in dataset_classes:
        print("used the janky loaded for HomogTreePairs")
        return dataset_classes[name] 
    
    raise RuntimeError(f'Dataset {name} not found in any of [{" ".join(import_paths)}]')
