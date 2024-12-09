import warnings
import numpy as np
from sklearn.preprocessing import LabelEncoder

from .utils import get_logger

def write_npz(path, images, masks, paths, **metadata):
    np.savez(path, masks=masks, images=images, paths=paths, **metadata)

# make label_type a string (Andrew)

def load_npzs(npzs, logger, n_samples=None, label_type=None):
    """Load data from NPZ files generated from lmcontrol crop command"""
    masks = []
    images = []
    paths = []
    metadata = dict()

    for npz_path in npzs:
        logger.debug(f"Reading {npz_path}")
        #npz = np.load(npz_path)
        npz = np.load(npz_path, mmap_mode='r', allow_pickle=True)
        
        total_samples = len(npz['masks'])
        indices = None
        if n_samples is not None and total_samples > n_samples:   
            indices = np.random.permutation(total_samples)[:n_samples]
            masks.append(npz['masks'][indices])
            images.append(npz['images'][indices])
            paths.append(npz['paths'][indices])
        else:                                                     
            if n_samples > total_samples:
                warnings.warn(f"{n_samples} is more samples than found in {npz_path}. Will use all samples")
            masks.append(npz['masks'])
            images.append(npz['images'])
            paths.append(npz['paths'])

        md_keys = set(npz.keys()) - {'paths', 'masks', 'images'}
        logger.debug(f"Found the following keys in {npz_path}: {' '.join(sorted(md_keys))}")
        
        for k in sorted(md_keys):
            if npz[k].ndim == 0:
                metadata.setdefault(k, []).extend([str(npz[k])] * images[-1].shape[0])
            else:
                if indices is not None:
                    metadata.setdefault(k, []).extend(npz[k][indices])
                else:
                    metadata.setdefault(k, []).extend(npz[k])

    logger.debug("Merging masks")
    masks = np.concatenate(masks, axis=0)
    logger.debug("Merging images")
    images = np.concatenate(images, axis=0)
    logger.debug("Merging paths")
    paths = np.concatenate(paths, axis=0)
    
    metadata = {k: np.array(v) for k, v in metadata.items()}

    target_len = len(masks)
    consistent_metadata = {}
    for k, v in metadata.items(): 
        if len(v) == target_len:
            consistent_metadata[k] = v
        else:
            logger.warning(f"Skipping metadata '{k}' due to length mismatch: expected {target_len}, got {len(v)}")

    return masks, images, paths, consistent_metadata

def encode_labels(labels, mode, classes=None, return_classes=False):
    """This is a wrapper for sklearn.preprocessing.LabelEncoder"""
    if mode == 'classification':
        enc = LabelEncoder()
        if classes is None:
            enc = LabelEncoder().fit(labels)
        else:
            enc.classes_ = classes
        if return_classes:
            return enc.transform(labels), enc.classes_
        else:
            return enc.transform(labels)
    elif mode == 'regression':
        labels = np.array(labels, dtype=np.float32)
        return labels
    else:
        raise ValueError("mode should be either classification or regression")
