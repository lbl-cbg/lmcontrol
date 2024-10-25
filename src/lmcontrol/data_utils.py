

import numpy as np
from sklearn.preprocessing import LabelEncoder

from .utils import get_logger

def write_npz(path, images, masks, paths, **metadata):
    np.savez(path, masks=masks, images=images, paths=paths, **metadata)

# make label_types a string (Andrew)

def load_npzs(npzs, logger, n_samples=None, label_types=None):
    """Load data from NPZ files generated from lmcontrol crop command"""
    masks = []
    images = []
    paths = []
    metadata = dict()

    for npz_path in npzs:
        logger.debug(f"Reading {npz_path}")
        npz = np.load(npz_path)
        
        total_samples = len(npz['masks'])
        
        if n is not None and total_samples > n_samples:
            indices = np.random.permutation(total_samples)[:n_samples]
        else:
            indices = np.arange(total_samples)
        
        masks.append(npz['masks'][indices])
        images.append(npz['images'][indices])
        paths.append(npz['paths'][indices])

        md_keys = set(npz.keys()) - {'paths', 'masks', 'images'}
        logger.debug(f"Found the following keys in {npz_path}: {' '.join(sorted(md_keys))}")
        
        for k in sorted(md_keys):
            if npz[k].ndim == 0:
                metadata.setdefault(k, []).extend([str(npz[k])] * len(indices))
            else:
                metadata.setdefault(k, []).extend(np.array(npz[k])[indices])

    logger.debug("Merging masks")
    masks = np.concatenate(masks, axis=0)
    logger.debug("Merging images")
    images = np.concatenate(images, axis=0)
    logger.debug("Merging paths")
    paths = np.concatenate(paths, axis=0)
    
    metadata = {k: np.array(v) for k, v in metadata.items()}

    target_len = len(masks)
    for k in metadata.keys():
        if len(metadata[k]) != target_len:
            logger.critical(f"Metadata '{k}' not found in all NPZ files")
            raise ValueError(f"Metadata '{k}' length mismatch: expected {target_len}, got {len(metadata[k])}")
    return masks, images, paths, metadata

def encode_labels(mode, labels, return_classes=True):
    """This is a wrapper for sklearn.preprocessing.LabelEncoder"""
    enc = LabelEncoder().fit(labels)
    if mode == 'classification':
        if return_classes:
            return enc.transform(labels), enc.classes_
        else:
            return enc.transform(labels)
    elif mode == 'regression':
        labels = np.array(labels, dtype=np.float32)
        return labels
    else:
        raise ValueError("mode must be either 'classification' or 'regression'")