

import numpy as np
from sklearn.preprocessing import LabelEncoder

from .utils import get_logger

def write_npz(path, images, masks, paths, **metadata):
    np.savez(path, masks=masks, images=images, paths=paths, **metadata)

# make label_types a string (Andrew)

def load_npzs(npzs, logger, n=None, label_types=None):
    """Load data from NPZ files generated from lmcontrol crop command"""
    masks = list()
    images = list()
    paths = list()

    metadata = dict()
    for npz_path in npzs:
        logger.debug(f"Reading {npz_path}")
        npz = np.load(npz_path)
        masks.append(npz['masks'])
        images.append(npz['images'])
        paths.append(npz['paths'])


        # read metadata found in NPZ files
        md_keys = set(npz.keys()) - {'paths', 'masks', 'images'}
        logger.debug(f"Found the following keys in {npz_path}: {' '.join(sorted(md_keys))}")
        for k in sorted(md_keys):
            v = metadata.setdefault(k, list())
            v.extend([str(npz[k])] * len(npz['masks']))


    # merge all image arrays
    logger.debug("Merging masks")
    masks = np.concatenate(masks)
    logger.debug("Merging images")
    images = np.concatenate(images)
    logger.debug("Merging paths")
    paths = np.concatenate(paths)
    #logger.info(f"Metadata keys: {metadata.keys()}")
    # label_types_array = np.array(label_types)
   
    if n is not None:
        metadata = {k: np.array(v) for k, v in metadata.items()}

        # # class_label_key = 'time' 
        # class_label_key = label_types_array
        # #classes = np.unique(metadata[class_label_key])
        # labels = np.array(metadata[class_label_key])
        # classes = np.unique(labels)
        
        

        selected_idx = []   \
        
        #UNCOMMENTED FOR EXPERIMENTATION
        # for label_key in label_types:
        #     labels = metadata[label_types]
        #     if not isinstance(labels, np.ndarray):
        #         labels = np.array(labels)
        #     classes = np.unique(labels)
        
        labels = metadata[label_types]
        classes = np.unique(labels)

        for cls in classes:
            class_indices = np.where(labels == cls)[0]

            if len(class_indices) > n:
                class_indices = np.random.permutation(class_indices)[:n]
            selected_idx.extend(class_indices)

        selected_idx = np.array(selected_idx)

        selected_idx = np.random.permutation(selected_idx)

        masks = masks[selected_idx]
        images = images[selected_idx]
        paths = paths[selected_idx]

        for k in metadata.keys():
            metadata[k] = metadata[k][selected_idx]


    target_len = len(masks)
    error = False

    # make sure all metadata keys were found in all NPZ files
    for k in metadata.keys():
        metadata[k] = np.array(metadata[k])
        if len(metadata[k]) != target_len:
            logger.critical(f"Metadata '{k}' not found in all NPZ files")
            error = True

    if error:
        raise ValueError("NPZ files do not have the same metadata keys. See CRITICAL messages in logs for more details")

    
    return masks, images, paths, metadata

def encode_labels(labels, return_classes=True):
    """This is a wrapper for sklearn.preprocessing.LabelEncoder"""
    enc = LabelEncoder().fit(labels)
    if return_classes:
        return enc.transform(labels), enc.classes_
    else:
        return enc.transform(labels)
