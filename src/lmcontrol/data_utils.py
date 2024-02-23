import numpy as np

from .utils import get_logger

def write_npz(path, images, masks, paths, **metadata):
    np.savez(path, masks=masks, images=images, paths=paths, **metadata)


def load_npzs(npzs, logger):
    """Load data from NPZ files generated from lmcontrol crop command"""
    masks = list()
    images = list()
    paths = list()

    metadata = dict()
    for npz_path in npzs:
        logger.info(f"Reading {npz_path}")
        npz = np.load(npz_path)
        masks.append(npz['masks'])
        images.append(npz['images'])
        paths.append(npz['paths'])


        # read metadata found in NPZ files
        md_keys = set(npz.keys()) - {'paths', 'masks', 'images'}
        logger.debug(f"Found the following keys in {npz_path}: {' '.join(sorted(md_keys))}")
        for k in md_keys:
            v = metadata.setdefault(k, list())
            v.extend([str(npz[k])] * len(npz['masks']))


    # merge all image arrays
    logger.debug("Merging masks")
    masks = np.concatenate(masks)
    logger.debug("Merging images")
    images = np.concatenate(images)
    logger.debug("Merging paths")
    paths = np.concatenate(paths)

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
