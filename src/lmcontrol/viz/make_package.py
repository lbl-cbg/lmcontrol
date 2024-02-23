import argparse
import logging
import sys
import warnings

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import numpy as np
from sklearn.preprocessing import scale

with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
    warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
    from umap import UMAP

from sklearn.preprocessing import LabelEncoder

from ..utils import get_logger
from ..data_utils import load_npzs


def compute_embedding(data, logger, metric='euclidean', two_d=False, center_images=False):
    ncomp = 2 if two_d else 3
    if center_images:
        logger.info("Centering images")
        center = data.mean()
        data = (scale(data.reshape(data.shape[0], -1), with_std=False, with_mean=True, axis=1) + center).reshape(data.shape)
    umap = UMAP(n_components=ncomp, min_dist=0.1, metric=metric, verbose=True, n_jobs=1, angular_rp_forest=False, n_neighbors=20)
    emb = umap.fit_transform(data.reshape(len(data), -1))
    return emb


def prepare_labels(metadata):
    """Encode labels and package into dictionary for easy saving"""
    ret = dict()
    for k in metadata:
        enc = LabelEncoder().fit(metadata[k])
        ret[k] = dict()
        ret[k + '_classes'] = enc.classes_
        ret[k + '_labels'] = enc.transform(metadata[k])
    return ret


def main(argv=None):
    """
    Make an NPZ archive with data necesary for interactive visualization of images
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("out_npz", help='the path to save the NPZ file for passing to the emb-viz command')
    parser.add_argument("npzs", nargs='+', help='the NPZ files produced from the crop command')
    parser.add_argument("-M", "--metric", default=None,
                        help="the distance metric to use with UMAP. Default is euclidean for images, jaccard for masks")
    parser.add_argument("-2", "--two_dim", action='store_true', default=False,
                        help='compute 2-dimensional embeddings. 3-dimensions are computed by default')
    parser.add_argument("-m", "--masks", action='store_true', default=False,
                        help='compute embeddings from masks. segmented images are used by default')
    parser.add_argument("-c", "--center", action='store_true', default=False,
                        help="center images independently before computing embedding")
    parser.add_argument("-v", "--verbose", action='store_true', default=False,
                        help="print detailed info on metadata found in individual NPZs")
    args = parser.parse_args(argv)

    logger = get_logger()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    masks, images, paths, metadata = load_npzs(args.npzs, logger)

    labels = prepare_labels(metadata)

    dat = images
    metric = 'euclidean'
    if args.masks:
        dat = masks
        metric = 'jaccard'
    metric = args.metric or metric

    emb = compute_embedding(dat, logger, metric=metric, two_d=args.two_dim, center_images=args.center)

    logger.info(f"Saving embeddings, images, and metadata to {args.out_npz}", )
    np.savez(args.out_npz, images=images, embedding=emb, **labels)


if __name__ == "__main__":
    main()