import argparse

import numpy as np
from skimage.filters import median
import skimage.io as sio
from skimage.measure import label
from skimage.morphology import closing
import tqdm


def outlier_threshold_tol(img, tol=0.001, max_med=10, ntrim=15, nsig=1):
    """
    Threshold gray scale image keeping only those pixels on the tails of the
    pixel value distribution.

    Args:
        tol (float)      : the tolerance for applying successive median filters.
                           If the change is less than this, stop applying filters
        max_med (int)    : the maximum number of median filters to apply
        ntrim (int)      : the size of the border to ignore
        nsig (int)       : the number of standard deviations above/below which to
                           set the threshold
    """
    mean = img.ravel().mean()
    std = img.ravel().std()
    mask = np.logical_or(img < mean - nsig*std, img > mean + nsig*std).astype(int)
    if ntrim > 0:
        mask[:ntrim] = 0
        mask[-ntrim:] = 0
        mask[:, :ntrim] = 0
        mask[:, -ntrim:] = 0

    for i in range(max_med):
        tmp = median(mask.astype(int))
        diff = (mask != tmp).mean()
        if diff < tol:
            break
        mask = tmp


def outlier_cluster(img, frac=0.3, tol=0.001, max_med=10, ntrim=15, nsig=1):
    """
    Find mask for cell using outlier thresholding.

    Args:
        tol (float)      : the tolerance for applying successive median filters.
                           If the change is less than this, stop applying filters
        max_med (int)    : the maximum number of median filters to apply
        ntrim (int)      : the size of the border to ignore
        nsig (int)       : the number of standard deviations above/below which to
                           keep set the threshold
        frac (float)     : the minimum fraction of pixels required to belong to the
                           main cluster for declaring a main cluster found
    """
    mask = outlier_threshold_tol(img, ntrim=15)
    mask = closing(mask)
    if frac is not None:
        labels = label(mask)
        counts = np.bincount(labels.ravel())[1:]
        counts = (counts / counts.sum())
        cell_cluster_id = np.where(counts >= frac)[0]
        if len(cell_cluster_id) == 0:
            raise ValueError("Unable to find one majority cluster")

        cell_cluster_id = cell_cluster_id[0] + 1
        mask = (labels == cell_cluster_id).astype(int)
    return mask


def trim_box(mask, img, size=None):
    """
    Crop image using bounding box calculated using the provided mask. The
    mask should have a single cluster for this function to have the intended
    effect.

    Args:
        mask (array)        : the mask to use for cropping the image
        img  (array)        : the image to crop
        size (tuple, list)  : the target size around the cluster found in *mask* that
                              *img* should be cropped to

    """
    mask = mask.astype(int)
    X, Y = np.where(mask == 1)
    Xx, Xn = X.max(), X.min()
    Yx, Yn = Y.max(), Y.min()
    if size is not None:
        Ydim = Yx - Yn
        Xdim = Xx - Xn
        Xpad = (size[0] - Xdim) // 2
        Ypad = (size[1] - Ydim) // 2
        Xx += Xpad
        Xn -= Xpad
        Yx += Ypad
        Yn -= Ypad
        if (Xx - Xn) != size[0]:
            Xx += 1
        if (Yx - Yn) != size[0]:
            Yx += 1
    return img[Xn: Xx, Yn: Yx]


def crop_image(img, size=None):
    """
    Crop image around the cell imaged in *img*.

    Args:
        img  (array)        : the image to crop
        size (tuple, list)  : the target size around the cluster found in *mask* that

    """
    mask = outlier_cluster(img)
    new_img = trim_box(mask, img, size=size)
    return new_img


def crop_images(argv=None):

	n_unseg = 0
	total = 0
	masks = list()
	for tif_path in tqdm(sorted(glob.glob(f"{image_path}/*/*.tif"))):
	    total += 1
	    target = tif_path.replace("S4/", "S4_seg/")
	    os.makedirs(os.path.dirname(target), exist_ok=True)
	    img = sio.imread(tif_path)[:, :, 0]
	    try:
	        #TODO: Add check for garbage images i.e. overdispersed images.
	        if img.std() > 15:
	            raise ValueError("StdDev of pixels is quite high, this is probably a bad image")
	        mask = outlier_cluster(img)
	    except ValueError:
	        target = tif_path.replace("S4/", "S4_unseg/")
	        os.makedirs(os.path.dirname(target), exist_ok=True)
	        sio.imsave(target, img)
	        n_unseg += 1
	        continue
	    new_img = trim_box(mask, img)
	    sio.imsave(target, new_img)
