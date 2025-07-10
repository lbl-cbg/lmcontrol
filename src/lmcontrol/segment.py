import argparse
import glob
import io
import os
from pathlib import Path
import re
from zipfile import BadZipFile, ZipFile
import uuid
import hashlib
import json


from typing import Dict, Iterable, Any, Tuple, Optional

import flowkit as fk
from numba import njit
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from skimage.filters import median
import skimage.io as sio
from skimage.measure import label
from skimage.morphology import closing
from PIL import Image
from tqdm import tqdm

from .utils import get_logger
from .command import spreadsheet
from .data_utils import write_npz

@njit
def _raw_outlier_mask(img):
    mean = img.ravel().mean()
    std = img.ravel().std()
    mask = np.logical_or(img < mean - std, img > mean + std).astype(np.int64)
    mask[:15] = 0
    mask[-15:] = 0
    mask[:, :15] = 0
    mask[:, -15:] = 0
    return mask


def outlier_threshold_tol(img, tol=0.001, max_med=10):
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

    mask = _raw_outlier_mask(img)

    for i in range(max_med):
        tmp = median(mask.astype(int))
        diff = (mask != tmp).mean()
        if diff < tol:
            break
        mask = tmp

    return mask


class UnsegmentableError(ValueError):
    """An Exception to raise when segmentation fails for known reasons"""
    pass


def outlier_cluster(img, frac=0.3, tol=0.001, max_med=10):
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
    mask = outlier_threshold_tol(img)
    mask = closing(mask)
    if mask.sum() / mask.size < 0.005:
        raise UnsegmentableError("Not enough masked pixels to to cluster")
    if frac is not None:
        labels = label(mask)
        counts = np.bincount(labels.ravel())[1:]
        counts = (counts / counts.sum())
        cell_cluster_id = np.where(counts >= frac)[0]
        if len(cell_cluster_id) == 0:
            raise UnsegmentableError("Unable to find one majority cluster")

        cell_cluster_id = cell_cluster_id[0] + 1
        mask = (labels == cell_cluster_id).astype(int)
    return mask


def _adjust_bounds(Xn, Xx, Yn, Yx, target_size):
        Ydim = Yx - Yn
        Xdim = Xx - Xn
        Xpad = (target_size[0] - Xdim) // 2
        Ypad = (target_size[1] - Ydim) // 2
        Xx += Xpad
        Xn -= Xpad
        Yx += Ypad
        Yn -= Ypad
        if (Xx - Xn) != target_size[0]:
            Xx += 1
        if (Yx - Yn) != target_size[1]:
            Yx += 1
        return Xn, Xx, Yn, Yx


def trim_box(mask, img, size=None, pad=True):
    """
    Crop image using bounding box calculated using the provided mask. The
    mask should have a single cluster for this function to have the intended
    effect.

    Args:
        mask (array)        : the mask to use for cropping the image
        img  (array)        : the image to crop
        size (tuple, list)  : the target size around the cluster found in *mask* that
                              *img* should be cropped to
        pad (bool)          : If True, pad with zeros around image to shape specified by *size*,
                              otherwise crop image to the specified size, centered on the mask.

    """
    mask = mask.astype(int)
    X, Y = np.where(mask == 1)
    Xx, Xn = X.max() + 1, X.min()
    Yx, Yn = Y.max() + 1, Y.min()

    if size is not None:
        if not pad:
            Xn, Xx, Yn, Yx = _adjust_bounds(Xn, Xx, Yn, Yx, size)
            if Xn < 0:
                Xn, Xx = 0, Xx - Xn
            elif Xx > mask.shape[0]:
                Xn, Xx = Xn - (Xx - mask.shape[0]), mask.shape[0]
            if Yn < 0:
                Yn, Yx = 0, Yx - Yn
            elif Yx > mask.shape[1]:
                Yn, Yx = Yn - (Yx - mask.shape[1]), mask.shape[1]
            ret = img[Xn: Xx, Yn: Yx]
        else:
            ret = img[Xn: Xx, Yn: Yx]
            image_height, image_width = ret.shape
            crop_height, crop_width = size
            padding_ltrb = [[0, 0], [0, 0]]
            w_d = crop_width - image_width
            if w_d > 0:
                padding_ltrb[1][0] = w_d // 2
                padding_ltrb[1][1] = w_d // 2 + int(w_d % 2)
            h_d = crop_height - image_height
            if h_d > 0:
                padding_ltrb[0][0] = h_d // 2
                padding_ltrb[0][1] = h_d // 2 + int(h_d % 2)

            mean_value = np.mean(ret)
            ret = np.pad(ret, padding_ltrb, mode='constant', constant_values=mean_value)
            Xn, Xx, Yn, Yx = _adjust_bounds(0, ret.shape[0], 0, ret.shape[1], size)
            ret = ret[Xn: Xx, Yn: Yx]
    else:
        ret = img[Xn: Xx, Yn: Yx]

    if 0 in ret.shape:
        raise ValueError(f"img has zero area, shape is {ret.shape}")
    return ret


def crop_image(img, size=None, return_mask=False):
    """
    Crop image around the cell imaged in *img*.

    Args:
        img  (array)        : the image to crop
        size (tuple, list)  : the target size around the cluster found in *mask* that

    """
    mask = outlier_cluster(img)
    new_img = trim_box(mask, img, size=size)
    if return_mask:
        return new_img, mask
    return new_img


def metadata_str(string):
    if len(string):
        try:
            ret = {k:v for k, v in (tuple(kv.split("=")) for kv in string.split(","))}
            for k in list(ret.keys()):
                v = ret[k]
                for t in (int, float, str):
                    try:
                        v = t(v)
                        break
                    except:
                        v = ret[k]
                ret[k] = v
        except:
            raise argparse.ArgumentTypeError()
        return ret
    return dict()


def add_metadata(argv=None):
    parser = argparse.ArgumentParser(description="Add metadata to a NPZ file produced by the crop command")
    parser.add_argument("npz", type=str, help='Path to the NPZ file')
    parser.add_argument("metadata", help="a comma-separated list of key=value pairs. e.g. ht=1,time=S4", default="", type=metadata_str)
    args = parser.parse_args(argv)

    npz = np.load(args.npz)
    data = dict(npz)
    data.update(args.metadata)
    np.savez(args.npz, **data)


def crop_image_center(image, crop_size, pad=True):
    """
    Crop the center of the given image to the specified size.
    If the image is smaller than the desired size, it will be padded with zeros.

    Args:
        image (array): The image to crop.
        crop_size (tuple): The target size (height, width) to crop the image to.
        pad (bool): If True, pad the image with zeros if it's smaller than the crop_size.

    Returns:
        np.array: The cropped (and possibly padded) image.
    """
    h, w = image.shape
    crop_h, crop_w = crop_size

    start_h = max(0, (h - crop_h) // 2)
    start_w = max(0, (w - crop_w) // 2)
    end_h = start_h + crop_h
    end_w = start_w + crop_w

    if pad:
        mean_value = np.mean(image)

        padding_h = max(0, crop_h - h)
        padding_w = max(0, crop_w - w)

        padded_image = np.pad(image,
                              ((padding_h // 2, padding_h - padding_h // 2),
                               (padding_w // 2, padding_w - padding_w // 2)),
                              mode='constant', constant_values=mean_value)

        h, w = padded_image.shape
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        end_h = start_h + crop_h
        end_w = start_w + crop_w

        return padded_image[start_h:end_h, start_w:end_w]

    return image[start_h:end_h, start_w:end_w]


def extract_s_ht_to_tif_mapping(zip_filepath):
    # Regular expressions to find S and HT values
    s_pattern = r'S(\d+)'
    ht_pattern = r'HT(\d+)'

    # Dictionary to store mapping from (S, HT) to list of TIF files
    s_ht_to_tifs = {}

    # Dictionary to store directory to (S, HT) mapping
    dir_to_s_ht = {}

    with ZipFile(zip_filepath, 'r') as zip_ref:
        # First, process all directories to get their (S, HT) values
        directories = set()
        for file_info in zip_ref.infolist():
            if file_info.is_dir():
                directories.add(file_info.filename)
            else:
                # Also add the directory part of files
                dir_path = os.path.dirname(file_info.filename)
                if dir_path:
                    directories.add(dir_path + '/')

        for directory in directories:
            # Find all matches for S and HT patterns
            s_matches = list(re.finditer(s_pattern, directory))
            ht_matches = list(re.finditer(ht_pattern, directory))

            if s_matches and ht_matches:
                # Use the last occurrence of S and HT in the path (most specific)
                s_value = int(s_matches[-1].group(1))
                ht_value = int(ht_matches[-1].group(1))
                dir_to_s_ht[directory] = (s_value, ht_value)

        # Now, find all TIF files and associate them with their (S, HT) values
        for file_info in zip_ref.infolist():
            if not file_info.is_dir() and file_info.filename.lower().endswith('.tif'):
                # Get the directory of this file
                dir_path = os.path.dirname(file_info.filename)
                if not dir_path.endswith('/'):
                    dir_path += '/'

                # Find the closest parent directory that has an (S, HT) mapping
                current_dir = dir_path
                s_ht_tuple = None

                while current_dir:
                    if current_dir in dir_to_s_ht:
                        s_ht_tuple = dir_to_s_ht[current_dir]
                        break
                    # Move up one directory level
                    current_dir = os.path.dirname(current_dir.rstrip('/')) + '/' if os.path.dirname(current_dir.rstrip('/')) else ''

                if s_ht_tuple:
                    # Add this TIF file to the mapping
                    if s_ht_tuple not in s_ht_to_tifs:
                        s_ht_to_tifs[s_ht_tuple] = []
                    s_ht_to_tifs[s_ht_tuple].append(file_info.filename)

    return s_ht_to_tifs


def dir_iterator(image_dir, logger):
    image_paths = glob.glob(os.path.join(image_dir, "*.tif"))
    logger.info(f"Found {len(image_paths)} images in {image_dir}")
    for tif in tqdm(image_paths):
        image = sio.imread(tif)
        if image.ndim == 3:
            image = image[:, :, 0]
        yield tif, image

def list_zip_files(zip_file_path):
    with ZipFile(zip_file_path, 'r') as zip_file:
        zip_contents = zip_file.namelist()
        files_only = [name for name in zip_contents if not name.endswith('/')]
    return files_only

def zip_iterator(zip_path, paths, logger):
    logger.info(f"Loading {len(paths)} images from {zip_path}")
    with ZipFile(zip_path, 'r') as zip_ref:
        for tiff_path in tqdm(paths):
            with zip_ref.open(tiff_path) as tiff_file:
                # Read the TIFF file into memory
                img_data = tiff_file.read()
                img = Image.open(io.BytesIO(img_data))
                # Force loading the image data since we're reading from a stream
                img.load()
                # Convert to NumPy array
                array = np.array(img)
                if array.ndim == 3:
                    array = array[:, :, 0]
                yield tiff_path, array

def build_metadata(campaign, ht_metadata, sample_metadata, **defaults):

    all_metadata = dict()
    for i, sample in sample_metadata.iterrows():
        for j, ht in ht_metadata.iterrows():
            if isinstance(ht['Reactor'], (np.floating, float)) and np.isnan(ht['Reactor']):
                # There are extra rows in the spreadsheet probably
                break
            md = dict(
                    campaign=campaign,
                    time=f"{sample['Time']:0.1f}",
                    ht=str(ht['Reactor'][2:]),
                    condition=ht['Process conditions/Comments'].strip(),
                    sample=sample['Sample'],
                    feed=ht['Carbon source'],
                    starting_media=ht['Carbon source'],
                )
            md = defaults | md
            all_metadata[f"{md['sample']}_HT{md['ht']}"] = md

    return all_metadata


def pad_images_to_max_size(image_list, padval=None):
    if not image_list:
        return []

    # Find the maximum dimensions
    max_height = max(img.shape[0] for img in image_list)
    max_width = max(img.shape[1] for img in image_list)

    padded_images = []

    if padval is None:
        padval_func = lambda img: np.median(img)
    else:
        padval_func = lambda img: padval

    for img in image_list:
        # Calculate padding needed
        pad_height = max_height - img.shape[0]
        pad_width = max_width - img.shape[1]

        # Calculate padding for each side
        top_pad = pad_height // 2
        bottom_pad = pad_height - top_pad
        left_pad = pad_width // 2
        right_pad = pad_width - left_pad

        # Pad the image
        padded_img = np.pad(
            img,
            ((top_pad, bottom_pad), (left_pad, right_pad)),
            mode='constant',
            constant_values=padval_func(img)
        )

        padded_images.append(padded_img)

    return padded_images


def extract_event(path):
    return int(os.path.basename(path)[:-4].split('_')[-1])


def segment_all(
        images: Iterable,
        output_dir: str,
        metadata: Dict[str, Any],
        logger,
        crop: Optional[Tuple[int, int]] = (64, 32),
        pad: bool = False,
        include_unseg: bool = False,
        save_unseg: bool = False,
        save_tifs: bool = False,
        fcs_sample_df: pd.DataFrame = None,
):
    dir_exists = False # only create unseg directory if there were any unsegmentable images
    unseg_dir = os.path.join(output_dir, 'unseg')

    seg_images = list()
    seg_masks = list()
    paths = list()
    orig_seg_images = list()
    n_unseg = 0


    logger.info(f"Saving segmented images to {output_dir}")
    if save_unseg:
        logger.info(f"Saving any unsegmented images to {unseg_dir}")
    else:
        logger.info("Discarding unsegmented images")

    npz_out = os.path.join(output_dir, "all_processed.npz")
    logger.info(f"Cropping images to {crop} and saving to {npz_out} for convenience")

    shapes = set()
    n_total_imgs = 0
    events = list()
    for image_path, image in images:
        n_total_imgs += 1
        try:
            shapes.add(image.shape)
            # Check for bad images caused by flow-cytometer and/or imaging errors
            if image.std() / image.mean() > 0.15:
                raise UnsegmentableError("StdDev of pixels is quite high, this is probably a bad image")
            mask = outlier_cluster(image)
            if save_tifs:
                segi = trim_box(mask, image)
                orig_seg_images.append(segi)  # Save segmented original image
            paths.append(image_path)

            if crop is None:
                segi = image
                segm = mask
            else:
                segi = trim_box(mask, image, size=crop, pad=pad)
                segm = trim_box(mask, mask, size=crop, pad=pad)

            seg_images.append(segi)
            seg_masks.append(segm)
            events.append(extract_event(image_path))
        except UnsegmentableError:
            n_unseg += 1
            if save_unseg:
                target = os.path.join(unseg_dir, os.path.basename(image_path))
                if not dir_exists:
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    dir_exists = True
                sio.imsave(target, image)
            if include_unseg:
                paths.append(image_path)
                if crop:
                    segi = crop_image_center(image, crop_size=crop, pad=pad)
                else:
                    segi = image
                segm = np.zeros_like(segi)
                seg_images.append(segi)
                seg_masks.append(segm)
                events.append(extract_event(image_path))
            continue

    logger.info(f"Done segmenting images. {n_unseg} ({100 * n_unseg / n_total_imgs:.1f}%) images were unsegmentable")


    if crop is None and len(shapes) > 1:
        seg_images = pad_images_to_max_size(seg_images)
        seg_masks = pad_images_to_max_size(seg_masks, padval=0)

    seg_images = np.array(seg_images, dtype=np.uint8)
    seg_masks = np.array(seg_masks, dtype=np.uint8)
    paths = np.array(paths)

    if fcs_sample_df is not None:
        metadata = metadata.copy()
        for c in fcs_sample_df.columns:
            if c[0] == "ImageFlag":
                continue
            values = fcs_sample_df[c].values
            if c[0] == "Event":
                values = values.astype(int)
            metadata[c[0]] = values

    os.makedirs(output_dir, exist_ok=True)
    # Save segmented original images
    if not save_tifs:
        for image, path in zip(orig_seg_images, paths):
            target = os.path.join(output_dir, os.path.basename(path))
            sio.imsave(target, image)

    # save cropped and rotate
    logger.info(f"Saving all cropped images to {npz_out}")
    write_npz(npz_out, seg_images, seg_masks, paths, **metadata)


def init_hasher(file_path):
    base_hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            base_hasher.update(byte_block)
    return base_hasher


def uniq_dir(base, strings=None):
    """Compute the MD5 hash of a file and a list of strings."""
    if isinstance(base, str):
        assert os.path.isfile(base)
        hasher = init_hasher(base)
    else:
        hasher = base.copy()

    # Process the list of strings
    for string in (strings or list()):
        hasher.update(string.encode('utf-8'))  # Encode string to bytes

    return hasher.hexdigest()


def make_dir(output_dir, metadata, digest):
    dirs = [output_dir,
            metadata['campaign'],
            'manual_scaling' if metadata['manual_scaling'] ==  1 else 'auto_scaling']
    if metadata['source'] == 'sample':
        dirs.append(metadata['sample'])
        dirs.append(f"HT{metadata['ht']}")
    elif metadata['source'] == 'background':
        dirs.append('background')
    elif metadata['source'] == 'water':
        dirs.append('water')

    dirs.append(digest)
    return os.path.join(*dirs)


def parse_acs_zip(zip_path):
    """Look for an FCS file and TIFF files in a file assumed to be an ACS Zip archive"""
    try:
        tif_paths = list()
        fcs_sample_df = None
        with ZipFile(zip_path, 'r') as zip_file:
            fcs_path = list()
            for path in list_zip_files(zip_path):
                if path.endswith(".tif"):
                    tif_paths.append(path)
                elif path.endswith(".fcs"):
                    fcs_path.append(path)
            if len(fcs_path) == 1:
                with zip_file.open(fcs_path[0]) as fcs_fh:
                    fcs_sample_df = fk.Sample(io.BytesIO(fcs_fh.read())).as_dataframe(source='raw')
            else:
                raise ValueError(f"Found more than one .fcs file in {zip_path}")
        return fcs_sample_df, tif_paths
    except BadZipFile:
        return None


def check_fcs(fcs_path):
    if fcs_path is None:
        return None
    return fk.Sample(fcs_path).as_dataframe(source='raw').iloc[events]


def main(argv=None):
    """
    Segment and images in a directory, saving segmented images to a
    new directory. This will also save all images cropped and stored in
    an ndarray for easy loading of cropped data.
    """

    def int_tuple(string):
        if len(string) == 0:
            return (64, 32)
        else:
            try:
                x, y = string.split(',')  ###
                x, y = int(x), int(y)
                return (x, y)
            except:
                raise ArgumentTypeError()

    desc = "Identify cells in flow-cytometer images and crop images down to single cells"
    epi = """
    Segmented and cropped images are saved to the same name in output_dir. Separately, all
    images are cropped to the same size and saved to a NPZ file (named all_processed.npz) in
    output_dir. Saving of individual images can be turned off using the -n flag.
    Metadata about all images in image_dir can be saved to the NPZ file using the -m flag.
    This feature is useful if you plan to merge multiple NPZ files for interactive viewing.
    Some images are unsegmentable. These images can be saved to the output_dir/unseg using
    the -u flag.
    """
    parser = argparse.ArgumentParser(description=desc, epilog=epi)
    parser.add_argument("images", type=str, help='The directory containing images to crop')
    parser.add_argument("output_dir", type=str, help='The directory to save cropped images to')
    parser.add_argument('campaign', help='The campaign name', default=None, nargs='?')
    parser.add_argument('ht_metadata', type=spreadsheet, help='a spreadsheet with metadata about HTs', default=None, nargs='?')
    parser.add_argument('sample_metadata', type=spreadsheet, help='a spreadsheet with metadata about samples', default=None, nargs='?')
    parser.add_argument("-u", "--save-unseg", action='store_true', default=False,
                        help="Save unsegmentable images in output_dir under directory 'unseg'")
    parser.add_argument("-U", "--include-unseg", action='store_true', default=False,
                        help="Included unsegmentable images in all_processed.npz, cropping from center if necessary")
    parser.add_argument("-T", "--save-tifs", action='store_true', default=False,
                        help="Do not save cropped TIF files i.e. only save the all_processed.npz file")
    parser.add_argument('-c', '--crop', type=int_tuple, default=None, metavar='SHAPE',
                        help='the size to crop images to (in pixels) for saving as ndarray. do not crop by default')
    parser.add_argument('-p', '--pad', default=False, action='store_true',
                        help='pad segmented image with zeros to size indicated with --crop. Otherwise use pad with original image contents')
    parser.add_argument("-m", "--metadata", help="a comma-separated list of key=value pairs. e.g. ht=1,time=S4", default="", type=metadata_str)
    parser.add_argument("-s", "--scaling-limits", type=int_tuple, default=(-1, -1), metavar='MIN_MAX',
                        help='the min and max values used for scaling')
    parser.add_argument("-S", "--source", type=str, choices=('sample', 'water', 'background'), default='sample',
                        help='the source of the image(s)')
    parser.add_argument("-A", "--acs", action='store_true', default=False,
                        help="images argument is a path to an acs file")
    parser.add_argument("-F", "--fcs", type=str, default=None, help='the FCS file with flow cytometry data')
    parser.add_argument("-d", "--debug", action='store_true', default=False,
                        help="Print output directory names and exit")

    args = parser.parse_args(argv)

    logger = get_logger("warning" if args.debug else "info")

    default_metadata = dict(scale_min=args.scaling_limits[0], scale_max=args.scaling_limits[1],
                            manual_scaling=0 if args.scaling_limits == (-1, -1) else 1,
                            source=args.source)

    if os.path.isdir(args.images):
        if args.campaign is not None:
            args.metadata['campaign'] = args.campaign

        metadata = default_metadata | args.metadata

        fcs_sample_df = None
        if args.fcs is not None:
            fcs_sample_df = check_fcs(args.fcs)

        it = dir_iterator(args.images, logger)
        segment_all(it, args.output_dir, metadata, logger,
                    crop=args.crop, pad=args.pad, include_unseg=args.include_unseg,
                    save_unseg=args.save_unseg, save_tifs=args.save_tifs, fcs_sample_df=fcs_sample_df)
    else:
        fcs_sample_df, tif_paths = parse_acs_zip(args.images)

        if fcs_sample_df is not None:
            it = zip_iterator(args.images, tif_paths, logger)
            metadata = default_metadata | args.metadata
            segment_all(it, args.output_dir, metadata, logger,
                        crop=args.crop, pad=args.pad, include_unseg=args.include_unseg,
                        save_unseg=args.save_unseg, save_tifs=args.save_tifs, fcs_sample_df=fcs_sample_df)
        else:
            if args.campaign is None or args.ht_metadata is None or args.sample_metadata is None:
                logger.error("If providing a Zip file, you must specify the campaign and provide metadata for HTs and Samples")
                exit(2)

            if args.source == 'sample':
                metadata = build_metadata(args.campaign, args.ht_metadata, args.sample_metadata, **default_metadata)

                all_paths = extract_s_ht_to_tif_mapping(args.images)
                keys = [f"S{s}_HT{ht}" for s, ht in all_paths]
                logger.info(f"Found the following samples in {args.images}\n" + "\n".join(keys))


                base_hasher = init_hasher(args.images)

                for s, ht in all_paths:
                    paths = all_paths[(s, ht)]
                    it = zip_iterator(args.images, paths, logger)
                    S_HT_md = metadata[f"S{s}_HT{ht}"]

                    outdir = make_dir(args.output_dir, S_HT_md, uniq_dir(base_hasher, paths))
                    if args.debug:
                        print(outdir)
                    else:
                        segment_all(it, outdir, S_HT_md, logger,
                                    crop=args.crop, pad=args.pad, include_unseg=args.include_unseg,
                                    save_unseg=args.save_unseg, save_tifs=args.save_tifs)
            else:
                if args.campaign is not None:
                    args.metadata['campaign'] = args.campaign

                metadata = default_metadata | args.metadata

                all_paths = list_zip_files(args.images)
                outdir = make_dir(args.output_dir, metadata, uniq_dir(args.images, all_paths))
                if args.debug:
                    print(outdir)
                else:
                    it = zip_iterator(args.images, all_paths, logger)
                    segment_all(it, outdir, metadata, logger,
                                crop=args.crop, pad=args.pad, include_unseg=args.include_unseg,
                                save_unseg=args.save_unseg, save_tifs=args.save_tifs)


if __name__ == "__main__":
    main()
