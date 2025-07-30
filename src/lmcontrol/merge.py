import argparse

from .utils import get_logger, get_metadata_info
from .data_utils import load_npzs

from hdmf.spec import GroupSpec, DatasetSpec, NamespaceBuilder
from hdmf.backends.hdf5 import HDF5IO, H5DataIO
from hdmf.common import DynamicTable, load_namespaces, get_hdf5io
import numpy as np
from sklearn.preprocessing import LabelEncoder


_scale_map = dict()

def scaler(key):
    global _scale_map
    def func(f):
        _scale_map[key] = f
        return f
    return func

def get_scaler(key):
    return _scale_map[key]

def ls_metadata(argv=None):
    """
    Prints the keys and values inside multiple NPZ files.
    If the value is a scalar or a single-element ndarray, it prints the value.
    If the value is an ndarray with more than one element, it prints the shape and data type.

    Parameters:
    argv (list): Command line arguments (default is None).
    """
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Print keys and values from multiple NPZ files.')
    parser.add_argument('file_paths', type=str, nargs='+', help='The paths to the NPZ files.')

    # Parse the command-line arguments
    args = parser.parse_args(argv)

    # Iterate over each file path provided
    for file_path in args.file_paths:
        print(f"\n{file_path}")
        try:
            with np.load(file_path) as data:
                for key in data.keys():
                    value = data[key]
                    if np.isscalar(value) or (isinstance(value, np.ndarray) and value.size == 1):
                        print(f"{key:<16}: {value.item()}")  # Use .item() to get the scalar value
                    else:
                        print(f"{key:<16}: shape={value.shape}, dtype={value.dtype}")
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")


@scaler("auto-arcsinh")
def suggest_arcsinh_cofactor(data):
    """
    Suggest arcsinh cofactor 'a' based on data distribution.

    Parameters:
    - data: 1D numpy array or Pandas Series of positive values (can include zeros and negatives)

    Returns:
    - suggested_cofactor: float, suggested cofactor based on heuristic
    """
    # Filter out negatives for statistics (arcsinh handles negatives, but cofactor depends on positive scale)
    positive_data = data[data > 0]

    if len(positive_data) == 0:
        raise ValueError("No positive values in data to base cofactor suggestion on.")

    median_val = np.median(positive_data)
    #p90_val = np.percentile(positive_data, 90)

    # Heuristic: cofactor ~ 1 / median or 1 / p90 (choose which feels better)
    suggested_cofactor = 1.0 / median_val


    scaled = np.arcsinh(data * suggested_cofactor)

    return scaled


def main(argv=None):

    desc = "Merge cropped images into a single HDF5 file"
    epi = """
    """
    parser = argparse.ArgumentParser(description=desc, epilog=epi)
    parser.add_argument("output_h5", help="the path of the HDF5 to write images to")
    parser.add_argument("npzs", nargs="+", help="the NPZ files with cropped images to merge")
    parser.add_argument("-f", "--force", action='store_true', default=False, help="ignore checks, just write my file")
    parser.add_argument("-c", "--compress", action='store_true', default=False, help="compress images and masks")

    args = parser.parse_args(argv)

    logger = get_logger()

    if not args.output_h5.endswith('.h5') and not args.force:
        logger.critical(f"output file {args.output_h5} does not end with .h5 - Naively failing to prevent overwriting your data")
        exit(1)

    metadata_info = get_metadata_info()

    masks, images, paths, metadata = load_npzs(args.npzs, logger=logger)

    table = DynamicTable(name='single_cell_images',
                         description='Single cell images from ABPDU',
                         id=np.arange(len(masks)))

    if args.compress:
        images = H5DataIO(images, compression='gzip', chunks=(100, 152, 152))
        masks = H5DataIO(masks.astype('uint8'), compression='gzip')
    else:
        masks = masks.astype('uint8')
    table.add_column('images', 'Cropped single cell images', data=images)
    table.add_column('masks', 'The segmentation masks for each image', data=masks)
    table.add_column('paths', 'the original file path for the raw image', data=paths)
    missing_desc = list()
    for key, value in metadata.items():
        if key not in metadata:
            logger.warning(f"Skipping key '{key}' - Not found in metadata info")
            continue

        if key not in metadata_info:
            missing_desc.append(key)
            continue
        if metadata_info[key]['enum']:
            enc = LabelEncoder().fit(value)
            value = enc.transform(value).astype(np.uint64)
            table.add_column(key, metadata_info[key]['description'], data=value, enum=enc.classes_)
        else:
            if 'scale' not in metadata_info[key]:
                breakpoint()
            if metadata_info[key]['scale'] is not None:
                value = get_scaler(metadata_info[key]['scale'])(value)
            table.add_column(key, metadata_info[key]['description'], data=value.astype(float))

    if len(missing_desc) > 0:
        missing_desc = ", ".join(missing_desc)
        if args.force:
            logger.info(f"Ignoring the following columns due to missing metadata: {missing_desc}")
        else:
            msg = f"The following keys do not have metadata info. Use --force flag to ignore them, or add their metadata info src/lmcontrol/metadata_info.yaml\n{missing_desc}"
            logger.info(msg)
            exit(1)

    logger.info(f"Writing to {args.output_h5}")
    with get_hdf5io(args.output_h5, 'w') as io:
        io.write(table)

    logger.info(f"DONE")

if __name__ == "__main__":
    main()
