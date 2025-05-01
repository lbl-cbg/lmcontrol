import h5py
import numpy as np
import os
import logging
import argparse
import datetime

def setup_logger(level=logging.INFO):
    """Set up logger with timestamp format."""
    logger = logging.getLogger('hdf5_downsampler')
    logger.setLevel(level)

    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # Create console handler and set level
    handler = logging.StreamHandler()
    handler.setLevel(level)

    # Create formatter with timestamp
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                 datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    return logger

logger = setup_logger()

def downsample_hdf5_file(input_file, output_file, downsample_factor):
    """
    Downsample an HDF5 file while preserving referenced objects and relationships.

    Args:
        input_file: Path to input HDF5 file
        output_file: Path to output HDF5 file
        downsample_factor: Factor by which to downsample (e.g., 10 means keep every 10th sample)
    """
    logger.info(f"Starting to downsample {input_file} to {output_file} with factor {downsample_factor}")

    # Open input file in read mode
    with h5py.File(input_file, 'r') as file_in:
        # Open output file in write mode
        with h5py.File(output_file, 'w') as file_out:

            # First pass: identify all objects that are referenced by other objects via attributes
            logger.info("Identifying referenced objects...")
            referenced_objects, reference_map = identify_referenced_objects(file_in)
            logger.info(f"Found {len(referenced_objects)} referenced objects and {len(reference_map)} objects with references")

            # Second pass: copy groups and their non-reference attributes
            logger.info("Copying groups and their attributes...")
            copy_groups(file_in, file_out)

            # Third pass: process and copy datasets
            logger.info("Processing and copying datasets...")
            process_datasets(file_in, file_out, referenced_objects, downsample_factor)

            # Fourth pass: update all reference attributes
            logger.info("Updating reference attributes...")
            update_references(file_out, reference_map)

    logger.info(f"Successfully completed downsampling {input_file} to {output_file}")

def identify_referenced_objects(file_handle):
    """Identify objects that are referenced by other objects via attributes."""
    referenced_objects = set()
    reference_map = {}  # Maps object names to their reference attributes

    def visit_func(name, obj):
        reference_attrs = {}
        # Check all attributes for references
        for attr_name in obj.attrs:
            if isinstance(obj.attrs[attr_name], h5py.Reference):
                try:
                    ref_obj = file_handle[obj.attrs[attr_name]]
                    ref_name = ref_obj.name
                    referenced_objects.add(ref_name)
                    reference_attrs[attr_name] = ref_name
                    logger.debug(f"Found reference in {name}.{attr_name} pointing to {ref_name}")
                except:
                    logger.warning(f"Could not resolve reference in {name}.{attr_name}")

        if reference_attrs:
            reference_map[name] = reference_attrs

    file_handle.visititems(visit_func)
    return referenced_objects, reference_map

def copy_groups(file_in, file_out):
    """Copy all groups and their non-reference attributes."""
    # Create root group attributes
    logger.info("Copying root group attributes")
    for attr_name, attr_value in file_in.attrs.items():
        if not isinstance(attr_value, h5py.Reference):
            file_out.attrs[attr_name] = attr_value
            logger.debug(f"Copied root attribute: {attr_name}")

    def visit_func(name, obj):
        if isinstance(obj, h5py.Group) and name != '/':
            # Create the group
            logger.info(f"Creating group: {name}")
            group_out = file_out.create_group(name)

            # Copy non-reference attributes
            for attr_name, attr_value in obj.attrs.items():
                if not isinstance(attr_value, h5py.Reference):
                    group_out.attrs[attr_name] = attr_value
                    logger.debug(f"Copied attribute {attr_name} for group {name}")

    file_in.visititems(visit_func)

def process_datasets(file_in, file_out, referenced_objects, downsample_factor):
    """Process and copy all datasets with appropriate downsampling."""

    def visit_func(name, obj):
        if isinstance(obj, h5py.Dataset):
            # Check if this is a referenced dataset that should be copied entirely
            if name in referenced_objects:
                logger.info(f"Copying referenced dataset entirely: {name} {obj.shape}")
                dataset_out = file_out.create_dataset(name, data=obj[:])
            else:
                # This is a regular dataset or an indexing dataset
                if len(obj.shape) > 0 and obj.shape[0] > downsample_factor:
                    # Downsample the first dimension
                    indices = np.arange(0, obj.shape[0], downsample_factor)
                    original_shape = obj.shape
                    if len(obj.shape) == 1:
                        dataset_out = file_out.create_dataset(name, data=obj[indices])
                    else:
                        dataset_out = file_out.create_dataset(name, data=obj[indices, ...])
                    logger.info(f"Downsampled dataset: {name} from {original_shape} to {dataset_out.shape}")
                else:
                    # Dataset is too small to downsample or has no dimensions
                    logger.info(f"Copying dataset without downsampling: {name} {obj.shape}")
                    dataset_out = file_out.create_dataset(name, data=obj[()])

            # Copy all non-reference attributes
            for attr_name, attr_value in obj.attrs.items():
                if not isinstance(attr_value, h5py.Reference):
                    dataset_out.attrs[attr_name] = attr_value
                    logger.debug(f"Copied attribute {attr_name} for dataset {name}")

    file_in.visititems(visit_func)

def update_references(file_out, reference_map):
    """Update all reference attributes to point to objects in the output file."""
    for obj_name, reference_attrs in reference_map.items():
        if obj_name in file_out:
            for attr_name, target_name in reference_attrs.items():
                if target_name in file_out:
                    # Create a new reference to the target in the output file
                    ref = file_out[target_name].ref
                    file_out[obj_name].attrs[attr_name] = ref
                    logger.info(f"Updated reference: {obj_name}.{attr_name} -> {target_name}")
                else:
                    logger.warning(f"Could not update reference: {obj_name}.{attr_name} -> {target_name} (target not found)")
        else:
            logger.warning(f"Could not update references for {obj_name} (object not found)")

def main():
    parser = argparse.ArgumentParser(description='Downsample an HDF5 file while preserving references.')
    parser.add_argument('input_file', help='Path to the input HDF5 file')
    parser.add_argument('output_file', help='Path to the output HDF5 file')
    parser.add_argument('--factor', type=int, default=10, help='Downsampling factor (default: 10)')

    # Add logging control arguments
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-v', '--verbose', action='store_true', help='Verbose output (include attribute information)')
    group.add_argument('-q', '--quiet', action='store_true', help='Quiet mode (no output)')

    args = parser.parse_args()

    # Set up logging based on verbosity flags
    if args.quiet:
        logger.setLevel(logging.ERROR)  # Only show errors
    elif args.verbose:
        logger.setLevel(logging.DEBUG)  # Show all messages including attribute details
    else:
        logger.setLevel(logging.INFO)   # Default: show groups and datasets

    downsample_hdf5_file(args.input_file, args.output_file, args.factor)

if __name__ == "__main__":
    main()
