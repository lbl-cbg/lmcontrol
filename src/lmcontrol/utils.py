from datetime import datetime
import logging
import sys

from importlib import resources
import yaml

def parse_logger(string, stream=sys.stderr, level='info'):
    if not string:
        ret = logging.getLogger()
        hdlr = logging.StreamHandler(stream)
    else:
        ret = logging.getLogger(string)
        hdlr = logging.FileHandler(string)

    ret.setLevel(getattr(logging, level.upper()))
    ret.addHandler(hdlr)
    hdlr.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    return ret


def get_logger(level='info'):
    return parse_logger('', level=level)


def get_metadata_info():
    with resources.as_file(resources.files(__package__) / 'metadata_info.yaml') as file_path:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
    return data


def get_seed():
    return int(datetime.now().timestamp()*1000000) % (2**32 -1)


def parse_seed(string):
    if string:
        try:
            i = int(string)
            if i > 2**32 - 1:
                raise ValueError(string)
            return i
        except :
            raise argparse.ArgumentTypeError(f'{string} is not a valid seed')
    else:
        return get_seed()

def format_time_diff(total_seconds):
    """Convert total seconds to a formatted string with days, hours, minutes, seconds"""
    days = int(total_seconds // 86400)  # 86400 seconds in a day
    hours = int((total_seconds % 86400) // 3600)  # 3600 seconds in an hour
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:  # Always show seconds if nothing else
        parts.append(f"{seconds}s")

    return " ".join(parts)
