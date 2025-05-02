from typing import Union, List

import os
import re
import tempfile
import contextlib
import tensorflow as tf

ENABLE_PATH_PREPROCESS = True

def is_hdf5_filepath(
    filepath: str,
) -> bool:
    return filepath.endswith(".h5") or filepath.endswith(".keras") or filepath.endswith(".hdf5")


def is_cloud_path(
    path: str,
) -> bool:
    """Check if the path is on cloud (which requires tf.io.gfile)

    Args:
        path (str): Path to directory or file

    Returns:
        bool: True if path is on cloud, False otherwise
    """
    return bool(re.match(r"^[a-z]+://", path))


def preprocess_paths(
    paths: Union[List[str], str],
    isdir: bool = False,
    enabled: bool = True,
    check_exists: bool = False,
) -> Union[List[str], str]:
    """Expand the path to the root "/" and makedirs

    Args:
        paths (Union[List, str]): A path or list of paths

    Returns:
        Union[List, str]: A processed path or list of paths, return None if it's not path
    """
    if not (enabled and ENABLE_PATH_PREPROCESS):
        return paths
    if isinstance(paths, (list, tuple)):
        paths = [path if is_cloud_path(path) else os.path.abspath(os.path.expanduser(path)) for path in paths]
        for i, path in enumerate(paths):
            dirpath = path if isdir else os.path.dirname(path)
            if not tf.io.gfile.exists(path):
                if check_exists:
                    paths[i] = None
                else:
                    if not tf.io.gfile.exists(dirpath):
                        tf.io.gfile.makedirs(dirpath)
        return list(filter(None, paths))
    if isinstance(paths, str):
        paths = paths if is_cloud_path(paths) else os.path.abspath(os.path.expanduser(paths))
        dirpath = paths if isdir else os.path.dirname(paths)
        if not tf.io.gfile.exists(paths):
            if check_exists:
                return None
            if not tf.io.gfile.exists(dirpath):
                tf.io.gfile.makedirs(dirpath)
        return paths
    return None


@contextlib.contextmanager
def save_file(
    filepath: str,
):
    if is_cloud_path(filepath):
        _, ext = os.path.splitext(filepath)
        with tempfile.NamedTemporaryFile(suffix=ext) as tmp:
            yield tmp.name
            tf.io.gfile.copy(tmp.name, filepath, overwrite=True)
    else:
        yield filepath


@contextlib.contextmanager
def read_file(
    filepath: str,
):
    if is_cloud_path(filepath):
        _, ext = os.path.splitext(filepath)
        with tempfile.NamedTemporaryFile(suffix=ext) as tmp:
            tf.io.gfile.copy(filepath, tmp.name, overwrite=True)
            yield tmp.name
    else:
        yield filepath