# Adapted from https://github.com/microsoft/recommenders/blob/main/recommenders/datasets/download_utils.py

from pathlib import Path

import os
import requests
import math
import zipfile
from tqdm import tqdm
from retrying import retry
from contextlib import contextmanager

from src import utils

log = utils.get_pylogger(__name__)


@retry(wait_random_min=1000, wait_random_max=5000, stop_max_attempt_number=5)
def maybe_download(url: str, filename: str, work_directory: Path) -> str:
    """Download a file if not already downloaded.

    Args:
        url (str): URL of the file to download.
        filename (str): File name.
        work_directory (str): Working directory.

    Returns:
        str: Filepath of the downloaded file.
    """

    os.makedirs(work_directory, exist_ok=True)
    filepath = os.path.join(work_directory, filename)
    
    if not os.path.exists(filepath):
        r = requests.get(url, stream=True)
        if r.status_code ==  200:
            log.info(f'Downloading {url}')
            total_size = int(r.headers.get('content-length', 0))
            block_size = 1024
            num_iterables = math.ceil(total_size / block_size)
            with open(filepath, 'wb') as file:
                for data in tqdm(
                        r.iter_content(block_size),
                        total=num_iterables,
                        unit='KB',
                        unit_scale=True,
                        ):
                    file.write(data)
        else:
            log.warning(f'Problem downloading {url}')
            r.raise_for_status()
    else:
        log.info(f'File {filepath} already downloaded.')

    return filepath

@contextmanager
def download_path(path: str):
    """Returns a path to download data.

    Args:
        path (str): Path to download data.

    Returns:
        str: Real path where the data will be downloaded.
    """
    path = os.path.realpath(path)
    yield path

def extract_file(archive_file: str, dst_dir: str, clean_archive: bool = False) -> None:
    """Extract files from a compressed file.

    Args:
        archive_file (str): Compressed file.
        dst_dir (str): Destination directory.
        clean_archive (bool): Whether to remove the compressed file.
    """

    fz = zipfile.ZipFile(archive_file, 'r')
    for file in fz.namelist():
        fz.extract(file, dst_dir)
    if clean_archive:
        os.remove(archive_file)

