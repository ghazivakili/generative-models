import requests
import tqdm
from pathlib import Path


def download_file_from_url(
    url: str, target_path: str, write_mode: str = "wb", chunk_size: int = 1024 * 1024
) -> None:
    """Downloads a file from a remote, publicly accessible, url.

    Args:
        url (str): file url.
        target_path (str): path to save file to.
        write_mode (str): write mode. Defaults to 'wb'.
        chunk_size (int, optional): number of bytes in every chunk of data read from the stream. Defaults to 1024*1024.

    Raises:
        RuntimeError: if GET request response code is not 200.
    """
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        dataset_bytes = response.iter_content(chunk_size)
        content_length = int(response.headers.get("Content-Length", 0))
        with open(target_path, write_mode) as f, tqdm.tqdm(
            response.iter_content(), total=content_length
        ) as pbar:
            pbar.set_description(f"Downloading file {url}")
            for chunk in dataset_bytes:
                f.write(chunk)
                pbar.update(chunk_size)
    else:
        raise RuntimeError(
            f"Got response status code {response.status_code}, but expected status code 200."
        )


source_file = Path("/root/zinc-15-drug-like-tranches.txt").resolve()
destination_dir = Path("/root/tranches/").resolve()


with open(str(source_file), "r") as f:
    urls: list[str] = f.read().split("\n")

for url in tqdm.tqdm(urls):
    tranche = url[25:].replace("/", "_")
    try:
        download_file_from_url(url=url, target_path=str(destination_dir / tranche))
    except Exception as e:
        print(f"Encountered error when downloading tranche {tranche}: {e}")
