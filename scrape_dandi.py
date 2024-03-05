"""Test reading NWB files from the DANDI Archive using remfile.

TODO: parallelize

Requires:
```
pip install remfile dandi
```
"""

import os
import sys
import time
import traceback
import warnings
from typing import List, Optional

from dandi.dandiapi import DandiAPIClient, RemoteDandiset
from tqdm import tqdm

from h5tojson import H5ToJson

# these take too long - skip when scraping dandisets in bulk
skip_dandisets: List[str] = []

# DANDI:000016 - a lot of groups
# Translation time: 253.93 s

# DANDI:000232 - a lot of groups
# Translation time: 169.02 s

# DANDI:000235 - large pixel mask datasets
# Translation time: 91.94 s

# DANDI:000236 - large pixel mask datasets
# Translation time: 116.44 s

# DANDI:000237 - large pixel mask datasets
# Translation time: 220.21 s

# DANDI:000238 - large pixel mask dataset(s)
# Translation time: 258.22 s

# DANDI:000294 - (icephys) a lot of groups
# Translation time: 773.42 s

# DANDI:000301
# Translation time: 124.68 s

# DANDI:000341 - (icephys) a lot of groups
# Translation time: 373.19 s

# DANDI:000488 - a lot of groups (images) and large string datasets
# Translation time: 737.30 s

# DANDI:000541 - a lot of voxel masks
# Translation time: 183.81 s

# DANDI:000546 - over 2300 ElectricalSeries
# Translation time: 917.51 s


def scrape_dandi_nwb_to_json(
    dandiset_indices_to_read: slice,
    num_assets_per_dandiset: int,
    output_dir: str,
    translation_times_path: str,
    overwrite: bool,
):
    """Test reading the first NWB asset from a random selection of 50 dandisets that uses NWB.

    Parameters
    ----------
    dandiset_indices_to_read : slice
        The slice of dandisets returned from `DandiAPIClient.get_dandisets()` to read.
    num_assets_per_dandiset : int
        The number of assets to translate per dandiset.
    output_dir : str
        The directory to write the JSON files to.
    translation_times_path : str
        The path to write the translation times to.
    overwrite : bool
        Whether to overwrite existing files.
    """

    if overwrite or not os.path.exists(translation_times_path):
        # overwrite translation times file so we don't append to it
        with open(translation_times_path, "w") as f:
            f.write("dandiset_id,asset_path,read_time\n")

    client = DandiAPIClient()
    dandisets = list(client.get_dandisets())

    dandisets_to_read = dandisets[dandiset_indices_to_read]
    print("Reading NWB files from the following dandisets:")
    print([d.get_raw_metadata()["identifier"] for d in dandisets_to_read])

    warnings.filterwarnings("ignore", message=r"Ignoring cached namespace.*", category=UserWarning)

    failed_reads = dict()
    dandiset_translation_times = dict()
    for i, dandiset in enumerate(dandisets_to_read):
        dandiset_metadata = dandiset.get_raw_metadata()

        # skip any dandisets that do not use NWB
        if not any(
            data_standard["identifier"] == "RRID:SCR_015242"  # this is the RRID for NWB
            for data_standard in dandiset_metadata["assetsSummary"].get("dataStandard", [])
        ):
            continue

        dandiset_identifier = dandiset_metadata["identifier"]
        print("-------------------")
        print(f"{i}: {dandiset_identifier}")
        if dandiset_identifier in skip_dandisets:
            continue

        try:
            dandiset_translation_times[dandiset] = translate_dandiset_assets(
                dandiset, num_assets_per_dandiset, output_dir, overwrite
            )
        except Exception as e:
            print(traceback.format_exc())
            failed_reads[dandiset] = e

        # append translation times to csv file after every read so we don't lose data if the script crashes
        with open(translation_times_path, "a") as f:
            for asset_path, translation_time in dandiset_translation_times[dandiset].items():
                f.write(f"{dandiset.identifier},{asset_path},{translation_time}\n")

    if failed_reads:
        print("Failed reads:")
        print(failed_reads)
        sys.exit(1)


def translate_dandiset_assets(
    dandiset: RemoteDandiset, num_assets: Optional[int], output_dir: str, overwrite: bool
) -> dict:
    """Process a single dandiset from a RemoteDandiset object.

    Parameters
    ----------
    dandiset : RemoteDandiset
        The dandiset to process.
    num_assets : int, optional
        The number of assets to translate. If None, then all assets are translated.
        Already translated assets count toward this limit.
    output_dir : str
        The directory to write the JSON file to.
    overwrite : bool
        Whether to overwrite existing files.

    Returns
    -------
    dict
        The time it took to translate each NWB file.
    """
    id = dandiset.identifier
    if id.startswith("DANDI:"):
        id = id[6:]

    # reading all into a list when we want only one may be inefficient
    # (can take 1-5 seconds depending on number of assets)
    # but it's negligible compared to the translation time
    assets = list(dandiset.get_assets())

    # remove non-NWB files (it could be MP4)
    assets = [a for a in assets if a.path.split(".")[-1] == "nwb"]

    if num_assets is None:
        num_assets = len(assets)

    if num_assets == 0:
        print("No NWB files?!")
        return dict()

    assets = assets[:num_assets]
    asset_translation_times = dict()

    for asset in tqdm(assets, desc=f"{id} assets"):
        json_path = f"{output_dir}/{id}/{asset.path}.json"
        if overwrite or not os.path.exists(json_path):
            os.makedirs(os.path.dirname(json_path), exist_ok=True)

            s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
            start = time.perf_counter()
            translator = H5ToJson(s3_url, json_path)
            translator.translate()
            end = time.perf_counter()

            asset_translation_times[asset.path] = end - start

    return asset_translation_times


def scrape_single_dandiset_nwb_to_json(dandiset_id: str, num_assets: int, output_dir: str, overwrite: bool):
    """Process a single dandiset from the dandiset ID.

    Parameters
    ----------
    dandiset_id : str
        The dandiset ID to process.
    num_assets : int
        The number of assets to translate.
    output_dir : str
        The directory to write the JSON file to.
    overwrite : bool
        Whether to overwrite existing files.
    """
    client = DandiAPIClient()
    dandiset = client.get_dandiset(dandiset_id)
    translate_dandiset_assets(dandiset, num_assets, output_dir, overwrite)


if __name__ == "__main__":
    # scrape_single_dandiset_nwb_to_json("000005", num_assets=None, output_dir="dandi_json", overwrite=True)
    # stop

    # NOTE perf_counter includes sleep time
    start = time.perf_counter()
    dandiset_indices_to_read = slice(None)  # slice(0, 1000)  # slice(None) = all
    scrape_dandi_nwb_to_json(
        dandiset_indices_to_read,
        num_assets_per_dandiset=1,
        output_dir="dandi_json",
        translation_times_path="dandi_json_translation_times.csv",
        overwrite=False,
    )
    end = time.perf_counter()
    print(f"Run time: {end - start:.2f} s")
