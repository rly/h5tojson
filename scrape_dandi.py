"""Test reading NWB files from the DANDI Archive using remfile.

Requires:
```
pip install remfile dandi
```
"""

from typing import Optional
from dandi.dandiapi import DandiAPIClient, RemoteDandiset
from linked_arrays import H5ToJson
import os
import sys
import time

# from tqdm import tqdm
import traceback
import warnings

# these take too long - skip when scraping dandisets in bulk
skip_dandisets = []

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
    dandiset_indices_to_read: slice, output_dir: str, translation_times_path: str, overwrite: bool
):
    """Test reading the first NWB asset from a random selection of 50 dandisets that uses NWB.

    Parameters
    ----------
    dandiset_indices_to_read : slice
        The slice of dandisets returned from `DandiAPIClient.get_dandisets()` to read.
    output_dir : str
        The directory to write the JSON files to.
    translation_times_path : str
        The path to write the translation times to.
    overwrite : bool
        Whether to overwrite existing files.
    """
    client = DandiAPIClient()
    dandisets = list(client.get_dandisets())

    dandisets_to_read = dandisets[dandiset_indices_to_read]
    print("Reading NWB files from the following dandisets:")
    print([d.get_raw_metadata()["identifier"] for d in dandisets_to_read])

    warnings.filterwarnings("ignore", message=r"Ignoring cached namespace.*", category=UserWarning)

    failed_reads = dict()
    translation_times = dict()
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
            translation_times[dandiset] = process_dandiset(dandiset, output_dir, overwrite)
        except Exception as e:
            print(traceback.format_exc())
            failed_reads[dandiset] = e

    # write translation times to a csv file
    # TODO append to a csv file after every read so we don't lose data if the script crashes
    with open(translation_times_path, "w") as f:
        f.write("dandiset_id,read_time\n")
        for dandiset, translation_time in translation_times.items():
            f.write(f"{dandiset.identifier},{translation_time}\n")

    if failed_reads:
        print("Failed reads:")
        print(failed_reads)
        sys.exit(1)


def process_dandiset(dandiset: RemoteDandiset, output_dir: str, overwrite: bool) -> Optional[float]:
    """Process a single dandiset from a RemoteDandiset object.

    Parameters
    ----------
    dandiset : RemoteDandiset
        The dandiset to process.
    output_dir : str
        The directory to write the JSON file to.
    overwrite : bool
        Whether to overwrite existing files.

    Returns
    -------
    float
        The time it took to translate the NWB file.
    """
    id = dandiset.identifier
    if id.startswith("DANDI:"):
        id = id[6:]

    # iterate through assets until we get an NWB file (it could be MP4)
    assets = dandiset.get_assets()
    first_asset = next(assets)
    while first_asset.path.split(".")[-1] != "nwb":
        first_asset = next(assets)
    if first_asset.path.split(".")[-1] != "nwb":
        print("No NWB files?!")
        return None

    asset = first_asset

    # read all assets
    # assets = list(dandiset.get_assets())
    # for asset in tqdm(assets):
    if asset.path.split(".")[-1] != "nwb":
        return None

    json_path = f"{output_dir}/{id}/{asset.path}.json"
    if os.path.exists(json_path) and not overwrite:
        return None
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
    start = time.perf_counter()
    translator = H5ToJson(s3_url, json_path, None)
    translator.translate()
    end = time.perf_counter()
    print(f"Translation time: {end - start:.2f} s")

    return end - start


def process_dandiset_from_id(dandiset_id: str, output_dir: str, overwrite: bool):
    """Process a single dandiset from the dandiset ID.

    Parameters
    ----------
    dandiset_id : str
        The dandiset ID to process.
    output_dir : str
        The directory to write the JSON file to.
    overwrite : bool
        Whether to overwrite existing files.
    """
    client = DandiAPIClient()
    dandiset = client.get_dandiset(dandiset_id)
    process_dandiset(dandiset, output_dir, overwrite)


if __name__ == "__main__":
    # process_dandiset_from_id("000546", output_dir="dandi_json", overwrite=True)

    # NOTE perf_counter includes sleep time
    start = time.perf_counter()
    dandiset_indices_to_read = slice(None)  # slice(0, 1000)  # slice(None) = all
    scrape_dandi_nwb_to_json(
        dandiset_indices_to_read,
        output_dir="dandi_json",
        translation_times_path="dandi_json_translation_times.csv",
        overwrite=False,
    )
    end = time.perf_counter()
    print(f"Run time: {end - start:.2f} s")
