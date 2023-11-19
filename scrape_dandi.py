"""Test reading NWB files from the DANDI Archive using fsspec.

Requires:
```
pip install fsspec requests aiohttp dandi
```
"""

from dandi.dandiapi import DandiAPIClient, RemoteDandiset
from linked_arrays import H5ToJson
import os
import sys
from tqdm import tqdm
import traceback
import warnings

# these take too long - skip when scraping dandisets in bulk
skip_dandisets = ["DANDI:000016", "DANDI:000226", "DANDI:000232", "DANDI:000341", "DANDI:000541"]
# DANDI:000226
# Read time: 770.77 s
# DANDI:000232
# Read time: 926.37 s
# DANDI:000341
# Read time: 389.20 s
# DANDI:000541
# Read time: 2064.57 s


def scrape_dandi_nwb_to_json(dandiset_indices_to_read: slice, overwrite: bool):
    """Test reading the first NWB asset from a random selection of 50 dandisets that uses NWB.

    Parameters
    ----------
    dandiset_indices_to_read : slice
        The slice of dandisets returned from `DandiAPIClient.get_dandisets()` to read.
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
            process_dandiset(dandiset, overwrite)
        except Exception as e:
            print(traceback.format_exc())
            failed_reads[dandiset] = e

        # separately, when streaming, is http fs or s3 fs faster?

    if failed_reads:
        print("Failed reads:")
        print(failed_reads)
        sys.exit(1)


def process_dandiset(dandiset: RemoteDandiset, overwrite: bool):
    """Process a single dandiset from a RemoteDandiset object.

    Parameters
    ----------
    dandiset : RemoteDandiset
        The dandiset to process.
    overwrite : bool
        Whether to overwrite existing files.
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
        return

    if first_asset:  # if not necessary but useful for testing on first asset
        asset = first_asset

        # read all assets
        # assets = list(dandiset.get_assets())
        # for asset in tqdm(assets):
        if asset.path.split(".")[-1] != "nwb":
            return

        json_path = f"dandi_json/{id}/{asset.path}.json"
        if os.path.exists(json_path) and not overwrite:
            return
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
        translator = H5ToJson(s3_url, json_path, None)
        translator.translate()


def process_dandiset_from_id(dandiset_id: str, overwrite: bool):
    """Process a single dandiset from the dandiset ID.

    Parameters
    ----------
    dandiset_id : str
        The dandiset ID to process.
    overwrite : bool
        Whether to overwrite existing files.
    """
    client = DandiAPIClient()
    dandiset = client.get_dandiset(dandiset_id)
    process_dandiset(dandiset, overwrite)


if __name__ == "__main__":
    # process_dandiset_from_id("000244", overwrite=True)
    dandiset_indices_to_read = slice(0, 1000)  # None = all
    scrape_dandi_nwb_to_json(dandiset_indices_to_read, overwrite=False)
