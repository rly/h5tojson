"""Test the example code in README.md.

NOTE: This requires the internet because it streams an NWB file from DANDI.
"""

import os

from dandi.dandiapi import DandiAPIClient

from h5tojson import H5ToJson

# Get the S3 URL of a particular NWB HDF5 file from Dandiset 000049
dandiset_id = "000049"  # ephys dataset from the Svoboda Lab
subject_id = "sub-661968859"
file_name = "sub-661968859_ses-681698752_behavior+ophys.nwb"
with DandiAPIClient() as client:
    path = f"{subject_id}/{file_name}"
    asset = client.get_dandiset(dandiset_id).get_asset_by_path(path)
    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)

# Create an output directory and set the output JSON path
output_dir = f"test_output/{dandiset_id}/{subject_id}"
os.makedirs(output_dir, exist_ok=True)
json_path = f"{output_dir}/sub-661968859_ses-681698752_behavior+ophys.nwb.json"

# Create the H5ToJson translator object and run it
translator = H5ToJson(s3_url, json_path)
translator.translate()

# Translate the same file, but save the DfOverF/data dataset as an individual HDF5 file
translator = H5ToJson(
    s3_url, json_path, datasets_as_hdf5=["/processing/brain_observatory_pipeline/Fluorescence/DfOverF/data"]
)
translator.translate()
