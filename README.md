# h5tojson
Experimental format for storing HDF5 data as JSON, with or without array chunk metadata

## Differences with [kerchunk](https://github.com/fsspec/kerchunk)

h5tojson provides the option to leave out the array chunk metadata for faster caching and processing of
the non-array metadata.

h5tojson is like kerchunk but with human-readable metadata (names, attributes, links, etc.) separated from
array data so that the array chunk references do not need to be downloaded if not needed.

In addition:

1. Kerchunk stores keys that represent zarr files, e.g. `"acquisition/ElectricalSeries/data/.zarray"`,
`"acquisition/ElectricalSeries/data/.zattrs"`, and `"acquisition/ElectricalSeries/data/0.0"`. The `.zarray` and
`.zattrs` values are JSON-encoded strings that must be parsed to get the array shape, dtype, and attributes. This
format makes those values difficult to parse or query without custom code to decode the JSON. In the
h5tojson format, those values are stored as JSON.
2. Kerchunk stores all keys as a flat list under the `"refs"` key, despite the keys holding a hierarchical
data format. This makes it difficult to query the data without custom code to parse the keys and find one key
based on a different key that matches a query, e.g., to find the key representing a dataset where the parent
group has a particular attribute, requires querying the parent group name and then building the key from the
parent group name and the dataset name. In the h5tojson format, the keys are stored as a hierarchical
structure, so that there is a common parent group between the attribute and the dataset.
3. Kerchunk stores the chunk locations for every dataset. This is good for its use case of indexing the
chunks for fast access, but for parsing and querying the non-dataset values of an HDF5 file, it is not necessary
and takes a significant amount of space in the JSON file. In the h5tojson format, the chunk locations are
omitted.
4. HDF5 links and references are encoded in the JSON.

HDMF can map HDF5 files into builders
HDF5Zarr can also translate HDF5 files into Zarr stores

Another, very verbose/detailed way to represent the HDF5 file as JSON is:
https://hdf5-json.readthedocs.io/en/latest/examples/tall.html

## Dev setup

Install:
```
git clone https://github.com/rly/h5tojson
cd h5tojson
mamba create -n h5tojson python=3.11 --yes
mamba activate h5tojson
pip install -e ".[dev]"
```

Optional: Install [`pre-commit`](https://pre-commit.com/) which runs several basic checks and
`ruff`, `isort`, `black`, `interrogate`, and `codespell`.
```
pip install pre-commit
pre-commit install
pre-commit run
```

Run tests and other dev checks individually:
```
pytest
black .
ruff .
codespell .
interrogate .
mypy .
isort .
```

To use notebooks, install `jupyterlab`.

## Example of how to run

1. Install using the steps above.
2. `pip install dandi` for an API to access the S3 URLs of NWB HDF5 files.
3. Run the code below.

```python
from dandi.dandiapi import DandiAPIClient
from h5tojson import H5ToJson
import os

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
```

## Run the translation and convert select datasets to an individual HDF5 file

```python
# Translate the same file, but save the DfOverF/data dataset as an individual HDF5 file
translator = H5ToJson(
    s3_url, json_path, datasets_as_hdf5=["/processing/brain_observatory_pipeline/Fluorescence/DfOverF/data"]
)
translator.translate()
```

This package supports taking an existing HDF5 file and translating it into:

1. Only metadata as JSON (no datasets) (set `skip_all_dataset_data` == True)
2. Metadata as JSON plus references to chunk locations for efficient streaming in a separate JSON (set `chunk_refs_file_path`)
3. Metadata as JSON plus inlined datasets as readable values in the same JSON (configure `dataset_inline_max_bytes`, `object_dataset_inline_max_bytes`, `compound_dtype_dataset_inline_max_bytes`)
4. Metadata as JSON plus select or all datasets stored as individual HDF5 files referenced by metadata JSON (set `datasets_as_hdf5`)


## What can we use these JSON for?

Answering queries across many dandisets, e.g.:
- What is the listed species of each dandiset?
- What neurodata types are used by each dandiset?
- What NWB extensions are used by each dandiset?
- What versions of NWB are used by each dandiset?
- What compression and chunking settings are used by each dataset within each dandiset?
- How many dandisets have raw data?
- How many dandisets have processed data from a particular pipeline?
- How many dandisets have the same data organization in all of their NWB files?

Run `scrape_dandi.py` to generate one JSON file for one NWB file from each dandiset.
Then see `queries.ipynb` for example code on how to run some of the above queries using those JSON files.

## Ideas for querying JSON (not used yet):
- JSONPath - older language. Works well.
  - Python implementation https://github.com/h2non/jsonpath-ng
- JMESPath - newer language, like JSONPath but has pros and cons. It is a complete grammar, can do joins/multiselects, yields single results, has pipes, and does not support recursion. Very popular. Used by AWS CLI.
  - Python implementation https://github.com/jmespath/jmespath.py
- jq - older language that is turing-complete and can do a lot
