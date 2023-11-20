# linked-arrays
Experimental format for storing HDF5 data as JSON.

Kerchunk with human-readable metadata (names, attributes, links, etc.) separated from array data so that the array chunk
references do not need to be downloaded if not needed.

1. Kerchunk stores keys that represent zarr files, e.g. `"acquisition/ElectricalSeries/data/.zarray"`,
`"acquisition/ElectricalSeries/data/.zattrs"`, and `"acquisition/ElectricalSeries/data/0.0"`. The `.zarray` and
`.zattrs` values are JSON-encoded strings that must be parsed to get the array shape, dtype, and attributes. This
format makes those values difficult to parse or query without custom code to decode the JSON. In the
linked-arrays format, those values are stored as JSON.
2. Kerchunk stores all keys as a flat list under the `"refs"` key, despite the keys holding a hierarchical
data format. This makes it difficult to query the data without custom code to parse the keys and find one key
based on a different key that matches a query, e.g., to find the key representing a dataset where the parent
group has a particular attribute, requires querying the parent group name and then building the key from the
parent group name and the dataset name. In the linked-arrays format, the keys are stored as a hierarchical
structure, so that there is a common parent group between the attribute and the dataset.
3. Kerchunk stores the chunk locations for every dataset. This is good for its use case of indexing the
chunks for fast access, but for parsing and querying the non-dataset values of an HDF5 file, it is not necessary
and takes a significant amount of space in the JSON file. In the linked-arrays format, the chunk locations are
omitted.
4. HDF5 links and references are encoded in the JSON.

HDMF can map HDF5 files into builders
HDF5Zarr can also translate HDF5 files into Zarr stores

Another, very verbose/detailed way to represent the HDF5 file as JSON is:
https://hdf5-json.readthedocs.io/en/latest/examples/tall.html

## Dev setup

Install:
```
git clone https://github.com/rly/linked-arrays
cd linked-arrays
mamba env create -n arrays python=3.11 --yes
conda activate arrays
pip install -e ".[dev]"
```

Run tests and other dev checks:
```
pytest
black .
ruff .
codespell .
interrogate .
```

To use notebooks, install `jupyterlab`.

TODO:
- [ ] Set up pre-commit hooks

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