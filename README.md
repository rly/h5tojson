# linked-arrays
Experimental format for datasets with linked arrays

Kerchunk with:
1. Human-readable metadata (names, attributes, links, etc.) separated from array data so that the array chunk references do not need to be downloaded if not needed.
2. Organization separated from array data
3. Support for locally stored array data according to a standard files and folders-based organization


HDMF can map HDF5 files into builders
HDF5Zarr can also translate HDF5 files into Zarr stores



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