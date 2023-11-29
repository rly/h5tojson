"""Functions to test translation from HDF5 to JSON using H5ToJson"""

import json
from pathlib import Path
from typing import Callable, Optional

import fsspec
import h5py
import numpy as np
import zarr
from dateutil.parser import parse

from linked_arrays.h5tojson import H5ToJson


def _create_translate(
    tmp_path: Path, callable: Callable[[h5py.File], None], kwargs: Optional[dict] = None
) -> tuple[dict, dict]:
    """Helper function to create an HDF5 file using the callable function, and translate it to JSON

    Parameters
    ----------
    tmp_path : Path
        Path to a temporary directory.
    callable : function
        A function that takes an h5py.File object as its only argument.
    kwargs : dict, optional
        Keyword arguments to pass to the H5ToJson constructor.

    Returns
    -------
    tuple[dict, dict]
        A tuple containing the JSON dictionary and the chunk_refs dictionary.
    """
    hdf5_file_path = tmp_path / "test.h5"
    with h5py.File(hdf5_file_path, "w") as f:
        callable(f)

    json_file_path = tmp_path / "test.json"
    chunk_refs_file_path = tmp_path / "chunk_refs.json"
    translator = H5ToJson(hdf5_file_path, json_file_path, chunk_refs_file_path, **(kwargs or {}))
    translator.translate()

    with open(json_file_path) as f:
        json_dict = json.load(f)

    with open(chunk_refs_file_path) as f:
        chunk_refs = json.load(f)

    return json_dict, chunk_refs


def _get_array_from_chunk_refs(chunk_refs: dict, dataset_name: str) -> np.ndarray:
    """Use Zarr to get an array from a chunk_refs dictionary."""
    mapper = fsspec.get_mapper("reference://", fo=chunk_refs)
    z = zarr.open(mapper)
    if z[dataset_name].shape == ():
        return z[dataset_name][()]
    return z[dataset_name][:]


def is_iso8601(date_string: str) -> bool:
    """Check if a string is a valid ISO 8601 date."""
    try:
        parse(date_string)
        return True
    except ValueError:
        return False


def test_translate_root(tmp_path):
    """Test translation of an empty root group."""

    def set_up_test_file(_: h5py.File):
        pass

    json_dict, _ = _create_translate(tmp_path, set_up_test_file)

    created_at = json_dict.pop("created_at")
    assert is_iso8601(created_at)

    expected_chunk_refs_file_path = f"file://{str(tmp_path / 'chunk_refs.json')}"
    expected = {
        "version": 1,
        "templates": {"c": expected_chunk_refs_file_path},
        "file": {},
        "translation_options": {
            "compound_dtype_dataset_inline_max_bytes": 2000,
            "dataset_inline_threshold_max_bytes": 500,
            "object_dataset_inline_max_bytes": 200000,
        },
    }

    assert json_dict == expected


def test_translate_subgroup(tmp_path):
    """Test translation of a simple subgroup."""

    def set_up_test_file(f: h5py.File):
        f.create_group("group1")

    json_dict, _ = _create_translate(tmp_path, set_up_test_file)

    expected = {
        "groups": {
            "group1": {},
        },
    }
    assert json_dict["file"] == expected


def test_translate_nested_group_links(tmp_path):
    """Test translation of nested groups with links."""

    def set_up_test_file(f: h5py.File):
        f.create_group("group1/group2")
        f["link_to_group2"] = h5py.SoftLink("/group1/group2")
        f["link_to_link_to_group2"] = h5py.SoftLink("/link_to_group2")
        f["group1/group2"]["cyclic_link"] = h5py.SoftLink("/group1")

    json_dict, _ = _create_translate(tmp_path, set_up_test_file)

    expected = {
        "groups": {
            "group1": {
                "groups": {
                    "group2": {
                        "soft_links": {
                            "cyclic_link": {
                                "path": "/group1",
                            },
                        },
                    },
                },
            },
        },
        "soft_links": {
            "link_to_group2": {
                "path": "/group1/group2",
            },
            "link_to_link_to_group2": {
                "path": "/link_to_group2",
            },
        },
    }
    assert json_dict["file"] == expected


def test_translate_scalar_int_dataset(tmp_path):
    """Test translation of a scalar int dataset."""

    def set_up_test_file(f: h5py.File):
        f.create_dataset("dataset1", data=2)

    json_dict, chunk_refs = _create_translate(tmp_path, set_up_test_file)

    expected = {
        "datasets": {
            "dataset1": {
                "data": 2,
                "dtype": "int64",
                "shape": [],
                "chunks": [],
                "compressor": None,
                "fill_value": None,
                "filters": [],
                "refs": {
                    "file": "{{c}}",
                    "prefix": "dataset1",
                },
            },
        },
    }
    assert json_dict["file"] == expected

    expected_uri = f"file://{str(tmp_path / 'test.h5')}"
    expected_refs = {
        "dataset1/.zarray": (
            '{"chunks":[],"compressor":null,"dtype":"int64","fill_value":null,"filters":[],'
            '"order":"C","shape":[],"zarr_format":2}'
        ),
        "dataset1/0": [expected_uri, 2048, 8],
    }
    assert chunk_refs == expected_refs

    translated_array = _get_array_from_chunk_refs(chunk_refs, "dataset1")
    expected_array = np.array(2, dtype="int64")
    assert translated_array == expected_array


def test_translate_1d_int_dataset(tmp_path):
    """Test translation of a simple 1D int dataset."""

    def set_up_test_file(f: h5py.File):
        f.create_dataset("dataset1", data=[1, 2, 3])

    json_dict, chunk_refs = _create_translate(tmp_path, set_up_test_file)

    expected = {
        "datasets": {
            "dataset1": {
                "data": [1, 2, 3],
                "dtype": "int64",
                "shape": [3],
                "chunks": [3],
                "compressor": None,
                "fill_value": None,
                "filters": [],
                "refs": {
                    "file": "{{c}}",
                    "prefix": "dataset1",
                },
            },
        },
    }
    assert json_dict["file"] == expected

    expected_uri = f"file://{str(tmp_path / 'test.h5')}"
    expected_refs = {
        "dataset1/.zarray": (
            '{"chunks":[3],"compressor":null,"dtype":"int64","fill_value":null,"filters":[],'
            '"order":"C","shape":[3],"zarr_format":2}'
        ),
        "dataset1/0": [expected_uri, 2048, 24],
    }
    assert chunk_refs == expected_refs

    translated_array = _get_array_from_chunk_refs(chunk_refs, "dataset1")
    expected_array = np.array([1, 2, 3], dtype="int64")
    assert (translated_array == expected_array).all()


def test_translate_1d_int_dataset_no_inline(tmp_path):
    """Test translation of a simple 1D int dataset where the data should be inlined."""

    def set_up_test_file(f: h5py.File):
        f.create_dataset("dataset1", data=[1, 2, 3])

    json_dict, chunk_refs = _create_translate(tmp_path, set_up_test_file, {"dataset_inline_max_bytes": 0})

    expected = {
        "datasets": {
            "dataset1": {
                "data": None,
                "dtype": "int64",
                "shape": [3],
                "chunks": [3],
                "compressor": None,
                "fill_value": None,
                "filters": [],
                "refs": {
                    "file": "{{c}}",
                    "prefix": "dataset1",
                },
            },
        },
    }
    assert json_dict["file"] == expected

    expected_uri = f"file://{str(tmp_path / 'test.h5')}"
    expected_refs = {
        "dataset1/.zarray": (
            '{"chunks":[3],"compressor":null,"dtype":"int64","fill_value":null,"filters":[],'
            '"order":"C","shape":[3],"zarr_format":2}'
        ),
        "dataset1/0": [expected_uri, 2048, 24],
    }
    assert chunk_refs == expected_refs

    translated_array = _get_array_from_chunk_refs(chunk_refs, "dataset1")
    expected_array = np.array([1, 2, 3], dtype="int64")
    assert (translated_array == expected_array).all()


def test_translate_2d_int_dataset_full(tmp_path):
    """Test translation of a chunked, compressed int dataset."""
    data = np.ones((1000, 1000))

    def set_up_test_file(f: h5py.File):
        f.create_dataset("dataset1", data=data, chunks=(900, 900), compression="gzip")

    json_dict, chunk_refs = _create_translate(tmp_path, set_up_test_file)

    expected = {
        "datasets": {
            "dataset1": {
                "data": None,
                "dtype": "float64",
                "shape": [1000, 1000],
                "chunks": [900, 900],
                "compressor": None,
                "fill_value": None,
                "filters": [{"id": "zlib", "level": 4}],
                "refs": {
                    "file": "{{c}}",
                    "prefix": "dataset1",
                },
            },
        },
    }
    assert json_dict["file"] == expected

    expected_uri = f"file://{str(tmp_path / 'test.h5')}"
    expected_refs = {
        "dataset1/.zarray": (
            '{"chunks":[900,900],"compressor":null,"dtype":"float64","fill_value":null,'
            '"filters":[{"id":"zlib","level":4}],"order":"C","shape":[1000,1000],"zarr_format":2}'
        ),
        "dataset1/0.0": [expected_uri, 4016, 9463],
        "dataset1/0.1": [expected_uri, 13479, 10404],
        "dataset1/1.0": [expected_uri, 23883, 6674],
        "dataset1/1.1": [expected_uri, 30557, 6780],
    }
    assert chunk_refs == expected_refs

    translated_array = _get_array_from_chunk_refs(chunk_refs, "dataset1")
    expected_array = data
    assert (translated_array == expected_array).all()


def test_translate_2d_int_dataset_full_inline(tmp_path):
    """Test inlined translation of a chunked, compressed int dataset where the data should be inlined."""
    data = np.ones((10, 10))

    def set_up_test_file(f: h5py.File):
        f.create_dataset("dataset1", data=data, chunks=(9, 9), compression="gzip")

    json_dict, chunk_refs = _create_translate(tmp_path, set_up_test_file, {"dataset_inline_max_bytes": 800})

    # NOTE: the inlined base64 data is gzipped, so it is not human-readable. TODO do we want this?
    expected = {
        "datasets": {
            "dataset1": {
                "data": [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
                "dtype": "float64",
                "shape": [10, 10],
                "chunks": [9, 9],
                "compressor": None,
                "fill_value": None,
                "filters": [{"id": "zlib", "level": 4}],
                "refs": {
                    "file": "{{c}}",
                    "prefix": "dataset1",
                },
            },
        },
    }
    assert json_dict["file"] == expected

    expected_uri = f"file://{str(tmp_path / 'test.h5')}"
    expected_refs = {
        "dataset1/.zarray": (
            '{"chunks":[9,9],"compressor":null,"dtype":"float64","fill_value":null,'
            '"filters":[{"id":"zlib","level":4}],"order":"C","shape":[10,10],"zarr_format":2}'
        ),
        "dataset1/0.0": [expected_uri, 4016, 21],
        "dataset1/0.1": [expected_uri, 4037, 25],
        "dataset1/1.0": [expected_uri, 4062, 22],
        "dataset1/1.1": [expected_uri, 4084, 20],
    }
    assert chunk_refs == expected_refs

    translated_array = _get_array_from_chunk_refs(chunk_refs, "dataset1")
    expected_array = data
    assert (translated_array == expected_array).all()


def test_translate_dataset_deep(tmp_path):
    """Test translation of a simple dataset lower in the file hierarchy."""

    def set_up_test_file(f: h5py.File):
        f.create_group("group1/group2")
        f["group1/group2"].create_dataset("dataset1", data=[1, 2, 3])

    json_dict, chunk_refs = _create_translate(tmp_path, set_up_test_file)

    expected = {
        "groups": {
            "group1": {
                "groups": {
                    "group2": {
                        "datasets": {
                            "dataset1": {
                                "data": [1, 2, 3],
                                "dtype": "int64",
                                "shape": [3],
                                "chunks": [3],
                                "compressor": None,
                                "fill_value": None,
                                "filters": [],
                                "refs": {
                                    "file": "{{c}}",
                                    "prefix": "group1/group2/dataset1",
                                },
                            },
                        },
                    },
                },
            },
        },
    }
    assert json_dict["file"] == expected

    expected_uri = f"file://{str(tmp_path / 'test.h5')}"
    expected_refs = {
        "group1/group2/dataset1/.zarray": (
            '{"chunks":[3],"compressor":null,"dtype":"int64","fill_value":null,'
            '"filters":[],"order":"C","shape":[3],"zarr_format":2}'
        ),
        "group1/group2/dataset1/0": [expected_uri, 3464, 24],
    }
    assert chunk_refs == expected_refs

    translated_array = _get_array_from_chunk_refs(chunk_refs, "group1/group2/dataset1")
    expected_array = np.array([1, 2, 3], dtype="int64")
    assert (translated_array == expected_array).all()


def test_translate_scalar_string_dataset(tmp_path):
    """Test translation of a scalar string dataset."""

    def set_up_test_file(f: h5py.File):
        f.create_dataset("dataset1", data="value1", dtype=h5py.string_dtype("utf-8"))

    json_dict, chunk_refs = _create_translate(tmp_path, set_up_test_file)

    expected = {
        "datasets": {
            "dataset1": {
                "data": "value1",
                "dtype": "object",
                "shape": [],
                "chunks": [],
                "compressor": None,
                "fill_value": None,
                "filters": [],
                "refs": {
                    "file": "{{c}}",
                    "prefix": "dataset1",
                },
            },
        },
    }
    assert json_dict["file"] == expected

    expected_uri = f"file://{str(tmp_path / 'test.h5')}"
    expected_refs = {
        "dataset1/.zarray": (
            '{"chunks":[],"compressor":null,"dtype":"object","fill_value":null,"filters":[],'
            '"order":"C","shape":[],"zarr_format":2}'
        ),
        "dataset1/0": [expected_uri, 2048, 16],
    }
    assert chunk_refs == expected_refs

    # NOTE it is not clear how to parse string/object data from the file URI using the chunk offset and size
    # so these datasets are always embedded in the JSON file.
    # for now, keep the chunk refs file and the reference to it in the JSON file in case it is useful in the future


def test_translate_1d_string_dataset(tmp_path):
    """Test translation of a simple 1D string dataset."""

    def set_up_test_file(f: h5py.File):
        f.create_dataset("dataset1", data=["value1", "value2", "value3"])

    json_dict, chunk_refs = _create_translate(tmp_path, set_up_test_file)

    expected = {
        "datasets": {
            "dataset1": {
                "data": ["value1", "value2", "value3"],
                "dtype": "object",
                "shape": [3],
                "chunks": [3],
                "compressor": None,
                "fill_value": None,
                "filters": [],
                "refs": {
                    "file": "{{c}}",
                    "prefix": "dataset1",
                },
            },
        },
    }
    assert json_dict["file"] == expected

    expected_uri = f"file://{str(tmp_path / 'test.h5')}"
    expected_refs = {
        "dataset1/.zarray": (
            '{"chunks":[3],"compressor":null,"dtype":"object","fill_value":null,"filters":[],'
            '"order":"C","shape":[3],"zarr_format":2}'
        ),
        "dataset1/0": [expected_uri, 2048, 48],
    }
    assert chunk_refs == expected_refs


def test_translate_2d_string_dataset(tmp_path):
    """Test translation of a simple 2D string dataset."""

    def set_up_test_file(f: h5py.File):
        f.create_dataset("dataset1", data=[["value1", "value2", "value3"], ["value4", "value5", "value6"]])

    json_dict, chunk_refs = _create_translate(tmp_path, set_up_test_file)

    expected = {
        "datasets": {
            "dataset1": {
                "data": [["value1", "value2", "value3"], ["value4", "value5", "value6"]],
                "dtype": "object",
                "shape": [2, 3],
                "chunks": [2, 3],
                "compressor": None,
                "fill_value": None,
                "filters": [],
                "refs": {
                    "file": "{{c}}",
                    "prefix": "dataset1",
                },
            },
        },
    }
    assert json_dict["file"] == expected

    expected_uri = f"file://{str(tmp_path / 'test.h5')}"
    expected_refs = {
        "dataset1/.zarray": (
            '{"chunks":[2,3],"compressor":null,"dtype":"object","fill_value":null,"filters":[],'
            '"order":"C","shape":[2,3],"zarr_format":2}'
        ),
        "dataset1/0.0": [expected_uri, 2048, 96],
    }
    assert chunk_refs == expected_refs


def test_translate_2d_string_dataset_full(tmp_path):
    """Test translation of a chunked, compressed 2D string dataset."""

    def set_up_test_file(f: h5py.File):
        f.create_dataset(
            "dataset1",
            data=[["value1", "value2", "value3"], ["value4", "value5", "value6"]],
            chunks=(2, 2),
            compression="gzip",
        )

    json_dict, chunk_refs = _create_translate(tmp_path, set_up_test_file)

    expected = {
        "datasets": {
            "dataset1": {
                "data": [["value1", "value2", "value3"], ["value4", "value5", "value6"]],
                "dtype": "object",
                "shape": [2, 3],
                "chunks": [2, 2],
                "compressor": None,
                "fill_value": None,
                "filters": [{"id": "zlib", "level": 4}],
                "refs": {
                    "file": "{{c}}",
                    "prefix": "dataset1",
                },
            },
        },
    }
    assert json_dict["file"] == expected

    expected_uri = f"file://{str(tmp_path / 'test.h5')}"
    expected_refs = {
        "dataset1/.zarray": (
            '{"chunks":[2,2],"compressor":null,"dtype":"object","fill_value":null,'
            '"filters":[{"id":"zlib","level":4}],"order":"C","shape":[2,3],"zarr_format":2}'
        ),
        "dataset1/0.0": [expected_uri, 8112, 30],
        "dataset1/0.1": [expected_uri, 8142, 29],
    }
    assert chunk_refs == expected_refs


def test_translate_scalar_dataset_object_refs(tmp_path):
    """Test translation of a 1D dataset containing object references."""

    def set_up_test_file(f: h5py.File):
        f.create_dataset("dataset1", shape=tuple(), dtype=h5py.ref_dtype)
        f["dataset1"][()] = f.ref

    json_dict, chunk_refs = _create_translate(tmp_path, set_up_test_file)

    expected = {
        "datasets": {
            "dataset1": {
                "data": {
                    # TODO should this be a dictionary or a string representation of the JSON like for datasets?
                    "dtype": "object_reference",
                    "path": "/",
                },
                "dtype": "object",
                "shape": [],
                "chunks": [],
                "compressor": None,
                "fill_value": None,
                "filters": [],
                "refs": {
                    "file": "{{c}}",
                    "prefix": "dataset1",
                },
            },
        },
    }
    assert json_dict["file"] == expected

    expected_uri = f"file://{str(tmp_path / 'test.h5')}"
    expected_refs = {
        "dataset1/.zarray": (
            '{"chunks":[],"compressor":null,"dtype":"object","fill_value":null,"filters":[],'
            '"order":"C","shape":[],"zarr_format":2}'
        ),
        "dataset1/0": [expected_uri, 2048, 8],
    }
    assert chunk_refs == expected_refs


def test_translate_1d_dataset_object_refs(tmp_path):
    """Test translation of a 1D dataset containing object references."""

    def set_up_test_file(f: h5py.File):
        f.create_dataset("dataset1", shape=(2,), dtype=h5py.ref_dtype)
        f["dataset1"][:] = f.ref

    json_dict, chunk_refs = _create_translate(tmp_path, set_up_test_file)

    expected = {
        "datasets": {
            "dataset1": {
                "data": [
                    {"dtype": "object_reference", "path": "/"},
                    {"dtype": "object_reference", "path": "/"},
                ],
                "dtype": "object",
                "shape": [2],
                "chunks": [2],
                "compressor": None,
                "fill_value": None,
                "filters": [],
                "refs": {
                    "file": "{{c}}",
                    "prefix": "dataset1",
                },
            },
        },
    }
    assert json_dict["file"] == expected

    expected_uri = f"file://{str(tmp_path / 'test.h5')}"
    expected_refs = {
        "dataset1/.zarray": (
            '{"chunks":[2],"compressor":null,"dtype":"object","fill_value":null,"filters":[],'
            '"order":"C","shape":[2],"zarr_format":2}'
        ),
        "dataset1/0": [expected_uri, 2048, 16],
    }
    assert chunk_refs == expected_refs


def test_translate_2d_dataset_object_refs(tmp_path):
    """Test translation of a 2D dataset containing object references."""

    def set_up_test_file(f: h5py.File):
        f.create_dataset("dataset1", shape=(3, 2), dtype=h5py.ref_dtype)
        f["dataset1"][:] = f.ref

    json_dict, chunk_refs = _create_translate(tmp_path, set_up_test_file)

    expected = {
        "datasets": {
            "dataset1": {
                "data": [
                    [{"dtype": "object_reference", "path": "/"}, {"dtype": "object_reference", "path": "/"}],
                    [{"dtype": "object_reference", "path": "/"}, {"dtype": "object_reference", "path": "/"}],
                    [{"dtype": "object_reference", "path": "/"}, {"dtype": "object_reference", "path": "/"}],
                ],
                "dtype": "object",
                "shape": [3, 2],
                "chunks": [3, 2],
                "compressor": None,
                "fill_value": None,
                "filters": [],
                "refs": {
                    "file": "{{c}}",
                    "prefix": "dataset1",
                },
            },
        },
    }
    assert json_dict["file"] == expected

    expected_uri = f"file://{str(tmp_path / 'test.h5')}"
    expected_refs = {
        "dataset1/.zarray": (
            '{"chunks":[3,2],"compressor":null,"dtype":"object","fill_value":null,"filters":[],'
            '"order":"C","shape":[3,2],"zarr_format":2}'
        ),
        "dataset1/0.0": [expected_uri, 2048, 48],
    }
    assert chunk_refs == expected_refs


def test_translate_scalar_compound_dataset(tmp_path):
    """Test translation of a scalar compound dataset."""

    def set_up_test_file(f: h5py.File):
        fixed_length_utf8_type = h5py.string_dtype("utf-8", 30)
        fixed_length_ascii_type = h5py.string_dtype("ascii", 30)
        cpd_type_int_vlen_flen_strings = np.dtype([
            ("int", int),
            ("vlen_utf8", h5py.string_dtype("utf-8")),
            ("vlen_ascii", h5py.string_dtype("ascii")),
            ("flen_utf8", fixed_length_utf8_type),
            ("flen_ascii", fixed_length_ascii_type),
        ])
        f.create_dataset(
            "dataset1", data=(2, "±", b"v", "±".encode("utf-8"), b"v"), dtype=cpd_type_int_vlen_flen_strings
        )

    json_dict, chunk_refs = _create_translate(tmp_path, set_up_test_file)

    expected = {
        "datasets": {
            "dataset1": {
                "data": [2, "±", "v", "±", "v"],
                "dtype": (
                    "[('int', '<i8'), ('vlen_utf8', 'O'), ('vlen_ascii', 'O'), ('flen_utf8', 'S30'), ('flen_ascii',"
                    " 'S30')]"
                ),
                "shape": [],
                "chunks": [],
                "compressor": None,
                "fill_value": None,
                "filters": [],
                "refs": {
                    "file": "{{c}}",
                    "prefix": "dataset1",
                },
            },
        },
    }
    assert json_dict["file"] == expected

    expected_uri = f"file://{str(tmp_path / 'test.h5')}"
    expected_refs = {
        "dataset1/.zarray": (
            "{\"chunks\":[],\"compressor\":null,\"dtype\":\"[('int', '<i8'), ('vlen_utf8', 'O'), ('vlen_ascii',"
            " 'O'), ('flen_utf8', 'S30'), ('flen_ascii',"
            ' \'S30\')]","fill_value":null,"filters":[],"order":"C","shape":[],"zarr_format":2}'
        ),
        "dataset1/0": [expected_uri, 2048, 100],
    }
    assert chunk_refs == expected_refs


def test_translate_1d_compound_dataset(tmp_path):
    """Test translation of a 1D compound dataset."""

    def set_up_test_file(f: h5py.File):
        fixed_length_utf8_type = h5py.string_dtype("utf-8", 30)
        fixed_length_ascii_type = h5py.string_dtype("ascii", 30)
        cpd_type_int_vlen_flen_strings = np.dtype([
            ("int", int),
            ("vlen_utf8", h5py.string_dtype("utf-8")),
            ("vlen_ascii", h5py.string_dtype("ascii")),
            ("flen_utf8", fixed_length_utf8_type),
            ("flen_ascii", fixed_length_ascii_type),
        ])
        f.create_dataset(
            "dataset1",
            data=[(2, "±", b"v", "±".encode("utf-8"), b"v"), (2, "💜", b"v", "💜".encode("utf-8"), b"v")],
            dtype=cpd_type_int_vlen_flen_strings,
        )

    json_dict, chunk_refs = _create_translate(tmp_path, set_up_test_file)

    expected = {
        "datasets": {
            "dataset1": {
                "data": [[2, "±", "v", "±", "v"], [2, "💜", "v", "💜", "v"]],
                "dtype": (
                    "[('int', '<i8'), ('vlen_utf8', 'O'), ('vlen_ascii', 'O'), ('flen_utf8', 'S30'), ('flen_ascii',"
                    " 'S30')]"
                ),
                "shape": [2],
                "chunks": [2],
                "compressor": None,
                "fill_value": None,
                "filters": [],
                "refs": {
                    "file": "{{c}}",
                    "prefix": "dataset1",
                },
            },
        },
    }
    assert json_dict["file"] == expected

    expected_uri = f"file://{str(tmp_path / 'test.h5')}"
    expected_refs = {
        "dataset1/.zarray": (
            "{\"chunks\":[2],\"compressor\":null,\"dtype\":\"[('int', '<i8'), ('vlen_utf8', 'O'), ('vlen_ascii',"
            " 'O'), ('flen_utf8', 'S30'), ('flen_ascii',"
            ' \'S30\')]","fill_value":null,"filters":[],"order":"C","shape":[2],"zarr_format":2}'
        ),
        "dataset1/0": [expected_uri, 2048, 200],
    }
    assert chunk_refs == expected_refs


def test_translate_scalar_compound_dataset_object_refs(tmp_path):
    """Test translation of a scalar compound dataset with object references."""

    def set_up_test_file(f: h5py.File):
        f.create_group("group1")
        fixed_length_utf8_type = h5py.string_dtype("utf-8", 30)
        fixed_length_ascii_type = h5py.string_dtype("ascii", 30)
        cpd_type_int_vlen_flen_strings = np.dtype([
            ("int", int),
            ("vlen_utf8", h5py.string_dtype("utf-8")),
            ("vlen_ascii", h5py.string_dtype("ascii")),
            ("flen_utf8", fixed_length_utf8_type),
            ("flen_ascii", fixed_length_ascii_type),
            ("ref1", h5py.ref_dtype),
        ])
        f.create_dataset(
            "dataset1",
            data=(2, "±", b"v", "±".encode("utf-8"), b"v", f["group1"].ref),
            dtype=cpd_type_int_vlen_flen_strings,
        )

    json_dict, chunk_refs = _create_translate(tmp_path, set_up_test_file)

    expected = {
        "datasets": {
            "dataset1": {
                "data": [2, "±", "v", "±", "v", {"dtype": "object_reference", "path": "/groups/group1"}],
                "dtype": (
                    "[('int', '<i8'), ('vlen_utf8', 'O'), ('vlen_ascii', 'O'), ('flen_utf8', 'S30'), ('flen_ascii',"
                    " 'S30'), ('ref1', 'O')]"
                ),
                "shape": [],
                "chunks": [],
                "compressor": None,
                "fill_value": None,
                "filters": [],
                "refs": {
                    "file": "{{c}}",
                    "prefix": "dataset1",
                },
            },
        },
        "groups": {
            "group1": {},
        },
    }
    assert json_dict["file"] == expected

    expected_uri = f"file://{str(tmp_path / 'test.h5')}"
    expected_refs = {
        "dataset1/.zarray": (
            "{\"chunks\":[],\"compressor\":null,\"dtype\":\"[('int', '<i8'), ('vlen_utf8', 'O'), ('vlen_ascii',"
            " 'O'), ('flen_utf8', 'S30'), ('flen_ascii', 'S30'), ('ref1',"
            ' \'O\')]","fill_value":null,"filters":[],"order":"C","shape":[],"zarr_format":2}'
        ),
        "dataset1/0": [expected_uri, 2280, 108],
    }
    assert chunk_refs == expected_refs


def test_translate_1d_compound_dataset_object_refs(tmp_path):
    """Test translation of a 1D compound dataset with object references."""

    def set_up_test_file(f: h5py.File):
        f.create_group("group1")
        fixed_length_utf8_type = h5py.string_dtype("utf-8", 30)
        fixed_length_ascii_type = h5py.string_dtype("ascii", 30)
        cpd_type_int_vlen_flen_strings = np.dtype([
            ("int", int),
            ("vlen_utf8", h5py.string_dtype("utf-8")),
            ("vlen_ascii", h5py.string_dtype("ascii")),
            ("flen_utf8", fixed_length_utf8_type),
            ("flen_ascii", fixed_length_ascii_type),
            ("ref1", h5py.ref_dtype),
        ])
        f.create_dataset(
            "dataset1",
            data=[
                (2, "±", b"v", "±".encode("utf-8"), b"v", f["group1"].ref),
                (2, "💜", b"v", "💜".encode("utf-8"), b"v", f.ref),
            ],
            dtype=cpd_type_int_vlen_flen_strings,
        )

    json_dict, chunk_refs = _create_translate(tmp_path, set_up_test_file)

    expected = {
        "datasets": {
            "dataset1": {
                "data": [
                    [2, "±", "v", "±", "v", {"dtype": "object_reference", "path": "/groups/group1"}],
                    [2, "💜", "v", "💜", "v", {"dtype": "object_reference", "path": "/"}],
                ],
                "dtype": (
                    "[('int', '<i8'), ('vlen_utf8', 'O'), ('vlen_ascii', 'O'), ('flen_utf8', 'S30'), ('flen_ascii',"
                    " 'S30'), ('ref1', 'O')]"
                ),
                "shape": [2],
                "chunks": [2],
                "compressor": None,
                "fill_value": None,
                "filters": [],
                "refs": {
                    "file": "{{c}}",
                    "prefix": "dataset1",
                },
            },
        },
        "groups": {
            "group1": {},
        },
    }
    assert json_dict["file"] == expected

    expected_uri = f"file://{str(tmp_path / 'test.h5')}"
    expected_refs = {
        "dataset1/.zarray": (
            "{\"chunks\":[2],\"compressor\":null,\"dtype\":\"[('int', '<i8'), ('vlen_utf8', 'O'), ('vlen_ascii',"
            " 'O'), ('flen_utf8', 'S30'), ('flen_ascii', 'S30'), ('ref1',"
            ' \'O\')]","fill_value":null,"filters":[],"order":"C","shape":[2],"zarr_format":2}'
        ),
        "dataset1/0": [expected_uri, 2296, 216],
    }
    assert chunk_refs == expected_refs


def test_translate_attrs(tmp_path):
    """Test translation of attrs with different attribute dtypes and encodings."""

    def set_up_test_file(f: h5py.File):
        fixed_length_utf8_type = h5py.string_dtype("utf-8", 30)
        fixed_length_ascii_type = h5py.string_dtype("ascii", 30)

        f.attrs["vlen_utf8"] = "2.0±0.1"  # written as variable-length with utf8 encoding, read as str
        f.attrs["vlen_ascii"] = b"value1"  # written as variable-length with ascii encoding, read as str
        f.attrs["int"] = 42  # written as 64-bit int
        f.attrs["float"] = 42.0  # written as 64-bit float
        f.attrs["bool"] = True  # written as 8-bit enum, read as bool
        f.attrs["empty_int"] = h5py.Empty(dtype=int)  # written as 64-bit int
        # written as fixed-length with utf8 encoding, read as numpy bytes arrays
        f.attrs["flen_utf8"] = np.array("2.0±0.1".encode("utf-8"), dtype=fixed_length_utf8_type)  # dtype: |S30
        # written as fixed-length with ascii encoding, read as numpy bytes arrays
        f.attrs["flen_ascii"] = np.array("value1", dtype=fixed_length_ascii_type)  # dtype: |S30
        # written as fixed-length with latin-1 encoding, raises UnicodeDecodeError - not supported
        # f.attrs["flen_latin"] = np.array(s.encode("latin-1"), dtype=ascii_type)  # dtype: |S30

        # translation of a scalar or array with compound dtype with non-scalar fields is not supported
        # cpd_type_ints = np.dtype([("val1", ('<i4', 2), )])
        # f.attrs.create("cpd_type_ints", data=([3, 3], ), dtype=cpd_type_ints)
        # arr_cpd_type_ints = np.array([([23, 3], ), ([120, 100], )], dtype=cpd_type_ints)
        # f.attrs["array[cpd_type_ints]"] = arr_cpd_type_ints
        # TODO test separately

        # writing an hdf5 attribute with compound dtype (int, vlen_utf8) or (int, vlen_ascii) is not supported by h5py
        # raises TypeError: Can't implicitly convert non-string objects to strings
        # cpd_type_int_vlen_utf8 = np.dtype([("number", int), ('astring', h5py.string_dtype("utf-8"))])
        # arr_cpd_type_int_vlen_utf8 = np.array([[3, "3"], [10, "10"]], dtype=cpd_type_int_vlen_utf8)
        # f.attrs["cpd_type_int_vlen_utf8"] = arr_cpd_type_int_vlen_utf8
        # cpd_type_int_vlen_ascii = np.dtype([("number", int), ('astring', h5py.string_dtype("ascii"))])
        # arr_cpd_type_int_vlen_ascii = np.array([[3, "3"], [10, "10"]], dtype=cpd_type_int_vlen_ascii)
        # f.attrs["cpd_type_int_vlen_ascii"] = arr_cpd_type_int_vlen_ascii

        # scalar compound dtype (int, fixed length utf8, fixed length ascii)
        cpd_type_int_flen_strings = np.dtype([
            ("int", int), ("flen_utf8", fixed_length_utf8_type), ("flen_ascii", fixed_length_ascii_type)
        ])
        f.attrs.create("cpd_type_int_flen_strings", data=(3, "3", b"3"), dtype=cpd_type_int_flen_strings)

        f.attrs["list[int]"] = [42, 314]
        f.attrs["list[float]"] = [42.0, 3.14]
        f.attrs["list[bool]"] = [True, False]
        f.attrs["list[vlen_utf8]"] = ["2.0±0.1", "3.0±0.2"]
        f.attrs["list[vlen_ascii]"] = [b"value1", b"value2"]
        f.attrs["list[flen_utf8]"] = np.array(
            ["2.0±0.1".encode("utf-8"), "3.0±0.2".encode("utf-8")], dtype=fixed_length_utf8_type
        )
        f.attrs["list[flen_ascii]"] = np.array(["value1", "value2"], dtype=fixed_length_ascii_type)

        # 1d array of compound dtype (int, fixed length utf8, fixed length ascii)
        cpd_type_int_flen_strings = np.dtype([
            ("int", int), ("flen_utf8", fixed_length_utf8_type), ("flen_ascii", fixed_length_ascii_type)
        ])
        f.attrs["1d_cpd_type_int_flen_strings"] = np.array(
            [(3, "3", "3"), (10, "10", "10")], dtype=cpd_type_int_flen_strings
        )

        f.attrs["2d_int"] = np.array([[1, 2, 3], [4, 5, 6]])
        f.attrs["2d_vlen_utf8"] = np.array(
            [["2.0±0.1", "3.0±0.2"], ["2.0±0.111", "3.0±0.211"]], dtype=h5py.string_dtype("utf-8")
        )
        f.attrs["2d_vlen_ascii"] = np.array(
            [["value1", "value2"], ["value3", "value4"]], dtype=h5py.string_dtype("ascii")
        )
        f.attrs["2d_flen_utf8"] = np.array(
            [
                ["2.0±0.1".encode("utf-8"), "3.0±0.2".encode("utf-8")],
                ["2.0±0.111".encode("utf-8"), "3.0±0.211".encode("utf-8")],
            ],
            dtype=fixed_length_utf8_type,
        )
        f.attrs["2d_flen_ascii"] = np.array([["value1", "value2"], ["value3", "value4"]], dtype=fixed_length_ascii_type)

        # translation of an array with compound dtype and ndim > 1 is not supported
        # f.attrs["2d_cpd_type_int_flen_strings"] = np.array(
        #     [[(3, "3", "3"), (10, "10", "10")], [(4, "4", "4"), (1, "1", "1")]], dtype=cpd_type_int_flen_strings
        # )

        f.create_group("group1")
        f["group1"].create_dataset("dataset1", data=[1, 2, 3])
        f.attrs["group_ref"] = f["group1"].ref
        f.attrs["dataset_ref"] = f["group1/dataset1"].ref
        f.attrs["root_ref"] = f.ref
        f.attrs["1d_root_ref"] = np.array([f.ref, f.ref], dtype=h5py.ref_dtype)
        f.attrs["2d_root_ref"] = np.array([[f.ref, f.ref], [f.ref, f.ref]], dtype=h5py.ref_dtype)

        cpd_type_int_flen_strings_ref = np.dtype([
            ("int", int),
            ("flen_utf8", fixed_length_utf8_type),
            ("flen_ascii", fixed_length_ascii_type),
            ("ref1", h5py.ref_dtype),
        ])
        f.attrs.create("cpd_type_int_flen_strings_ref", data=(3, "3", b"3", f.ref), dtype=cpd_type_int_flen_strings_ref)

        f.attrs.create(
            "1d_cpd_type_int_flen_strings_ref", data=[(3, "3", b"3", f.ref)], dtype=cpd_type_int_flen_strings_ref
        )

        # translation of an array with compound dtype and ndim > 1 is not supported
        # f.attrs.create(
        #     "2d_cpd_type_int_flen_strings_ref", data=[[(3, "3", b"3", f.ref)]], dtype=cpd_type_int_flen_strings_ref
        # )

    json_dict, _ = _create_translate(tmp_path, set_up_test_file)

    attrs = json_dict["file"]["attributes"]
    assert attrs["vlen_utf8"] == "2.0±0.1"
    assert attrs["vlen_ascii"] == "value1"
    assert attrs["int"] == 42
    assert attrs["float"] == 42.0
    assert attrs["bool"]
    assert attrs["empty_int"] == ""
    assert attrs["flen_utf8"] == "2.0±0.1"
    assert attrs["flen_ascii"] == "value1"
    assert attrs["list[int]"] == [42, 314]
    assert attrs["list[float]"] == [42.0, 3.14]
    assert attrs["list[bool]"] == [True, False]
    assert attrs["list[vlen_utf8]"] == ["2.0±0.1", "3.0±0.2"]
    assert attrs["list[vlen_ascii]"] == ["value1", "value2"]
    assert attrs["list[flen_utf8]"] == ["2.0±0.1", "3.0±0.2"]
    assert attrs["list[flen_ascii]"] == ["value1", "value2"]
    assert attrs["1d_cpd_type_int_flen_strings"] == [[3, "3", "3"], [10, "10", "10"]]
    assert attrs["2d_int"] == [[1, 2, 3], [4, 5, 6]]
    assert attrs["2d_vlen_utf8"] == [["2.0±0.1", "3.0±0.2"], ["2.0±0.111", "3.0±0.211"]]
    assert attrs["2d_vlen_ascii"] == [["value1", "value2"], ["value3", "value4"]]
    assert attrs["2d_flen_utf8"] == [["2.0±0.1", "3.0±0.2"], ["2.0±0.111", "3.0±0.211"]]
    assert attrs["2d_flen_ascii"] == [["value1", "value2"], ["value3", "value4"]]
    assert attrs["group_ref"] == {
        # TODO should this be a dictionary or a string representation of the JSON like for datasets?
        "dtype": "object_reference",
        "path": "/groups/group1",
    }
    assert attrs["dataset_ref"] == {
        "dtype": "object_reference",
        "path": "/groups/group1/datasets/dataset1",
    }
    assert attrs["root_ref"] == {
        "dtype": "object_reference",
        "path": "/",
    }
    assert attrs["1d_root_ref"] == [
        {
            "dtype": "object_reference",
            "path": "/",
        },
        {
            "dtype": "object_reference",
            "path": "/",
        },
    ]
    assert attrs["2d_root_ref"] == [
        [
            {
                "dtype": "object_reference",
                "path": "/",
            },
            {
                "dtype": "object_reference",
                "path": "/",
            },
        ],
        [
            {
                "dtype": "object_reference",
                "path": "/",
            },
            {
                "dtype": "object_reference",
                "path": "/",
            },
        ],
    ]
    assert attrs["cpd_type_int_flen_strings_ref"] == [
        3,
        "3",
        "3",
        {
            "dtype": "object_reference",
            "path": "/",
        },
    ]
    assert attrs["1d_cpd_type_int_flen_strings_ref"] == [
        [
            3,
            "3",
            "3",
            {
                "dtype": "object_reference",
                "path": "/",
            },
        ]
    ]


def test_translate_nested_group_attrs(tmp_path):
    """Test translation of nested groups with attributes."""

    def set_up_test_file(f: h5py.File):
        f.create_group("group1/group2")
        f.create_group("group1/group3")
        f["group1"].attrs["int"] = 42
        f["group1/group2"].attrs["list[vlen_utf8]"] = ["value1", "value2"]

    json_dict, _ = _create_translate(tmp_path, set_up_test_file)

    expected = {
        "groups": {
            "group1": {
                "groups": {
                    "group2": {
                        "attributes": {
                            "list[vlen_utf8]": ["value1", "value2"],
                        },
                    },
                    "group3": {},
                },
                "attributes": {
                    "int": 42,
                },
            },
        },
    }
    assert json_dict["file"] == expected


def test_translate_dataset_attrs(tmp_path):
    """Test translation of a simple dataset with attributes."""

    def set_up_test_file(f: h5py.File):
        f.create_dataset("dataset1", data=[1, 2, 3])
        f["dataset1"].attrs["int"] = 42
        f["dataset1"].attrs["list[vlen_utf8]"] = ["value1", "value2"]

    json_dict, _ = _create_translate(tmp_path, set_up_test_file)

    expected = {
        "datasets": {
            "dataset1": {
                "data": [1, 2, 3],
                "dtype": "int64",
                "shape": [3],
                "chunks": [3],
                "compressor": None,
                "fill_value": None,
                "filters": [],
                "refs": {
                    "file": "{{c}}",
                    "prefix": "dataset1",
                },
                "attributes": {
                    "int": 42,
                    "list[vlen_utf8]": ["value1", "value2"],
                },
            },
        },
    }
    assert json_dict["file"] == expected
