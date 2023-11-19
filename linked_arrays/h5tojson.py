"""Convert an HDF5 file to JSON with Kerchunk-style references to dataset chunks.

Not supported:
- References to external files (HDF5 1.12+) -- these are not currently detected and may not even be supported by h5py
- Region references and references to named dtypes
- Attributes and datasets with compound data types and more than 1 dimension (scalar and 1-dim are supported)
- Attributes and datasets with compound data types with non-scalar fields
- External datasets (I think)


Datasets with object dtype (strings, references) or compound dtype are read into memory and stored in the JSON file.
WARNING: The translation of a dataset of strings, references, or compound dtypes to JSON will use a lot of memory
and result in a bloated JSON file.

Note that Zarr stores object arrays using an object codec:
https://zarr.readthedocs.io/en/stable/tutorial.html#object-arrays

Note that in HDF5, an edge chunk takes up the same amount of space as an interior chunk even if the
edge chunk is not full, if the chunks are not compressed.

######
# TODO change how attributes are stored so that the dtype and shape are revealed in the case of compound dtypes?
# TODO should an HDF5 dataset at /group1/group1.1/dataset1 be stored in the JSON file as
#      /groups/group1/groups/group1.1/datasets/dataset1 or /groups/group1.1/dataset1? how would attributes be stored?
######

All attributes will be serialized into JSON.

Limitations:
- Attributes with compound data types will lose their dtype and shape information, which means they cannot be
roundtripped. The dtype and shape information are in the HDF5 file and the numpy array in memory, but not
written to the JSON file.

In addition to the list of not supported data types above, HDMF does not support:
- Attributes with compound data types that contain references
- Scalar datasets with compound data types that contain references
- Empty datasets and attributes (I think)
- Compact datasets (raw data is stored in the object header of the dataset) (I think)

This does not use HDMF HDF5IO methods which are very similar, because attributes that are references
are de-referenced immediately.
Scalar datasets with object reference dtype are read as DatasetBuilders with data=ReferenceBuilder (OK).
Non-scalar datasets with object reference dtype are read as BuilderH5ReferenceDataset.
Non-scalar datasets with compound dtype, regardless of whether it contains strings or references,
are read as BuilderH5TableDataset.
Most likely, scalar datasets with compound dtype are OK to read by HDMF, but
scalar datasets with compound dtype that contains references are not supported.

Decoding datasets of variable-length strings from the HDF5 file was challenging and remains to be done.
Attempts were made to use the custom VLenHDF5String codec from the HDF5Zarr package
https://github.com/catalystneuro/HDF5Zarr/blob/ac96afda37e079086f264dcc0d0230b7cf9d3397/hdf5zarr/hdf5zarr.py#L69
but it did not work because the `buf` argument was not the right shape and it is not clear what is going on.

NOTE: NaN, Inf, and -Inf are not valid JSON. They are written out anyway and can be read back in.
One solution is to replace all NaN with "NaN" before writing to JSON.
Encoding all float dataset as base64-encoded strings may also solve this problem.
Floating point attributes can still be NaN though...

Adapted from kerchunk/hdf.py version 0.2.2. The following is the license for the adapted kerchunk code.

NOTE: as of 2023-11-18:
If you attempt to add or update a file on GitHub that is larger than 50 MiB, you will receive a warning from Git.
The changes will still successfully push to your repository, but you can consider removing the commit to minimize
performance impact. GitHub blocks files larger than 100 MiB.

MIT License

Copyright (c) 2020 Intake

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from collections import defaultdict
import warnings
import fsspec
import h5py
import json
import logging
import numcodecs
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from typing import Union
import ujson

logger = logging.getLogger("h5tojson")


class FloatJSONEncoder(json.JSONEncoder):
    """JSON encoder that converts NaN, Inf, and -Inf to strings."""

    def encode(self, obj, *args, **kwargs):
        """Convert NaN, Inf, and -Inf to strings."""
        obj = FloatJSONEncoder._convert_nan(obj)
        return super().encode(obj, *args, **kwargs)

    def iterencode(self, obj, *args, **kwargs):
        """Convert NaN, Inf, and -Inf to strings."""
        obj = FloatJSONEncoder._convert_nan(obj)
        return super().iterencode(obj, *args, **kwargs)

    @staticmethod
    def _convert_nan(obj):
        """Convert NaN, Inf, and -Inf from a JSON object to strings."""
        if isinstance(obj, dict):
            return {k: FloatJSONEncoder._convert_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [FloatJSONEncoder._convert_nan(v) for v in obj]
        elif isinstance(obj, float):
            return FloatJSONEncoder._nan_to_string(obj)
        return obj

    @staticmethod
    def _nan_to_string(obj: float):
        """Convert NaN, Inf, and -Inf from a float to a string."""
        if np.isnan(obj):
            return "NaN"
        elif np.isinf(obj):
            if obj > 0:
                return "Infinity"
            else:
                return "-Infinity"
        else:
            return float(obj)


class H5ToJson:
    """Class to manage translation of an HDF5 file to JSON with Kerchunk-style references to dataset chunks."""

    def __init__(
        self,
        hdf5_file_path: Union[str, Path],  # or BinaryIO
        json_file_path: Union[str, Path],
        chunk_refs_file_path: Union[str, Path],
        dataset_inline_threshold=500,
        object_dataset_inline_threshold=200000,
        storage_options=None,
    ):
        """Create an object to translate an HDF5 file to JSON with Kerchunk-style references to dataset chunks.

        Parameters
        ----------
        hdf5_file_path : Union[str, Path]
            Path to the HDF5 file to convert.
        json_file_path : Union[str, Path]
            Path to the JSON file to write.
        chunk_refs_file_path : Union[str, Path]
            Path to the JSON file to write chunk references to. If None, then chunk references will not be written.
        dataset_inline_threshold : int, optional
            Maximum number of bytes per dataset that does not have an object dtype to store inline in the JSON file.
            Default is 500.
        object_dataset_inline_threshold : int, optional
            Maximum number of bytes per dataset that has an object dtype to store inline in the JSON file.
            Default is 200000.
        storage_options : dict, optional
            Options to pass to fsspec when opening the HDF5 file. Default is None.
        """
        self.hdf5_file_path = str(hdf5_file_path)
        fs, path = fsspec.core.url_to_fs(self.hdf5_file_path, **(storage_options or {}))
        self._input_file = fs.open(path, "rb")
        self._h5f = h5py.File(self._input_file, mode="r")
        self.dataset_inline_threshold = dataset_inline_threshold
        self.object_dataset_inline_threshold = object_dataset_inline_threshold
        self.json_file_path = str(json_file_path)
        self.chunk_refs_file_path = str(chunk_refs_file_path) if chunk_refs_file_path else None
        self._uri = "file://" + self.hdf5_file_path

    def close(self):
        """Close the HDF5 file."""
        self._h5f.close()

    def translate(self):
        """Translate the HDF5 file to JSON with Kerchunk-style references to dataset chunks."""
        self.json_dict = {
            "version": 1,
            "refs": {},  # the only key will be the root group "/"
        }
        if self.chunk_refs_file_path:
            self.json_dict["templates"] = {"c": "file://" + self.chunk_refs_file_path}  # use kerchunk-style template
        self.chunk_refs = {}

        # translate the root group and all its contents
        self.translate_group(self._h5f, self.json_dict["refs"])

        # write the dictionary to a human-readable JSON file
        # NOTE: spaces take up a lot of space in the JSON file...
        with open(self.json_file_path, "w") as f:
            # ujson.dump(self.json_dict, f, indent=2, escape_forward_slashes=False)
            # NOTE: ujson does not allow for custom encoders, so use the standard library json module
            json.dump(self.json_dict, f, indent=2, allow_nan=False, cls=FloatJSONEncoder)

        # write the chunk refs dictionary to a human-readable JSON file
        # NOTE: spaces take up a lot of space in the JSON file...
        if self.chunk_refs_file_path:
            with open(self.chunk_refs_file_path, "w") as f:
                ujson.dump(self.chunk_refs, f, indent=2, escape_forward_slashes=False)

    def translate_group(self, h5obj: h5py.Group, group_dict: dict):
        """Translate a group and all its contents in the HDF5 file to a dictionary.

        Parameters
        ----------
        h5obj : h5py.Group
            An HDF5 group.
        group_dict : dict
            A dictionary representing the groups contained within the parent group.
        """
        logger.debug(f"HDF5 group: {h5obj.name}")
        if h5obj.name == "/":  # root group
            name = "/"
        else:
            name = os.path.basename(h5obj.name)
        logger.debug(f"Group name: {name}")

        # create a new key for this group in the dictionary of groups contained in the parent group.
        # use defaultdict so that we can add to the "soft_links", "external_links", etc. keys more easily.
        group_dict[name] = defaultdict(dict)

        # add soft links, external links, subgroups, datasets, and attributes to this group dict
        for sub_obj_name, sub_obj in h5obj.items():
            link_type = h5obj.get(sub_obj_name, getlink=True)

            if isinstance(link_type, h5py.SoftLink):
                logger.debug(f"Adding HDF5 soft link: {sub_obj_name}")
                group_dict[name]["soft_links"][sub_obj_name] = {"path": link_type.path}

            elif isinstance(link_type, h5py.ExternalLink):
                logger.debug(f"Adding HDF5 external link: {sub_obj_name}")
                group_dict[name]["external_links"][sub_obj_name] = {
                    "path": link_type.path,
                    "filename": link_type.filename,
                }

            else:  # link_type must be an h5py.HardLink
                if isinstance(sub_obj, h5py.Group):
                    logger.debug(f"Adding HDF5 subgroup: {sub_obj.name}")
                    self.translate_group(sub_obj, group_dict[name]["groups"])

                else:  # sub_obj must be an h5py.Dataset
                    logger.debug(f"Adding HDF5 dataset: {sub_obj.name}")
                    self.translate_dataset(sub_obj, group_dict[name]["datasets"])

        attrs = self.translate_attrs(h5obj)
        if attrs:
            group_dict[name]["attributes"] = attrs

    # NOTE many of these methods are instance methods so that HDF5 references can be dereferenced using self._h5f
    def translate_attrs(
        self,
        h5obj: Union[h5py.Dataset, h5py.Group],
    ) -> dict:
        """Copy attributes from an HDF5 object to a new "attributes" key in the group/dataset dictionary.

        Parameters
        ----------
        h5obj : h5py.Group or h5py.Dataset
            An HDF5 group or dataset.
        object_dict : dict
            A dictionary representing the translated HDF5 group or dataset.
        """
        ret = {}
        for n, v in h5obj.attrs.items():
            logger.debug(f"Adding HDF5 attribute of {h5obj.name}: {n} = {v} (type {type(v)}, shape {np.shape(v)})))")
            ret[n] = self._translate_data(v)
            # TODO: store dtype, shape, etc.? probably not necessary, but if there are attributes
            # that have compound dtypes, then the field names are lost and those could be misread as
            # attributes with an additional dimension.
            # see https://hdf5-json.readthedocs.io/en/latest/examples/classic.html
        return ret

    def _translate_data(self, value):
        """Translate a scalar/array attribute or scalar dataset from HDF5 to JSON.

        TODO check docstring

        """
        if isinstance(value, np.void):
            # scalar with compound dtype
            value = self._translate_compound_dtype_array(value)
        elif isinstance(value, np.ndarray):
            logger.debug(f"Translating numpy array with dtype {value.dtype}")
            if value.dtype.kind == "S":
                # decode from byte array to python string so that it is json-serializable
                value = value.astype("U").tolist()
            elif value.dtype.kind == "V":
                # array with compound dtype
                value = self._translate_compound_dtype_array(value)
            elif value.ndim == 0:
                # convert a 0D non-compound dtype numpy array to python int/float so that it is json-serializable
                value = value.item()
            else:
                value = self._translate_object_array_to_list(value)
                # value2 = value.ravel()
                # # modify value array in place
                # for i, val in enumerate(value2):
                #     if isinstance(val, bytes):
                #         value2[i] = val.decode()
                #     elif isinstance(val, h5py.h5r.Reference):
                #         value2[i] = self._translate_object_ref(val)
                # # convert array to nested lists and convert numpy numeric types to python int/float
                # # so that it is json-serializable
                # value = value.tolist()
        elif isinstance(value, h5py.Empty):
            # HDF5 has the concept of Empty or Null datasets and attributes. These are
            # not the same as an array with a shape of (), or a scalar dataspace in
            # HDF5 terms. Instead, it is a dataset with an associated type, no data,
            # and no shape. In h5py, this is represented as either a dataset with shape
            # None, or an instance of h5py.Empty. The purpose appears to be to hold
            # attributes, particularly for netcdf4 files.
            value = ""
        else:
            # scalar or 0D array with non-compound dtype
            value = self._translate_single_value(value)
        return value

    def _translate_compound_dtype_array(self, array) -> list:
        """Translate a numpy array with a compound dtype to a list or list of lists."""
        # compound dtype, return a list of lists. use dtype to see fields
        # TODO would it be better to return a dict?
        field_names = array.dtype.names
        for field_name in field_names:
            if array.dtype[field_name].shape != ():
                raise RuntimeError("Compound dtypes with non-scalar fields are not supported.")

        if array.ndim == 0:  # scalar or 0D array
            return self._translate_compound_dtype_element(field_names, array)
        elif array.ndim == 1:  # 1D array
            outer = []
            for val in array:  # read array into memory
                inner = self._translate_compound_dtype_element(field_names, val)
                outer.append(inner)
            return outer
        else:  # ndim > 1 not supported, but could be added
            raise RuntimeError("Compound dtypes with ndim > 1 are not supported.")

    def _translate_compound_dtype_element(self, field_names, val) -> list:
        """Translate a single element (row) from a compound dtype array to a list."""
        # iterate through the fields, and process each value specially if needed
        ret = [self._translate_single_value(val[field_name]) for field_name in field_names]
        return ret

    def _translate_single_value(self, value):
        """Decode bytes, dereference HDF5 object references, and convert numpy numeric types to python int/float."""
        if isinstance(value, bytes):
            value = value.decode()  # str
        elif isinstance(value, h5py.h5r.Reference):
            value = self._translate_object_ref(value)  # dict
        elif isinstance(value, (np.number, np.bool_)):
            # convert a scalar to python int/float so that it is json-serializable
            value = value.item()
        elif isinstance(value, h5py.h5r.RegionReference):
            raise RuntimeError("Region references are not supported.")
        return value

    def _translate_object_ref(self, ref: h5py.h5r.Reference) -> dict:
        """Encode an HDF5 object reference as a JSON dictionary."""
        target = self._h5f[ref]
        return {"dtype": "object_reference", "path": H5ToJson.get_json_path(target)}

    # @staticmethod
    # def _translate_array_object_strings(values: np.ndarray):
    #     """Encode a non-scalar array of HDF5 strings as a dict that maps indices to a custom dict."""
    #     data = {}
    #     it = np.nditer(values, flags=["refs_ok", "multi_index"])
    #     for i in it:
    #         index_str = ".".join(map(str, it.multi_index))
    #         if isinstance(i.item(), bytes):
    #             data[index_str] = i.item().decode()
    #         else:
    #             data[index_str] = i.item()
    #     return data

    # def _translate_array_object_refs(self, values: np.ndarray):
    #     """Encode a non-scalar array of HDF5 object refs as a dict that maps indices to a custom dict."""
    #     data = {}
    #     it = np.nditer(values, flags=["refs_ok", "multi_index"])
    #     for i in it:
    #         index_str = ".".join(map(str, it.multi_index))
    #         data[index_str] = self._translate_object_ref(i.item())
    #     return data

    def _translate_object_array_to_list(self, values: np.ndarray) -> list:
        """Encode a non-scalar array of HDF5 objects as a nested list of decoded, dereferenced values."""
        # this will modify the array values in-place.

        value2 = values.ravel()
        for i, val in enumerate(value2):
            if isinstance(val, bytes):
                value2[i] = val.decode()
            elif isinstance(val, h5py.h5r.Reference):
                value2[i] = self._translate_object_ref(val)
            elif isinstance(val, h5py.h5r.RegionReference):
                raise RuntimeError("Region references are not supported.")
        # convert array to nested lists and convert numpy numeric types to python int/float
        # so that it is json-serializable
        values = values.tolist()
        return values

    @staticmethod
    def get_json_path(h5obj: Union[h5py.Group, h5py.Dataset]) -> str:
        """Get the translated JSON path of an HDF5 object.

        For an object with HDF5 name /foo/bar/baz where foo and bar are groups and baz is a dataset.
        this will return /groups/foo/groups/bar/datasets/baz.

        Parameters
        ----------
        h5obj : h5py.Group or h5py.Dataset
            An HDF5 group or dataset.
        """
        if h5obj.name == "/":
            return "/"

        if isinstance(h5obj, h5py.Group):
            base = "groups/" + os.path.basename(h5obj.name)
        else:
            base = "datasets/" + os.path.basename(h5obj.name)

        if h5obj.parent.name == "/":
            return "/" + base

        return H5ToJson.get_json_path(h5obj.parent) + "/" + base

    def translate_dataset(self, h5dataset: h5py.Dataset, datasets_dict: dict):
        """Translate an HDF5 dataset to a new dict within the given dict.

        Parameters
        ----------
        h5dataset : h5py.Dataset
            An HDF5 dataset.
        datasets_dict : dict
            A dictionary representing the datasets contained in the parent group.
        """
        logger.debug(f"HDF5 dataset: {h5dataset.name}")
        logger.debug(f"HDF5 compression: {h5dataset.compression}")

        if h5dataset.file != self._h5f:
            raise RuntimeError("External datasets are not supported.")

        name = os.path.basename(h5dataset.name)

        # create a new key for this dataset in the dictionary of datasets contained in the parent group
        datasets_dict[name] = {}
        this_dset_dict = datasets_dict[name]

        fill = h5dataset.fillvalue or None
        if h5dataset.id.get_create_plist().get_layout() == h5py.h5d.COMPACT:
            # TODO Only do if h5obj.nbytes < self.inline??
            this_dset_dict["data"] = h5dataset[:]
            fill = None
            filters = []
        else:
            filters = H5ToJson.decode_filters(h5dataset)

        if isinstance(fill, (np.number, np.bool_)):
            fill = fill.item()  # convert to python int/float so that it is json-serializable

        # Get storage info of this HDF5 dataset
        # this_dset_dict["storage_info"] = H5ToJson.storage_info(h5dataset)  # dict
        # NOTE: the keys below are used by zarr.
        # HDF5 uses slightly different keys:
        # HDF5 {"chunks": False} = Zarr {"chunks": dataset.shape}
        # HDF5 "compression" = Zarr "compressor" or it is placed in "filters"
        this_dset_dict["filters"] = filters  # list
        this_dset_dict["dtype"] = str(h5dataset.dtype)
        this_dset_dict["shape"] = h5dataset.shape
        this_dset_dict["chunks"] = h5dataset.chunks or h5dataset.shape
        this_dset_dict["compressor"] = None
        this_dset_dict["fill_value"] = fill

        # TODO how to represent in the JSON file that string data are vlen strings
        # so that zarr can decode the binary data if desired.
        # NOTE this is not really necessary because the entire string dataset
        # will be inlined in the main JSON file.
        # kerchunk uses an ID-to-value mapping in the vlen_encode = embed option
        # but that requires kerchunk to be available and is efficient primarily if
        # there is a lot of repetition in the strings.
        # this_dset_dict["object_codec"] = object_codec

        if self.chunk_refs_file_path:
            kerchunk_refs = H5ToJson.get_kerchunk_refs(self._uri, h5dataset, this_dset_dict)
            self.chunk_refs.update(kerchunk_refs)
            this_dset_dict["refs"] = {
                "file": "{{c}}",  # use template
                "prefix": H5ToJson.get_ref_key_prefix(h5dataset),
            }

        data = None

        # if the dataset is a scalar, then store the value directly into the "data" key
        # just like for attributes
        if h5dataset.shape == ():
            value = h5dataset[()]
            data = self._translate_data(value)  # a value, not a dict

        dset_size = np.prod(h5dataset.shape) * h5dataset.dtype.itemsize

        # handle object dtype datasets (strings, references)
        # store the entire dataset in the "data" key as a dict
        if data is None and h5dataset.dtype.kind == "O":
            if dset_size <= self.object_dataset_inline_threshold:
                # scalar case is handled above
                values = h5dataset[:]  # read the entire dataset into memory
                # array.flat[0] returns the first element of the flattened array - useful for testing dtype
                if not isinstance(values.flat[0], (bytes, str, h5py.h5r.Reference)):
                    raise RuntimeError(f"Unexpected object dtype for dataset {h5dataset.name}: {type(values.flat[0])}")
                data = self._translate_object_array_to_list(values)
            else:
                warnings.warn(
                    f"Dataset with name {h5dataset.name} has object dtype and is too large to inline: {dset_size} > "
                    f"{self.object_dataset_inline_threshold}. Increase `object_dataset_inline_threshold` to inline "
                    "this dataset."
                )

            # if isinstance(values.flat[0], (bytes, str)):
            #     data = H5ToJson._translate_array_object_strings(values)
            # elif isinstance(values.flat[0], h5py.h5r.Reference):
            #     data = self._translate_array_object_refs(values)
            # else:
            #     raise RuntimeError(f"Unexpected object dtype for dataset {h5dataset.name}")

        # handle fixed-length string datasets
        # store the entire dataset in the "data" key as a list
        if data is None and h5dataset.dtype.kind == "S":
            if dset_size <= self.object_dataset_inline_threshold:
                # decode from byte array to python string so that it is json-serializable
                data = h5dataset[:].astype("U").tolist()

        # handle compound dtype datasets
        # store the entire dataset in the "data" key as a list
        if data is None and h5dataset.dtype.kind == "V":
            # scalar case is handled above
            values = h5dataset[:]  # read the entire dataset into memory
            data = self._translate_compound_dtype_array(values)

        if data is None:
            # if dataset is small enough, inline it into "data" key
            if dset_size <= self.dataset_inline_threshold:
                # NOTE this is OK for ints, but floats will be stored as strings which means potential
                # loss of precision. an alternative is to read the floats into memory (decode with all
                # filters) and then encode it in base64 and then store it in JSON.
                data = h5dataset[:].tolist()

        # even if we have unfiltered the data, report the original chunks/compression below
        this_dset_dict["data"] = data or None  # can be list or dict. should this be written if it is none?

        attrs = self.translate_attrs(h5dataset)
        if attrs:
            this_dset_dict["attributes"] = attrs

        # TODO handle fillvalue, fletcher32, scaleoffset -- see kerchunk/HDF5Zarr

    @staticmethod
    def decode_filters(h5obj: h5py.Dataset) -> list:
        """Return a list of numcodecs filters for the given HDF5 object.

        Note that HDF5 does not use numcodecs filters. This translates the HDF5 filters to numcodecs filters.
        Only some filters are supported.

        Supported filters:
        - Shuffled object types
        - Blosc (ID: 32001)
        - Zstd (ID: 32015)
        - Gzip (ID: gzip)

        Not supported filters:
        - Scaleoffset (ID: scaleoffset)
        - Szip (ID: szip)
        - LZF (ID: lzf)
        - LZ4 (ID: 32004)
        - Bitshuffle (ID: 32008)
        """
        filters_json = []
        if h5obj.scaleoffset:
            warnings.warn(f"{h5obj.name} uses HDF5 scaleoffset filter - not supported")
            filters_json.append({"id": "scaleoffset"})
            # raise RuntimeError(f"{h5obj.name} uses HDF5 scaleoffset filter - not supported")
        if h5obj.compression in ("szip", "lzf"):
            warnings.warn(f"{h5obj.name} uses szip or lzf compression - not supported")
            filters_json.append({"id": h5obj.compression})
            # raise RuntimeError(f"{h5obj.name} uses szip or lzf compression - not supported")
        filters = []
        if h5obj.shuffle and h5obj.dtype.kind != "O":
            # cannot use shuffle if we materialised objects
            filters.append(numcodecs.Shuffle(elementsize=h5obj.dtype.itemsize))
        for filter_id, properties in h5obj._filters.items():
            if str(filter_id) == "32001":
                blosc_compressors = (
                    "blosclz",
                    "lz4",
                    "lz4hc",
                    "snappy",
                    "zlib",
                    "zstd",
                )
                (
                    _1,
                    _2,
                    bytes_per_num,
                    total_bytes,
                    clevel,
                    shuffle,
                    compressor,
                ) = properties
                pars = dict(
                    blocksize=total_bytes,
                    clevel=clevel,
                    shuffle=shuffle,
                    cname=blosc_compressors[compressor],
                )
                filters.append(numcodecs.Blosc(**pars))
            elif str(filter_id) == "32015":
                filters.append(numcodecs.Zstd(level=properties[0]))
            elif str(filter_id) == "gzip":
                filters.append(numcodecs.Zlib(level=properties))
            elif str(filter_id) == "32004":
                raise RuntimeError(f"{h5obj.name} uses lz4 compression - not supported")
            elif str(filter_id) == "32008":
                raise RuntimeError(f"{h5obj.name} uses bitshuffle compression - not supported")
            elif str(filter_id) == "shuffle":
                # already handled before this loop
                pass
            else:
                # raise RuntimeError(
                #     f"{h5obj.name} uses filter id {filter_id} with properties {properties}, not supported."
                # )
                warnings.warn(f"{h5obj.name} uses filter id {filter_id} with properties {properties}, not supported.")
                filters_json.append({"id": filter_id, "properties": properties})
        filters_json.extend([f.get_config() for f in filters])  # json-serializable list of filters
        # use numcodecs.get_codec to get the codec object from the config dict
        return filters_json

    @staticmethod
    def storage_info(dset: h5py.Dataset) -> dict:
        """Get storage information of an HDF5 dataset in the HDF5 file.

        NOTE: currently unused. using get_kerchunk_refs instead

        Storage information consists of file offset and size (length) for every
        chunk of the HDF5 dataset.

        Parameters
        ----------
        dset : h5py.Dataset
            HDF5 dataset for which to collect storage information.

        Returns
        -------
        dict
            HDF5 dataset storage information. Dict keys are chunk array offsets
            as tuples. Dict values are pairs with chunk file offset and size
            integers.
        """
        # Empty (null) dataset...
        if dset.shape is None:
            return dict()

        dsid = dset.id
        if dset.chunks is None:
            # Contiguous dataset...
            if dsid.get_offset() is None:
                # No data ever written...
                return dict()
            else:
                key = (0,) * (len(dset.shape) or 1)
                return {key: {"offset": dsid.get_offset(), "size": dsid.get_storage_size()}}
        else:
            # Chunked dataset...
            num_chunks = dsid.get_num_chunks()
            if num_chunks == 0:
                # No data ever written...
                return dict()

            # Go over all the dataset chunks...
            stinfo = dict()
            chunk_size = dset.chunks

            def get_key(blob):
                """Get indexing key for the chunk."""
                return tuple([a // b for a, b in zip(blob.chunk_offset, chunk_size)])

            def store_chunk_info(blob):
                """Store chunk index, offset, and size in the dict."""
                stinfo[get_key(blob)] = {"offset": blob.byte_offset, "size": blob.size}

            has_chunk_iter = callable(getattr(dsid, "chunk_iter", None))

            if has_chunk_iter:
                dsid.chunk_iter(store_chunk_info)
            else:
                for index in range(num_chunks):
                    store_chunk_info(dsid.get_chunk_info(index))

            return stinfo

    @staticmethod
    def get_ref_key_prefix(dset: h5py.Dataset) -> str:
        """Get the prefix used in keys in the references JSON for an HDF5 dataset.

        This is the dataset name without the leading slash.

        Parameters
        ----------
        h5obj : h5py.Dataset
            An HDF5 dataset.

        Returns
        -------
        str
            The dataset name without the leading slash.
        """
        return dset.name[1:]

    @staticmethod
    def get_kerchunk_refs(uri: str, dset: h5py.Dataset, this_dset_dict: dict) -> dict:
        """Get kerchunk-style chunk references of an HDF5 dataset in the HDF5 file.

        This is the format expected by fsspec.implementations.reference.ReferenceFileSystem.

        TODO refactor with the above function if the above one is still useful (it might not be)

        Storage information consists of file offset and size (length) for every
        chunk of the HDF5 dataset. Dataset-level information is stored in the main
        JSON file. This chunk-level information is stored in a separate file.

        TODO should this include zarray? That metadata is the only way
        to make sense of the chunks and it allows the array to be recreated at least
        in zarr, but an interpreter could br written to read the chunks into other
        formats. It is kind of like the header of a binary file. Storing the
        contents of .zarray as a text string instead of a JSON object is kind of
        inelegant though.

        Example output:
        {
            "data/.zarray": ("{\"chunks\":[600,8],\"compressor\":null,\"dtype\":\"<i8\",\"fill_value\":null,"
                             "\"filters\":null,\"order\":\"C\",\"shape\":[1000,10],\"zarr_format\":2}"),
            "data/0.0": ["file://test.h5", 4016, 38400],
            "data/0.1": ["file://test.h5", 42416, 38400],
            "data/1.0": ["file://test.h5", 80816, 38400],
            "data/1.1": ["file://test.h5", 119216, 38400]
        }

        Parameters
        ----------
        uri : str
            URI of the HDF5 file.
        dset : h5py.Dataset
            HDF5 dataset for which to collect storage information.
        this_dset_dict : dict
            A dictionary representing the dataset in the JSON file.

        Returns
        -------
        dict
            HDF5 dataset storage information. Dict keys are chunk array offsets
            with dot separators, appended to the dataset path. Dict values are
            lists of uri, chunk file offset, and chunk size.
        """
        name = H5ToJson.get_ref_key_prefix(dset)
        ret = {
            # alternatively store as a dictionary instead of a JSON string
            f"{name}/.zarray": ujson.dumps({
                "chunks": this_dset_dict["chunks"],
                "compressor": this_dset_dict["compressor"],
                "dtype": this_dset_dict["dtype"],
                "fill_value": this_dset_dict["fill_value"],
                "filters": this_dset_dict["filters"],
                "order": "C",
                "shape": this_dset_dict["shape"],
                "zarr_format": 2,
            })
        }

        # Empty (null) dataset...
        if dset.shape is None:
            return ret

        dsid = dset.id
        if dset.chunks is None:
            # Contiguous dataset...
            if dsid.get_offset() is None:
                # No data ever written...
                return ret
            else:
                key = name + "/" + ".".join(map(str, (0,) * (len(dset.shape) or 1)))
                ret[key] = [uri, dsid.get_offset(), dsid.get_storage_size()]
                return ret
        else:
            # Chunked dataset...
            num_chunks = dsid.get_num_chunks()
            if num_chunks == 0:
                # No data ever written...
                return ret

            # Go over all the dataset chunks...
            chunk_size = dset.chunks

            def get_key(blob):
                """Get indexing key for the chunk."""
                return name + "/" + ".".join(map(str, tuple([a // b for a, b in zip(blob.chunk_offset, chunk_size)])))

            def store_chunk_info(blob):
                """Store uri, chunk index, offset, and size in the list."""
                key = get_key(blob)
                ret[key] = [uri, blob.byte_offset, blob.size]

            has_chunk_iter = callable(getattr(dsid, "chunk_iter", None))

            if has_chunk_iter:
                dsid.chunk_iter(store_chunk_info)
            else:
                if num_chunks >= 1e4:
                    warnings.warn(
                        f"Dataset with name {dset.name} has {num_chunks} chunks. This may take a long time to read."
                    )
                    for index in tqdm(range(num_chunks)):
                        store_chunk_info(dsid.get_chunk_info(index))
                else:
                    # no progress bar
                    for index in range(num_chunks):
                        store_chunk_info(dsid.get_chunk_info(index))

            return ret
