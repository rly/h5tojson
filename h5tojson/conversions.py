"""Functions to convert HDF5 files to objects, dictionaries, and JSON files."""

from typing import Any, Dict, Optional, Union, List

from pydantic import BaseModel, Field

from .h5tojson import H5ToJson, _remove_empty_dicts_in_dict
from .models import H5ToJsonFile


class H5ToJsonOpts(BaseModel):
    """H5ToJsonOpts model"""

    chunk_refs_file_path: Optional[str] = Field(None, description="Path to the chunk refs file")
    dataset_inline_max_bytes: int = Field(500, description="Max bytes for inline dataset")
    object_dataset_inline_max_bytes: int = Field(200000, description="Max bytes for inline object dataset")
    compound_dtype_dataset_inline_max_bytes: int = Field(
        2000, description="Max bytes for inline compound dtype dataset"
    )
    skip_all_dataset_data: bool = Field(False, description="Skip all data in datasets")
    datasets_as_hdf5: Optional[Union[List[str], bool]] = Field(
        None, description="Paths to datasets to be saved in individual hdf5 files"
    )
    storage_options: Optional[dict] = Field(None, description="Storage options for the file")


def h5_to_object(hdf5_file_path: str, opts: Optional[H5ToJsonOpts] = H5ToJsonOpts()) -> H5ToJsonFile:
    """Convert an HDF5 file to an object"""
    X = H5ToJson(
        hdf5_file_path=hdf5_file_path,
        json_file_path=None,
        chunk_refs_file_path=opts.chunk_refs_file_path,
        dataset_inline_max_bytes=opts.dataset_inline_max_bytes,
        object_dataset_inline_max_bytes=opts.object_dataset_inline_max_bytes,
        compound_dtype_dataset_inline_max_bytes=opts.compound_dtype_dataset_inline_max_bytes,
        skip_all_dataset_data=opts.skip_all_dataset_data,
        datasets_as_hdf5=opts.datasets_as_hdf5,
        storage_options=opts.storage_options,
    )
    X.translate()
    ret: H5ToJsonFile = X.file_object
    return ret


def h5_to_dict(hdf5_file_path: str, opts: Optional[H5ToJsonOpts] = H5ToJsonOpts()) -> Dict[str, Any]:
    """Convert an HDF5 file to a dictionary"""
    obj = h5_to_object(hdf5_file_path=hdf5_file_path, opts=opts)
    return _remove_empty_dicts_in_dict(obj.model_dump())


def h5_to_json_file(hdf5_file_path: str, json_file_path: str, opts: Optional[H5ToJsonOpts] = H5ToJsonOpts()) -> None:
    """Convert an HDF5 file to a JSON file"""
    X = H5ToJson(
        hdf5_file_path=hdf5_file_path,
        json_file_path=json_file_path,
        chunk_refs_file_path=opts.chunk_refs_file_path,
        dataset_inline_max_bytes=opts.dataset_inline_max_bytes,
        object_dataset_inline_max_bytes=opts.object_dataset_inline_max_bytes,
        compound_dtype_dataset_inline_max_bytes=opts.compound_dtype_dataset_inline_max_bytes,
        skip_all_dataset_data=opts.skip_all_dataset_data,
        datasets_as_hdf5=opts.datasets_as_hdf5,
        storage_options=opts.storage_options,
    )
    X.translate()
