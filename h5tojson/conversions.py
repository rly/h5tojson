from typing import Any, Dict, Optional
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
    storage_options: Optional[dict] = Field(None, description="Storage options for the file")


def h5_to_object(h5_file_path: str, opts: Optional[H5ToJsonOpts] = H5ToJsonOpts()) -> H5ToJsonFile:
    """Convert an HDF5 file to an object"""
    X = H5ToJson(
        h5_file_path=h5_file_path,
        json_file_path=None,
        chunk_refs_file_path=opts.chunk_refs_file_path,
        dataset_inline_max_bytes=opts.dataset_inline_max_bytes,
        object_dataset_inline_max_bytes=opts.object_dataset_inline_max_bytes,
        compound_dtype_dataset_inline_max_bytes=opts.compound_dtype_dataset_inline_max_bytes,
        storage_options=opts.storage_options,
    )
    X.translate()
    ret: H5ToJsonFile = X.file_object
    return ret


def h5_to_dict(h5_file_path: str, opts: Optional[H5ToJsonOpts] = H5ToJsonOpts()) -> Dict[str, Any]:
    """Convert an HDF5 file to a dictionary"""
    obj = h5_to_object(h5_file_path=h5_file_path, opts=opts)
    return _remove_empty_dicts_in_dict(obj.dict())


def h5_to_json_file(h5_file_path: str, json_file_path: str, opts: Optional[H5ToJsonOpts] = H5ToJsonOpts()) -> None:
    """Convert an HDF5 file to a JSON file"""
    X = H5ToJson(
        h5_file_path=h5_file_path,
        json_file_path=json_file_path,
        chunk_refs_file_path=opts.chunk_refs_file_path,
        dataset_inline_max_bytes=opts.dataset_inline_max_bytes,
        object_dataset_inline_max_bytes=opts.object_dataset_inline_max_bytes,
        compound_dtype_dataset_inline_max_bytes=opts.compound_dtype_dataset_inline_max_bytes,
        storage_options=opts.storage_options,
    )
    X.translate()
