from typing import Any, Dict, Optional, Union, Tuple

from pydantic import BaseModel, Field


class H5ToJsonDatasetRefs(BaseModel):
    """H5ToJsonDatasetRefs model"""

    file: str = Field(description="Path to the chunk refs file")
    prefix: str = Field(description="Prefix to use for the key")


class H5ToJsonDataset(BaseModel):
    """H5ToJsonDataset model"""

    data: Optional[Any] = Field(None, description="Dataset data")
    attributes: dict = Field({}, description="Dataset attributes")
    filters: list = Field([], description="Dataset filters")
    dtype: Optional[str] = Field(None, description="Dataset dtype")
    shape: tuple = Field([], description="Dataset shape")
    chunks: tuple = Field([], description="Dataset chunks")
    compressor: Optional[str] = Field(None, description="Dataset compressor")
    fill_value: Optional[Union[str, int, float]] = Field(None, description="Dataset fill value")
    refs: Optional[H5ToJsonDatasetRefs] = Field(None, description="Dataset chunk refs")


class H5ToJsonSoftLink(BaseModel):
    """H5ToJsonSoftLink model"""

    path: str = Field(description="Path to the target")


class H5ToJsonExternalLink(BaseModel):
    """H5ToJsonExternalLink model"""

    path: str = Field(description="Path to the target")
    filename: str = Field(description="Filename of the target")


class H5ToJsonGroup(BaseModel):
    """H5ToJsonGroup model"""

    soft_links: Dict[str, H5ToJsonSoftLink] = Field({}, description="Soft links")
    external_links: Dict[str, H5ToJsonExternalLink] = Field({}, description="External links")
    groups: Dict[str, "H5ToJsonGroup"] = Field({}, description="Subgroups")
    datasets: Dict[str, H5ToJsonDataset] = Field({}, description="Datasets")
    attributes: dict = Field({}, description="Attributes")


class H5ToJsonTranslationOptions(BaseModel):
    """H5ToJsonTranslationOptions model"""

    dataset_inline_threshold_max_bytes: int = Field(description="Max bytes for inline dataset")
    object_dataset_inline_max_bytes: int = Field(description="Max bytes for inline object dataset")
    compound_dtype_dataset_inline_max_bytes: int = Field(description="Max bytes for inline compound dtype dataset")
    skip_all_dataset_data: bool = Field(description="Skip all data in datasets")


class H5ToJsonFile(BaseModel):
    """H5ToJsonFile model"""

    version: int = Field(description="Version of the file")
    created_at: str = Field(description="Date of creation in ISO format")
    translation_options: H5ToJsonTranslationOptions = Field(description="Translation options")
    templates: dict = Field({}, description="Templates")
    file: H5ToJsonGroup = Field(description="Root group")

    @staticmethod
    def from_dict(d: dict) -> "H5ToJsonFile":
        """Create an H5ToJsonFile from a dictionary"""
        return H5ToJsonFile(**d)

    @staticmethod
    def from_json_file(json_file_path: str) -> "H5ToJsonFile":
        """Create an H5ToJsonFile from a JSON file"""
        import json

        with open(json_file_path, "r") as f:
            d = json.load(f)
        return H5ToJsonFile.from_dict(d)

    def get_all_groups_and_datasets(self) -> Tuple[Dict[str, H5ToJsonGroup], Dict[str, H5ToJsonDataset]]:
        """Get all groups and datasets in the file (flattens the tree)"""
        groups: Dict[str, H5ToJsonGroup] = {}
        datasets: Dict[str, H5ToJsonDataset] = {}

        def _helper(g: H5ToJsonGroup, path: str):
            nonlocal groups
            groups[path] = g
            for k, v in g.groups.items():
                _helper(v, path + "/" + k)
            for k, v in g.datasets.items():
                datasets[path + "/" + k] = v

        _helper(self.file, "/")
        return groups, datasets
