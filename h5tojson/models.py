from typing import Any, Dict, Optional, Union

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


class H5ToJsonFile(BaseModel):
    """H5ToJsonFile model"""

    version: int = Field(description="Version of the file")
    created_at: str = Field(description="Date of creation in ISO format")
    translation_options: H5ToJsonTranslationOptions = Field(description="Translation options")
    templates: dict = Field({}, description="Templates")
    file: H5ToJsonGroup = Field(description="Root group")
