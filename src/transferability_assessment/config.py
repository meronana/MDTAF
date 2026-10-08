"""Configuration for transferability assessment pipeline."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PreprocessConfig:
    """Molecular preprocessing configuration."""

    remove_salts: bool = True
    """Remove salt and solvent fragments."""

    keep_stereochemistry: bool = True
    """Preserve stereochemistry in standardization."""

    deduplication_method: Literal["inchikey", "smiles", "none"] = "inchikey"
    """Method for compound deduplication."""

    duplicate_aggregation: Literal["first", "mean", "all"] = "mean"
    """How to handle duplicate labels."""

    label_type: Literal["regression", "classification"] = "regression"
    """Type of target variable."""

    log_scale_detection: bool = True
    """Automatically detect if labels are log-scaled."""


@dataclass
class ValidationConfig:
    """Input validation configuration."""

    allow_empty_datasets: bool = False
    """Allow processing of empty datasets."""

    min_valid_smiles_ratio: float = 0.5
    """Minimum fraction of valid SMILES required."""

    outlier_detection: Literal["iqr", "zscore", "none"] = "iqr"
    """Method for detecting label outliers."""

    outlier_threshold: float = 1.5
    """IQR multiplier (1.5) or z-score threshold."""

    min_overlap_for_correlation: int = 3
    """Minimum overlap compounds to compute correlation."""


@dataclass
class AssessmentConfig:
    """Overall assessment configuration."""

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    seed: int = 42
    """Random seed for reproducibility."""

    verbose: bool = True
    """Print progress messages."""

    output_format: Literal["json", "dict"] = "dict"
    """Output format for assessment results."""
