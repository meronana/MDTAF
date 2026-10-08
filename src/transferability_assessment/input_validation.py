"""Input data validation for source-target dataset pairs."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


@dataclass
class SMILESValidationResult:
    """Result of SMILES validation."""

    is_valid: bool
    mol: Chem.Mol | None = None
    error_message: str = ""


@dataclass
class DatasetValidationReport:
    """Comprehensive validation report for a single dataset."""

    dataset_name: str
    original_count: int = 0
    valid_smiles_count: int = 0
    missing_labels_count: int = 0
    duplicate_count: int = 0
    invalid_labels_count: int = 0
    outlier_count: int = 0

    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    label_stats: dict = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if dataset is valid for processing."""
        return len(self.errors) == 0

    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            f"Dataset: {self.dataset_name}",
            f"  Original count: {self.original_count}",
            f"  Valid SMILES: {self.valid_smiles_count}",
            f"  Missing labels: {self.missing_labels_count}",
            f"  Duplicates: {self.duplicate_count}",
            f"  Invalid labels: {self.invalid_labels_count}",
            f"  Outliers detected: {self.outlier_count}",
        ]
        if self.errors:
            lines.append("  Errors:")
            for err in self.errors:
                lines.append(f"    - {err}")
        if self.warnings:
            lines.append("  Warnings:")
            for warn in self.warnings:
                lines.append(f"    - {warn}")
        return "\n".join(lines)


def validate_smiles(smiles: str) -> SMILESValidationResult:
    """Validate a single SMILES string.

    Parameters
    ----------
    smiles : str
        SMILES string to validate

    Returns
    -------
    SMILESValidationResult
        Validation result with parsed molecule if valid
    """
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return SMILESValidationResult(
                is_valid=False, error_message="RDKit parsing returned None"
            )
        if mol.GetNumHeavyAtoms() == 0:
            return SMILESValidationResult(
                is_valid=False, error_message="Molecule has no heavy atoms"
            )
        return SMILESValidationResult(is_valid=True, mol=mol)
    except Exception as e:
        return SMILESValidationResult(is_valid=False, error_message=str(e))


def validate_smiles_list(smiles_list) -> tuple[list[Chem.Mol | None], np.ndarray]:
    """Validate list of SMILES strings.

    Parameters
    ----------
    smiles_list : list-like
        List of SMILES strings

    Returns
    -------
    tuple
        (molecules, validity_mask) where molecules[i] is None if invalid
    """
    molecules = []
    validity = []
    for s in smiles_list:
        result = validate_smiles(s)
        molecules.append(result.mol if result.is_valid else None)
        validity.append(result.is_valid)
    return molecules, np.array(validity)


def validate_labels(
    labels, data_type: Literal["regression", "classification"] = "regression"
) -> tuple[np.ndarray, np.ndarray]:
    """Validate labels and return cleaned array with validity mask.

    Parameters
    ----------
    labels : array-like
        Target labels
    data_type : {"regression", "classification"}
        Type of labels

    Returns
    -------
    tuple
        (cleaned_labels, validity_mask)
    """
    labels_arr = np.asarray(labels)
    validity = np.ones(len(labels_arr), dtype=bool)

    # Check for NaN/None
    if labels_arr.dtype == object:
        validity = np.array(
            [
                x is not None
                and x != ""
                and (isinstance(x, (int, float)) or (isinstance(x, str) and x.strip()))
                for x in labels_arr
            ]
        )
    else:
        validity = ~np.isnan(labels_arr.astype(float))

    # Validate by type
    if data_type == "regression":
        try:
            clean_labels = labels_arr[validity].astype(float)
            return clean_labels, validity
        except (ValueError, TypeError):
            return np.array([]), validity

    elif data_type == "classification":
        clean_labels = labels_arr[validity]
        if len(np.unique(clean_labels)) < 2:
            return np.array([]), np.zeros(len(labels_arr), dtype=bool)
        return clean_labels, validity

    return labels_arr[validity], validity


def detect_outliers(
    labels: np.ndarray,
    method: Literal["iqr", "zscore"] = "iqr",
    threshold: float = 1.5,
) -> np.ndarray:
    """Detect outliers in labels.

    Parameters
    ----------
    labels : np.ndarray
        Array of numerical labels
    method : {"iqr", "zscore"}
        Outlier detection method
    threshold : float
        IQR multiplier or z-score threshold

    Returns
    -------
    np.ndarray
        Boolean mask of outliers
    """
    labels = np.asarray(labels, dtype=float)

    if method == "iqr":
        q1, q3 = np.percentile(labels, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outliers = (labels < lower_bound) | (labels > upper_bound)
    elif method == "zscore":
        z_scores = np.abs((labels - np.mean(labels)) / (np.std(labels) + 1e-8))
        outliers = z_scores > threshold
    else:
        outliers = np.zeros(len(labels), dtype=bool)

    return outliers


def validate_dataset(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    label_col: str = "label",
    dataset_name: str = "dataset",
    data_type: Literal["regression", "classification"] = "regression",
    allow_empty: bool = False,
    min_valid_ratio: float = 0.5,
    outlier_method: Literal["iqr", "zscore", "none"] = "iqr",
    outlier_threshold: float = 1.5,
) -> DatasetValidationReport:
    """Validate a dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    smiles_col : str
        Column name for SMILES
    label_col : str
        Column name for labels
    dataset_name : str
        Dataset identifier
    data_type : {"regression", "classification"}
        Type of labels
    allow_empty : bool
        Allow processing empty datasets
    min_valid_ratio : float
        Minimum ratio of valid compounds
    outlier_method : {"iqr", "zscore", "none"}
        Outlier detection method
    outlier_threshold : float
        Outlier threshold

    Returns
    -------
    DatasetValidationReport
        Comprehensive validation report
    """
    report = DatasetValidationReport(dataset_name=dataset_name)
    report.original_count = len(df)

    # Check basic structure
    if smiles_col not in df.columns:
        report.errors.append(f"SMILES column '{smiles_col}' not found")
        return report
    if label_col not in df.columns:
        report.errors.append(f"Label column '{label_col}' not found")
        return report

    # Validate SMILES
    smiles_list = df[smiles_col].tolist()
    mols, smiles_valid = validate_smiles_list(smiles_list)
    report.valid_smiles_count = np.sum(smiles_valid)

    if report.valid_smiles_count < len(df) * min_valid_ratio:
        report.errors.append(
            f"Valid SMILES ratio ({report.valid_smiles_count}/{len(df)}) "
            f"below minimum ({min_valid_ratio})"
        )

    # Validate labels
    labels = df[label_col].tolist()
    clean_labels, labels_valid = validate_labels(labels, data_type=data_type)
    report.missing_labels_count = np.sum(~labels_valid)
    report.invalid_labels_count = np.sum(
        ~np.isnan(np.asarray(labels, dtype=float))
    ) - len(clean_labels)

    # Label statistics
    if len(clean_labels) > 0:
        report.label_stats = {
            "mean": float(np.mean(clean_labels)),
            "median": float(np.median(clean_labels)),
            "std": float(np.std(clean_labels)),
            "min": float(np.min(clean_labels)),
            "max": float(np.max(clean_labels)),
            "q1": float(np.percentile(clean_labels, 25)),
            "q3": float(np.percentile(clean_labels, 75)),
            "count": int(len(clean_labels)),
        }

        # Outlier detection
        if outlier_method != "none":
            outliers = detect_outliers(clean_labels, method=outlier_method, threshold=outlier_threshold)
            report.outlier_count = np.sum(outliers)
            if report.outlier_count > 0:
                report.warnings.append(
                    f"{report.outlier_count} outliers detected "
                    f"({100*report.outlier_count/len(clean_labels):.1f}%)"
                )

    # Overall validity check
    combined_valid = smiles_valid & labels_valid
    valid_count = np.sum(combined_valid)

    if valid_count == 0 and not allow_empty:
        report.errors.append("No valid SMILES-label pairs after validation")
    elif valid_count == 0:
        report.warnings.append("Dataset is empty after validation")
    else:
        report.info.append(f"Valid compounds: {valid_count}/{len(df)}")

    return report
