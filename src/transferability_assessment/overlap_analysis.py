"""Compound overlap and label correspondence analysis."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class LabelCorrelationStats:
    """Label correspondence statistics for overlapping compounds."""

    n_overlap: int = 0
    pearson_r: float = np.nan
    pearson_pvalue: float = np.nan
    spearman_rho: float = np.nan
    spearman_pvalue: float = np.nan
    r_squared: float = np.nan
    rmse: float = np.nan
    mean_bias: float = np.nan
    calibration_slope: float = np.nan
    confidence_level: Literal["high", "medium", "low"] = "low"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "n_overlap": self.n_overlap,
            "pearson_r": float(self.pearson_r) if not np.isnan(self.pearson_r) else None,
            "pearson_pvalue": float(self.pearson_pvalue) if not np.isnan(self.pearson_pvalue) else None,
            "spearman_rho": float(self.spearman_rho) if not np.isnan(self.spearman_rho) else None,
            "spearman_pvalue": float(self.spearman_pvalue) if not np.isnan(self.spearman_pvalue) else None,
            "r_squared": float(self.r_squared) if not np.isnan(self.r_squared) else None,
            "rmse": float(self.rmse) if not np.isnan(self.rmse) else None,
            "mean_bias": float(self.mean_bias) if not np.isnan(self.mean_bias) else None,
            "calibration_slope": float(self.calibration_slope) if not np.isnan(self.calibration_slope) else None,
            "confidence_level": self.confidence_level,
        }


@dataclass
class OverlapAnalysisResult:
    """Result of source-target overlap analysis."""

    source_name: str = ""
    target_name: str = ""

    source_total: int = 0
    target_total: int = 0
    overlap_count: int = 0

    source_coverage: float = 0.0
    target_coverage: float = 0.0

    label_correspondence: LabelCorrelationStats = field(default_factory=LabelCorrelationStats)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "source_name": self.source_name,
            "target_name": self.target_name,
            "source_total": self.source_total,
            "target_total": self.target_total,
            "overlap_count": self.overlap_count,
            "source_coverage_pct": self.source_coverage,
            "target_coverage_pct": self.target_coverage,
            "label_correspondence": self.label_correspondence.to_dict(),
        }


def find_compound_overlap(
    source_smiles_or_keys: list[str],
    target_smiles_or_keys: list[str],
    method: Literal["exact", "inchikey"] = "exact",
) -> tuple[np.ndarray, np.ndarray]:
    """Find overlapping compounds between source and target.

    Parameters
    ----------
    source_smiles_or_keys : list[str]
        Source SMILES or InChIKeys
    target_smiles_or_keys : list[str]
        Target SMILES or InChIKeys
    method : {"exact", "inchikey"}
        Matching method (currently both work with exact string match)

    Returns
    -------
    tuple
        (source_indices, target_indices) where compounds match
    """
    source_set = {s: i for i, s in enumerate(source_smiles_or_keys)}
    target_set = {t: i for i, t in enumerate(target_smiles_or_keys)}

    # Find intersection
    common_ids = set(source_set.keys()) & set(target_set.keys())

    # Get indices
    source_indices = np.array([source_set[c] for c in common_ids])
    target_indices = np.array([target_set[c] for c in common_ids])

    # Sort by source index for consistency
    sort_idx = np.argsort(source_indices)
    source_indices = source_indices[sort_idx]
    target_indices = target_indices[sort_idx]

    return source_indices, target_indices


def compute_label_correspondence(
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    min_overlap: int = 3,
) -> LabelCorrelationStats:
    """Compute label correlation statistics.

    Parameters
    ----------
    source_labels : np.ndarray
        Source labels for overlapping compounds
    target_labels : np.ndarray
        Target labels for overlapping compounds
    min_overlap : int
        Minimum overlap required to compute correlation

    Returns
    -------
    LabelCorrelationStats
        Correlation and regression statistics
    """
    stats_result = LabelCorrelationStats(n_overlap=len(source_labels))

    if len(source_labels) < min_overlap:
        stats_result.confidence_level = "low"
        return stats_result

    # Convert to float
    source_labels = np.asarray(source_labels, dtype=float)
    target_labels = np.asarray(target_labels, dtype=float)

    # Pearson correlation
    try:
        r, pval = stats.pearsonr(source_labels, target_labels)
        stats_result.pearson_r = r
        stats_result.pearson_pvalue = pval
    except Exception:
        pass

    # Spearman correlation
    try:
        rho, pval = stats.spearmanr(source_labels, target_labels)
        stats_result.spearman_rho = rho
        stats_result.spearman_pvalue = pval
    except Exception:
        pass

    # R² (from linear regression)
    try:
        ss_res = np.sum((target_labels - source_labels) ** 2)
        ss_tot = np.sum((target_labels - np.mean(target_labels)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        stats_result.r_squared = r2
    except Exception:
        pass

    # RMSE
    try:
        rmse = np.sqrt(np.mean((target_labels - source_labels) ** 2))
        stats_result.rmse = rmse
    except Exception:
        pass

    # Mean bias
    try:
        bias = np.mean(target_labels - source_labels)
        stats_result.mean_bias = bias
    except Exception:
        pass

    # Calibration slope (linear regression: target = slope * source + intercept)
    try:
        A = np.vstack([source_labels, np.ones(len(source_labels))]).T
        slope, intercept = np.linalg.lstsq(A, target_labels, rcond=None)[0]
        stats_result.calibration_slope = slope
    except Exception:
        pass

    # Confidence level based on sample size
    if stats_result.n_overlap >= 10:
        stats_result.confidence_level = "high"
    elif stats_result.n_overlap >= 5:
        stats_result.confidence_level = "medium"
    else:
        stats_result.confidence_level = "low"

    return stats_result


def analyze_overlap(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    source_id_col: str = "_inchi_key",
    target_id_col: str = "_inchi_key",
    source_label_col: str = "label",
    target_label_col: str = "label",
    source_name: str = "source",
    target_name: str = "target",
    min_overlap_for_correlation: int = 3,
) -> OverlapAnalysisResult:
    """Analyze compound overlap between source and target datasets.

    Parameters
    ----------
    source_df : pd.DataFrame
        Source dataset
    target_df : pd.DataFrame
        Target dataset
    source_id_col : str
        Column name for compound ID (typically InChIKey)
    target_id_col : str
        Column name for compound ID (typically InChIKey)
    source_label_col : str
        Column name for labels in source
    target_label_col : str
        Column name for labels in target
    source_name : str
        Source dataset name
    target_name : str
        Target dataset name
    min_overlap_for_correlation : int
        Minimum overlap required to compute correlation

    Returns
    -------
    OverlapAnalysisResult
        Comprehensive overlap analysis
    """
    result = OverlapAnalysisResult(source_name=source_name, target_name=target_name)

    # Get total counts
    result.source_total = len(source_df)
    result.target_total = len(target_df)

    # Find overlap
    source_ids = source_df[source_id_col].tolist()
    target_ids = target_df[target_id_col].tolist()

    source_idx, target_idx = find_compound_overlap(source_ids, target_ids)
    result.overlap_count = len(source_idx)

    # Coverage percentages
    if result.source_total > 0:
        result.source_coverage = 100.0 * result.overlap_count / result.source_total
    if result.target_total > 0:
        result.target_coverage = 100.0 * result.overlap_count / result.target_total

    # Compute label correspondence
    if result.overlap_count >= min_overlap_for_correlation:
        source_labels = source_df[source_label_col].iloc[source_idx].values
        target_labels = target_df[target_label_col].iloc[target_idx].values

        result.label_correspondence = compute_label_correspondence(
            source_labels, target_labels, min_overlap=min_overlap_for_correlation
        )
    else:
        result.label_correspondence.n_overlap = result.overlap_count
        result.label_correspondence.confidence_level = "low"

    return result
