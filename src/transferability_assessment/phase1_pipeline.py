"""Phase 1: Input Validation & Preprocessing Pipeline."""

from pathlib import Path
from typing import Optional

import json
import pandas as pd

from .config import AssessmentConfig
from .input_validation import validate_dataset, DatasetValidationReport
from .preprocessing import preprocess_dataset
from .overlap_analysis import analyze_overlap, OverlapAnalysisResult


class Phase1Pipeline:
    """Phase 1: Comprehensive input validation and preprocessing."""

    def __init__(self, config: Optional[AssessmentConfig] = None):
        """Initialize pipeline.

        Parameters
        ----------
        config : AssessmentConfig, optional
            Configuration for the pipeline
        """
        self.config = config or AssessmentConfig()
        self.source_df: Optional[pd.DataFrame] = None
        self.target_df: Optional[pd.DataFrame] = None
        self.source_validation: Optional[DatasetValidationReport] = None
        self.target_validation: Optional[DatasetValidationReport] = None
        self.overlap_analysis: Optional[OverlapAnalysisResult] = None

    def load_datasets(
        self,
        source_path: str | Path,
        target_path: str | Path,
        source_name: str = "source",
        target_name: str = "target",
        **read_kwargs
    ) -> None:
        """Load source and target datasets.

        Parameters
        ----------
        source_path : str | Path
            Path to source dataset file (CSV/TSV)
        target_path : str | Path
            Path to target dataset file (CSV/TSV)
        source_name : str
            Name for source dataset
        target_name : str
            Name for target dataset
        **read_kwargs
            Additional arguments to pass to pd.read_csv
        """
        self.source_df = pd.read_csv(source_path, **read_kwargs)
        self.target_df = pd.read_csv(target_path, **read_kwargs)

        if self.config.verbose:
            print(f"Loaded source: {len(self.source_df)} rows")
            print(f"Loaded target: {len(self.target_df)} rows")

    def validate_datasets(
        self,
        source_smiles_col: str = "smiles",
        source_label_col: str = "label",
        target_smiles_col: str = "smiles",
        target_label_col: str = "label",
    ) -> tuple[DatasetValidationReport, DatasetValidationReport]:
        """Validate source and target datasets.

        Parameters
        ----------
        source_smiles_col : str
            SMILES column name in source
        source_label_col : str
            Label column name in source
        target_smiles_col : str
            SMILES column name in target
        target_label_col : str
            Label column name in target

        Returns
        -------
        tuple
            (source_validation_report, target_validation_report)
        """
        if self.source_df is None or self.target_df is None:
            raise ValueError("Datasets not loaded. Call load_datasets() first.")

        # Validate source
        self.source_validation = validate_dataset(
            self.source_df,
            smiles_col=source_smiles_col,
            label_col=source_label_col,
            dataset_name="source",
            data_type=self.config.preprocess.label_type,
            allow_empty=self.config.validation.allow_empty_datasets,
            min_valid_ratio=self.config.validation.min_valid_smiles_ratio,
            outlier_method=self.config.validation.outlier_detection,
            outlier_threshold=self.config.validation.outlier_threshold,
        )

        # Validate target
        self.target_validation = validate_dataset(
            self.target_df,
            smiles_col=target_smiles_col,
            label_col=target_label_col,
            dataset_name="target",
            data_type=self.config.preprocess.label_type,
            allow_empty=self.config.validation.allow_empty_datasets,
            min_valid_ratio=self.config.validation.min_valid_smiles_ratio,
            outlier_method=self.config.validation.outlier_detection,
            outlier_threshold=self.config.validation.outlier_threshold,
        )

        if self.config.verbose:
            print("\n" + self.source_validation.summary())
            print("\n" + self.target_validation.summary())

        return self.source_validation, self.target_validation

    def preprocess_datasets(
        self,
        source_smiles_col: str = "smiles",
        source_label_col: str = "label",
        target_smiles_col: str = "smiles",
        target_label_col: str = "label",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess source and target datasets.

        Parameters
        ----------
        source_smiles_col : str
            SMILES column name in source
        source_label_col : str
            Label column name in source
        target_smiles_col : str
            SMILES column name in target
        target_label_col : str
            Label column name in target

        Returns
        -------
        tuple
            (preprocessed_source_df, preprocessed_target_df)
        """
        if self.source_df is None or self.target_df is None:
            raise ValueError("Datasets not loaded. Call load_datasets() first.")

        # Preprocess source
        self.source_df = preprocess_dataset(
            self.source_df,
            smiles_col=source_smiles_col,
            label_col=source_label_col,
            remove_salts=self.config.preprocess.remove_salts,
            keep_stereochemistry=self.config.preprocess.keep_stereochemistry,
            deduplication_method=self.config.preprocess.deduplication_method,
            duplicate_aggregation=self.config.preprocess.duplicate_aggregation,
            drop_invalid=True,
        )

        # Preprocess target
        self.target_df = preprocess_dataset(
            self.target_df,
            smiles_col=target_smiles_col,
            label_col=target_label_col,
            remove_salts=self.config.preprocess.remove_salts,
            keep_stereochemistry=self.config.preprocess.keep_stereochemistry,
            deduplication_method=self.config.preprocess.deduplication_method,
            duplicate_aggregation=self.config.preprocess.duplicate_aggregation,
            drop_invalid=True,
        )

        if self.config.verbose:
            print(f"After preprocessing - source: {len(self.source_df)} rows")
            print(f"After preprocessing - target: {len(self.target_df)} rows")

        return self.source_df, self.target_df

    def analyze_overlap(
        self,
        source_smiles_col: str = "smiles",
        target_smiles_col: str = "smiles",
        source_label_col: str = "label",
        target_label_col: str = "label",
    ) -> OverlapAnalysisResult:
        """Analyze compound overlap between datasets.

        Parameters
        ----------
        source_smiles_col : str
            SMILES column name in source
        target_smiles_col : str
            SMILES column name in target
        source_label_col : str
            Label column name in source
        target_label_col : str
            Label column name in target

        Returns
        -------
        OverlapAnalysisResult
            Overlap analysis results
        """
        if self.source_df is None or self.target_df is None:
            raise ValueError("Datasets not loaded. Call load_datasets() first.")

        # Ensure InChIKey is available
        if "_inchi_key" not in self.source_df.columns:
            from .preprocessing import standardize_smiles
            self.source_df["_inchi_key"] = [
                standardize_smiles(s).inchi_key
                for s in self.source_df[source_smiles_col]
            ]
        if "_inchi_key" not in self.target_df.columns:
            from .preprocessing import standardize_smiles
            self.target_df["_inchi_key"] = [
                standardize_smiles(s).inchi_key
                for s in self.target_df[target_smiles_col]
            ]

        # Analyze overlap
        self.overlap_analysis = analyze_overlap(
            self.source_df,
            self.target_df,
            source_id_col="_inchi_key",
            target_id_col="_inchi_key",
            source_label_col=source_label_col,
            target_label_col=target_label_col,
            source_name="source",
            target_name="target",
            min_overlap_for_correlation=self.config.validation.min_overlap_for_correlation,
        )

        if self.config.verbose:
            print(f"\nOverlap Analysis:")
            print(f"  Shared compounds: {self.overlap_analysis.overlap_count}")
            print(f"  Source coverage: {self.overlap_analysis.source_coverage:.1f}%")
            print(f"  Target coverage: {self.overlap_analysis.target_coverage:.1f}%")
            if self.overlap_analysis.label_correspondence.n_overlap > 0:
                corr = self.overlap_analysis.label_correspondence
                print(f"  Label correlation (n={corr.n_overlap}):")
                print(f"    Pearson r: {corr.pearson_r:.3f} (p={corr.pearson_pvalue:.3e})")
                print(f"    Spearman ρ: {corr.spearman_rho:.3f} (p={corr.spearman_pvalue:.3e})")
                print(f"    R²: {corr.r_squared:.3f}")
                print(f"    RMSE: {corr.rmse:.3f}")
                print(f"    Confidence: {corr.confidence_level}")

        return self.overlap_analysis

    def run(
        self,
        source_path: str | Path,
        target_path: str | Path,
        source_name: str = "source",
        target_name: str = "target",
        source_smiles_col: str = "smiles",
        source_label_col: str = "label",
        target_smiles_col: str = "smiles",
        target_label_col: str = "label",
        **read_kwargs
    ) -> dict:
        """Run complete Phase 1 pipeline.

        Parameters
        ----------
        source_path : str | Path
            Path to source dataset
        target_path : str | Path
            Path to target dataset
        source_name : str
            Name for source dataset
        target_name : str
            Name for target dataset
        source_smiles_col : str
            SMILES column in source
        source_label_col : str
            Label column in source
        target_smiles_col : str
            SMILES column in target
        target_label_col : str
            Label column in target
        **read_kwargs
            Additional arguments for pd.read_csv

        Returns
        -------
        dict
            Complete Phase 1 report
        """
        # Load
        self.load_datasets(source_path, target_path, source_name, target_name, **read_kwargs)

        # Validate
        source_val, target_val = self.validate_datasets(
            source_smiles_col=source_smiles_col,
            source_label_col=source_label_col,
            target_smiles_col=target_smiles_col,
            target_label_col=target_label_col,
        )

        if not (source_val.is_valid() and target_val.is_valid()):
            if self.config.verbose:
                print("\nValidation failed. Aborting.")
            return self.get_report()

        # Preprocess
        self.preprocess_datasets(
            source_smiles_col=source_smiles_col,
            source_label_col=source_label_col,
            target_smiles_col=target_smiles_col,
            target_label_col=target_label_col,
        )

        # Overlap analysis
        self.analyze_overlap(
            source_smiles_col=source_smiles_col,
            target_smiles_col=target_smiles_col,
            source_label_col=source_label_col,
            target_label_col=target_label_col,
        )

        return self.get_report()

    def get_report(self) -> dict:
        """Get Phase 1 report in dictionary format.

        Returns
        -------
        dict
            Complete Phase 1 analysis report
        """
        report = {
            "phase": 1,
            "stage": "Input Validation & Preprocessing",
            "source_validation": (
                {
                    "name": self.source_validation.dataset_name,
                    "original_count": self.source_validation.original_count,
                    "valid_smiles": self.source_validation.valid_smiles_count,
                    "missing_labels": self.source_validation.missing_labels_count,
                    "duplicates": self.source_validation.duplicate_count,
                    "outliers": self.source_validation.outlier_count,
                    "is_valid": self.source_validation.is_valid(),
                    "label_stats": self.source_validation.label_stats,
                    "warnings": self.source_validation.warnings,
                    "errors": self.source_validation.errors,
                }
                if self.source_validation
                else None
            ),
            "target_validation": (
                {
                    "name": self.target_validation.dataset_name,
                    "original_count": self.target_validation.original_count,
                    "valid_smiles": self.target_validation.valid_smiles_count,
                    "missing_labels": self.target_validation.missing_labels_count,
                    "duplicates": self.target_validation.duplicate_count,
                    "outliers": self.target_validation.outlier_count,
                    "is_valid": self.target_validation.is_valid(),
                    "label_stats": self.target_validation.label_stats,
                    "warnings": self.target_validation.warnings,
                    "errors": self.target_validation.errors,
                }
                if self.target_validation
                else None
            ),
            "overlap_analysis": self.overlap_analysis.to_dict() if self.overlap_analysis else None,
        }
        return report

    def save_report(self, output_path: str | Path) -> None:
        """Save Phase 1 report to JSON file.

        Parameters
        ----------
        output_path : str | Path
            Path to save JSON report
        """
        report = self.get_report()
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        if self.config.verbose:
            print(f"\nReport saved to {output_path}")

    def save_datasets(
        self,
        source_output: str | Path,
        target_output: str | Path,
    ) -> None:
        """Save preprocessed datasets.

        Parameters
        ----------
        source_output : str | Path
            Path to save preprocessed source
        target_output : str | Path
            Path to save preprocessed target
        """
        if self.source_df is not None:
            self.source_df.to_csv(source_output, index=False)
            if self.config.verbose:
                print(f"Saved preprocessed source to {source_output}")
        if self.target_df is not None:
            self.target_df.to_csv(target_output, index=False)
            if self.config.verbose:
                print(f"Saved preprocessed target to {target_output}")
