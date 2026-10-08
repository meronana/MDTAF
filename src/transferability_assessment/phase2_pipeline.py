"""Phase 2: Chemical Space Analysis Pipeline."""

from pathlib import Path
from typing import Optional

import json
import pandas as pd

from .config import AssessmentConfig
from .preprocessing import get_molecule_list
from .chemical_space_analysis import (
    analyze_morgan_fingerprints,
    analyze_scaffolds,
    analyze_descriptor_distributions,
    MorganFingerprintStats,
    ScaffoldStats,
    DescriptorStats,
)


class Phase2Pipeline:
    """Phase 2: Chemical Space Analysis."""

    def __init__(self, config: Optional[AssessmentConfig] = None):
        """Initialize Phase 2 pipeline.

        Parameters
        ----------
        config : AssessmentConfig, optional
            Configuration for the pipeline
        """
        self.config = config or AssessmentConfig()
        self.source_df: Optional[pd.DataFrame] = None
        self.target_df: Optional[pd.DataFrame] = None

        self.morgan_stats: Optional[MorganFingerprintStats] = None
        self.scaffold_stats: Optional[ScaffoldStats] = None
        self.descriptor_stats: Optional[list[DescriptorStats]] = None

    def load_datasets(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> None:
        """Load preprocessed datasets from Phase 1.

        Parameters
        ----------
        source_df : pd.DataFrame
            Preprocessed source dataframe (with SMILES column)
        target_df : pd.DataFrame
            Preprocessed target dataframe (with SMILES column)
        """
        self.source_df = source_df.copy()
        self.target_df = target_df.copy()

        if self.config.verbose:
            print(f"Loaded source: {len(self.source_df)} compounds")
            print(f"Loaded target: {len(self.target_df)} compounds")

    def load_from_files(
        self,
        source_path: str | Path,
        target_path: str | Path,
        smiles_col: str = "smiles",
    ) -> None:
        """Load preprocessed datasets from CSV files.

        Parameters
        ----------
        source_path : str | Path
            Path to preprocessed source CSV
        target_path : str | Path
            Path to preprocessed target CSV
        smiles_col : str
            Name of SMILES column
        """
        self.source_df = pd.read_csv(source_path)
        self.target_df = pd.read_csv(target_path)

        if self.config.verbose:
            print(f"Loaded source from {source_path}: {len(self.source_df)} compounds")
            print(f"Loaded target from {target_path}: {len(self.target_df)} compounds")

    def analyze_chemical_space(
        self,
        source_smiles_col: str = "smiles",
        target_smiles_col: str = "smiles",
        morgan_radius: int = 2,
        morgan_nbits: int = 2048,
    ) -> None:
        """Analyze chemical space metrics.

        Parameters
        ----------
        source_smiles_col : str
            SMILES column in source
        target_smiles_col : str
            SMILES column in target
        morgan_radius : int
            Morgan fingerprint radius
        morgan_nbits : int
            Number of bits in Morgan fingerprint
        """
        if self.source_df is None or self.target_df is None:
            raise ValueError("Datasets not loaded. Call load_datasets() first.")

        if self.config.verbose:
            print("\n" + "=" * 60)
            print("Phase 2: Chemical Space Analysis")
            print("=" * 60)

        # Get molecules
        if self.config.verbose:
            print("Converting SMILES to molecules...")
        source_mols = get_molecule_list(self.source_df, smiles_col=source_smiles_col)
        target_mols = get_molecule_list(self.target_df, smiles_col=target_smiles_col)

        # Morgan fingerprints
        if self.config.verbose:
            print("Computing Morgan fingerprint similarity...")
        self.morgan_stats = analyze_morgan_fingerprints(
            source_mols,
            target_mols,
            radius=morgan_radius,
            n_bits=morgan_nbits,
        )

        if self.config.verbose:
            print(f"  Source self-similarity: {self.morgan_stats.source_self_sim_mean:.3f}")
            print(f"  Target self-similarity: {self.morgan_stats.target_self_sim_mean:.3f}")
            print(f"  Cross similarity: {self.morgan_stats.cross_sim_mean:.3f}")
            print(
                f"  Target w/ similar source NN: "
                f"{self.morgan_stats.target_compounds_with_similar_source * 100:.1f}%"
            )

        # Scaffolds
        if self.config.verbose:
            print("Computing scaffold statistics...")
        self.scaffold_stats = analyze_scaffolds(source_mols, target_mols)

        if self.config.verbose:
            print(f"  Source scaffolds: {self.scaffold_stats.source_n_scaffolds}")
            print(f"  Target scaffolds: {self.scaffold_stats.target_n_scaffolds}")
            print(f"  Shared scaffolds: {self.scaffold_stats.shared_scaffolds}")
            print(
                f"  Scaffold Jaccard: {self.scaffold_stats.source_target_scaffold_sim:.3f}"
            )

        # Descriptors
        if self.config.verbose:
            print("Computing descriptor distributions...")
        self.descriptor_stats = analyze_descriptor_distributions(
            source_mols, target_mols
        )

        if self.config.verbose:
            print(f"  Analyzed {len(self.descriptor_stats)} descriptors")
            # Print top KS differences
            ks_vals = [
                (d.descriptor_name, d.ks_statistic)
                for d in self.descriptor_stats
                if not pd.isna(d.ks_statistic)
            ]
            ks_vals.sort(key=lambda x: x[1], reverse=True)
            if ks_vals:
                print(f"  Largest KS gaps:")
                for desc_name, ks_val in ks_vals[:3]:
                    print(f"    {desc_name}: {ks_val:.3f}")

    def get_report(self) -> dict:
        """Get Phase 2 report in dictionary format.

        Returns
        -------
        dict
            Complete Phase 2 analysis report
        """
        report = {
            "phase": 2,
            "stage": "Chemical Space Analysis",
            "morgan_fingerprints": (
                self.morgan_stats.to_dict() if self.morgan_stats else None
            ),
            "scaffolds": self.scaffold_stats.to_dict() if self.scaffold_stats else None,
            "descriptors": (
                [d.to_dict() for d in self.descriptor_stats]
                if self.descriptor_stats
                else None
            ),
        }
        return report

    def save_report(self, output_path: str | Path) -> None:
        """Save Phase 2 report to JSON file.

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

    def run(
        self,
        source_df: Optional[pd.DataFrame] = None,
        target_df: Optional[pd.DataFrame] = None,
        source_path: Optional[str | Path] = None,
        target_path: Optional[str | Path] = None,
        source_smiles_col: str = "smiles",
        target_smiles_col: str = "smiles",
        morgan_radius: int = 2,
        morgan_nbits: int = 2048,
    ) -> dict:
        """Run complete Phase 2 pipeline.

        Parameters
        ----------
        source_df : pd.DataFrame, optional
            Source dataframe (if None, load from source_path)
        target_df : pd.DataFrame, optional
            Target dataframe (if None, load from target_path)
        source_path : str | Path, optional
            Path to source CSV file
        target_path : str | Path, optional
            Path to target CSV file
        source_smiles_col : str
            SMILES column in source
        target_smiles_col : str
            SMILES column in target
        morgan_radius : int
            Morgan fingerprint radius
        morgan_nbits : int
            Number of bits

        Returns
        -------
        dict
            Complete Phase 2 report
        """
        # Load data
        if source_df is not None and target_df is not None:
            self.load_datasets(source_df, target_df)
        elif source_path is not None and target_path is not None:
            self.load_from_files(source_path, target_path, smiles_col=source_smiles_col)
        else:
            raise ValueError(
                "Either (source_df, target_df) or (source_path, target_path) must be provided"
            )

        # Analyze
        self.analyze_chemical_space(
            source_smiles_col=source_smiles_col,
            target_smiles_col=target_smiles_col,
            morgan_radius=morgan_radius,
            morgan_nbits=morgan_nbits,
        )

        return self.get_report()
