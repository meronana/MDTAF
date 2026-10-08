"""Chemical space analysis: fingerprints, scaffolds, descriptors."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Scaffolds
from scipy import stats

RDLogger.DisableLog("rdApp.*")


@dataclass
class MorganFingerprintStats:
    """Morgan fingerprint similarity statistics."""

    source_self_sim_mean: float = np.nan
    target_self_sim_mean: float = np.nan
    cross_sim_mean: float = np.nan
    cross_sim_std: float = np.nan

    source_target_nearest_mean: float = np.nan
    target_source_nearest_mean: float = np.nan

    target_compounds_with_similar_source: float = np.nan
    """Fraction of target compounds with source NN similarity > 0.5"""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "source_self_sim_mean": float(self.source_self_sim_mean),
            "target_self_sim_mean": float(self.target_self_sim_mean),
            "cross_sim_mean": float(self.cross_sim_mean),
            "cross_sim_std": float(self.cross_sim_std),
            "source_target_nearest_mean": float(self.source_target_nearest_mean),
            "target_source_nearest_mean": float(self.target_source_nearest_mean),
            "target_with_similar_source_pct": float(self.target_compounds_with_similar_source * 100),
        }


@dataclass
class ScaffoldStats:
    """Bemis-Murcko scaffold statistics."""

    source_n_scaffolds: int = 0
    target_n_scaffolds: int = 0
    shared_scaffolds: int = 0
    source_unique_scaffolds: int = 0
    target_unique_scaffolds: int = 0

    source_unique_pct: float = np.nan
    target_unique_pct: float = np.nan
    shared_pct: float = np.nan

    source_target_scaffold_sim: float = np.nan
    """Jaccard similarity of scaffold sets"""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "source_n_scaffolds": self.source_n_scaffolds,
            "target_n_scaffolds": self.target_n_scaffolds,
            "shared_scaffolds": self.shared_scaffolds,
            "source_unique_scaffolds": self.source_unique_scaffolds,
            "target_unique_scaffolds": self.target_unique_scaffolds,
            "source_unique_pct": float(self.source_unique_pct),
            "target_unique_pct": float(self.target_unique_pct),
            "shared_pct": float(self.shared_pct),
            "source_target_scaffold_jaccard": float(self.source_target_scaffold_sim),
        }


@dataclass
class DescriptorStats:
    """Descriptor distribution gap statistics."""

    descriptor_name: str = ""
    source_mean: float = np.nan
    target_mean: float = np.nan
    source_std: float = np.nan
    target_std: float = np.nan
    ks_statistic: float = np.nan
    ks_pvalue: float = np.nan
    wasserstein_distance: float = np.nan
    standardized_mean_diff: float = np.nan

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "descriptor": self.descriptor_name,
            "source_mean": float(self.source_mean),
            "target_mean": float(self.target_mean),
            "source_std": float(self.source_std),
            "target_std": float(self.target_std),
            "ks_statistic": float(self.ks_statistic),
            "ks_pvalue": float(self.ks_pvalue),
            "wasserstein_distance": float(self.wasserstein_distance),
            "standardized_mean_diff": float(self.standardized_mean_diff),
        }


def get_morgan_fingerprints(
    mols_list: list[Chem.Mol | None],
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Get Morgan fingerprints for molecules.

    Parameters
    ----------
    mols_list : list[Chem.Mol | None]
        List of RDKit molecules
    radius : int
        Morgan radius
    n_bits : int
        Number of bits in fingerprint

    Returns
    -------
    np.ndarray
        Array of shape (N, n_bits) with fingerprints as binary vectors
    """
    fingerprints = []
    for mol in mols_list:
        if mol is None:
            fingerprints.append(np.zeros(n_bits, dtype=np.uint8))
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros(n_bits, dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fingerprints.append(arr)
    return np.array(fingerprints)


def compute_pairwise_similarity(fps1: np.ndarray, fps2: np.ndarray) -> np.ndarray:
    """Compute Tanimoto similarity between two sets of fingerprints using bitwise operations.

    Parameters
    ----------
    fps1 : np.ndarray
        Fingerprints of shape (N1, n_bits), binary values
    fps2 : np.ndarray
        Fingerprints of shape (N2, n_bits), binary values

    Returns
    -------
    np.ndarray
        Similarity matrix of shape (N1, N2)
    """
    # Ensure binary
    fps1 = fps1.astype(bool).astype(int)
    fps2 = fps2.astype(bool).astype(int)

    N1, N2 = fps1.shape[0], fps2.shape[0]
    similarities = np.zeros((N1, N2), dtype=np.float32)

    for i in range(N1):
        for j in range(N2):
            # Tanimoto = (A AND B) / (A OR B)
            intersection = np.sum(fps1[i] & fps2[j])
            union = np.sum(fps1[i] | fps2[j])

            if union > 0:
                similarities[i, j] = intersection / union
            else:
                similarities[i, j] = 0.0

    return similarities


def analyze_morgan_fingerprints(
    source_mols: list[Chem.Mol | None],
    target_mols: list[Chem.Mol | None],
    radius: int = 2,
    n_bits: int = 2048,
) -> MorganFingerprintStats:
    """Analyze Morgan fingerprint similarity between source and target.

    Parameters
    ----------
    source_mols : list[Chem.Mol | None]
        Source molecules
    target_mols : list[Chem.Mol | None]
        Target molecules
    radius : int
        Morgan radius
    n_bits : int
        Number of bits

    Returns
    -------
    MorganFingerprintStats
        Morgan fingerprint statistics
    """
    stats_result = MorganFingerprintStats()

    # Get fingerprints
    source_fps = get_morgan_fingerprints(source_mols, radius=radius, n_bits=n_bits)
    target_fps = get_morgan_fingerprints(target_mols, radius=radius, n_bits=n_bits)

    # Within-dataset similarity
    if len(source_fps) > 1:
        source_sim = compute_pairwise_similarity(source_fps, source_fps)
        # Exclude diagonal
        source_sim_upper = source_sim[np.triu_indices_from(source_sim, k=1)]
        stats_result.source_self_sim_mean = float(np.mean(source_sim_upper))

    if len(target_fps) > 1:
        target_sim = compute_pairwise_similarity(target_fps, target_fps)
        # Exclude diagonal
        target_sim_upper = target_sim[np.triu_indices_from(target_sim, k=1)]
        stats_result.target_self_sim_mean = float(np.mean(target_sim_upper))

    # Cross-dataset similarity
    cross_sim = compute_pairwise_similarity(source_fps, target_fps)
    stats_result.cross_sim_mean = float(np.mean(cross_sim))
    stats_result.cross_sim_std = float(np.std(cross_sim))

    # Nearest neighbor similarity
    if cross_sim.shape[0] > 0:
        source_nn = np.max(cross_sim, axis=1)  # Best target match for each source
        stats_result.source_target_nearest_mean = float(np.mean(source_nn))

    if cross_sim.shape[1] > 0:
        target_nn = np.max(cross_sim, axis=0)  # Best source match for each target
        stats_result.target_source_nearest_mean = float(np.mean(target_nn))
        # Fraction with similarity > 0.5
        stats_result.target_compounds_with_similar_source = float(
            np.mean(target_nn > 0.5)
        )

    return stats_result


def get_bemis_murcko_scaffold(mol: Chem.Mol | None) -> str | None:
    """Get Bemis-Murcko scaffold SMILES.

    Parameters
    ----------
    mol : Chem.Mol | None
        Molecule

    Returns
    -------
    str | None
        Scaffold SMILES or None if invalid
    """
    if mol is None:
        return None
    try:
        scaffold = Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return None


def analyze_scaffolds(
    source_mols: list[Chem.Mol | None],
    target_mols: list[Chem.Mol | None],
) -> ScaffoldStats:
    """Analyze Bemis-Murcko scaffold overlap.

    Parameters
    ----------
    source_mols : list[Chem.Mol | None]
        Source molecules
    target_mols : list[Chem.Mol | None]
        Target molecules

    Returns
    -------
    ScaffoldStats
        Scaffold statistics
    """
    stats_result = ScaffoldStats()

    # Get scaffolds
    source_scaffolds = set()
    for mol in source_mols:
        scaffold = get_bemis_murcko_scaffold(mol)
        if scaffold is not None:
            source_scaffolds.add(scaffold)

    target_scaffolds = set()
    for mol in target_mols:
        scaffold = get_bemis_murcko_scaffold(mol)
        if scaffold is not None:
            target_scaffolds.add(scaffold)

    stats_result.source_n_scaffolds = len(source_scaffolds)
    stats_result.target_n_scaffolds = len(target_scaffolds)

    # Overlap
    shared = source_scaffolds & target_scaffolds
    stats_result.shared_scaffolds = len(shared)

    # Unique
    stats_result.source_unique_scaffolds = len(source_scaffolds - target_scaffolds)
    stats_result.target_unique_scaffolds = len(target_scaffolds - source_scaffolds)

    # Percentages
    total_source = len(source_scaffolds)
    total_target = len(target_scaffolds)

    if total_source > 0:
        stats_result.source_unique_pct = (
            stats_result.source_unique_scaffolds / total_source
        )
    if total_target > 0:
        stats_result.target_unique_pct = (
            stats_result.target_unique_scaffolds / total_target
        )

    # Jaccard similarity
    total_union = len(source_scaffolds | target_scaffolds)
    if total_union > 0:
        stats_result.source_target_scaffold_sim = (
            len(shared) / total_union
        )
        stats_result.shared_pct = len(shared) / total_union

    return stats_result


def compute_descriptors(
    mols: list[Chem.Mol | None],
) -> dict[str, np.ndarray]:
    """Compute molecular descriptors.

    Parameters
    ----------
    mols : list[Chem.Mol | None]
        Molecules

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping descriptor names to arrays
    """
    descriptor_functions = {
        "MW": Descriptors.MolWt,
        "logP": Crippen.MolLogP,
        "TPSA": Descriptors.TPSA,
        "HBD": Descriptors.NumHDonors,
        "HBA": Descriptors.NumHAcceptors,
        "RotBonds": Descriptors.NumRotatableBonds,
        "FracCSP3": Descriptors.FractionCSP3,
        "AromaticRings": Descriptors.NumAromaticRings,
        "HeavyAtomCount": Descriptors.HeavyAtomCount,
        "RingCount": Descriptors.RingCount,
    }

    descriptors = {name: [] for name in descriptor_functions}

    for mol in mols:
        if mol is None:
            for name in descriptor_functions:
                descriptors[name].append(np.nan)
        else:
            for name, func in descriptor_functions.items():
                try:
                    value = func(mol)
                    descriptors[name].append(float(value))
                except Exception:
                    descriptors[name].append(np.nan)

    # Convert to numpy arrays and handle NaNs
    for name in descriptors:
        arr = np.array(descriptors[name])
        # Fill NaNs with column mean
        if np.any(np.isnan(arr)):
            col_mean = np.nanmean(arr)
            arr[np.isnan(arr)] = col_mean if not np.isnan(col_mean) else 0.0
        descriptors[name] = arr

    return descriptors


def analyze_descriptor_distributions(
    source_mols: list[Chem.Mol | None],
    target_mols: list[Chem.Mol | None],
) -> list[DescriptorStats]:
    """Analyze descriptor distribution gaps.

    Parameters
    ----------
    source_mols : list[Chem.Mol | None]
        Source molecules
    target_mols : list[Chem.Mol | None]
        Target molecules

    Returns
    -------
    list[DescriptorStats]
        Descriptor statistics for each descriptor
    """
    source_desc = compute_descriptors(source_mols)
    target_desc = compute_descriptors(target_mols)

    results = []
    for desc_name in source_desc.keys():
        source_vals = source_desc[desc_name]
        target_vals = target_desc[desc_name]

        # Skip if no valid values
        if len(source_vals) == 0 or len(target_vals) == 0:
            continue

        stats_result = DescriptorStats(descriptor_name=desc_name)
        stats_result.source_mean = float(np.mean(source_vals))
        stats_result.target_mean = float(np.mean(target_vals))
        stats_result.source_std = float(np.std(source_vals))
        stats_result.target_std = float(np.std(target_vals))

        # KS test
        try:
            ks_stat, ks_pval = stats.ks_2samp(source_vals, target_vals)
            stats_result.ks_statistic = float(ks_stat)
            stats_result.ks_pvalue = float(ks_pval)
        except Exception:
            pass

        # Wasserstein distance
        try:
            w_dist = stats.wasserstein_distance(source_vals, target_vals)
            stats_result.wasserstein_distance = float(w_dist)
        except Exception:
            pass

        # Standardized mean difference (Cohen's d equivalent)
        try:
            pooled_std = np.sqrt(
                (stats_result.source_std**2 + stats_result.target_std**2) / 2
            )
            if pooled_std > 1e-8:
                smd = (stats_result.source_mean - stats_result.target_mean) / pooled_std
                stats_result.standardized_mean_diff = float(smd)
        except Exception:
            pass

        results.append(stats_result)

    return results
