"""Molecular preprocessing and standardization."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MolStandardize

RDLogger.DisableLog("rdApp.*")


@dataclass
class PreprocessingResult:
    """Result of molecular preprocessing."""

    canonical_smiles: str | None = None
    inchi_key: str | None = None
    mol: Chem.Mol | None = None
    is_valid: bool = False
    error_message: str = ""


def standardize_smiles(
    smiles: str, remove_salts: bool = True, keep_stereochemistry: bool = True
) -> PreprocessingResult:
    """Standardize and canonicalize a SMILES string.

    Parameters
    ----------
    smiles : str
        Input SMILES string
    remove_salts : bool
        Remove salt/solvent fragments
    keep_stereochemistry : bool
        Preserve stereochemical information

    Returns
    -------
    PreprocessingResult
        Standardized SMILES and molecular info
    """
    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return PreprocessingResult(error_message="Invalid SMILES")

        # Remove salts if requested
        if remove_salts:
            try:
                fc = MolStandardize.rdMolStandardize.LargestFragmentChooser()
                mol = fc.choose(mol)
            except Exception:
                pass

        # Get canonical SMILES
        if keep_stereochemistry:
            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        else:
            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)

        # Get InChIKey
        try:
            inchi_key = Chem.MolToInchiKey(mol)
        except Exception:
            inchi_key = None

        return PreprocessingResult(
            canonical_smiles=canonical_smiles,
            inchi_key=inchi_key,
            mol=mol,
            is_valid=True,
        )

    except Exception as e:
        return PreprocessingResult(error_message=str(e))


def preprocess_dataset(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    label_col: str = "label",
    remove_salts: bool = True,
    keep_stereochemistry: bool = True,
    deduplication_method: Literal["inchikey", "smiles", "none"] = "inchikey",
    duplicate_aggregation: Literal["first", "mean", "all"] = "mean",
    drop_invalid: bool = True,
) -> pd.DataFrame:
    """Preprocess molecular dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    smiles_col : str
        Column name for SMILES
    label_col : str
        Column name for labels
    remove_salts : bool
        Remove salt/solvent fragments
    keep_stereochemistry : bool
        Preserve stereochemistry
    deduplication_method : {"inchikey", "smiles", "none"}
        Method for identifying duplicates
    duplicate_aggregation : {"first", "mean", "all"}
        How to handle duplicate labels
    drop_invalid : bool
        Drop compounds that fail standardization

    Returns
    -------
    pd.DataFrame
        Preprocessed dataframe
    """
    # Make copy to avoid modifying original
    df = df.copy()

    # Standardize SMILES
    results = []
    for i, smiles in enumerate(df[smiles_col]):
        result = standardize_smiles(
            smiles, remove_salts=remove_salts, keep_stereochemistry=keep_stereochemistry
        )
        results.append(result)

    # Extract results
    df["_canonical_smiles"] = [r.canonical_smiles for r in results]
    df["_inchi_key"] = [r.inchi_key for r in results]
    df["_is_valid"] = [r.is_valid for r in results]

    # Drop invalid if requested
    if drop_invalid:
        n_before = len(df)
        df = df[df["_is_valid"]].copy()
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            print(f"Dropped {n_dropped} invalid compounds")
        # Drop _is_valid after filtering
        df = df.drop(columns=["_is_valid"])
    else:
        # Drop _is_valid either way
        if "_is_valid" in df.columns:
            df = df.drop(columns=["_is_valid"])

    # Deduplication
    if deduplication_method != "none":
        dedup_col = "_inchi_key" if deduplication_method == "inchikey" else "_canonical_smiles"

        if duplicate_aggregation == "first":
            df = df.drop_duplicates(subset=[dedup_col], keep="first").copy()
        elif duplicate_aggregation == "mean":
            # Aggregate labels for duplicates
            numeric_label_col = label_col
            df[numeric_label_col] = pd.to_numeric(df[numeric_label_col], errors="coerce")

            agg_dict = {numeric_label_col: "mean"}
            # Add other columns to keep (last value)
            for col in df.columns:
                if col not in [dedup_col, numeric_label_col, "_canonical_smiles", "_inchi_key"]:
                    agg_dict[col] = "last"

            df = df.groupby(dedup_col, as_index=False).agg(agg_dict)

        elif duplicate_aggregation == "all":
            # Keep all duplicates, add duplicate ID
            df["_duplicate_group"] = df.groupby(dedup_col, sort=False).ngroup()

    # Update SMILES column with canonical version
    if "_canonical_smiles" in df.columns:
        df[smiles_col] = df["_canonical_smiles"]
        df = df.drop(columns=["_canonical_smiles"])

    return df


def get_molecule_list(df: pd.DataFrame, smiles_col: str = "smiles"):
    """Get list of RDKit molecules from dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with preprocessed SMILES
    smiles_col : str
        Column name for SMILES

    Returns
    -------
    list[Chem.Mol | None]
        List of molecules (None if invalid)
    """
    molecules = []
    for smiles in df[smiles_col]:
        mol = Chem.MolFromSmiles(str(smiles))
        molecules.append(mol)
    return molecules
