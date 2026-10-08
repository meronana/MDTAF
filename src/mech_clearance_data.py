"""Data loaders for mechanism-aware clearance model.

Key difference from script 15 approach:
- Load raw 'value' columns (NOT z-scored)
- Work in log-scale without within-source z-scoring
- IVIVE transformation operates in natural log-scale
- Z-score only applied for final metric comparison
"""
import os

import numpy as np
import pandas as pd


class ClearanceRegressionLoader:
    """Load clearance regression data in log-scale WITHOUT within-source z-scoring."""

    def __init__(self, data_dir="data/final_reg"):
        self.data_dir = data_dir
        self._eps = 1e-8

    def _log_transform(self, values):
        """Safe log10 transformation."""
        v = np.asarray(values, dtype=float)
        v = np.where(v > 0, v, self._eps)
        return np.log10(v)

    def load_raw_values(self, split, task="clearance"):
        """Load raw values in log-scale (NOT z-scored).

        Args:
            split: "source", "target_train", "target_val", "target_test"
            task: "clearance", "half_life", "fu"

        Returns:
            df: DataFrame with columns [smiles, value, y_log, y_zscored]
                - y_log: raw log10 transform of value (NO z-score)
                - y_zscored: z-scored version (from final_reg)
        """
        filepath = os.path.join(self.data_dir, f"{task}_{split}.csv")
        df = pd.read_csv(filepath)

        # Ensure we have both raw value and z-scored y
        if "value" not in df.columns or "y" not in df.columns:
            raise ValueError(f"Missing columns in {filepath}")

        # Create log-transformed column
        df["y_log"] = self._log_transform(df["value"])

        # Rename existing z-scored column
        df["y_zscored"] = df["y"]

        return df[["smiles", "value", "y_log", "y_zscored"]].copy()

    def compute_zscores(self, y_log_train):
        """Compute z-score statistics from training data."""
        mean = np.nanmean(y_log_train)
        std = np.nanstd(y_log_train)
        std = std if std > 1e-9 else 1.0
        return mean, std

    def apply_zscore(self, y_log, mean, std):
        """Apply z-score transformation."""
        return (y_log - mean) / std


# Convenience functions for common workflows

def load_clearance_raw(data_dir="data/final_reg"):
    """Load all clearance data in log-scale (not z-scored)."""
    loader = ClearanceRegressionLoader(data_dir)

    source = loader.load_raw_values("source", task="clearance")
    target_train = loader.load_raw_values("target_train", task="clearance")
    target_val = loader.load_raw_values("target_val", task="clearance")
    target_test = loader.load_raw_values("target_test", task="clearance")

    return {
        "source": source,
        "target_train": target_train,
        "target_val": target_val,
        "target_test": target_test,
        "loader": loader,
    }


def load_clearance_for_ivive(data_dir="data/final_reg"):
    """Load clearance data prepared for IVIVE training.

    Returns:
        dict with keys:
            - "source_y": (n_source,) in log-scale
            - "target_train_y": (n_train,) in log-scale
            - "target_val_y": (n_val,) in log-scale
            - "target_test_y": (n_test,) in log-scale
            - "target_zscale_mean": for converting back
            - "target_zscale_std": for converting back
    """
    data = load_clearance_raw(data_dir)

    # Use target_train to compute z-score statistics (NO LEAKAGE)
    target_train_y_log = data["target_train"]["y_log"].values
    zscale_mean, zscale_std = data["loader"].compute_zscores(target_train_y_log)

    return {
        "source_y": data["source"]["y_log"].values,
        "target_train_y": data["target_train"]["y_log"].values,
        "target_val_y": data["target_val"]["y_log"].values,
        "target_test_y": data["target_test"]["y_log"].values,
        "target_zscale_mean": zscale_mean,
        "target_zscale_std": zscale_std,
    }


if __name__ == "__main__":
    # Test the loader
    data = load_clearance_raw()
    print("Source shape:", data["source"].shape)
    print("Target train shape:", data["target_train"].shape)
    print("Target test shape:", data["target_test"].shape)

    print("\nSource y_log stats:")
    print(f"  mean={data['source']['y_log'].mean():.3f}, std={data['source']['y_log'].std():.3f}")

    print("\nTarget train y_log stats:")
    print(f"  mean={data['target_train']['y_log'].mean():.3f}, std={data['target_train']['y_log'].std():.3f}")

    print("\nTarget test y_log stats:")
    print(f"  mean={data['target_test']['y_log'].mean():.3f}, std={data['target_test']['y_log'].std():.3f}")

    # Check that value and y_log are consistent
    print("\nSample values:")
    sample_idx = 0
    val = data["source"]["value"].iloc[sample_idx]
    y_log = data["source"]["y_log"].iloc[sample_idx]
    expected_y_log = np.log10(val)
    print(f"  value={val:.3f}, y_log={y_log:.6f}, expected={expected_y_log:.6f}")
