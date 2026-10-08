"""Parametric scaling model: Simpler alternative to full IVIVE.

Instead of trying to learn CLint and fu separately, learn a direct
parametric transformation from in-vitro to in-vivo clearance.

Model:
    log(CL_human) = α × log(CL_invitro) + β + ε_encoder

Where:
    α: scaling factor (physiological: typically 0.5-1.0 for IVIVE scaling)
    β: offset (physiological: log of non-metabolic clearance)
    ε_encoder: learned feature transformation via encoder

This is "IVIVE-inspired" but empirical - no explicit CLint modeling.
"""
import torch
import torch.nn as nn


class ParametricScaleModel(nn.Module):
    """Parametric scaling model with learned encoder.

    Flow:
        Graphormer 768-d (frozen)
            ↓
        Encoder: 768→512→256 (learns feature transformation)
            ↓
        Two-head output:
            - Main head: 256→128→1 (correction term)
            - Scale head: 256→128→1 (learns α, β)
            ↓
        Parametric transformation:
            α × log(source_value) + β + correction_term
    """

    def __init__(self, d_in=768, d_hidden=256, d_final=128,
                 dropout=0.2, dropout_head=0.1,
                 init_alpha=1.0, init_beta=0.0):
        super().__init__()

        self.register_buffer("init_alpha_val", torch.tensor(float(init_alpha)))
        self.register_buffer("init_beta_val", torch.tensor(float(init_beta)))

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, d_hidden), nn.ReLU()
        )

        # Correction head: learns residual correction to parametric scaling
        self.correction_head = nn.Sequential(
            nn.Linear(d_hidden, d_final), nn.ReLU(), nn.Dropout(dropout_head),
            nn.Linear(d_final, 1)
        )

        # Scale parameter head: learns α and β
        self.alpha_head = nn.Sequential(
            nn.Linear(d_hidden, d_final), nn.ReLU(), nn.Dropout(dropout_head),
            nn.Linear(d_final, 1)
        )

        self.beta_head = nn.Sequential(
            nn.Linear(d_hidden, d_final), nn.ReLU(), nn.Dropout(dropout_head),
            nn.Linear(d_final, 1)
        )

        # Learnable global scale factors with initialization
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(float(init_alpha))))
        self.beta = nn.Parameter(torch.tensor(float(init_beta)))

    def forward(self, x, source_y_log=None, use_global_params=True,
                return_intermediate=False):
        """
        Args:
            x: Graphormer embedding (batch, 768)
            source_y_log: Source values in log-scale (batch,) for parametric transform
            use_global_params: If True, use learnable α,β. If False, predict per-sample.
            return_intermediate: If True, return intermediate values

        Returns:
            y_pred: Predicted log(human CL)
            (optional) dict with intermediate values
        """
        # Encode
        z = self.encoder(x)  # (batch, d_hidden)

        # Correction term
        correction = self.correction_head(z).squeeze(-1)  # (batch,)

        if use_global_params and source_y_log is not None:
            # Use global learnable parameters: log(CL_H) = α × log(CL_inv) + β + correction
            alpha = torch.exp(self.log_alpha)
            y_pred = alpha * source_y_log + self.beta + correction

            if return_intermediate:
                return y_pred, {
                    "alpha": alpha.detach().cpu().item(),
                    "beta": self.beta.detach().cpu().item(),
                    "correction": correction,
                    "alpha_learned_head": None,
                    "beta_learned_head": None,
                }
        else:
            # Predict α and β per-sample
            alpha_pred = torch.exp(self.alpha_head(z).squeeze(-1))  # (batch,)
            beta_pred = self.beta_head(z).squeeze(-1)  # (batch,)

            if source_y_log is not None:
                y_pred = alpha_pred * source_y_log + beta_pred + correction
            else:
                # Fallback if no source values
                y_pred = correction

            if return_intermediate:
                return y_pred, {
                    "alpha_sample_mean": alpha_pred.mean().detach().cpu().item(),
                    "beta_sample_mean": beta_pred.mean().detach().cpu().item(),
                    "correction": correction,
                }

        return y_pred

    def get_params(self):
        """Return current scale parameters."""
        alpha = torch.exp(self.log_alpha).detach().cpu().item()
        beta = self.beta.detach().cpu().item()
        return {
            "alpha": alpha,
            "beta": beta,
            "log_alpha_raw": self.log_alpha.detach().cpu().item(),
        }


class SimpleDirectModel(nn.Module):
    """Baseline: Direct regression without any parametric scaling.

    Just for comparison to see if parametric scaling adds value.
    """

    def __init__(self, d_in=768, d_hidden=256, d_final=128,
                 dropout=0.2, dropout_head=0.1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(d_in, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, d_hidden), nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(d_hidden, d_final), nn.ReLU(), nn.Dropout(dropout_head),
            nn.Linear(d_final, 1)
        )

    def forward(self, x, **kwargs):
        z = self.encoder(x)
        y_pred = self.head(z).squeeze(-1)
        return y_pred


class SourceSupervisionModel(nn.Module):
    """Learn CLint prediction from source data.

    Use source measurements to learn an explicit CLint predictor,
    then apply parametric IVIVE transformation.

    This requires training two stages:
    1. Source: predict source labels
    2. Target: use learned source model + apply transformation
    """

    def __init__(self, d_in=768, d_hidden=256, d_final=128,
                 dropout=0.2, dropout_head=0.1):
        super().__init__()

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, d_hidden), nn.ReLU()
        )

        # CLint prediction head (trained on source)
        self.clint_head = nn.Sequential(
            nn.Linear(d_hidden, d_final), nn.ReLU(), nn.Dropout(dropout_head),
            nn.Linear(d_final, 1)
        )

        # Scaling parameters
        self.log_alpha = nn.Parameter(torch.tensor(0.0))  # log(1) = 0
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        z = self.encoder(x)
        log_clint = self.clint_head(z).squeeze(-1)
        return log_clint  # Returns log_CLint, not log_CL_H

    def predict_target_cl(self, x):
        """Predict human CL using learned transformation."""
        z = self.encoder(x)
        log_clint = self.clint_head(z).squeeze(-1)

        # Apply parametric IVIVE
        alpha = torch.exp(self.log_alpha)
        log_cl_h = alpha * log_clint + self.beta

        return log_cl_h

    def get_params(self):
        alpha = torch.exp(self.log_alpha).detach().cpu().item()
        beta = self.beta.detach().cpu().item()
        return {
            "alpha": alpha,
            "beta": beta,
        }
