"""Mechanism-aware clearance model with IVIVE-inspired transformation.

Core idea: Instead of forcing source-target alignment in latent space,
learn the biological transformation from in-vitro CLint-like measurements
to human systemic clearance using IVIVE principles.

Flow:
  Graphormer 768-d (frozen)
    ↓
  Encoder (768→512→256)
    ↓
  Two heads:
    - head_clint: 256 → 128 → 1 (log CLint)
    - head_fu: 256 → 128 → 1 (logit unbound fraction)
    ↓
  IVIVE transformation layer
    ↓
  log(human clearance)
"""
import torch
import torch.nn as nn


class IVIVETransformationLayer(nn.Module):
    """
    Differentiable IVIVE transformation based on well-stirred hepatic model.

    Original formula:
      CL_H = (Q_H × fu × CLint) / (Q_H + fu × CLint)

    Log-space version with learnable parameters:
      log(CL_H) = α × log(fu × CLint / (β + fu × CLint)) + γ

    Where:
      α (alpha):    Scaling factor for metabolic rate heterogeneity
      β (beta):     Normalized hepatic blood flow (default ≈ 1)
      γ (gamma):    Offset to account for non-hepatic clearance
    """

    def __init__(self, init_alpha=1.0, init_beta=1.0, init_gamma=0.0,
                 learnable_alpha=True, learnable_beta=True, learnable_gamma=True):
        super().__init__()

        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        else:
            self.register_buffer("alpha", torch.tensor(float(init_alpha)))

        if learnable_beta:
            self.beta = nn.Parameter(torch.tensor(float(init_beta)))
        else:
            self.register_buffer("beta", torch.tensor(float(init_beta)))

        if learnable_gamma:
            self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))
        else:
            self.register_buffer("gamma", torch.tensor(float(init_gamma)))

    def forward(self, log_clint, fu, compute_intermediate=False):
        """
        Args:
            log_clint: Log of intrinsic clearance (batch,) [log scale]
            fu: Unbound fraction (batch,) [0, 1] from sigmoid
            compute_intermediate: If True, return intermediate values

        Returns:
            log_cl_h: Log of hepatic clearance (batch,)
            (optional) intermediate dict
        """
        # Convert log_clint back to linear for IVIVE computation
        clint = torch.exp(torch.clamp(log_clint, min=-10, max=10))  # (batch,)

        # Clamp fu to valid range [0.01, 0.99] to avoid numerical issues
        fu = torch.clamp(fu, min=0.01, max=0.99)

        # Well-stirred hepatic model: CL_H = (Q_H * fu * CLint) / (Q_H + fu * CLint)
        # In log space: log(CL_H) = log(fu * CLint / (β + fu * CLint))
        #              = log(fu) + log(CLint) - log(β + fu * CLint)

        # Clamp beta to reasonable range to avoid division issues
        beta_clamped = torch.clamp(self.beta, min=0.1, max=10.0)

        # Compute ratio: (fu * CLint) / (β + fu * CLint)
        numerator = fu * clint  # (batch,)
        denominator = beta_clamped + fu * clint  # (batch,)

        # Avoid log(0) with small epsilon
        ratio = numerator / (denominator + 1e-8)
        ratio = torch.clamp(ratio, min=1e-8, max=1.0)  # Ensure valid range

        # IVIVE transformation: log(CL_H) = α * log(ratio) + γ
        log_ratio = torch.log(ratio)
        log_ratio = torch.where(torch.isnan(log_ratio), torch.zeros_like(log_ratio), log_ratio)
        log_ratio = torch.where(torch.isinf(log_ratio), torch.ones_like(log_ratio) * -20, log_ratio)

        log_cl_h = self.alpha * log_ratio + self.gamma
        log_cl_h = torch.where(torch.isnan(log_cl_h), torch.zeros_like(log_cl_h), log_cl_h)

        if compute_intermediate:
            return log_cl_h, {
                "clint": clint,
                "fu": fu,
                "numerator": numerator,
                "denominator": denominator,
                "ratio": ratio,
                "alpha": self.alpha.detach().cpu().item(),
                "beta": beta_clamped.detach().cpu().item(),
                "gamma": self.gamma.detach().cpu().item(),
            }
        return log_cl_h


class MechanismAwareClearanceModel(nn.Module):
    """
    Full mechanism-aware clearance prediction model.

    Architecture:
      Input: Frozen Graphormer embedding (768-d)
             ↓
      Encoder: 768 → 512 → 256 (trainable)
             ↓
      Split into two streams:
        - CLint prediction head: 256 → 128 → 1 (log scale)
        - Fu prediction head: 256 → 128 → 1 (logit scale, then sigmoid)
             ↓
      IVIVE transformation layer
             ↓
      Output: log(human clearance)
    """

    def __init__(self, d_in=768, d_hidden=256, d_final=128,
                 dropout=0.2, dropout_head=0.1, use_ivive=True,
                 ivive_learnable_alpha=True,
                 ivive_learnable_beta=True,
                 ivive_learnable_gamma=True):
        """
        Args:
            d_in: Input dimension (Graphormer embedding)
            d_hidden: Encoder hidden dimension
            d_final: Final layer before output
            dropout: Dropout in encoder
            dropout_head: Dropout in regression heads
            use_ivive: If True, use IVIVE transformation; else direct output
            ivive_learnable_*: Which IVIVE parameters are trainable
        """
        super().__init__()

        self.use_ivive = use_ivive

        # Shared encoder (Graphormer output → latent representation)
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, d_hidden), nn.ReLU()
        )

        # CLint regression head (predict log intrinsic clearance)
        self.head_clint = nn.Sequential(
            nn.Linear(d_hidden, d_final), nn.ReLU(), nn.Dropout(dropout_head),
            nn.Linear(d_final, 1)
        )

        # Fu regression head (predict logit unbound fraction)
        self.head_fu = nn.Sequential(
            nn.Linear(d_hidden, d_final), nn.ReLU(), nn.Dropout(dropout_head),
            nn.Linear(d_final, 1)
        )

        # IVIVE transformation layer
        if use_ivive:
            self.ivive = IVIVETransformationLayer(
                init_alpha=1.0,
                init_beta=1.0,
                init_gamma=0.0,
                learnable_alpha=ivive_learnable_alpha,
                learnable_beta=ivive_learnable_beta,
                learnable_gamma=ivive_learnable_gamma,
            )
        else:
            # Fallback: simple linear head
            self.head_direct = nn.Sequential(
                nn.Linear(d_hidden, d_final), nn.ReLU(), nn.Dropout(dropout_head),
                nn.Linear(d_final, 1)
            )

    def forward(self, x, compute_intermediate=False, return_all_outputs=False):
        """
        Args:
            x: Graphormer embedding (batch, 768)
            compute_intermediate: If True, return intermediate computations
            return_all_outputs: If True, return all head outputs

        Returns:
            y_pred: Predicted log(human clearance) (batch,)
            (optional) dict with intermediate values
        """
        # Encode
        z = self.encoder(x)  # (batch, d_hidden)

        # Predict CLint (log scale)
        log_clint = self.head_clint(z).squeeze(-1)  # (batch,)

        # Predict fu (unbounded logit, then sigmoid to [0,1])
        logit_fu = self.head_fu(z).squeeze(-1)  # (batch,)
        fu = torch.sigmoid(logit_fu)  # (batch,) in (0, 1)

        # Transform to clearance via IVIVE
        if self.use_ivive:
            y_pred, intermediate = self.ivive(
                log_clint, fu, compute_intermediate=True
            )
        else:
            y_pred = self.head_direct(z).squeeze(-1)
            intermediate = {}

        # Build output dict if requested
        if compute_intermediate or return_all_outputs:
            output_dict = {
                "y_pred": y_pred,
                "log_clint": log_clint,
                "logit_fu": logit_fu,
                "fu": fu,
            }
            if intermediate:
                output_dict.update(intermediate)
            return output_dict

        return y_pred

    def get_ivive_params(self):
        """Return current IVIVE parameter values."""
        if self.use_ivive:
            return {
                "alpha": self.ivive.alpha.detach().cpu().item(),
                "beta": self.ivive.beta.detach().cpu().item(),
                "gamma": self.ivive.gamma.detach().cpu().item(),
            }
        return None


# Variants for ablation studies

class IVIVEWithoutFu(nn.Module):
    """IVIVE model where fu is fixed or not used."""

    def __init__(self, d_in=768, d_hidden=256, d_final=128,
                 dropout=0.2, dropout_head=0.1, fixed_fu=0.5):
        super().__init__()

        self.fixed_fu = fixed_fu

        self.encoder = nn.Sequential(
            nn.Linear(d_in, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, d_hidden), nn.ReLU()
        )

        self.head_clint = nn.Sequential(
            nn.Linear(d_hidden, d_final), nn.ReLU(), nn.Dropout(dropout_head),
            nn.Linear(d_final, 1)
        )

        self.ivive = IVIVETransformationLayer(
            init_alpha=1.0,
            init_beta=1.0,
            init_gamma=0.0,
        )

    def forward(self, x):
        z = self.encoder(x)
        log_clint = self.head_clint(z).squeeze(-1)
        fu_fixed = torch.full_like(log_clint, self.fixed_fu)
        y_pred = self.ivive(log_clint, fu_fixed)
        return y_pred


class DirectRegressionModel(nn.Module):
    """Baseline: Direct regression without IVIVE (for comparison)."""

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

    def forward(self, x):
        z = self.encoder(x)
        y_pred = self.head(z).squeeze(-1)
        return y_pred
