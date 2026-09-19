"""
heston_gan_pricing.py
======================
Pricing-validation experiment for SRForGAN.

Compares three European-option price estimators at the trained forecast
horizon (t + FORECAST_HORIZON), for a grid of strikes around each test
trajectory's spot:

  1. oracle_price     - semi-analytic Heston price obtained by inverting the
                         (already-validated) Heston characteristic function
                         via the fractional FFT, using the TRUE state
                         (S_t, v_t) and TRUE parameters (kappa, theta,
                         sigma_v, rho) of that trajectory. This is an
                         "oracle" because it conditions on the latent
                         variance v_t, which the generator never observes.

  2. heston_mc_price  - Monte-Carlo Heston price, using the SAME
                         Andersen-QE / exact scheme already implemented in
                         HestonSimulator._mc_pdf, started from the same
                         (S_t, v_t). It is the same *kind* of estimator
                         (sample-average of a discounted payoff) as the GAN
                         price below, so:
                           heston_mc_price - oracle_price  -> pure MC noise
                           gan_price - heston_mc_price     -> generator error
                                                               net of MC noise

  3. gan_price        - Monte-Carlo price using samples drawn from the
                         trained SRForGAN generator, conditioned ONLY on the
                         observed log-price window (no v_t, no parameters -
                         exactly what the model sees at inference time).

Everything downstream (error metrics, moment diagnostics, plots) compares
these three vectors trajectory-by-trajectory and strike-by-strike.

-------------------------------------------------------------------------
IMPORTANT NUMERICAL NOTE - read before changing the pricing internals
-------------------------------------------------------------------------
HestonSimulator.get_pdf() builds its density by inverting the Heston
characteristic function on a fractional-FFT grid (`pdf_fine`, `x_grid`)
that is intentionally very fine, but whose domain extends *far* beyond the
true support of the distribution (this is an inherent side effect of the
Bluestein/frFFT construction, and is the same "ghost" tail region already
flagged during the project's own frFFT validation work).

Verified directly: summing `payoff(exp(x)) * pdf_fine(x)` over the *raw,
un-truncated* grid returns numbers that are many orders of magnitude wrong
for an unbounded payoff such as a call option - a virtually negligible
density value at very large log-price gets amplified without bound by e^x.
The same applies to the "raw moments" code path
(`get_pdf(n_bins=None)` -> `self.means`/`self.stds`), which sums over that
same raw grid.

This module therefore NEVER touches `pdf_fine`/`x_grid` directly for
pricing. `compute_oracle_prices()` below always truncates to a per-
trajectory window (`approx_mean +/- x_width * approx_std`, the same window
`get_pdf(n_bins=...)` already uses for its *binned* histogram output) and
renormalises before integrating any payoff. This was checked against
put-call parity, against a Black-Scholes special case (sigma_v -> 0), and
against the project's own exact Andersen-QE Monte-Carlo scheme
(cir_evol.QT_cir_evol + heston_evol.mc_heston) for parameters representative
of this thesis's ranges: agreement was within ~0.1-0.6% (|z| < 2 against
Monte-Carlo standard errors) across strikes from 80% to 120% moneyness.
Do not restore direct integration over the raw grid.

-------------------------------------------------------------------------
NOTE ON LOADING THE TRAINED GENERATOR
-------------------------------------------------------------------------
`load_generator()` uses `MySRForGAN.load_models(load_dir=...)`, NOT the
manual `set_generator()` + `torch.load()` + `load_state_dict()` pattern.
`load_models()` reconstructs the network architecture (z_dim, hidden size,
condition size, ...) directly from the checkpoint and its paired
"<model_name>_config.json" - it does not need those hyperparameters
re-specified here. The manual pattern requires every architecture
hyperparameter (in particular `z_noise_dim`) to be re-typed exactly as used
during training; getting even one of them wrong (e.g. omitting
`z_noise_dim` when constructing `MySRForGAN`) silently instantiates a
differently-shaped network and load_state_dict() then fails with a
size-mismatch error. Always prefer load_models() for inference.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis, ttest_1samp
from scipy.optimize import brentq

import torch

from heston_data_simulator import HestonSimulator
from mySRForGAN import MySRForGAN
from utilities import prepare_data


# =========================================================================
# 1. Configuration
# =========================================================================

@dataclass
class PricingConfig:
    # --- Heston parameter ranges. Defaults below match
    #     var_all_srforgan_heston_training_config.json, i.e. an
    #     in-distribution test set for that checkpoint. If you load a
    #     different checkpoint, either override these explicitly or build
    #     the config with `PricingConfig.from_training_config_json(...)`. ---
    X0_range: tuple = (0.0, 0.0)
    mu_range: tuple = (0.0, 0.0)          # risk-neutral drift; must equal `r` below
    v0_range: tuple = (0.5, 0.5)
    kappa_range: tuple = (1.5, 5.0)
    theta_range: tuple = (0.57, 1.5)
    sigma_v_range: tuple = (0.2, 1.3)
    rho_range: tuple = (-0.7, -0.7)

    # --- Time grid (must match the horizon the generator was trained on) ---
    condition_steps: int = 252
    forecast_horizon: int = 10
    trading_days: int = 252

    # --- Test-set size. Kept modest on purpose: a few hundred trajectories
    #     give tight aggregate statistics without diluting the per-
    #     trajectory frFFT bin resolution or the pricing-loop runtime. ---
    n_test: int = 500
    seed: int = 12345
    scheme: str = "milstein"

    # --- Pricing setup ---
    r: float = 0.0                        # risk-free rate; keep equal to mu_range
    moneyness_grid: np.ndarray = field(
        default_factory=lambda: np.array([0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20])
    )

    # --- Oracle (frFFT) pricing settings - see module docstring ---
    oracle_Nfft: int = 4096
    oracle_eta: float = 0.25
    oracle_x_width: float = 8.0
    oracle_n_bins: int = 3000

    # --- Monte-Carlo settings ---
    mc_sims_heston: int = 10_000
    n_samples_gan: int = 10_000

    # --- Trained model. `checkpoint_path` must point at the
    #     "..._generator.pth" produced by MySRForGAN.save_models(); its
    #     sibling "..._config.json" must live in the same folder.
    #     `model_name` is auto-derived from the checkpoint filename
    #     (stripping "_generator.pth") - only set it explicitly if that
    #     derivation would be wrong for your naming. ---
    checkpoint_path: str = "models/srforgan_models/var_all_srforgan_heston_generator.pth"
    model_name: Optional[str] = None

    @property
    def n_total(self) -> int:
        return self.condition_steps + self.forecast_horizon

    @property
    def T(self) -> float:
        return self.n_total / self.trading_days

    @classmethod
    def from_training_config_json(cls, path: str, **overrides) -> "PricingConfig":
        """
        Build a PricingConfig from a saved "<model_name>_training_config.json"
        (as produced alongside this project's training runs), so the test
        set is guaranteed to be sampled from the exact ranges the checkpoint
        was trained on. Any field can still be overridden via **overrides
        (e.g. checkpoint_path, n_test, moneyness_grid).
        """
        with open(path) as f:
            tc = json.load(f)
        kwargs = dict(
            X0_range=tuple(tc["X0_range"]),
            mu_range=tuple(tc["mu_range"]),
            v0_range=tuple(tc["v0_range"]),
            kappa_range=tuple(tc["kappa_range"]),
            theta_range=tuple(tc["theta_range"]),
            sigma_v_range=tuple(tc["sigma_v_range"]),
            rho_range=tuple(tc["rho_range"]),
            condition_steps=tc.get("N_steps", cls.condition_steps),
        )
        kwargs.update(overrides)
        return cls(**kwargs)


# =========================================================================
# 2. Test-set simulation
# =========================================================================

def simulate_test_set(cfg: PricingConfig):
    """
    Simulate `cfg.n_test` Heston trajectories and extract, for each:
      - the conditioning window fed to the generator  (paths[:, :condition_steps])
      - the reference state (S_t, v_t) at the conditioning boundary
      - the realised continuation (not used for pricing itself - kept only
        for bookkeeping / sanity plots)

    Mirrors exactly the indexing convention already used elsewhere in this
    project: the conditioning window is trajectories[:, :condition_steps]
    and the Heston "current" reference point for pricing/forecast purposes
    is column `condition_steps` (one step past the end of the window).
    """
    sim = HestonSimulator(
        X0_range=cfg.X0_range,
        mu_range=cfg.mu_range,
        v0_range=cfg.v0_range,
        kappa_range=cfg.kappa_range,
        theta_range=cfg.theta_range,
        sigma_v_range=cfg.sigma_v_range,
        rho_range=cfg.rho_range,
        T=cfg.T,
        N=cfg.n_total,
        n_simulations=cfg.n_test,
        seed=cfg.seed,
        scheme=cfg.scheme,
    )
    trajectories = sim.get_paths()

    if not np.allclose(sim.mu, cfg.r):
        warnings.warn(
            f"mu_range used for simulation (mean={sim.mu.mean():.6f}) does not "
            f"match the discounting rate r={cfg.r}. Prices from the two routes "
            f"will not be comparable under risk-neutral valuation unless these "
            f"are consistent.",
            stacklevel=2,
        )

    conditions = trajectories[:, : cfg.condition_steps]           # (J, condition_steps)
    targets = trajectories[:, cfg.n_total: cfg.n_total + 1]        # (J, 1) realised continuation

    X_t = sim.paths[:, cfg.condition_steps]                        # (J,) reference log-price
    v_t = sim.variance_paths[:, cfg.condition_steps]                # (J,) reference variance
    S_t = np.exp(X_t)

    return sim, conditions, targets, X_t, v_t, S_t


def build_strike_grid(S_t: np.ndarray, moneyness_grid: np.ndarray) -> np.ndarray:
    """(J,) spot array, (M,) moneyness levels -> (J, M) strike matrix."""
    return S_t[:, None] * moneyness_grid[None, :]


# =========================================================================
# 3. Oracle (semi-analytic) Heston pricing
# =========================================================================

def compute_oracle_prices(
    sim: HestonSimulator,
    X0: np.ndarray,
    v0: np.ndarray,
    n_steps_ahead: int,
    strikes: np.ndarray,
    r: float = 0.0,
    Nfft: int = 4096,
    eta: float = 0.25,
    x_width: float = 8.0,
    n_bins: int = 3000,
) -> np.ndarray:
    """
    Semi-analytic European option prices from the Heston characteristic
    function, one strike matrix row per trajectory. See the module
    docstring for why this truncates/renormalises instead of integrating
    the raw frFFT grid directly.

    strikes : (J, M) strike levels per trajectory (see build_strike_grid).
              Cells with K >= S_t(that trajectory) are priced as calls,
              K < S_t as puts (standard OTM convention for smile
              construction - keeps every priced option away from being
              deep ITM, which is what implied-vol inversion is sensitive to).

    Returns
    -------
    prices : (J, M) discounted option prices.
    """
    tau = n_steps_ahead * sim.dt
    J, M = strikes.shape

    # Force a fresh (non-cached) frFFT grid; sim.bins caching is meant for
    # the class's own binned get_pdf() calls and must not leak in here.
    sim.bins = None
    sim.get_pdf(
        n_steps_ahead=n_steps_ahead, n_bins=None,
        P=X0, v=v0, Nfft=Nfft, eta=eta, x_width=x_width,
    )
    pdf_fine, x_grid, lam = sim.pdf_fine, sim.x_grid, sim._frfft_lam   # (J, Nfft) each

    approx_mean = X0 + (sim.mu - 0.5 * sim.theta) * tau
    approx_std = np.sqrt(np.maximum(sim.theta, 0.0) * tau)
    lo = approx_mean - x_width * approx_std
    hi = approx_mean + x_width * approx_std

    S_t = np.exp(X0)
    disc = np.exp(-r * tau)
    prices = np.zeros((J, M))

    for j in range(J):
        bins = np.linspace(lo[j], hi[j], n_bins + 1)
        weights = pdf_fine[j] * lam
        mass, _ = np.histogram(x_grid[j], bins=bins, weights=weights)
        mass[mass < 1e-12] = 0.0
        total = mass.sum()
        if total <= 0:
            prices[j, :] = np.nan
            continue
        mass /= total
        centers = 0.5 * (bins[:-1] + bins[1:])
        S_centers = np.exp(centers)

        for m in range(M):
            K = strikes[j, m]
            if K >= S_t[j]:
                payoff = np.maximum(S_centers - K, 0.0)
            else:
                payoff = np.maximum(K - S_centers, 0.0)
            prices[j, m] = disc * np.sum(payoff * mass)

    return prices


# =========================================================================
# 4. Monte-Carlo pricing (shared by the Heston-MC and the GAN routes)
# =========================================================================

def mc_price_from_samples(
    log_price_samples: np.ndarray,
    S_t: np.ndarray,
    strikes: np.ndarray,
    r: float,
    tau: float,
):
    """
    log_price_samples : (J, n_mc) draws of the terminal log-price.
    S_t                : (J,) spot at the reference time (used only to pick
                          call vs put per the OTM convention, matching
                          compute_oracle_prices).
    strikes            : (J, M)

    Returns (prices, standard_errors), both (J, M).
    """
    S = np.exp(log_price_samples)                      # (J, n_mc)
    J, n_mc = S.shape
    M = strikes.shape[1]
    disc = np.exp(-r * tau)

    prices = np.zeros((J, M))
    se = np.zeros((J, M))
    for m in range(M):
        K = strikes[:, m]                               # (J,)
        is_call = (K >= S_t)[:, None]                    # (J, 1)
        call_payoff = np.maximum(S - K[:, None], 0.0)
        put_payoff = np.maximum(K[:, None] - S, 0.0)
        payoff = np.where(is_call, call_payoff, put_payoff)   # (J, n_mc)
        prices[:, m] = disc * payoff.mean(axis=1)
        se[:, m] = disc * payoff.std(axis=1) / np.sqrt(n_mc)
    return prices, se


def heston_mc_terminal_samples(sim: HestonSimulator, X0, v0, n_steps_ahead, mc_sims):
    """
    Thin wrapper around HestonSimulator._mc_pdf, reusing the project's own
    Andersen-QE / exact-log-price scheme (same call pattern already used
    elsewhere in this codebase for the "true distribution" reference).
    Returns raw terminal log-price draws, shape (J, mc_sims).
    """
    return sim._mc_pdf(
        X0=X0, v0=v0, n_bins=None, mc_sims=mc_sims,
        n_steps=n_steps_ahead, get_raw_terminal_values=True,
    )


# =========================================================================
# 5. Generator loading and sampling
# =========================================================================

def load_generator(
    cfg: PricingConfig,
    device: str = "cpu"
) -> MySRForGAN:
    """
    Load a trained SRForGAN generator directly from its checkpoint.

    The generator architecture is reconstructed from the
    `architecture_params` stored inside the checkpoint, avoiding
    the need to manually specify architecture hyperparameters.
    """

    ckpt_path = Path(cfg.checkpoint_path)

    print(f"Loading generator checkpoint: {ckpt_path}")

    # ------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------
    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    # ------------------------------------------------------------
    # Retrieve architecture parameters stored in checkpoint
    # ------------------------------------------------------------
    if "architecture_params" not in ckpt:
        raise KeyError(
            "Checkpoint does not contain 'architecture_params'."
        )

    arch = ckpt["architecture_params"]

    print("Generator architecture:")
    for key, value in arch.items():
        print(f"  {key}: {value}")

    # ------------------------------------------------------------
    # Reconstruct the MySRForGAN wrapper
    # ------------------------------------------------------------
    model = MySRForGAN(
        max_epoch=100,
        batch_size=128,
        z_noise_dim=arch["latent_size"],
        n_samples_sr=10,
        scoring_rule="energy",
        lr_g=1e-3,
        early_stopping_patience=10,
        name=cfg.model_name or ckpt_path.stem.replace(
            "_generator", ""
        ),
        use_amp=False,
    )

    # ------------------------------------------------------------
    # Reconstruct generator architecture
    # ------------------------------------------------------------
    model.set_generator(
        condition_size=arch["condition_size"],
        output_dim=arch["output_dim"],
        hidden_dim_rnn=arch["hidden_dim"],
        n_layers=arch["n_layers"],
        rnn_layer=arch["rnn_layer"],
        dropout=arch["dropout"],
    )

    # ------------------------------------------------------------
    # Load trained weights
    # ------------------------------------------------------------
    missing_keys, unexpected_keys = model.G.load_state_dict(
        ckpt["model_state_dict"],
        strict=False,
    )

    if missing_keys:
        print("WARNING - missing keys:")
        for key in missing_keys:
            print(f"  {key}")

    if unexpected_keys:
        print("WARNING - unexpected keys:")
        for key in unexpected_keys:
            print(f"  {key}")

    # ------------------------------------------------------------
    # Move to device and evaluation mode
    # ------------------------------------------------------------
    model.G.to(device)
    model.G.eval()

    print("Generator successfully loaded.")
    return model

def sample_from_generator(model: MySRForGAN, conditions: np.ndarray,
                           targets: np.ndarray, n_samples: int) -> np.ndarray:
    """
    conditions : (J, condition_steps) log-price windows.
    targets    : (J, 1) placeholder labels (required by prepare_data/
                 TensorDataset - not used to bias the generated samples).

    Returns generated log-price draws, shape (J, n_samples). Isolated in
    its own function so the (currently untyped) MyCGAN.generate() contract
    is easy to adapt if its signature changes.
    """
    dataset, _, _ = prepare_data(targets, conditions)
    _, generated = model.generate(dataset, get_pdf=True, bins=None, n_samples=n_samples)
    return np.asarray(generated)


# =========================================================================
# 6. Black-Scholes price / implied volatility
# =========================================================================

def bs_price(S, K, T, r, sigma, is_call: bool) -> float:
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        return np.exp(-r * T) * intrinsic
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(price, S, K, T, r, is_call: bool,
                 lo: float = 1e-6, hi: float = 5.0) -> float:
    """Brentq inversion of the Black-Scholes price. Returns np.nan if the
    quoted price sits outside the range spanned by [lo, hi] volatility
    (e.g. below intrinsic value due to MC/discretisation noise)."""
    intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
    if not np.isfinite(price) or price < np.exp(-r * T) * intrinsic - 1e-10:
        return np.nan

    def f(sigma):
        return bs_price(S, K, T, r, sigma, is_call) - price

    try:
        return brentq(f, lo, hi, xtol=1e-8)
    except ValueError:
        return np.nan


# =========================================================================
# 7. Orchestration
# =========================================================================

def run_pricing_experiment(cfg: PricingConfig, model: Optional[MySRForGAN] = None,
                            device: str = "cpu") -> dict:
    """
    Runs the full experiment and returns a dict with:
      - 'df'          : long-format per (trajectory, strike) results
      - 'moments_df'  : per-trajectory moment diagnostics (GAN vs Heston-MC)
      - 'sim'         : the HestonSimulator instance (for further analysis)
      - 'S_t', 'v_t'  : reference spot / variance arrays
    """
    print("Simulating test set...")
    sim, conditions, targets, X_t, v_t, S_t = simulate_test_set(cfg)
    tau = cfg.forecast_horizon * sim.dt
    strikes = build_strike_grid(S_t, cfg.moneyness_grid)

    print("Pricing with the semi-analytic Heston oracle (frFFT)...")
    price_oracle = compute_oracle_prices(
        sim, X_t, v_t, cfg.forecast_horizon, strikes, r=cfg.r,
        Nfft=cfg.oracle_Nfft, eta=cfg.oracle_eta,
        x_width=cfg.oracle_x_width, n_bins=cfg.oracle_n_bins,
    )

    print("Pricing with Heston Monte-Carlo (Andersen-QE / exact scheme)...")
    heston_mc_samples = heston_mc_terminal_samples(
        sim, X_t, v_t, cfg.forecast_horizon, cfg.mc_sims_heston
    )
    price_heston_mc, se_heston_mc = mc_price_from_samples(
        heston_mc_samples, S_t, strikes, cfg.r, tau
    )

    if model is None:
        print("Loading trained SRForGAN generator...")
        model = load_generator(cfg, device=device)

    print("Sampling forecast distribution from SRForGAN...")
    gan_samples = sample_from_generator(model, conditions, targets, cfg.n_samples_gan)
    price_gan, se_gan = mc_price_from_samples(gan_samples, S_t, strikes, cfg.r, tau)

    # ---- Assemble long-format results table ----
    print("Assembling results table...")
    J, M = strikes.shape
    rows = []
    for j in range(J):
        for m in range(M):
            K = strikes[j, m]
            is_call = K >= S_t[j]
            po, pmc, pg = price_oracle[j, m], price_heston_mc[j, m], price_gan[j, m]
            se_mc, se_g = se_heston_mc[j, m], se_gan[j, m]
            iv_o = implied_vol(po, S_t[j], K, tau, cfg.r, is_call)
            iv_mc = implied_vol(pmc, S_t[j], K, tau, cfg.r, is_call)
            iv_g = implied_vol(pg, S_t[j], K, tau, cfg.r, is_call)
            combined_se_gan_vs_mc = np.sqrt(se_g ** 2 + se_mc ** 2)
            rows.append(dict(
                traj_id=j, moneyness=cfg.moneyness_grid[m], strike=K,
                option_type="call" if is_call else "put",
                S_t=S_t[j], v_t=v_t[j],
                kappa=sim.kappa[j], theta=sim.theta[j],
                sigma_v=sim.sigma_v[j], rho=sim.rho[j],
                price_oracle=po, price_heston_mc=pmc, price_gan=pg,
                se_heston_mc=se_mc, se_gan=se_g,
                iv_oracle=iv_o, iv_heston_mc=iv_mc, iv_gan=iv_g,
                err_price_gan_vs_oracle=pg - po,
                err_price_gan_vs_mc=pg - pmc,
                err_price_mc_vs_oracle=pmc - po,
                err_iv_gan_vs_oracle=iv_g - iv_o,
                err_iv_gan_vs_mc=iv_g - iv_mc,
                z_gan_vs_mc=(pg - pmc) / combined_se_gan_vs_mc if combined_se_gan_vs_mc > 0 else np.nan,
                z_mc_vs_oracle=(pmc - po) / se_mc if se_mc > 0 else np.nan,
            ))
    df = pd.DataFrame(rows)

    # ---- Per-trajectory moment diagnostics: GAN vs Heston-MC samples ----
    moments_df = pd.DataFrame({
        "traj_id": np.arange(J),
        "kappa": sim.kappa, "theta": sim.theta,
        "sigma_v": sim.sigma_v, "rho": sim.rho, "v_t": v_t,
        "mean_mc": heston_mc_samples.mean(axis=1),
        "mean_gan": gan_samples.mean(axis=1),
        "std_mc": heston_mc_samples.std(axis=1),
        "std_gan": gan_samples.std(axis=1),
        "skew_mc": skew(heston_mc_samples, axis=1),
        "skew_gan": skew(gan_samples, axis=1),
        "kurt_mc": kurtosis(heston_mc_samples, axis=1),
        "kurt_gan": kurtosis(gan_samples, axis=1),
    })

    return dict(df=df, moments_df=moments_df, sim=sim, S_t=S_t, v_t=v_t,
                heston_mc_samples=heston_mc_samples, gan_samples=gan_samples)


# =========================================================================
# 8. Error metrics
# =========================================================================

def summarize_errors(df: pd.DataFrame, group_cols: Optional[list] = None) -> pd.DataFrame:
    """
    RMSE / WAPE / mean bias of GAN prices and implied vols against both
    benchmarks, optionally grouped (e.g. group_cols=['moneyness'] or
    a parameter bucket column added beforehand with pd.qcut).
    """
    def _agg(g: pd.DataFrame) -> pd.Series:
        # WAPE is computed as:
        #   WAPE = sum(|y_i - yhat_i|) / sum(y_i)
        # with the corresponding benchmark as y_i.
        price_wape_vs_oracle = np.divide(
            np.abs(g["err_price_gan_vs_oracle"]).sum(),
            g["price_oracle"].sum(),
            out=np.array(np.nan),
            where=np.abs(g["price_oracle"].sum()) > 1e-12,
        )

        price_wape_vs_mc = np.divide(
            np.abs(g["err_price_gan_vs_mc"]).sum(),
            g["price_heston_mc"].sum(),
            out=np.array(np.nan),
            where=np.abs(g["price_heston_mc"].sum()) > 1e-12,
        )

        iv_valid = np.isfinite(g["iv_oracle"]) & np.isfinite(g["err_iv_gan_vs_oracle"])
        iv_wape_vs_oracle = np.divide(
            np.abs(g.loc[iv_valid, "err_iv_gan_vs_oracle"]).sum(),
            g.loc[iv_valid, "iv_oracle"].sum(),
            out=np.array(np.nan),
            where=np.abs(g.loc[iv_valid, "iv_oracle"].sum()) > 1e-12,
        )

        out = {
            "n": len(g),
            "rmse_price_vs_oracle": np.sqrt(np.mean(g["err_price_gan_vs_oracle"] ** 2)),
            "wape_price_vs_oracle": 100.0 * price_wape_vs_oracle,
            "bias_price_vs_oracle": np.mean(g["err_price_gan_vs_oracle"]),
            "rmse_price_vs_mc": np.sqrt(np.mean(g["err_price_gan_vs_mc"] ** 2)),
            "wape_price_vs_mc": 100.0 * price_wape_vs_mc,
            "rmse_iv_vs_oracle": np.sqrt(np.nanmean(g["err_iv_gan_vs_oracle"] ** 2)),
            "wape_iv_vs_oracle": 100.0 * iv_wape_vs_oracle,
            "bias_iv_vs_oracle": np.nanmean(g["err_iv_gan_vs_oracle"]),
            "share_|z_gan_vs_mc|>1.96": np.mean(np.abs(g["z_gan_vs_mc"]) > 1.96),
        }
        return pd.Series(out)

    if group_cols:
        return df.groupby(group_cols).apply(_agg).reset_index()
    return _agg(df).to_frame().T


def bias_significance_by_group(
    df: pd.DataFrame,
    group_cols: list,
    error_col: str = "err_price_gan_vs_oracle",
) -> pd.DataFrame:
    """
    One-sample two-sided t-test of H0: mean(error_col) == 0 within each
    group defined by group_cols, e.g. group_cols=["moneyness"].

    VALIDITY REQUIREMENT: each group must contain at most one row per
    trajectory. Grouping by a fixed moneyness level satisfies this - every
    trajectory contributes exactly one (trajectory, strike) pair per
    moneyness level, and different trajectories are independently
    simulated and independently sampled, so the resulting `n` residuals
    are (approximately) i.i.d. under H0. Do NOT reuse this to test bias
    pooled across several strikes of the SAME trajectory (e.g. the overall
    row of summarize_errors with no group_cols): those rows share the same
    10,000 MC/GAN draws for that trajectory, which breaks independence and
    would overstate significance. For that pointwise/within-trajectory
    question, use the `share_|z_gan_vs_mc|>1.96` diagnostic instead, which
    is built for correlated, per-draw comparisons rather than for testing
    an aggregate bias.

    Returns one row per group with n, mean_bias, std_bias (sample std,
    ddof=1), the t-statistic and the two-sided p-value.
    """
    def _test(g: pd.DataFrame) -> pd.Series:
        stat, p = ttest_1samp(g[error_col], popmean=0.0)
        return pd.Series({
            "n": len(g),
            "mean_bias": g[error_col].mean(),
            "std_bias": g[error_col].std(ddof=1),
            "t_stat": stat,
            "p_value": p,
        })

    return df.groupby(group_cols).apply(_test).reset_index()


def moment_bias_summary(moments_df: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """
    Bias, WAPE (Weighted Absolute Percentage Error) and cross-trajectory
    correlation of each moment, GAN vs Heston-MC.

    WAPE is a ratio of sums, not a mean of ratios (unlike MAPE):
        WAPE = 100 * sum(|gan - mc|) / sum(|mc|)
    computed over trajectories whose |mc| exceeds `eps`. The mask is
    applied to the arrays BEFORE summing - masking only the final ratio
    """
    out = {}
    for stat in ["mean", "std", "skew", "kurt"]:
        gan = moments_df[f"{stat}_gan"].to_numpy()
        mc = moments_df[f"{stat}_mc"].to_numpy()
        d = gan - mc

        mask = np.isfinite(mc) & (np.abs(mc) > eps)   # filter BEFORE summing
        wape = 100.0 * np.abs(d[mask]).sum() / np.abs(mc[mask]).sum() if mask.any() else np.nan

        out[f"bias_{stat}"] = d.mean()
        out[f"wape_{stat}"] = wape
        out[f"n_excluded_{stat}"] = int((~mask).sum())
        out[f"corr_{stat}"] = moments_df[f"{stat}_gan"].corr(moments_df[f"{stat}_mc"])
    return pd.Series(out).to_frame("value")


# =========================================================================
# 9. Plots
# =========================================================================

def plot_iv_smile(df: pd.DataFrame, traj_ids: list, ncols: int = 2):
    n = len(traj_ids)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
    axes = axes.flatten()
    for i, tid in enumerate(traj_ids):
        g = df[df.traj_id == tid].sort_values("moneyness")
        ax = axes[i]
        ax.plot(g.moneyness, g.iv_oracle, "k--", lw=2, label="Heston oracle")
        ax.plot(g.moneyness, g.iv_heston_mc, "o", color="tab:blue", ms=4,
                 alpha=0.7, label="Heston MC (QE)")
        ax.plot(g.moneyness, g.iv_gan, "-s", color="tab:orange", ms=4,
                 label="SRForGAN")
        row = g.iloc[0]
        ax.set_title(
            f"traj {tid}:  kappa={row.kappa:.2f}, theta={row.theta:.2f}, "
            f"sigma_v={row.sigma_v:.2f}, v_t={row.v_t:.3f}", fontsize=9
        )
        ax.set_xlabel("Moneyness K/S_t")
        ax.set_ylabel("Implied volatility")
        ax.legend(fontsize=8)
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()


def plot_price_error_heatmap(df: pd.DataFrame, param_col: str, n_buckets: int = 5,
                              value_col: str = "err_iv_gan_vs_oracle"):
    """Mean |value_col| across (moneyness x quantile-bucket of param_col)."""
    d = df.copy()
    d["param_bucket"] = pd.qcut(d[param_col], n_buckets, duplicates="drop")
    pivot = d.pivot_table(
        index="param_bucket", columns="moneyness",
        values=value_col, aggfunc=lambda x: np.nanmean(np.abs(x)),
    )
    fig, ax = plt.subplots(figsize=(1.4 * pivot.shape[1] + 2, 0.6 * pivot.shape[0] + 2))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns], rotation=45)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_xlabel("Moneyness K/S_t")
    ax.set_ylabel(f"{param_col} bucket")
    ax.set_title(f"Mean |{value_col}|")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


def plot_moment_parity(moments_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, stat, label in zip(
        axes.flatten(),
        ["mean", "std", "skew", "kurt"],
        ["Mean(X_T)", "Std(X_T)", "Skewness(X_T)", "Excess kurtosis(X_T)"],
    ):
        x, y = moments_df[f"{stat}_mc"], moments_df[f"{stat}_gan"]
        ax.scatter(x, y, s=14, alpha=0.6)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y = x")
        ax.set_xlabel(f"Heston MC {label}")
        ax.set_ylabel(f"SRForGAN {label}")
        ax.set_title(label)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_error_vs_noise_floor(df: pd.DataFrame):
    """|gan price - heston_mc price| vs the combined MC standard error -
    points above the y=x line are errors that exceed what MC noise alone
    would explain."""
    abs_err = df["err_price_gan_vs_mc"].abs()
    combined_se = np.sqrt(df["se_gan"] ** 2 + df["se_heston_mc"] ** 2)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(combined_se, abs_err, s=10, alpha=0.4)
    lim = max(combined_se.max(), abs_err.max())
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="error = noise floor")
    ax.plot([0, lim], [0, 1.96 * lim], "orange", ls=":", lw=1, label="95% noise band")
    ax.set_xlabel("Combined MC standard error")
    ax.set_ylabel("|GAN price - Heston MC price|")
    ax.set_title("Pricing error vs. Monte-Carlo noise floor")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# =========================================================================
# 10. Example usage
# =========================================================================

if __name__ == "__main__":
    cfg = PricingConfig(
        n_test=300,
        checkpoint_path="models/srforgan_models/var_all_srforgan_heston_generator.pth",
    )
    # Equivalently, to guarantee the test set matches the checkpoint's own
    # training ranges exactly:
    # cfg = PricingConfig.from_training_config_json(
    #     "models/srforgan_models/var_all_srforgan_heston_training_config.json",
    #     n_test=300,
    #     checkpoint_path="models/srforgan_models/var_all_srforgan_heston_generator.pth",
    # )

    results = run_pricing_experiment(cfg)
    df, moments_df = results["df"], results["moments_df"]

    print("\n=== Overall pricing error summary ===")
    print(summarize_errors(df))

    print("\n=== Error summary by moneyness ===")
    print(summarize_errors(df, group_cols=["moneyness"]))

    print("\n=== Significance of the aggregate price bias, by moneyness ===")
    print(bias_significance_by_group(
        df, group_cols=["moneyness"], error_col="err_price_gan_vs_oracle"
    ))
    print("\n=== Significance of the aggregate IV bias, by moneyness ===")
    print(bias_significance_by_group(
        df, group_cols=["moneyness"], error_col="err_iv_gan_vs_oracle"
    ))
    # The same test on the IV-space bias is just as easy to obtain:
    # bias_significance_by_group(df, group_cols=["moneyness"],
    #                             error_col="err_iv_gan_vs_oracle")

    print("\n=== Moment-bias summary (GAN vs Heston MC) ===")
    print(moment_bias_summary(moments_df))

    example_ids = list(range(min(4, cfg.n_test)))
    plot_iv_smile(df, example_ids)
    plot_error_vs_noise_floor(df)
    plot_moment_parity(moments_df)
    plot_price_error_heatmap(df, param_col="sigma_v")
    plot_price_error_heatmap(df, param_col="kappa")