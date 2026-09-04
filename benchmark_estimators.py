"""
Config-driven Monte Carlo benchmark of causal estimators on the PHAIR_EHR
semi-synthetic DGP.

An *arm* is one full analysis specification -- propensity model, outcome
model, cross-fitting choice, calibration procedure, estimator set. Arms share
the same replicate seeds, so differences between them are attributable to the
pieces you varied.

    python benchmark_estimators.py --config benchmark_config.yaml
    python benchmark_estimators.py --arms logit gb_raw --replicates 50
    
"""

# NOTE: this must run BEFORE anything imports rpy2. rpy2 emits its R_HOME,
# R version and library paths at INFO *during import*, so a suppression placed
# further down the file is too late -- the lines have already printed.
import logging as _logging  # noqa: E402

_logging.getLogger("rpy2").setLevel(_logging.WARNING)

import argparse
import logging
import os
import time
import warnings
from itertools import product
from os.path import join
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml
from CausalEstimate import MultiEstimator
from CausalEstimate.estimators import AIPW, IPW, TMLE
from CausalEstimate.utils.constants import EFFECT as EFFECT_KEY
from joblib import Parallel, delayed
from scipy import stats
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

try:
    from betacal import BetaCalibration

    _HAS_BETACAL = True
except ImportError:  # pragma: no cover
    _HAS_BETACAL = False

def _load_hal():
    """Import hal_r LAZILY, on first use.

    Importing it at module level means every parallel worker pulls in rpy2 as
    soon as it imports this module -- which happens before silence() runs in
    that process, so rpy2's import-time INFO chatter escapes once per worker
    (8 lines at --n-jobs 8). Deferring the import to build_learner() puts it
    after silence(), which run_one() calls first thing in every worker.

    Side benefit: runs with no HAL arm never import rpy2 or start R at all.
    """
    from hal_r import HAL9001

    return HAL9001

SIM_CONFIG = "./corebehrt/configs/causal/simulate_semisynthetic.yaml"

# Default number of folds for cross-fitting nuisances and for the internal
# split used by calibration. Overridable per run via `run.n_folds`. Note the
# jackknife overrides this with V, so that the cross-fitting refits double as
# its leave-fold-out refits.
DEFAULT_N_FOLDS = 5
CACHE = "./outputs/causal/benchmark_cache"


def silence(enabled=True):
    """Mute warnings and library logging, including inside loky workers."""
    if not enabled:
        return
    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
    for name in ("simulate", "corebehrt", "CausalEstimate", "root", "rpy2"):
        logging.getLogger(name).setLevel(logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)


silence()


# ----------------------------------------------------------------------
# Learner registry
# ----------------------------------------------------------------------
def build_learner(spec):
    """Instantiate a learner from its config block.

    `keep` degrades the learner to its first k features (0 = intercept-only),
    which is how misspecification arms are expressed.
    """
    kind = spec["type"]
    if kind == "logistic":
        model = LogisticRegression(max_iter=spec.get("max_iter", 2000))
    elif kind == "gradient_boosting":
        model = HistGradientBoostingClassifier(
            max_iter=spec.get("max_iter", 150), random_state=0
        )
    elif kind == "hal":
        try:
            HAL9001 = _load_hal()
        except Exception as exc:  # pragma: no cover - needs rpy2 + R
            raise RuntimeError(
                "learner type 'hal' needs hal_r.py (rpy2 + R hal9001) importable."
            ) from exc
        model = HAL9001(
            max_degree=spec.get("max_degree", 2),
            smoothness_orders=spec.get("smoothness_orders", 0),
            num_knots=spec.get("num_knots", 6),
        )
    else:
        raise ValueError(f"unknown learner type: {kind!r}")
    return model, spec.get("keep", None)


def _restrict(X, keep):
    if keep is None:
        return X
    if keep <= 0:
        return np.zeros((X.shape[0], 1))
    return X[:, :keep]


def fit_predict(spec, X, y, crossfit, seed, extra=None,
                n_folds=DEFAULT_N_FOLDS,
                return_models=False, folds=None):
    """Fit a learner, returning in-sample or out-of-fold probabilities.

    `extra` is appended as a column (used to put treatment into the outcome
    model). When present, predictions under A=1 and A=0 are both returned.

    crossfit=False is appropriate for Donsker-class learners (parametric
    models, HAL with a bounded sectional variation norm), where the empirical
    process term is already negligible. Flexible learners like gradient
    boosting need crossfit=True.
    """
    _, keep = build_learner(spec)
    Xr = _restrict(X, keep)
    design = Xr if extra is None else np.column_stack([Xr, extra])
    n = len(y)

    def _fit(idx):
        m, _ = build_learner(spec)
        return m.fit(design[idx], y[idx])

    def _pred(m, idx):
        if extra is None:
            return m.predict_proba(design[idx])[:, 1], None
        k = len(idx)
        p1 = m.predict_proba(np.column_stack([Xr[idx], np.ones(k)]))[:, 1]
        p0 = m.predict_proba(np.column_stack([Xr[idx], np.zeros(k)]))[:, 1]
        return p1, p0

    def _finish(o1, o0, models):
        res = (o1, o0 if extra is not None else None)
        return (*res, models) if return_models else res

    if y.sum() < 10 or (n - y.sum()) < 10:  # degenerate draw
        flat = np.full(n, y.mean(), dtype=float)
        return _finish(flat, flat, [])

    if not crossfit:
        allidx = np.arange(n)
        o1, o0 = _pred(_fit(allidx), allidx)
        return _finish(o1, o0 if o0 is not None else o1, [])

    out1, out0, models = np.zeros(n), np.zeros(n), []
    # An explicit partition matters for the jackknife: StratifiedKFold
    # stratifies on its label, so folds built from A and folds built from Y
    # differ even under the same seed (~90% overlap). theta_(-v) must be
    # computed on ONE retained set with every nuisance excluding exactly that
    # fold, so the caller passes a single shared partition.
    if folds is None:
        folds = list(
            StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(design, y)
        )
    for tr, te in folds:
        m = _fit(tr)
        p1, p0 = _pred(m, te)
        out1[te] = p1
        if p0 is not None:
            out0[te] = p0
        models.append((tr, m))
    return _finish(out1, out0, models)


# ----------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------
def calibrate_ps(ps, A, method, seed, bounds, n_folds=DEFAULT_N_FOLDS):
    """Cross-fitted recalibration of propensity scores.

    Flexible learners return badly calibrated probabilities -- raw gradient
    boosting has a calibration slope near 0.16 on this data -- and every
    inverse-probability method inherits that error. Mirrors the repo's own
    calibrate_exp_y.py stage. Output is clipped because isotonic regression is
    a step function and returns exact 0 and 1 on held-out folds.
    """
    if method == "none":
        return ps
    ps = np.clip(np.asarray(ps, dtype=float), 1e-6, 1 - 1e-6)
    out = np.zeros(len(ps))
    for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(
        ps.reshape(-1, 1), A
    ):
        if method == "isotonic":
            out[te] = (
                IsotonicRegression(out_of_bounds="clip").fit(ps[tr], A[tr]).predict(ps[te])
            )
        elif method == "beta":
            if not _HAS_BETACAL:
                raise RuntimeError("calibrate: beta needs `pip install betacal`")
            out[te] = BetaCalibration("abm").fit(ps[tr], A[tr]).predict(ps[te])
        elif method == "sigmoid":
            z = logit(ps).reshape(-1, 1)
            out[te] = (
                LogisticRegression(max_iter=2000)
                .fit(z[tr], A[tr])
                .predict_proba(z[te])[:, 1]
            )
        else:
            raise ValueError(f"unknown calibration: {method!r}")
    return np.clip(out, bounds[0], bounds[1])


def calibration_slope(ps, A):
    """Slope of A on logit(ps): 1.0 = calibrated, <1 = overconfident."""
    z = logit(np.clip(ps, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    try:
        return float(LogisticRegression(max_iter=2000).fit(z, A).coef_[0][0])
    except Exception:
        return float("nan")


# ----------------------------------------------------------------------
# Estimators
# ----------------------------------------------------------------------
def _weight_kwargs(cfg_like):
    """clip_percentile / eps, from arm then run, with library defaults."""
    return dict(
        clip_percentile=float(cfg_like.get("clip_percentile", 1.0)),
        eps=float(cfg_like.get("eps", 1e-9)),
    )


def build_bootstrap_estimators(names, estimand="ATE", wkw=None):
    wkw = wkw or {}
    common = dict(
        effect_type=estimand, treatment_col="exposure", outcome_col="outcome",
        ps_col="ps"
    )
    out = []
    if "IPW" in names:
        out.append(IPW(**common, **wkw))
    if "AIPW" in names:
        out.append(
            AIPW(**common, probas_t1_col="probas_exposed",
                 probas_t0_col="probas_control", **wkw)
        )
    if "TMLE" in names:
        out.append(
            TMLE(
                **common,
                probas_col="probas",
                probas_t1_col="probas_exposed",
                probas_t0_col="probas_control",
                **wkw,
            )
        )
    return out


def point_estimates(names, df, estimand="ATE", wkw=None):
    """Point estimate per estimator on one dataset -- no bootstrap, no CI."""
    wkw = wkw or {}
    out = {}
    common = dict(effect_type=estimand, treatment_col="exposure",
                  outcome_col="outcome", ps_col="ps")
    if "IPW" in names:
        out["IPW"] = IPW(
            **common, **wkw
        ).compute_effect(df)[EFFECT_KEY]
    if "AIPW" in names:
        out["AIPW"] = AIPW(
            **common, probas_t1_col="probas_exposed",
            probas_t0_col="probas_control", **wkw
        ).compute_effect(df)[EFFECT_KEY]
    if "TMLE" in names:
        out["TMLE"] = TMLE(
            **common, probas_col="probas", probas_t1_col="probas_exposed",
            probas_t0_col="probas_control", **wkw,
        ).compute_effect(df)[EFFECT_KEY]
    return out


def jackknife_ci(names, full_df, theta_full, retained, V, estimand="ATE",
                 wkw=None):
    """V-fold (delete-a-group) jackknife CIs.

    Li, Ertefaie & van der Laan (2026), arXiv:2607.22493. Needs only V
    leave-fold-out refits and no influence function. When the arm is already
    cross-fitted with V folds, those refits ARE the cross-fitting fits, so the
    only extra work is scoring them on the retained rows -- measured at ~0.04s
    versus ~1.8s for 500 bootstrap draws.

    Pseudo-values are theta~_v = V*theta_hat - (V-1)*theta_(-v). For FIXED V
    the Studentized statistic converges to t with V-1 df, so the critical
    value is t_{V-1}, NOT 1.96 -- the variance estimator does not converge in
    probability, and the heavier tails are what make the interval valid.
    Larger V tightens it (t_9 = 2.26 vs t_4 = 2.78) at negligible cost.

    `retained` is a list of (index, df) for each leave-fold-out dataset.
    """
    tcrit = stats.t.ppf(0.975, V - 1)
    per_fold = {k: [] for k in theta_full}
    for _idx, df_v in retained:
        for k, val in point_estimates(names, df_v, estimand, wkw).items():
            per_fold[k].append(val)

    rows = []
    for k, theta in theta_full.items():
        tv = np.asarray(per_fold[k], dtype=float)
        if len(tv) < 2 or not np.all(np.isfinite(tv)):
            rows.append({"method": k, "effect": theta,
                         "CI95_lower": np.nan, "CI95_upper": np.nan})
            continue
        # Pseudo-values supply the VARIANCE only. The point estimate stays
        # theta_hat: the Studentized statistic (theta_hat - theta)/se is what
        # converges to t_{V-1}. Reporting the pseudo-value mean instead gives
        # the bias-corrected jackknife estimate, a different and much noisier
        # quantity -- with gb it inflated the empirical SD ~10x.
        pseudo = V * theta - (V - 1) * tv
        se = pseudo.std(ddof=1) / np.sqrt(V)
        rows.append({"method": k, "effect": theta,
                     "CI95_lower": theta - tcrit * se,
                     "CI95_upper": theta + tcrit * se,
                     "crit": tcrit})
    return rows


def build_treatment_generator(X, A_obs, names, design, learners):
    """Construct g(W) for design.treatment == "generate".

    Two things matter for a fair comparison:

    1. FORM. Fitting a logistic model to the observed treatment makes a
       logistic propensity model correctly specified BY CONSTRUCTION, so it
       wins by fiat and flexible learners are penalised for flexibility they
       do not need. `nonlinear` instead uses a thresholded, interacted,
       non-monotone predictor that no candidate learner matches exactly.

    2. BOUNDS. ATT control weights are g/(1-g), which explodes as g -> 1. An
       unbounded fitted propensity reaching 0.999 leaves the ATT resting on a
       handful of controls (ESS/n ~ 0.12 on this data). Squashing g into
       `generator_bounds` raises that to ~0.4-0.5 without removing the signal.
    """
    kind = design.get("treatment_generator", "logit")
    lo, hi = design.get("generator_bounds", [0.10, 0.75])

    if kind == "nonlinear":
        col = {n: X[:, i] for i, n in enumerate(names)}

        def f(name, default=0.0):
            return col.get(name, np.full(len(A_obs), default))

        z = (
            1.2 * (f("utilization_intensity") > 0.5).astype(float)  # threshold
            + 0.9 * f("disease_burden") * f("chronic_disease_count")  # interaction
            - 0.8 * np.abs(f("age"))  # non-monotone
            + 0.6 * f("code_diversity")
        )
    else:
        spec = learners[kind]
        g_raw, _ = fit_predict(spec, X, A_obs, False, 0)
        z = logit(np.clip(np.asarray(g_raw), 1e-6, 1 - 1e-6))

    if lo is None or hi is None:  # opt out of bounding
        return np.clip(expit(z), 1e-6, 1 - 1e-6)
    p = expit(z - z.mean())
    rng_p = p.max() - p.min()
    p = (p - p.min()) / rng_p if rng_p > 0 else np.full_like(p, 0.5)
    return lo + (hi - lo) * p


def att_effective_sample_size(ps, A):
    """ESS of the ATT control weights w = ps/(1-ps), as a fraction of n.

    The ATT reweights controls to resemble the treated. If a few controls with
    ps near 1 dominate, the contrast rests on very little information and every
    estimator inherits the same bias. Below ~0.2, treat ATT results as noise.
    """
    ctrl = A == 0
    if ctrl.sum() == 0:
        return float("nan")
    w = ps[ctrl] / (1 - ps[ctrl])
    if not np.isfinite(w).all() or w.sum() <= 0:
        return float("nan")
    return float(w.sum() ** 2 / (w**2).sum() / len(A))


def overlap_diagnostics(ps, lo, hi):
    return {
        "pct_ps_outside": 100 * float(((ps < lo) | (ps > hi)).mean()),
        "ps_min": float(ps.min()),
        "ps_max": float(ps.max()),
        "max_weight": float(1.0 / np.minimum(ps, 1 - ps).min()),
    }


# ----------------------------------------------------------------------
# DGP (computed once, cached)
# ----------------------------------------------------------------------
def load_dgp(outcome_name, rebuild=False):
    """Return (X, A, P0, P1, feature_names) for the fixed cohort.

    P0/P1 come from the NOISELESS risk surface, so the true ATE is an exact
    constant -- which is what a simulation study wants as its target.
    """
    key = f"{CACHE}.{outcome_name}.npz"
    if os.path.exists(key) and not rebuild:
        d = np.load(key, allow_pickle=True)
        return d["X"], d["A"], d["P0"], d["P1"], list(d["names"])

    # Imported lazily: building the DGP pulls in corebehrt (and torch), but a
    # cached run needs neither.
    from corebehrt.modules.features.loader import ShardLoader
    from corebehrt.modules.setup.config import load_config as load_corebehrt_config
    from corebehrt.modules.simulation.config_semisynthetic import (
        create_semisynthetic_config,
    )
    from corebehrt.modules.simulation.semisynthetic_simulator import (
        SemiSyntheticCausalSimulator,
    )

    cfg = load_corebehrt_config(SIM_CONFIG)
    sim = SemiSyntheticCausalSimulator(create_semisynthetic_config(cfg))
    sim.compute_global_feature_stats(ShardLoader(cfg.paths.data, cfg.paths.splits))

    feats, exposed, p1s, p0s = [], [], [], []
    for shard, _ in ShardLoader(cfg.paths.data, cfg.paths.splits)():
        out = sim.extract_features_and_probabilities(shard)
        if out is None:
            continue
        features_df, _pids, is_exposed, probas, _tau = out
        feats.append(features_df)
        exposed.append(is_exposed)
        p1s.append(probas[outcome_name]["P1"])
        p0s.append(probas[outcome_name]["P0"])

    features = pd.concat(feats)
    X = features.to_numpy(dtype=float)
    A = np.concatenate(exposed).astype(int)
    P1, P0 = np.concatenate(p1s), np.concatenate(p0s)
    os.makedirs(os.path.dirname(key) or ".", exist_ok=True)
    np.savez(key, X=X, A=A, P0=P0, P1=P1, names=np.array(features.columns))
    return X, A, P0, P1, list(features.columns)


# ----------------------------------------------------------------------
# One replicate of one arm
# ----------------------------------------------------------------------
def run_one(rep, arm, cfg, X, A_obs, P0, P1, g_gen, quiet=True):
    silence(quiet)
    design, run, learners = cfg["design"], cfg["run"], cfg["learners"]
    lo, hi = design.get("ps_clip", [0.01, 0.99])
    rng = np.random.default_rng(run.get("seed", 0) * 100_000 + 10_000 + rep)
    n = len(A_obs)

    idx = rng.integers(0, n, n) if design["resample"] == "patients" else np.arange(n)
    Xr, P0r, P1r = X[idx], P0[idx], P1[idx]

    if design.get("treatment", "sample") == "generate":
        Ar = rng.binomial(1, g_gen[idx])
    else:
        Ar = A_obs[idx]

    Yr = np.where(Ar == 1, rng.binomial(1, P1r), rng.binomial(1, P0r))
    crossfit = arm.get("crossfit", True)

    estimand = str(arm.get("estimand", run.get("estimand", "ATE"))).upper()
    ci_method = arm.get("ci_method", run.get("ci_method", "bootstrap"))
    V = int(arm.get("jackknife_V", run.get("jackknife_V", 10)))
    base_folds = int(run.get("n_folds", DEFAULT_N_FOLDS))
    ps_spec = learners[arm["ps_model"]]
    out_spec = learners[arm["outcome_model"]]
    cal = arm.get("calibrate", "none")
    # arm-level clip_percentile / eps override the run-level values
    wkw = _weight_kwargs({**run, **arm})

    def pipeline(idx):
        """Run the COMPLETE nuisance + calibration pipeline on rows `idx`.

        The jackknife needs theta_(-v) to be the same estimator applied to the
        reduced sample -- which, for a cross-fitted estimator, means
        cross-fitting again *within* the retained rows. Reusing the outer fold
        models and scoring them on their own training rows is NOT equivalent:
        those predictions are in-sample and badly optimistic (gb's calibration
        slope is ~3.5 in-sample versus ~0.12 out-of-fold), and the (V-1)
        multiplier in the pseudo-values turns that gap into nonsense.

        So the cost is (V+1) full pipelines, not V cheap re-scorings.
        """
        Xi, Ai, Yi = Xr[idx], Ar[idx], Yr[idx]
        ps_i, _ = fit_predict(ps_spec, Xi, Ai, crossfit, rep, n_folds=base_folds)
        ps_i = calibrate_ps(np.asarray(ps_i), Ai, cal, rep, (lo, hi),
                            n_folds=base_folds)
        ps_i = np.clip(ps_i, 1e-6, 1 - 1e-6)
        q1, q0 = fit_predict(out_spec, Xi, Yi, crossfit, rep, extra=Ai,
                             n_folds=base_folds)
        return pd.DataFrame({
            "exposure": Ai, "outcome": Yi, "ps": ps_i,
            "probas_exposed": q1, "probas_control": q0,
            "probas": np.where(Ai == 1, q1, q0),
        })

    _t0 = time.perf_counter()
    df = pipeline(np.arange(len(Ar)))
    t_fit = time.perf_counter() - _t0
    _t1 = time.perf_counter()
    ps = df["ps"].to_numpy()

    names = arm.get(
        "estimators",
        ["IPW", "AIPW", "TMLE", "AIPW_EIF", "TMLE_EIF"],
    )
    rows = [
        {
            "method": "naive",
            "effect": Yr[Ar == 1].mean() - Yr[Ar == 0].mean(),
            "CI95_lower": np.nan,
            "CI95_upper": np.nan,
        }
    ]

    boot_names = [x for x in names if x in ("IPW", "AIPW", "TMLE")]
    if boot_names and ci_method == "jackknife":
        # One shared partition; theta_(-v) re-runs the whole pipeline on the
        # retained rows. StratifiedKFold on Ar keeps the exposure ratio stable
        # across folds.
        retained = [
            (tr, pipeline(tr))
            for tr, _te in StratifiedKFold(
                V, shuffle=True, random_state=rep
            ).split(Xr, Ar)
        ]
        if len(retained) >= 2:
            rows.extend(
                jackknife_ci(boot_names, df,
                             point_estimates(boot_names, df, estimand, wkw),
                             retained, len(retained), estimand, wkw)
            )
        else:  # crossfit=False leaves no fold models; fall back rather than fail
            for k, v in point_estimates(boot_names, df, estimand, wkw).items():
                rows.append({"method": k, "effect": v,
                             "CI95_lower": np.nan, "CI95_upper": np.nan})
    elif boot_names:
        res = MultiEstimator(
            estimators=build_bootstrap_estimators(boot_names, estimand, wkw),
            verbose=False
        ).compute_effects(df, n_bootstraps=run.get("n_bootstrap", 100))
        for name, r in res.items():
            rows.append(
                {
                    "method": name,
                    "effect": r["effect"],
                    "CI95_lower": r["CI95_lower"],
                    "CI95_upper": r["CI95_upper"],
                }
            )

    if "AIPW_EIF" in names:
        # AIPW with CausalEstimate's analytic influence-function CI, the same
        # compute_ci that backs TMLE_EIF. AIPW and TMLE share the efficient
        # influence function, so these two rows differ only in which nuisance
        # the residual is taken against (initial Q-hat vs targeted Q*).
        r = AIPW(
            effect_type=estimand,
            treatment_col="exposure",
            outcome_col="outcome",
            ps_col="ps",
            probas_t1_col="probas_exposed",
            probas_t0_col="probas_control",
            **wkw,
        ).compute_effect(df)
        rows.append(
            {
                "method": "AIPW_EIF",
                "effect": r[EFFECT_KEY],
                "CI95_lower": r["CI95_lower"],
                "CI95_upper": r["CI95_upper"],
            }
        )

    if "TMLE_EIF" in names:
        r = TMLE(
            effect_type=estimand,
            treatment_col="exposure",
            outcome_col="outcome",
            ps_col="ps",
            probas_col="probas",
            probas_t1_col="probas_exposed",
            probas_t0_col="probas_control",
            **wkw,
        ).compute_effect(df)
        rows.append(
            {
                "method": "TMLE_EIF",
                "effect": r["effect"],
                "CI95_lower": r["CI95_lower"],
                "CI95_upper": r["CI95_upper"],
            }
        )

    t_ci = time.perf_counter() - _t1

    diag = overlap_diagnostics(ps, lo, hi)
    diag["cal_slope"] = calibration_slope(ps, Ar)
    diag["att_ess"] = att_effective_sample_size(np.asarray(ps), Ar)
    # Per-replicate CPU cost. Wall-clock per ARM is meaningless once arms
    # interleave across workers, so we sum these instead: total CPU-seconds is
    # parallelism-independent, and t_fit vs t_ci says where the time went.
    diag["t_fit"] = t_fit
    diag["t_ci"] = t_ci
    diag["t_total"] = t_fit + t_ci
    for r in rows:
        r.update(diag)
        r["replicate"] = rep
        r["arm"] = arm["name"]
    return rows


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def summarize(raw, true_ate):
    recs = []
    for (arm, method), g in raw.groupby(["arm", "method"], sort=False):
        est = g["effect"].to_numpy()
        bias = est - true_ate
        sd = est.std(ddof=1)
        rec = {
            "arm": arm,
            "method": method,
            "mean_est": est.mean(),
            "bias": bias.mean(),
            "bias_mcse": sd / np.sqrt(len(est)),
            "emp_SD": sd,
            "RMSE": np.sqrt((bias**2).mean()),
        }
        if g["CI95_lower"].notna().all():
            width = g["CI95_upper"] - g["CI95_lower"]
            # Back the SE out with the critical value THAT ROW ACTUALLY USED.
            # Jackknife rows use t_{V-1}, not 1.96, so dividing everything by
            # 1.96 inflates their apparent SE by t_{V-1}/1.96 -- 1.15x at V=10,
            # 1.42x at V=5 -- and makes SE/SD incomparable across methods.
            crit = g["crit"].fillna(1.96) if "crit" in g else 1.96
            se = (width / (2 * crit)).mean()
            cov = ((g.CI95_lower <= true_ate) & (g.CI95_upper >= true_ate)).mean()
            rec.update(
                {
                    "mean_SE": se,
                    "SE/SD": se / sd,
                    "coverage": 100 * cov,
                    "cov_mcse": 100 * np.sqrt(cov * (1 - cov) / len(est)),
                }
            )
        recs.append(rec)
    return pd.DataFrame(recs)


def run_benchmark(cfg):
    run, design = cfg["run"], cfg["design"]
    silence(run.get("quiet", True))
    X, A_obs, P0, P1, names = load_dgp(run.get("outcome", "OUTCOME"))
    estimand = str(run.get("estimand", "ATE")).upper()

    degenerate = [n for i, n in enumerate(names) if X[:, i].std() == 0]
    if degenerate:
        print(f"  NOTE: constant feature(s): {degenerate} -- coefficients inert.")

    g_gen = None
    if design.get("treatment", "sample") == "generate":
        g_gen = build_treatment_generator(
            X, A_obs, names, design, cfg["learners"]
        )
        print(
            f"  GENERATOR ({design.get('treatment_generator', 'logit')}): "
            f"g in [{g_gen.min():.3f}, {g_gen.max():.3f}]  sd={g_gen.std():.3f}  "
            f"ATT ESS/n={att_effective_sample_size(g_gen, np.zeros(len(g_gen), int)):.2f}"
        )

    # The ATE averages the individual effect over EVERYONE; the ATT averages it
    # over the treated only, so the two targets genuinely differ. Under
    # treatment=generate nobody is deterministically treated, so the treated
    # population is weighted by g(W).
    ite = P1 - P0
    if estimand == "ATE":
        true_ate = float(ite.mean())
    elif estimand == "ATT":
        true_ate = (
            float((ite * g_gen).sum() / g_gen.sum())
            if g_gen is not None
            else float(ite[A_obs == 1].mean())
        )
    else:
        raise ValueError(f"unknown estimand: {estimand!r} (use ATE or ATT)")

    jobs = [
        (rep, arm)
        for arm, rep in product(cfg["arms"], range(run.get("replicates", 200)))
    ]
    _wall0 = time.perf_counter()
    stream = Parallel(n_jobs=run.get("n_jobs", 1), verbose=0, return_as="generator")(
        delayed(run_one)(rep, arm, cfg, X, A_obs, P0, P1, g_gen, run.get("quiet", True))
        for rep, arm in jobs
    )
    batches = cast(
        "list[list[dict[str, Any]]]",
        list(
            tqdm(
                stream,
                total=len(jobs),
                desc="arm x replicate",
                unit="fit",
                disable=not run.get("progress", True),
            )
        ),
    )
    wall = time.perf_counter() - _wall0
    raw = pd.DataFrame([r for b in batches for r in b])
    summary = summarize(raw, true_ate)

    print("\n" + "=" * 100)
    print(f"  TRUE {estimand} = {true_ate:.4f}   n = {len(A_obs)}")
    print(
        f"  design: estimand={estimand}  resample={design['resample']}  "
        f"treatment={design.get('treatment', 'sample')}  "
        f"replicates={run.get('replicates')}  bootstrap={run.get('n_bootstrap')}"
    )
    print("=" * 100)

    per_rep = raw.drop_duplicates(["arm", "replicate"]).groupby("arm", sort=False)
    print("\n  ARM DIAGNOSTICS (propensity model only)")
    print(
        per_rep[["cal_slope", "att_ess", "pct_ps_outside", "ps_min", "ps_max",
              "max_weight"]]
        .mean()
        .to_string(float_format=lambda v: f"{v:10.4f}")
    )

    tsum = per_rep[["t_fit", "t_ci", "t_total"]].agg(["mean", "sum"])
    tsum.columns = ["fit/rep", "fit_tot", "ci/rep", "ci_tot", "tot/rep", "cpu_tot"]
    tsum = tsum[["fit/rep", "ci/rep", "tot/rep", "cpu_tot"]]
    tsum["% CPU"] = 100 * tsum["cpu_tot"] / tsum["cpu_tot"].sum()
    print(f"\n  TIMING (seconds; wall clock for the whole run: {wall:.1f}s "
          f"at n_jobs={run.get('n_jobs', 1)})")
    print(tsum.to_string(float_format=lambda v: f"{v:9.3f}"))
    print("  fit = nuisance pipeline on the full sample; ci = interval construction")
    print("  (for ci_method=jackknife, ci includes the V leave-fold-out pipelines)")

    for arm in cfg["arms"]:
        sub = summary[summary.arm == arm["name"]].drop(columns=["arm"])
        print(
            f"\n  ARM: {arm['name']}   ps={arm['ps_model']}  out={arm['outcome_model']}"
            f"  crossfit={arm.get('crossfit', True)}  "
            f"calibrate={arm.get('calibrate', 'none')}"
        )
        print(sub.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))

    print("\n" + "=" * 100)
    print("  bias_mcse = MC SE of the bias; SE/SD = estimated SE / empirical SD")
    print("  cov_mcse  = MC SE of coverage, in points\n")

    if run.get("save"):
        os.makedirs(run["save"], exist_ok=True)
        raw.to_csv(join(run["save"], "benchmark_raw.csv"), index=False)
        summary.to_csv(join(run["save"], "benchmark_summary.csv"), index=False)
        print(f"  saved -> {run['save']}\n")
    return summary, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="benchmark_config.yaml")
    ap.add_argument("--arms", nargs="*", help="run only these arm names")
    ap.add_argument("--replicates", type=int, help="override run.replicates")
    ap.add_argument("--n-bootstrap", type=int, help="override run.n_bootstrap")
    ap.add_argument("--n-jobs", type=int, help="override run.n_jobs")
    ap.add_argument("--treatment", choices=["sample", "generate"])
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    if a.arms:
        cfg["arms"] = [x for x in cfg["arms"] if x["name"] in a.arms]
        if not cfg["arms"]:
            raise SystemExit(f"no arms matched {a.arms}")
    for key, val in [
        ("replicates", a.replicates),
        ("n_bootstrap", a.n_bootstrap),
        ("n_jobs", a.n_jobs),
    ]:
        if val is not None:
            cfg["run"][key] = val
    if a.treatment:
        cfg["design"]["treatment"] = a.treatment
    run_benchmark(cfg)


if __name__ == "__main__":
    main()