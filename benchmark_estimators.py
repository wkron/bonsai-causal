"""
Monte Carlo benchmark of IPW / AIPW / TMLE on the semi-synthetic DGP.

The expensive part (feature extraction + risk surface) is computed ONCE and
cached; replicates then re-draw outcomes, refit nuisances, and re-estimate.

Misspecification: none | ps | outcome | both.
Two resampling modes, and the difference between them matters:

  --resample outcomes   Cohort is frozen (X, A, P0, P1 fixed); only the
                        Bernoulli outcome draws vary. Target = ATE of THIS
                        cohort. Isolates outcome-sampling noise.

  --resample patients   Draw n patients with replacement, then draw outcomes.
                        Target = ATE of the full 1302-patient population.
                        Includes patient-sampling noise, which is what the
                        bootstrap CIs are actually estimating.

Coverage is only expected to hit nominal 95% in `patients` mode. See the
SE/SD column: it diagnoses whether the interval width matches the true
replicate-to-replicate spread.

Usage:
    python benchmark_estimators.py --replicates 300
    python benchmark_estimators.py --replicates 300 --resample outcomes
    python benchmark_estimators.py --replicates 300 --misspecify ps --n-jobs 8
"""

import argparse
import logging
import os
import warnings
from os.path import join
from typing import Any, cast

import numpy as np
import pandas as pd
from CausalEstimate import MultiEstimator
from CausalEstimate.estimators import AIPW, IPW, TMLE
from joblib import Parallel, delayed
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from betacal import BetaCalibration
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from corebehrt.modules.features.loader import ShardLoader
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.simulation.config_semisynthetic import (
    create_semisynthetic_config,
)
from corebehrt.modules.simulation.semisynthetic_simulator import (
    SemiSyntheticCausalSimulator,
)


def silence(enabled: bool = True) -> None:
    """Mute warnings + corebehrt/CausalEstimate logging.

    Called at module import AND at the top of every worker task, because
    joblib's loky backend spawns fresh processes that do not inherit the
    parent's warning filters or logging levels.
    """
    if not enabled:
        return
    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
    for name in ("simulate", "corebehrt", "CausalEstimate", "root"):
        logging.getLogger(name).setLevel(logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)


silence()

SIM_CONFIG = "./corebehrt/configs/causal/simulate_semisynthetic.yaml"
CACHE = "./outputs/causal/benchmark_cache.npz"


# ----------------------------------------------------------------------
# DGP setup (run once)
# ----------------------------------------------------------------------
def load_dgp(outcome_name: str, cache_path: str = CACHE, rebuild: bool = False):
    """Return (X, A, P0, P1, feature_names) for the fixed cohort.

    P0/P1 come from extract_features_and_probabilities(), which is the
    NOISELESS risk surface -- so the true ATE is an exact constant, which is
    what a simulation study wants as its target.
    """
    key = f"{cache_path}.{outcome_name}.npz"
    if os.path.exists(key) and not rebuild:
        d = np.load(key, allow_pickle=True)
        return d["X"], d["A"], d["P0"], d["P1"], list(d["names"])

    cfg = load_config(SIM_CONFIG)
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
    P1 = np.concatenate(p1s)
    P0 = np.concatenate(p0s)
    os.makedirs(os.path.dirname(key) or ".", exist_ok=True)
    np.savez(key, X=X, A=A, P0=P0, P1=P1, names=np.array(features.columns))
    return X, A, P0, P1, list(features.columns)


# ----------------------------------------------------------------------
# Nuisance fitting
# ----------------------------------------------------------------------
def _model(kind):
    if kind == "gb":
        return HistGradientBoostingClassifier(max_iter=150, random_state=0)
    return LogisticRegression(max_iter=2000)


def cross_fit_ps(X, A, kind, seed=0):
    oof = np.zeros(len(A))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, A):
        oof[te] = _model(kind).fit(X[tr], A[tr]).predict_proba(X[te])[:, 1]
    return oof


def cross_fit_outcome(X, A, Y, kind, seed=0):
    XA = np.column_stack([X, A])
    p1, p0 = np.zeros(len(Y)), np.zeros(len(Y))
    if Y.sum() < 10 or (1 - Y).sum() < 10:  # degenerate draw
        return np.full(len(Y), Y.mean()), np.full(len(Y), Y.mean())
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(XA, Y):
        m = _model(kind).fit(XA[tr], Y[tr])
        p1[te] = m.predict_proba(np.column_stack([X[te], np.ones(len(te))]))[:, 1]
        p0[te] = m.predict_proba(np.column_stack([X[te], np.zeros(len(te))]))[:, 1]
    return p1, p0


def build_estimators():
    common = dict(effect_type="ATE", treatment_col="exposure",
                  outcome_col="outcome", ps_col="ps")
    return [
        IPW(**common, clip_percentile=1.0),
        AIPW(**common, probas_t1_col="probas_exposed",
             probas_t0_col="probas_control"),
        TMLE(**common, probas_col="probas", probas_t1_col="probas_exposed",
             probas_t0_col="probas_control", clip_percentile=1.0),
    ]


# ----------------------------------------------------------------------
# One replicate
# ----------------------------------------------------------------------
def calibrate_ps(ps, A, method="none", seed=0, bounds=(0.01, 0.99)):
    """Cross-fitted recalibration of propensity scores.

    Flexible learners (gradient boosting especially) return badly calibrated
    probabilities -- on this data raw gb has a calibration slope of 0.16,
    i.e. wildly overconfident. Every inverse-probability method inherits that
    error, so recalibrating is not cosmetic. This mirrors what the repo's own
    calibrate_exp_y.py stage does before estimate.py.

    Calibration is cross-fitted: fitting it on all rows and then scoring the
    same rows would be optimistic. Output is clipped, because isotonic
    regression legitimately returns exact 0 and 1 on held-out folds (it is a
    step function), which would make the inverse weights infinite.
    """
    if method == "none":
        return ps
    ps = np.clip(np.asarray(ps, dtype=float), 1e-6, 1 - 1e-6)
    out = np.zeros(len(ps))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(
            ps.reshape(-1, 1), A):
        if method == "isotonic":
            cal = IsotonicRegression(out_of_bounds="clip").fit(ps[tr], A[tr])
            out[te] = cal.predict(ps[te])
        elif method == "beta":
            # Beta calibration (Kull, Silva Filho & Flach 2017). "abm" is the
            # full three-parameter family and is what the repo's own
            # calibrate_exp_y.py stage uses via corebehrt.functional.trainer.
            cal = BetaCalibration("abm").fit(ps[tr], A[tr])
            out[te] = cal.predict(ps[te])
        else:  # Platt scaling
            z = logit(np.clip(ps, 1e-6, 1 - 1e-6)).reshape(-1, 1)
            cal = LogisticRegression(max_iter=2000).fit(z[tr], A[tr])
            out[te] = cal.predict_proba(z[te])[:, 1]
    return np.clip(out, bounds[0], bounds[1])


def brier(ps, A):
    return float(((np.asarray(ps) - np.asarray(A)) ** 2).mean())


def calibration_slope(ps, A):
    """Slope of A regressed on logit(ps). 1.0 = calibrated, <1 = overconfident."""
    z = logit(np.clip(ps, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    try:
        return float(LogisticRegression(max_iter=2000).fit(z, A).coef_[0][0])
    except Exception:
        return float("nan")


def aipw_eif(A, Y, ps, Y1_hat, Y0_hat):
    """The SAME estimator as CausalEstimate's AIPW, but with an analytic CI.

    Mirrors the library exactly: Hajek (self-normalized) arm weights, no
    clipping. So the two AIPW rows in the output differ only in how the point
    estimate and interval are produced:

      AIPW      point estimate = MEAN over bootstrap resamples
                interval       = percentile bootstrap
      AIPW_EIF  point estimate = the estimate on the actual data
                interval       = influence function, sd(psi)/sqrt(n)

    Those two point estimates converge as n_bootstraps grows but differ at
    finite B. The intervals are the substantive comparison: the percentile
    bootstrap is biased narrow for small B (~20% at B=20, converging by
    B~100), while the influence-function SE does not depend on B at all.

    psi is the inverse-weighted-residual form of AIPW, which is algebraically
    identical to base-minus-augmentation.
    """
    w1, w0 = A / ps, (1 - A) / (1 - ps)
    s1, s0 = w1 / w1.mean(), w0 / w0.mean()
    psi = s1 * (Y - Y1_hat) - s0 * (Y - Y0_hat) + Y1_hat - Y0_hat
    return psi.mean(), psi.std(ddof=1) / np.sqrt(len(psi))


def overlap_diagnostics(ps, lo=0.01, hi=0.99):
    """Share of propensity scores in the tails, and the worst IPW weight."""
    return {
        "pct_ps_below": 100 * float((ps < lo).mean()),
        "pct_ps_above": 100 * float((ps > hi).mean()),
        "pct_ps_outside": 100 * float(((ps < lo) | (ps > hi)).mean()),
        "ps_min": float(ps.min()),
        "ps_max": float(ps.max()),
        "max_weight": float(1.0 / np.minimum(ps, 1 - ps).min()),
    }


def _restrict(X, misspecify, which, cripple_to=1):
    """Return the feature matrix a given nuisance model is allowed to see.

    cripple_to = how many features a crippled model keeps.
      0 -> intercept-only (a dummy zero column); the model can only fit
           marginal rates, which is a guaranteed, total misspecification.
      k -> the first k features.

    Under 'both', the two models are crippled on DIFFERENT features, so
    neither can back into the other's information.

    WARNING: the oracle features are strongly intercorrelated (disease_burden
    vs utilization_intensity is 0.85), so cripple_to=1 is a WEAK
    misspecification -- one feature proxies much of the confounding. Use
    cripple_to=0 for a clean 'everything is broken' demonstration.
    """
    if misspecify == "none":
        return X
    crippled = (misspecify == "both") or (misspecify == which)
    if not crippled:
        return X
    if cripple_to <= 0:
        return np.zeros((X.shape[0], 1))
    if misspecify == "both":
        # different slices for the two models
        cols = range(cripple_to) if which == "ps" else \
            range(cripple_to, 2 * cripple_to)
        cols = [c % X.shape[1] for c in cols]
        return X[:, cols]
    return X[:, :cripple_to]


def one_replicate(rep, X, A, P0, P1, kind, misspecify, n_bootstrap, resample,
                  ps_fixed, quiet=True, cripple_to=1, ps_clip=(0.01, 0.99),
                  calibrate="none"):
    silence(quiet)  # loky workers start clean; re-mute inside the worker
    rng = np.random.default_rng(10_000 + rep)
    n = len(A)

    if resample == "patients":
        idx = rng.integers(0, n, n)
    else:
        idx = np.arange(n)

    Xr, Ar, P0r, P1r = X[idx], A[idx], P0[idx], P1[idx]
    Yr = np.where(Ar == 1, rng.binomial(1, P1r), rng.binomial(1, P0r))

    X_ps = _restrict(Xr, misspecify, "ps", cripple_to)
    X_out = _restrict(Xr, misspecify, "outcome", cripple_to)

    # PS depends only on (X, A): reusable when the cohort is frozen.
    ps = ps_fixed if (resample == "outcomes" and ps_fixed is not None) \
        else cross_fit_ps(X_ps, Ar, kind, seed=rep)
    ps = calibrate_ps(np.asarray(ps), Ar, calibrate, seed=rep, bounds=ps_clip)
    p1, p0 = cross_fit_outcome(X_out, Ar, Yr, kind, seed=rep)

    df = pd.DataFrame({
        "exposure": Ar, "outcome": Yr, "ps": ps,
        "probas_exposed": p1, "probas_control": p0,
        "probas": np.where(Ar == 1, p1, p0),
    })

    rows = []
    naive = Yr[Ar == 1].mean() - Yr[Ar == 0].mean()
    rows.append({"method": "naive", "effect": naive,
                 "CI95_lower": np.nan, "CI95_upper": np.nan})

    # NOTE: with n_bootstraps > 1, CausalEstimate returns the MEAN effect across
    # bootstrap resamples as "effect" -- not the estimate on the actual data.
    # The CI is the percentile interval. So "TMLE" below is a bootstrap-mean
    # point estimate, whereas "TMLE_obs_EIF" is the observed-data estimate with
    # an influence-function CI. They are different point estimates.
    res = MultiEstimator(estimators=build_estimators(), verbose=False)\
        .compute_effects(df, n_bootstraps=n_bootstrap)
    for name, r in res.items():
        rows.append({"method": name, "effect": r["effect"],
                     "CI95_lower": r["CI95_lower"], "CI95_upper": r["CI95_upper"]})

    # Observed-data TMLE + analytic influence-function CI (no bootstrap).
    tmle_obs = TMLE(effect_type="ATE", treatment_col="exposure",
                    outcome_col="outcome", ps_col="ps", probas_col="probas",
                    probas_t1_col="probas_exposed",
                    probas_t0_col="probas_control",
                    clip_percentile=1.0).compute_effect(df)
    rows.append({"method": "TMLE_obs_EIF", "effect": tmle_obs["effect"],
                 "CI95_lower": tmle_obs["CI95_lower"],
                 "CI95_upper": tmle_obs["CI95_upper"]})

    # Same estimator as the AIPW row above, but observed point estimate + EIF CI.
    est, se = aipw_eif(Ar, Yr, ps, p1, p0)
    rows.append({"method": "AIPW_EIF", "effect": est,
                 "CI95_lower": est - 1.96 * se,
                 "CI95_upper": est + 1.96 * se})

    diag = overlap_diagnostics(np.asarray(ps), *ps_clip)
    diag["cal_slope"] = calibration_slope(np.asarray(ps), Ar)
    diag["brier"] = brier(np.asarray(ps), Ar)
    for r in rows:
        r["replicate"] = rep
        r.update(diag)
    return rows


# ----------------------------------------------------------------------
# Benchmark driver
# ----------------------------------------------------------------------
def run_benchmark(replicates=300, outcome="OUTCOME", kind="logit",
                  misspecify="none", n_bootstrap=100, resample="patients",
                  n_jobs=1, save=None, quiet=True, progress=True,
                  cripple_to=1, ps_clip=(0.01, 0.99), calibrate="none"):
    silence(quiet)
    X, A, P0, P1, names = load_dgp(outcome)
    true_ate = float((P1 - P0).mean())

    degenerate = [n for i, n in enumerate(names) if X[:, i].std() == 0]
    if degenerate:
        print(f"  NOTE: constant feature(s) in this dataset: {degenerate} "
              f"-- their DGP coefficients have no effect.")

    ps_fixed = None
    if resample == "outcomes":
        ps_fixed = cross_fit_ps(
            _restrict(X, misspecify, "ps", cripple_to), A, kind, seed=0)

    jobs = (
        delayed(one_replicate)(r, X, A, P0, P1, kind, misspecify,
                               n_bootstrap, resample, ps_fixed, quiet,
                               cripple_to, ps_clip, calibrate)
        for r in range(replicates)
    )
    # return_as="generator" lets tqdm advance as each replicate lands.
    stream = Parallel(n_jobs=n_jobs, verbose=0, return_as="generator")(jobs)
    batches = cast(
        "list[list[dict[str, Any]]]",
        list(tqdm(stream, total=replicates, desc=f"replicates ({misspecify})",
                  unit="rep", disable=not progress)),
    )
    raw = pd.DataFrame([row for batch in batches for row in batch])

    recs = []
    for method, g in raw.groupby("method", sort=False):
        est = g["effect"].to_numpy()
        bias = est - true_ate
        emp_sd = est.std(ddof=1)
        rec = {
            "method": method,
            "mean_est": est.mean(),
            "bias": bias.mean(),
            "bias_mcse": emp_sd / np.sqrt(len(est)),
            "emp_SD": emp_sd,
            "RMSE": np.sqrt((bias ** 2).mean()),
        }
        if g["CI95_lower"].notna().all():
            width = g["CI95_upper"] - g["CI95_lower"]
            mean_se = (width / (2 * 1.96)).mean()
            cov = ((g["CI95_lower"] <= true_ate) &
                   (g["CI95_upper"] >= true_ate)).mean()
            rec.update({
                "mean_SE": mean_se,
                "SE/SD": mean_se / emp_sd,
                "coverage": 100 * cov,
                "cov_mcse": 100 * np.sqrt(cov * (1 - cov) / len(est)),
                "CI_width": width.mean(),
            })
        recs.append(rec)
    summary = pd.DataFrame(recs)
    summary.attrs['true_ate'] = true_ate

    print("\n" + "=" * 96)
    print(f"  MONTE CARLO BENCHMARK   outcome={outcome}  nuisance={kind}  "
          f"misspecified={misspecify}"
          + (f" (cripple_to={cripple_to})" if misspecify != "none" else ""))
    print(f"  replicates={replicates}  bootstrap={n_bootstrap}  "
          f"resample={resample}  n={len(A)}")
    print(f"  TRUE ATE = {true_ate:.4f}")
    d = raw.drop_duplicates("replicate")
    print(f"  OVERLAP (mean over replicates, bounds {ps_clip[0]}-{ps_clip[1]}): "
          f"ps outside = {d.pct_ps_outside.mean():.2f}% "
          f"({d.pct_ps_below.mean():.2f}% below / {d.pct_ps_above.mean():.2f}% above)")
    print(f"           calibration slope = {d.cal_slope.mean():.3f} "
          f"(1.0 = calibrated, <1 = overconfident; calibrate={calibrate})")
    print(f"           ps range [{d.ps_min.mean():.4f}, {d.ps_max.mean():.4f}]   "
          f"worst IPW weight = {d.max_weight.mean():.1f}   "
          f"(no estimator clips; all run clip_percentile=1.0 = NO clipping)")
    print("=" * 96)
    cols = ["method", "mean_est", "bias", "bias_mcse", "emp_SD", "RMSE",
            "mean_SE", "SE/SD", "coverage", "cov_mcse", "CI_width"]
    print(summary[[c for c in cols if c in summary]]
          .to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    print("=" * 96)
    print("  bias_mcse = Monte Carlo SE of the bias (is bias real or noise?)")
    print("  SE/SD     = mean estimated SE / empirical SD  (1.0 = honest variance)")
    print("              NB: bootstrap CIs are percentile-based, so SE is backed")
    print("              out as width/(2*1.96) and assumes rough symmetry.")
    print("  coverage  = % of replicates whose CI contains the TRUE ATE")
    print("  AIPW vs AIPW_EIF = the SAME estimator, two inference routes.")
    print("              AIPW     : bootstrap-mean estimate, percentile CI.")
    print("              AIPW_EIF : observed estimate, influence-function CI.")
    print("              Percentile CIs are biased narrow below ~100")
    print("              bootstraps; the EIF SE does not depend on B.")
    print("  cov_mcse  = Monte Carlo SE of coverage = sqrt(p(1-p)/R), in points.")
    print("              Coverage is only 'off' if it misses 95 by >~2*cov_mcse.\n")

    if save:
        os.makedirs(save, exist_ok=True)
        raw.to_csv(join(save, "benchmark_raw.csv"), index=False)
        summary.to_csv(join(save, "benchmark_summary.csv"), index=False)
        print(f"  saved -> {save}\n")
    return summary, raw


def compare_calibrations(methods=("none", "sigmoid", "isotonic", "beta"),
                         estimators=("IPW", "AIPW", "TMLE", "AIPW_EIF"),
                         **kwargs):
    """Run the benchmark once per calibration method and tabulate the result.

    Two blocks are printed. The first scores the propensity model itself
    (slope, Brier, tail mass, worst weight) -- this is a property of the
    calibration alone, independent of any estimator. The second scores the
    downstream estimators, which is what actually matters: a calibration that
    looks better on Brier score but leaves an estimator biased has not helped.
    """
    kwargs.pop("calibrate", None)
    kwargs.setdefault("save", None)
    quality, performance, true_ate = [], [], None

    for m in methods:
        print(f"\n>>> calibration = {m}")
        summary, raw = run_benchmark(calibrate=m, quiet=True, **kwargs)
        d = raw.drop_duplicates("replicate")
        true_ate = summary.attrs.get("true_ate", true_ate)
        quality.append({
            "calibration": m,
            "cal_slope": d.cal_slope.mean(),
            "brier": d.brier.mean(),
            "pct_outside": d.pct_ps_outside.mean(),
            "max_weight": d.max_weight.mean(),
        })
        for _, row in summary.iterrows():
            if row["method"] in estimators:
                performance.append({
                    "calibration": m,
                    "estimator": row["method"],
                    "bias": row["bias"],
                    "emp_SD": row["emp_SD"],
                    "RMSE": row["RMSE"],
                    "SE/SD": row.get("SE/SD", np.nan),
                    "coverage": row.get("coverage", np.nan),
                })

    q = pd.DataFrame(quality)
    p = pd.DataFrame(performance)

    print("\n" + "=" * 96)
    print("  CALIBRATION QUALITY  (propensity model only)")
    print("=" * 96)
    print(q.to_string(index=False, float_format=lambda v: f"{v:10.4f}"))

    print("\n" + "=" * 96)
    print("  DOWNSTREAM ESTIMATOR PERFORMANCE")
    print("=" * 96)
    for est in estimators:
        sub = p[p.estimator == est]
        if sub.empty:
            continue
        print(f"\n  {est}")
        print(sub.drop(columns=["estimator"])
                 .to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
    print("\n" + "=" * 96)
    print("  A better calibration slope does not guarantee a better estimator.")
    print("  Read the bias/coverage block, not just the Brier score.\n")
    return q, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=300)
    ap.add_argument("--outcome", default="OUTCOME")
    ap.add_argument("--model", dest="kind", default="logit",
                    choices=["logit", "gb"])
    ap.add_argument("--misspecify", default="none",
                    choices=["none", "ps", "outcome", "both"],
                    help="'both' cripples PS and outcome on DIFFERENT features; "
                         "no estimator is expected to survive it.")
    ap.add_argument("--n-bootstrap", type=int, default=100)
    ap.add_argument("--resample", default="patients",
                    choices=["patients", "outcomes"])
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--save", default="./outputs/causal/estimator_benchmark")
    ap.add_argument("--compare-calibration", action="store_true",
                    help="Run once per calibration method and tabulate "
                         "calibration quality against estimator performance.")
    ap.add_argument("--verbose", action="store_true",
                    help="Restore warnings and library logging (default: muted).")
    ap.add_argument("--no-progress", action="store_true",
                    help="Disable the tqdm progress bar.")
    ap.add_argument("--calibrate", default="none",
                    choices=["none", "isotonic", "sigmoid", "beta"],
                    help="Cross-fitted recalibration of the propensity score. "
                         "Strongly recommended with --model gb.")
    ap.add_argument("--ps-clip", type=float, nargs=2, default=[0.01, 0.99],
                    metavar=("LO", "HI"),
                    help="Absolute propensity bounds used for the overlap "
                         "report and to bound calibrated scores. Estimators "
                         "themselves are unclipped. Default 0.01 0.99.")
    ap.add_argument("--cripple-to", type=int, default=1,
                    help="Features a misspecified model keeps. 0 = intercept-"
                         "only (total misspecification). Default 1 is WEAK "
                         "because the oracle features are intercorrelated.")
    a = ap.parse_args()
    common = dict(replicates=a.replicates, outcome=a.outcome, kind=a.kind,
                  misspecify=a.misspecify, n_bootstrap=a.n_bootstrap,
                  resample=a.resample, n_jobs=a.n_jobs,
                  progress=not a.no_progress, cripple_to=a.cripple_to,
                  ps_clip=tuple(a.ps_clip))
    if a.compare_calibration:
        compare_calibrations(**common)
    else:
        run_benchmark(save=a.save, quiet=not a.verbose,
                      calibrate=a.calibrate, **common)


if __name__ == "__main__":
    main()