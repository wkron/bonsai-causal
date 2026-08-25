"""
Run IPW / AIPW / TMLE on the semi-synthetic simulation output, without the
transformer pipeline.

Nuisance models (propensity + outcome) are fitted on the same oracle features
the DGP used, via the repo's own extract_oracle_features(). That makes this a
*well-specified* benchmark: every estimator should land near the true ATE.

Usage:
    python try_estimators.py                       # OUTCOME, logistic nuisances
    python try_estimators.py --outcome OUTCOME_NULL
    python try_estimators.py --model gb            # gradient boosting nuisances
    python try_estimators.py --misspecify ps       # break the PS model on purpose
"""

import argparse
from os.path import join

import numpy as np
import pandas as pd
from CausalEstimate import MultiEstimator
from CausalEstimate.estimators import AIPW, IPW, TMLE
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from corebehrt.constants.data import PID_COL
from corebehrt.modules.features.loader import ShardLoader
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.simulation.config_semisynthetic import (
    create_semisynthetic_config,
)
from corebehrt.modules.simulation.semisynthetic_simulator import (
    SemiSyntheticCausalSimulator,
)

SIM_CONFIG = "./corebehrt/configs/causal/simulate_semisynthetic.yaml"


def build_analysis_frame(outcome_name: str):
    """Re-run feature extraction to get the oracle features, and join the
    simulated outcomes / counterfactuals produced by the simulator."""
    cfg = load_config(SIM_CONFIG)
    sim_cfg = create_semisynthetic_config(cfg)
    sim = SemiSyntheticCausalSimulator(sim_cfg)

    # Pass 1: global feature means/stds (same two-pass logic as the simulator)
    sim.compute_global_feature_stats(ShardLoader(cfg.paths.data, cfg.paths.splits))

    frames = []
    for shard, _ in ShardLoader(cfg.paths.data, cfg.paths.splits)():
        out = sim.extract_features_and_probabilities(shard)
        if out is None:
            continue
        features_df, pids, _is_exposed, _probas, _tau = out
        frames.append(features_df)
    features = pd.concat(frames)  # already indexed by subject_id

    cf = pd.read_csv(join(cfg.paths.outcomes, "counterfactuals.csv"))
    ite = pd.read_csv(join(cfg.paths.outcomes, "ite.csv"))

    df = features.join(cf.set_index(PID_COL), how="inner")
    df = df.join(ite.set_index(PID_COL)[[f"ite_{outcome_name}"]], how="inner")
    df = df.rename(columns={f"outcome_{outcome_name}": "outcome"})
    df["true_ite"] = df[f"ite_{outcome_name}"]
    feature_cols = [c for c in features.columns]
    return df.reset_index(), feature_cols


def make_model(kind: str):
    if kind == "gb":
        return HistGradientBoostingClassifier(max_iter=200, random_state=0)
    return LogisticRegression(max_iter=2000)


def cross_fit(X, y, kind, seed=0):
    """Out-of-fold predictions, so nuisance models don't overfit the same rows
    the estimator then evaluates on."""
    oof = np.zeros(len(y))
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        m = make_model(kind)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def cross_fit_outcome(X, A, y, kind, seed=0):
    """Outcome regression with treatment as a feature; returns predictions
    under A=1 and A=0 for every patient (out-of-fold)."""
    XA = np.column_stack([X, A])
    p1 = np.zeros(len(y))
    p0 = np.zeros(len(y))
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, te in skf.split(XA, y):
        m = make_model(kind)
        m.fit(XA[tr], y[tr])
        X1 = np.column_stack([X[te], np.ones(len(te))])
        X0 = np.column_stack([X[te], np.zeros(len(te))])
        p1[te] = m.predict_proba(X1)[:, 1]
        p0[te] = m.predict_proba(X0)[:, 1]
    return p1, p0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", default="OUTCOME")
    ap.add_argument("--model", default="logit", choices=["logit", "gb"])
    ap.add_argument("--n-bootstrap", type=int, default=200)
    ap.add_argument(
        "--misspecify",
        default="none",
        choices=["none", "ps", "outcome"],
        help="Deliberately cripple one nuisance model to see which estimators survive.",
    )
    args = ap.parse_args()

    df, feature_cols = build_analysis_frame(args.outcome)
    X = df[feature_cols].to_numpy(dtype=float)
    A = df["exposure"].to_numpy(dtype=int)
    Y = df["outcome"].to_numpy(dtype=int)

    # A crippled model sees only one weak feature instead of all ten.
    X_ps = X[:, [0]] if args.misspecify == "ps" else X
    X_out = X[:, [0]] if args.misspecify == "outcome" else X

    df["ps"] = cross_fit(X_ps, A, args.model)
    p1, p0 = cross_fit_outcome(X_out, A, Y, args.model)
    df["probas_exposed"] = p1
    df["probas_control"] = p0
    df["probas"] = np.where(A == 1, p1, p0)

    true_ate = df["true_ite"].mean()
    naive = Y[A == 1].mean() - Y[A == 0].mean()

    estimators = [
        IPW(effect_type="ATE", treatment_col="exposure", outcome_col="outcome",
            ps_col="ps", clip_percentile=1.0),
        AIPW(effect_type="ATE", treatment_col="exposure", outcome_col="outcome",
             ps_col="ps", probas_t1_col="probas_exposed",
             probas_t0_col="probas_control"),
        TMLE(effect_type="ATE", treatment_col="exposure", outcome_col="outcome",
             ps_col="ps", probas_col="probas", probas_t1_col="probas_exposed",
             probas_t0_col="probas_control", clip_percentile=1.0),
    ]
    results = MultiEstimator(estimators=estimators, verbose=False).compute_effects(
        df, n_bootstraps=args.n_bootstrap
    )

    print("\n" + "=" * 74)
    print(f"  outcome={args.outcome}   nuisance={args.model}   "
          f"misspecified={args.misspecify}   n={len(df)}")
    print("=" * 74)
    print(f"  PS overlap: [{df.ps.min():.3f}, {df.ps.max():.3f}]   "
          f"exposed mean {df.ps[A == 1].mean():.3f} / "
          f"control mean {df.ps[A == 0].mean():.3f}")
    print("-" * 74)
    print(f"{'method':<14}{'estimate':>10}{'95% CI':>20}{'bias':>10}{'covers?':>10}")
    print("-" * 74)
    print(f"{'TRUE ATE':<14}{true_ate:>10.4f}{'':>20}{'':>10}{'':>10}")
    print(f"{'naive (crude)':<14}{naive:>10.4f}{'':>20}{naive - true_ate:>10.4f}"
          f"{'':>10}")
    for name, r in results.items():
        lo, hi = r["CI95_lower"], r["CI95_upper"]
        covers = "yes" if lo <= true_ate <= hi else "NO"
        print(f"{name:<14}{r['effect']:>10.4f}"
              f"{f'[{lo:.4f}, {hi:.4f}]':>20}"
              f"{r['effect'] - true_ate:>10.4f}{covers:>10}")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()