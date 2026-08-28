#!/usr/bin/env python3
"""Generate the XGBoost model fixtures the Go scorer is tested against.

Every fixture is a pair: a model saved by XGBoost in its own JSON format, and a
file of predictions produced by XGBoost itself for a fixed input matrix. The Go
scorer in internal/runtime/xgboost is required to reproduce those numbers, so
the fixtures are the specification — if XGBoost changes how it scores a tree,
the Go tests fail rather than quietly drifting.

Both raw margins and post-link probabilities are recorded. The margin catches a
tree-traversal or base_score bug on its own; comparing only probabilities would
let a sigmoid hide a small margin error in the saturated tails.

Run:  python3 tools/fixtures/generate.py
"""

import json
import pathlib

import numpy as np
import xgboost as xgb

OUT = pathlib.Path(__file__).resolve().parents[2] / "testdata" / "xgboost"


def dump(name, model, X, *, objective):
    """Save a model plus XGBoost's own predictions for X."""
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_model(str(OUT / f"{name}.model.json"))

    booster = model.get_booster()
    dm = xgb.DMatrix(X, missing=np.nan)
    margin = booster.predict(dm, output_margin=True)
    prob = booster.predict(dm)

    payload = {
        "objective": objective,
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        # None is how JSON carries NaN here; json.dump would otherwise emit a
        # bare NaN token, which Go's encoding/json rejects.
        "inputs": [[None if np.isnan(v) else float(v) for v in row] for row in X],
        "margins": [[float(v)] if margin.ndim == 1 else [float(c) for c in v] for v in margin],
        "predictions": [[float(v)] if prob.ndim == 1 else [float(c) for c in v] for v in prob],
    }
    (OUT / f"{name}.expected.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {name}: {X.shape[0]} rows x {X.shape[1]} features ({objective})")


def main():
    rs = np.random.RandomState(20260828)

    # 1. Binary classification with a non-default base_score. 0.5 would make the
    #    intercept logit(0.5) = 0 and hide whether base_score is applied in
    #    probability or margin space, which is exactly the bug worth catching.
    X = rs.randn(400, 6).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] - X[:, 3] > 0).astype(int)
    m = xgb.XGBClassifier(n_estimators=25, max_depth=4, base_score=0.35, random_state=0)
    m.fit(X, y)
    dump("binary_logistic", m, X[:64], objective="binary:logistic")

    # 2. Missing values. Training on a matrix that is 30% NaN is what makes
    #    XGBoost actually learn default directions; a model trained on dense
    #    data leaves every default_left at 0 and the NaN branch untested.
    Xm = rs.randn(600, 5).astype(np.float32)
    Xm[rs.rand(600, 5) < 0.3] = np.nan
    ym = (np.nan_to_num(Xm[:, 0]) + np.nan_to_num(Xm[:, 1]) > 0).astype(int)
    mm = xgb.XGBClassifier(n_estimators=20, max_depth=4, base_score=0.42, random_state=0)
    mm.fit(Xm, ym)
    dump("binary_missing", mm, Xm[:64], objective="binary:logistic")

    # 3. Squared-error regression: identity link, so a link applied by mistake
    #    shows up immediately.
    Xr = rs.randn(400, 4).astype(np.float32)
    yr = 3.0 * Xr[:, 0] - 2.0 * Xr[:, 2] + rs.randn(400) * 0.1
    mr = xgb.XGBRegressor(n_estimators=20, max_depth=4, random_state=0)
    mr.fit(Xr, yr)
    dump("reg_squarederror", mr, Xr[:64], objective="reg:squarederror")

    # 4. Multi-class softprob. Trees are interleaved by class via tree_info and
    #    base_score is a per-class vector already in margin space.
    Xc = rs.randn(500, 5).astype(np.float32)
    yc = rs.randint(0, 4, 500)
    mc = xgb.XGBClassifier(
        n_estimators=8, max_depth=3, objective="multi:softprob", num_class=4, random_state=0
    )
    mc.fit(Xc, yc)
    dump("multi_softprob", mc, Xc[:64], objective="multi:softprob")

    # 5. Poisson regression: an exp link, so the intercept is log(base_score)
    #    rather than logit(base_score). Included because the rule the loader
    #    implements — the intercept is the inverse of the objective's output
    #    transform applied to base_score — is only worth stating if more than
    #    one link exercises it.
    Xp = rs.randn(400, 4).astype(np.float32)
    yp = rs.poisson(np.exp(0.5 * Xp[:, 0] + 0.2 * Xp[:, 1]))
    mp = xgb.XGBRegressor(
        n_estimators=15, max_depth=3, objective="count:poisson", random_state=0
    )
    mp.fit(Xp, yp)
    dump("count_poisson", mp, Xp[:64], objective="count:poisson")

    # 6. multi:softmax returns a class index rather than a distribution, so the
    #    Go side has to argmax the same margins it would otherwise softmax.
    Xk = rs.randn(400, 5).astype(np.float32)
    yk = rs.randint(0, 3, 400)
    mk = xgb.XGBClassifier(
        n_estimators=8, max_depth=3, objective="multi:softmax", num_class=3, random_state=0
    )
    mk.fit(Xk, yk)
    dump("multi_softmax", mk, Xk[:64], objective="multi:softmax")

    # 7. A single stump. Small enough to reason about by hand when a larger
    #    fixture fails and it is not obvious whether the bug is traversal or
    #    aggregation.
    Xs = rs.randn(100, 2).astype(np.float32)
    ys = (Xs[:, 0] > 0).astype(int)
    ms = xgb.XGBClassifier(n_estimators=1, max_depth=1, base_score=0.5, random_state=0)
    ms.fit(Xs, ys)
    dump("binary_stump", ms, Xs[:16], objective="binary:logistic")


if __name__ == "__main__":
    main()
