"""Train, compare and tune regression models on the California Housing dataset.

Steps
1. stratified 80/20 split on income category (the test set is touched exactly once, at the end)
2. EDA plots
3. 10-fold cross-validated RMSE for 5 models, with and without feature engineering
4. randomised hyper-parameter search for the best model family
5. final test-set RMSE / MAE / R^2 with a 95% confidence interval, permutation feature importance,
   residual plot, and the fitted pipeline saved to models/housing_model.joblib

Usage:  python src/train.py [--quick]     ->  results/, models/
"""
import argparse
import json
import os
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.stats import loguniform, randint  # noqa: E402
from sklearn.dummy import DummyRegressor  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LinearRegression, Ridge  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error  # noqa: E402
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit, cross_val_score  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from features import TARGET, make_model  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES, MODELS = os.path.join(ROOT, "results"), os.path.join(ROOT, "models")
SEED = 42


def load_split():
    df = pd.read_csv(os.path.join(ROOT, "data", "housing.csv"))
    income_cat = pd.cut(df["median_income"], bins=[0, 1.5, 3, 4.5, 6, np.inf], labels=[1, 2, 3, 4, 5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(split.split(df, income_cat))
    return df, df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True), income_cat


def eda(df):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(df["longitude"], df["latitude"], s=df["population"] / 120, c=df[TARGET] / 1000,
                    cmap="viridis", alpha=0.4)
    fig.colorbar(sc, label="Median house value ($k)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("California districts (size = population)")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "geo_prices.png"), dpi=120)
    plt.close(fig)

    corr = df.select_dtypes("number").corr()[TARGET].drop(TARGET).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh(corr.index, corr.values, color=["#d93025" if v < 0 else "#1a73e8" for v in corr.values])
    ax.set_title("Correlation with median house value")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "correlations.png"), dpi=120)
    plt.close(fig)
    return corr.round(3).to_dict()


def candidates(quick):
    n_trees = 50 if quick else 150
    return {
        "Mean baseline": DummyRegressor(),
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(random_state=SEED),
        "Random Forest": RandomForestRegressor(n_estimators=n_trees, n_jobs=-1, random_state=SEED),
        "Gradient Boosting": HistGradientBoostingRegressor(random_state=SEED),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer folds/trees/search iterations")
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    folds = 5 if args.quick else 10

    df, train, test, income_cat = load_split()
    X, y = train.drop(columns=TARGET), train[TARGET]
    X_test, y_test = test.drop(columns=TARGET), test[TARGET]
    out = {"rows": len(df), "train_rows": len(train), "test_rows": len(test),
           "missing_total_bedrooms": int(df["total_bedrooms"].isna().sum()),
           "capped_at_500001": int((df[TARGET] >= 500001).sum())}
    full_props = income_cat.value_counts(normalize=True).sort_index()
    test_props = pd.cut(test["median_income"], [0, 1.5, 3, 4.5, 6, np.inf], labels=[1, 2, 3, 4, 5]
                        ).value_counts(normalize=True).sort_index()
    out["stratification_max_error_pct"] = float((abs(test_props - full_props) / full_props).max() * 100)
    out["correlation_with_target"] = eda(df)

    # 1. model comparison: 10-fold CV RMSE, plain vs engineered features
    cv = {}
    for engineered in (False, True):
        for name, est in candidates(args.quick).items():
            t = time.perf_counter()
            scores = -cross_val_score(make_model(est, engineered), X, y, cv=folds,
                                      scoring="neg_root_mean_squared_error", n_jobs=1)
            key = f"{name} ({'engineered' if engineered else 'plain'})"
            cv[key] = {"rmse_mean": float(scores.mean()), "rmse_std": float(scores.std()),
                       "seconds": round(time.perf_counter() - t, 1)}
            print(f"{key:40s} CV RMSE {scores.mean():>9,.0f} ± {scores.std():,.0f}")
    out["cross_validation"] = cv

    # 2. tune gradient boosting (best family in the comparison) with randomised search
    search = RandomizedSearchCV(
        make_model(HistGradientBoostingRegressor(random_state=SEED)),
        {"model__learning_rate": loguniform(0.03, 0.3), "model__max_iter": randint(200, 800),
         "model__max_leaf_nodes": randint(15, 80), "model__min_samples_leaf": randint(10, 60),
         "model__l2_regularization": loguniform(1e-3, 10), "prep__geo__n_clusters": randint(5, 30)},
        n_iter=8 if args.quick else 30, cv=min(folds, 5), scoring="neg_root_mean_squared_error",
        random_state=SEED, n_jobs=-1)
    t = time.perf_counter()
    search.fit(X, y)
    out["tuning"] = {"best_cv_rmse": float(-search.best_score_), "seconds": round(time.perf_counter() - t, 1),
                     "best_params": {k: (float(v) if isinstance(v, (float, np.floating)) else int(v))
                                     for k, v in search.best_params_.items()}}
    print("Tuned CV RMSE", f"{-search.best_score_:,.0f}", search.best_params_)
    model = search.best_estimator_

    # 3. final evaluation on the untouched test set
    pred = model.predict(X_test)
    sq = (pred - y_test) ** 2
    ci = np.sqrt(stats.t.interval(0.95, len(sq) - 1, loc=sq.mean(), scale=stats.sem(sq)))
    baseline_rmse = root_mean_squared_error(y_test, np.full_like(pred, y.mean()))
    lin = make_model(LinearRegression(), engineered=False).fit(X, y)
    lin_rmse = root_mean_squared_error(y_test, lin.predict(X_test))
    out["test"] = {"rmse": float(root_mean_squared_error(y_test, pred)), "rmse_95ci": [float(ci[0]), float(ci[1])],
                   "mae": float(mean_absolute_error(y_test, pred)), "r2": float(r2_score(y_test, pred)),
                   "median_abs_pct_error": float(np.median(np.abs(pred - y_test) / y_test) * 100),
                   "within_20pct": float(np.mean(np.abs(pred - y_test) / y_test <= 0.2) * 100),
                   "mean_baseline_rmse": float(baseline_rmse), "linear_baseline_rmse": float(lin_rmse),
                   "linear_baseline_r2": float(r2_score(y_test, lin.predict(X_test)))}
    print("Test", out["test"])

    # 4. permutation importance on the raw input columns
    imp = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=SEED,
                                 scoring="neg_root_mean_squared_error", n_jobs=-1)
    importance = pd.Series(imp.importances_mean, index=X_test.columns).sort_values()
    out["permutation_importance_rmse_increase"] = importance.round(0).to_dict()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh(importance.index, importance.values / 1000, color="#1a73e8")
    ax.set_xlabel("Test RMSE increase when shuffled ($k)")
    ax.set_title("Permutation feature importance")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "feature_importance.png"), dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(y_test / 1000, pred / 1000, s=4, alpha=0.3)
    axes[0].plot([0, 500], [0, 500], "r--", lw=1)
    axes[0].set_xlabel("Actual ($k)")
    axes[0].set_ylabel("Predicted ($k)")
    axes[0].set_title(f"Test set: R² = {out['test']['r2']:.3f}")
    names = [k for k in cv if "baseline" not in k]
    axes[1].barh([n.replace(" (", "\n(") for n in names], [cv[n]["rmse_mean"] / 1000 for n in names],
                 xerr=[cv[n]["rmse_std"] / 1000 for n in names],
                 color=["#1a73e8" if "engineered" in n else "#9aa0a6" for n in names])
    axes[1].axvline(out["tuning"]["best_cv_rmse"] / 1000, color="#188038", ls="--", label="tuned GBM")
    axes[1].set_xlabel(f"{folds}-fold CV RMSE ($k)")
    axes[1].legend(loc="lower right")
    axes[1].tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "model_comparison.png"), dpi=120)
    plt.close(fig)

    joblib.dump(model, os.path.join(MODELS, "housing_model.joblib"), compress=3)
    with open(os.path.join(RES, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
