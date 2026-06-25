#!/usr/bin/env python3
"""
訓練汽車/機車估價模型
Usage: python scripts/train_model.py --type auto|moto
"""

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

GRADE_MAP = {
    "A+": 7, "A": 6, "B+": 5, "B": 4, "C": 3, "D": 2, "E": 1, "N": 0,
    "A+.W": 7, "A.W": 6, "B+.W": 5, "B.W": 4, "C.W": 3, "D.W": 2, "E.W": 1, "N.W": 0,
}


class QuantileRandomForest:
    """分位數隨機森林。每棵樹預測單一值，最終取所有樹預測的分位數。"""

    def __init__(self, quantile=0.5, n_estimators=200, max_depth=15,
                 min_samples_leaf=5, random_state=42, n_jobs=-1):
        self.quantile = quantile
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.trees = []
        self._fitted = False

    def fit(self, X, y):
        """訓練隨機森林（每棵樹學習同一分位數）"""
        np.random.seed(self.random_state)
        X = np.asarray(X)
        y = np.asarray(y)
        n = len(y)
        self.trees = []
        for i in range(self.n_estimators):
            idx = np.random.choice(n, size=n, replace=True)
            X_boot = X[idx]
            y_boot = y[idx]
            tree = RandomForestRegressor(
                n_estimators=1,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                bootstrap=False,
                random_state=self.random_state + i,
                criterion="absolute_error",
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        self._fitted = True
        return self

    def predict(self, X):
        """所有樹預測結果的分位數"""
        X = np.asarray(X)
        preds = np.array([tree.predict(X) for tree in self.trees])
        return np.percentile(preds, self.quantile * 100, axis=0)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def compute_age_years(row):
    try:
        ad = pd.to_datetime(row["auction_date"])
        return ((ad.year - int(row["year"])) * 12 + (ad.month - int(row["month"]))) / 12.0
    except:
        return np.nan


def prepare_features_auto(df):
    df = df.copy()
    df["age_years"] = df.apply(compute_age_years, axis=1)
    df["mileage_km"] = df["mileage_km"].fillna(0).astype(float)
    df["mileage_available"] = df["mileage_available"].fillna(0).astype(int)
    df["cc"] = df["cc"].fillna(df["cc"].median()).astype(float)
    df["grade_enc"] = df["grade"].map(GRADE_MAP).fillna(0).astype(int)
    df["tax_total"] = (df["tax"].fillna(0) + df["violation"].fillna(0) + df["strong_violation"].fillna(0)).astype(float)
    df["is_automatic"] = df["transmission"].map({"自": 1, "手自": 1, "手": 0}).fillna(0).astype(int)
    df["brand_model"] = df["brand"].astype(str) + "_" + df["model"].astype(str)
    df["auction_year"] = pd.to_datetime(df["auction_date"]).dt.year.astype(int)
    le_brand = LabelEncoder()
    df["brand_enc"] = le_brand.fit_transform(df["brand"].astype(str))
    le_model = LabelEncoder()
    df["model_enc"] = le_model.fit_transform(df["brand_model"].astype(str))
    return df, le_brand, le_model


def prepare_features_moto(df):
    df = df.copy()
    df["age_years"] = df.apply(compute_age_years, axis=1)
    df["mileage_km"] = df["mileage_km"].fillna(0).astype(float)
    df["mileage_available"] = df["mileage_available"].fillna(0).astype(int)
    df["cc"] = df["cc"].fillna(df["cc"].median()).astype(float)
    df["grade_enc"] = df["grade"].map(GRADE_MAP).fillna(0).astype(int)
    df["brand_model"] = df["brand"].astype(str) + "_" + df["model"].astype(str)
    le_brand = LabelEncoder()
    df["brand_enc"] = le_brand.fit_transform(df["brand"].astype(str))
    le_model = LabelEncoder()
    df["model_enc"] = le_model.fit_transform(df["brand_model"].astype(str))
    return df, le_brand, le_model


def impute_mileage_auto(df):
    df = df.copy()
    df["mileage_imputed"] = df["mileage_km"].copy()
    missing = df["mileage_available"] == 0
    for idx in df[missing].index:
        row = df.loc[idx]
        for fallback_cols in [["year", "brand", "grade"], ["brand", "grade"], []]:
            mask = (df["mileage_available"] == 1)
            for col in fallback_cols:
                if col:
                    mask &= df[col] == row[col]
            subset = df[mask]
            if len(subset) >= 3:
                df.loc[idx, "mileage_imputed"] = subset["mileage_km"].median()
                break
    return df


AUTO_FEATURE_COLS = ["age_years", "mileage_imputed", "mileage_available",
                      "cc", "brand_enc", "model_enc", "grade_enc",
                      "tax_total", "is_automatic", "auction_year"]

MOTO_FEATURE_COLS = ["age_years", "mileage_imputed", "mileage_available",
                     "cc", "brand_enc", "model_enc", "grade_enc"]


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true, y_pred, lower, upper):
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "Coverage_95": np.mean((y_true >= lower) & (y_true <= upper)) * 100,
        "CP_score": (np.mean(y_pred) - np.mean(y_true)) / np.mean(y_pred) * 100,
    }


def print_metrics(metrics, prefix=""):
    print(f"\n{'=' * 50}")
    if prefix:
        print(f"{prefix} Metrics")
        print(f"{'=' * 50}")
    for k, v in metrics.items():
        if k == "MAE":
            print(f"  {k}: NT$ {v:,.0f}")
        elif k in ("MAPE", "Coverage_95", "CP_score"):
            print(f"  {k}: {v:.2f}%")
        else:
            print(f"  {k}: {v:.4f}")
    print(f"{'=' * 50}")


# ============================================================
# PREDICTOR CLASSES
# ============================================================

class _BasePredictor:
    def __init__(self, model_path):
        # Ensure QuantileRandomForest is accessible from __main__ (used by pickle)
        import __main__, scripts.train_model as stm
        __main__.QuantileRandomForest = stm.QuantileRandomForest

        with open(model_path, "rb") as f:
            a = pickle.load(f)
        self.rf_median = a["rf_median"]
        self.rf_lower = a["rf_lower"]
        self.rf_upper = a["rf_upper"]
        self.feature_cols = a["feature_cols"]

    def _widen(self, lo, hi, avail):
        if avail == 0:
            w = hi - lo
            c = (hi + lo) / 2
            lo, hi = c - w * 0.9, c + w * 0.9
        return lo, hi

    def _prepare(self, d):
        # Build full row with all required CSV columns
        defaults = {
            "year": 2020, "month": 1, "cc": 2000, "grade": "C",
            "mileage_km": 50000, "mileage_available": 1,
            "transmission": "自", "tax": 0, "violation": 0, "strong_violation": 0,
            "auction_date": "2026-01-01",
        }
        row = {**defaults, **d}
        df = pd.DataFrame([row])
        # mileage_imputed is a training feature — ensure it always exists
        df["mileage_imputed"] = df["mileage_km"]
        ad = pd.to_datetime(row["auction_date"])
        df["auction_date"] = ad
        df["auction_year"] = ad.year
        df["tax_total"] = float(row.get("tax_total", 0))
        df["is_automatic"] = {"自": 1, "手自": 1, "手": 0}.get(str(row.get("transmission", "")), 0)
        return df[self.feature_cols].fillna(0).values

    def predict(self, input_dict):
        X = self._prepare(input_dict)
        med = float(self.rf_median.predict(X)[0])
        lo = float(self.rf_lower.predict(X)[0])
        hi = float(self.rf_upper.predict(X)[0])
        lo, hi = self._widen(lo, hi, input_dict.get("mileage_available", 1))
        return {"price_estimate": med, "lower": round(lo, -3), "upper": round(hi, -3)}


class AutomobilePredictor(_BasePredictor):
    def __init__(self, model_path="data/models/automobile_v1.pkl"):
        super().__init__(model_path)

    def _prepare(self, d):
        defaults = {
            "year": 2020, "month": 1, "cc": 2000, "grade": "C",
            "mileage_km": 50000, "mileage_available": 1,
            "transmission": "自", "tax": 0, "violation": 0, "strong_violation": 0,
            "auction_date": "2026-01-01",
        }
        row = {**defaults, **d}
        df = pd.DataFrame([row])
        ad = pd.to_datetime(row["auction_date"])
        df["auction_date"] = ad
        df["auction_year"] = ad.year
        df["tax_total"] = float(row.get("tax_total", 0))
        df["is_automatic"] = {"自": 1, "手自": 1, "手": 0}.get(str(row.get("transmission", "")), 0)
        # mileage_imputed must exist (training uses it); mileage_available=0 uses median below
        df["mileage_imputed"] = df["mileage_km"]
        if row.get("mileage_available", 1) == 0:
            # Use global median (mimics impute_mileage_auto fallback)
            df.loc[0, "mileage_imputed"] = 80000
        df, _, _ = prepare_features_auto(df)
        return df[self.feature_cols].fillna(0).values


class MotorcyclePredictor(_BasePredictor):
    def __init__(self, model_path="data/models/motorcycle_v1.pkl"):
        super().__init__(model_path)

    def _prepare(self, d):
        defaults = {
            "year": 2020, "month": 1, "cc": 150, "grade": "C",
            "mileage_km": 30000, "mileage_available": 1,
            "auction_date": "2026-01-01",
        }
        row = {**defaults, **d}
        df = pd.DataFrame([row])
        ad = pd.to_datetime(row["auction_date"])
        df["auction_date"] = ad
        df["auction_year"] = ad.year
        df["mileage_imputed"] = df["mileage_km"]
        df, _, _ = prepare_features_moto(df)
        return df[self.feature_cols].fillna(0).values


# ============================================================
# TRAINING
# ============================================================

def train_model(vehicle_type="auto", data_path=None, model_save_path=None):
    repo_root = Path(__file__).parent.parent.resolve()
    data_dir = Path(data_path) if data_path else repo_root / "data" / "parsed"
    model_dir = Path(model_save_path) if model_save_path else repo_root / "data" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / ("automobiles.csv" if vehicle_type == "auto" else "motorcycles.csv")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} records from {csv_path}")

    if vehicle_type == "moto":
        df = df[df["brand"] != "GOGORO"].reset_index(drop=True)
        print(f"[INFO] Excluded GOGORO: {len(df)} records remain")

    if vehicle_type == "auto":
        df, le_brand, le_model = prepare_features_auto(df)
        df = impute_mileage_auto(df)
    else:
        df, le_brand, le_model = prepare_features_moto(df)
        missing = df["mileage_available"] == 0
        if missing.sum():
            med = df[df["mileage_available"] == 1]["mileage_km"].median()
            df.loc[missing, "mileage_imputed"] = med

    feat_cols = AUTO_FEATURE_COLS if vehicle_type == "auto" else MOTO_FEATURE_COLS
    X = df[feat_cols].fillna(0)
    y = df["price"].values.astype(float)

    n = len(df)
    np.random.seed(42)
    perm = np.random.permutation(n)
    tr_idx, te_idx = perm[:int(n * 0.8)], perm[int(n * 0.8):]
    X_tr, X_te = X.iloc[tr_idx].values, X.iloc[te_idx].values
    y_tr, y_te = y[tr_idx], y[te_idx]
    miss_te = df.iloc[te_idx]["mileage_available"].values == 0

    print(f"[INFO] Train: {len(X_tr)}, Test: {len(X_te)}")
    print(f"[INFO] Training 3 quantile models (q=0.025, 0.5, 0.975)...")

    rf_m = QuantileRandomForest(0.5, 200, 15, 5, 42).fit(X_tr, y_tr)
    rf_l = QuantileRandomForest(0.025, 200, 15, 5, 42).fit(X_tr, y_tr)
    rf_u = QuantileRandomForest(0.975, 200, 15, 5, 42).fit(X_tr, y_tr)

    y_pred = rf_m.predict(X_te)
    y_lo = rf_l.predict(X_te)
    y_hi = rf_u.predict(X_te)

    if vehicle_type == "auto" and miss_te.sum():
        print(f"[INFO] Widening interval for {miss_te.sum()} imputed-mileage test samples")
        w = y_hi - y_lo
        c = (y_hi + y_lo) / 2
        y_lo = y_lo.copy()
        y_hi = y_hi.copy()
        y_lo[miss_te] = c[miss_te] - w[miss_te] * 0.9
        y_hi[miss_te] = c[miss_te] + w[miss_te] * 0.9

    m = compute_metrics(y_te, y_pred, y_lo, y_hi)
    print_metrics(m, prefix="Automobile" if vehicle_type == "auto" else "Motorcycle")

    model_path = model_dir / ("automobile_v1.pkl" if vehicle_type == "auto" else "motorcycle_v1.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "rf_median": rf_m, "rf_lower": rf_l, "rf_upper": rf_u,
            "le_brand": le_brand, "le_model": le_model,
            "feature_cols": feat_cols, "grade_map": GRADE_MAP,
            "vehicle_type": vehicle_type,
        }, f)
    print(f"[INFO] Saved: {model_path}")
    return m


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--type", choices=["auto", "moto"], required=True)
    p.add_argument("--data", default=None)
    p.add_argument("--output", default=None)
    args = p.parse_args()
    print(f"\n{'#' * 56}\n# Training {'Automobile' if args.type == 'auto' else 'Motorcycle'} Model\n{'#' * 56}")
    train_model(args.type, args.data, args.output)
