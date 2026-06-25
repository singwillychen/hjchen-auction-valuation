#!/usr/bin/env python3
"""
訓練汽車/機車估價模型
Usage: python scripts/train_model.py --type auto|moto
"""

import argparse
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ============================================================
# GRADE MAP
# ============================================================
GRADE_MAP = {
    "A+": 7,
    "A": 6,
    "B+": 5,
    "B": 4,
    "C": 3,
    "D": 2,
    "E": 1,
    "N": 0,
    # 带 .W 后缀的也映射到相同等级
    "A+.W": 7,
    "A.W": 6,
    "B+.W": 5,
    "B.W": 4,
    "C.W": 3,
    "D.W": 2,
    "E.W": 1,
    "N.W": 0,
}

# ============================================================
# CUSTOM QUANTILE RANDOM FOREST
# ============================================================

class QuantileRandomForest:
    """
    使用_pinball_loss_gradient近似實現的分位數隨機森林。
    每棵樹預測單一值（葉節點均值），最終預測為所有樹預測的中位數。
    這是一種近似但計算效率高的方法。
    """

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
            # Bootstrap 採樣
            idx = np.random.choice(n, size=n, replace=True)
            X_boot = X[idx]
            y_boot = y[idx]

            # 建立 CART 樹，使用 absolute_error（對分位數估計較好）
            tree = RandomForestRegressor(
                n_estimators=1,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                bootstrap=False,  # 已經自己做 bootstrap
                random_state=self.random_state + i,
                criterion="absolute_error",
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

        self._fitted = True
        return self

    def predict(self, X):
        """預測：所有樹預測結果的加權分位數"""
        X = np.asarray(X)
        preds = np.array([tree.predict(X) for tree in self.trees])  # (n_trees, n_samples)

        # 對每個樣本，取所有樹預測的分位數
        result = np.percentile(preds, self.quantile * 100, axis=0)
        return result


class TweedieRandomForest:
    """使用friedman_mse criterion的標準隨機森林，用於點估計（中位數近似）"""

    def __init__(self, n_estimators=200, max_depth=15, min_samples_leaf=5,
                 random_state=42, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.rf = None

    def fit(self, X, y):
        self.rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            criterion="friedman_mse",
        )
        self.rf.fit(X, y)
        return self

    def predict(self, X):
        return self.rf.predict(X)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def compute_age_years(row):
    """計算車齡（年）：拍賣年月 - 出廠年月"""
    try:
        auction_date = pd.to_datetime(row["auction_date"])
        auction_year = auction_date.year
        auction_month = auction_date.month
        manufacture_year = int(row["year"])
        manufacture_month = int(row["month"])
        total_months = (auction_year - manufacture_year) * 12 + (auction_month - manufacture_month)
        return total_months / 12.0
    except:
        return np.nan


def prepare_features_auto(df):
    """汽車特徵工程"""
    df = df.copy()

    # age_years
    df["age_years"] = df.apply(compute_age_years, axis=1)

    # mileage_km (keep as-is, mileage_available tells us if valid)
    df["mileage_km"] = df["mileage_km"].fillna(0).astype(float)
    df["mileage_available"] = df["mileage_available"].fillna(0).astype(int)

    # cc
    cc_median = df["cc"].median()
    df["cc"] = df["cc"].fillna(cc_median).astype(float)

    # grade ordinal
    df["grade_enc"] = df["grade"].map(GRADE_MAP).fillna(0).astype(int)

    # tax_total
    df["tax_total"] = (
        df["tax"].fillna(0) + df["violation"].fillna(0) + df["strong_violation"].fillna(0)
    ).astype(float)

    # is_automatic: 手自/自=1, 手=0
    auto_map = {"自": 1, "手自": 1, "手": 0}
    df["is_automatic"] = df["transmission"].map(auto_map).fillna(0).astype(int)

    # brand_model
    df["brand_model"] = df["brand"].astype(str) + "_" + df["model"].astype(str)

    # auction_year
    df["auction_year"] = pd.to_datetime(df["auction_date"]).dt.year.astype(int)

    # encode brand and brand_model
    le_brand = LabelEncoder()
    df["brand_enc"] = le_brand.fit_transform(df["brand"].astype(str))

    le_model = LabelEncoder()
    df["model_enc"] = le_model.fit_transform(df["brand_model"].astype(str))

    return df, le_brand, le_model


def prepare_features_moto(df):
    """機車特徵工程（精簡版）"""
    df = df.copy()

    # age_years
    df["age_years"] = df.apply(compute_age_years, axis=1)

    # mileage_km
    df["mileage_km"] = df["mileage_km"].fillna(0).astype(float)
    df["mileage_available"] = df["mileage_available"].fillna(0).astype(int)

    # cc
    cc_median = df["cc"].median()
    df["cc"] = df["cc"].fillna(cc_median).astype(float)

    # grade ordinal
    df["grade_enc"] = df["grade"].map(GRADE_MAP).fillna(0).astype(int)

    # brand_model
    df["brand_model"] = df["brand"].astype(str) + "_" + df["model"].astype(str)

    # encode brand and brand_model
    le_brand = LabelEncoder()
    df["brand_enc"] = le_brand.fit_transform(df["brand"].astype(str))

    le_model = LabelEncoder()
    df["model_enc"] = le_model.fit_transform(df["brand_model"].astype(str))

    return df, le_brand, le_model


# ============================================================
# MILEAGE IMPUTATION (AUTO ONLY)
# ============================================================

def impute_mileage_auto(df, target_col="mileage_km"):
    """
    對 mileage_available=0 的車，用「同年份+同廠牌+同評價」的中位數里程填補。
    若樣本數<3，則擴大到「同廠牌+同評價」。
    回傳填補後的 mileage_imputed 欄位。
    """
    df = df.copy()
    df["mileage_imputed"] = df[target_col].copy()

    missing_mask = df["mileage_available"] == 0
    missing_idx = df[missing_mask].index.tolist()

    for idx in missing_idx:
        row = df.loc[idx]
        year = row["year"]
        brand = row["brand"]
        grade = row["grade"]

        # 先嘗試：同年份+同廠牌+同評價
        subset = df[
            (df["year"] == year)
            & (df["brand"] == brand)
            & (df["grade"] == grade)
            & (df["mileage_available"] == 1)
        ]

        if len(subset) >= 3:
            median_mileage = subset["mileage_km"].median()
        else:
            # 擴大到：同廠牌+同評價
            subset2 = df[
                (df["brand"] == brand)
                & (df["grade"] == grade)
                & (df["mileage_available"] == 1)
            ]
            if len(subset2) >= 3:
                median_mileage = subset2["mileage_km"].median()
            else:
                # 最後手段：全體中位數
                median_mileage = df[df["mileage_available"] == 1]["mileage_km"].median()

        df.loc[idx, "mileage_imputed"] = median_mileage

    return df


# ============================================================
# FEATURE COLUMNS
# ============================================================

AUTO_FEATURE_COLS = [
    "age_years",
    "mileage_imputed",
    "mileage_available",
    "cc",
    "brand_enc",
    "model_enc",
    "grade_enc",
    "tax_total",
    "is_automatic",
    "auction_year",
]

MOTO_FEATURE_COLS = [
    "age_years",
    "mileage_imputed",
    "mileage_available",
    "cc",
    "brand_enc",
    "model_enc",
    "grade_enc",
]


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true, y_pred, lower, upper):
    """計算模型驗證指標"""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    # Coverage
    coverage = np.mean((y_true >= lower) & (y_true <= upper)) * 100

    # CP_score: (預測均價 - 得標價) / 預測均價 × 100
    cp_score = (np.mean(y_pred) - np.mean(y_true)) / np.mean(y_pred) * 100

    return {
        "R2": r2,
        "MAE": mae,
        "MAPE": mape,
        "Coverage_95": coverage,
        "CP_score": cp_score,
    }


def print_metrics(metrics, prefix=""):
    print(f"\n{'=' * 50}")
    if prefix:
        print(f"{prefix} Metrics")
        print(f"{'=' * 50}")
    print(f"  R² (解釋力):       {metrics['R2']:.4f}")
    print(f"  MAE (NT$):         {metrics['MAE']:,.0f}")
    print(f"  MAPE (%):          {metrics['MAPE']:.2f}%")
    print(f"  95% Coverage:      {metrics['Coverage_95']:.2f}%")
    print(f"  CP_score (%):      {metrics['CP_score']:.2f}%")
    print(f"{'=' * 50}")


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_model(vehicle_type="auto", data_path=None, model_save_path=None):
    assert vehicle_type in ("auto", "moto"), "--type must be 'auto' or 'moto'"

    repo_root = Path(__file__).parent.parent.resolve()
    data_dir = Path(data_path) if data_path else repo_root / "data" / "parsed"
    model_dir = Path(model_save_path) if model_save_path else repo_root / "data" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if vehicle_type == "auto":
        csv_path = data_dir / "automobiles.csv"
    else:
        csv_path = data_dir / "motorcycles.csv"

    print(f"\n[INFO] Loading {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} records")

    # ============================================================
    # EXCLUDE GOGORO for motorcycles
    # ============================================================
    if vehicle_type == "moto":
        n_before = len(df)
        df = df[df["brand"] != "GOGORO"].reset_index(drop=True)
        print(f"[INFO] Excluded GOGORO: {n_before} -> {len(df)} records")

    # ============================================================
    # FEATURE ENGINEERING
    # ============================================================
    if vehicle_type == "auto":
        df, le_brand, le_model = prepare_features_auto(df)
        # Mileage imputation for auto
        df = impute_mileage_auto(df)
    else:
        df, le_brand, le_model = prepare_features_moto(df)
        # For moto, simple median imputation for missing mileage
        df["mileage_imputed"] = df["mileage_km"].copy()
        missing_mask = df["mileage_available"] == 0
        if missing_mask.sum() > 0:
            median_mileage = df[df["mileage_available"] == 1]["mileage_km"].median()
            df.loc[missing_mask, "mileage_imputed"] = median_mileage

    # Check for missing values in key features
    feature_cols = AUTO_FEATURE_COLS if vehicle_type == "auto" else MOTO_FEATURE_COLS
    print(f"\n[INFO] Using features: {feature_cols}")

    X = df[feature_cols].copy()
    y = df["price"].values.astype(float)

    # Fill any remaining NaN
    X = X.fillna(0)

    print(f"[INFO] Feature matrix shape: {X.shape}")

    # ============================================================
    # TRAIN / TEST SPLIT (80/20)
    # ============================================================
    n = len(df)
    np.random.seed(42)
    perm = np.random.permutation(n)
    train_size = int(n * 0.8)
    train_idx = perm[:train_size]
    test_idx = perm[train_size:]

    X_train = X.iloc[train_idx].values
    X_test = X.iloc[test_idx].values
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Track which test samples had missing mileage
    test_missing_mileage = df.iloc[test_idx]["mileage_available"].values == 0

    print(f"[INFO] Train size: {len(X_train)}, Test size: {len(X_test)}")

    # ============================================================
    # TRAIN QUANTILE MODELS
    # ============================================================
    print(f"\n[INFO] Training quantile models...")

    # Point estimate (0.5 quantile - median)
    print("  - Training median model (q=0.50)...")
    rf_median = QuantileRandomForest(quantile=0.5, n_estimators=200, max_depth=15,
                                      min_samples_leaf=5, random_state=42)
    rf_median.fit(X_train, y_train)

    # Lower bound (0.025)
    print("  - Training lower bound model (q=0.025)...")
    rf_lower = QuantileRandomForest(quantile=0.025, n_estimators=200, max_depth=15,
                                     min_samples_leaf=5, random_state=42)
    rf_lower.fit(X_train, y_train)

    # Upper bound (0.975)
    print("  - Training upper bound model (q=0.975)...")
    rf_upper = QuantileRandomForest(quantile=0.975, n_estimators=200, max_depth=15,
                                     min_samples_leaf=5, random_state=42)
    rf_upper.fit(X_train, y_train)

    print("[INFO] Training complete!")

    # ============================================================
    # PREDICTIONS
    # ============================================================
    print("[INFO] Making predictions...")
    y_pred = rf_median.predict(X_test)
    y_lower = rf_lower.predict(X_test)
    y_upper = rf_upper.predict(X_test)

    # Widen interval for missing mileage samples (95% -> 99%)
    if vehicle_type == "auto":
        widen_mask = test_missing_mileage
        if widen_mask.sum() > 0:
            print(f"\n[INFO] Widening prediction interval for {widen_mask.sum()} samples with imputed mileage")
            # Compute current interval width and expand (95% -> 99%)
            current_width = y_upper - y_lower
            expand_factor = 1.8
            center = (y_upper + y_lower) / 2
            y_lower = y_lower.copy()
            y_upper = y_upper.copy()
            y_lower[widen_mask] = center[widen_mask] - (current_width[widen_mask] * expand_factor / 2)
            y_upper[widen_mask] = center[widen_mask] + (current_width[widen_mask] * expand_factor / 2)

    # ============================================================
    # METRICS
    # ============================================================
    metrics = compute_metrics(y_test, y_pred, y_lower, y_upper)
    print_metrics(metrics, prefix=f"{'Automobile' if vehicle_type == 'auto' else 'Motorcycle'}")

    # ============================================================
    # SAVE MODEL
    # ============================================================
    model_name = "automobile_v1.pkl" if vehicle_type == "auto" else "motorcycle_v1.pkl"
    model_path = model_dir / model_name

    model_artifacts = {
        "rf_median": rf_median,
        "rf_lower": rf_lower,
        "rf_upper": rf_upper,
        "le_brand": le_brand,
        "le_model": le_model,
        "feature_cols": feature_cols,
        "grade_map": GRADE_MAP,
        "vehicle_type": vehicle_type,
    }

    with open(model_path, "wb") as f:
        pickle.dump(model_artifacts, f)

    print(f"\n[INFO] Model saved to: {model_path}")

    # Print some sample predictions
    print(f"\n[INFO] Sample predictions (first 5 test samples):")
    print(f"{'True Price':>12} {'Predicted':>12} {'Lower':>12} {'Upper':>12} {'In Range':>10}")
    print("-" * 62)
    for i in range(min(5, len(y_test))):
        in_range = "✓" if (y_test[i] >= y_lower[i] and y_test[i] <= y_upper[i]) else "✗"
        print(f"{y_test[i]:>12,.0f} {y_pred[i]:>12,.0f} {y_lower[i]:>12,.0f} {y_upper[i]:>12,.0f} {in_range:>10}")

    return metrics


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train automobile or motorcycle valuation model")
    parser.add_argument(
        "--type",
        choices=["auto", "moto"],
        required=True,
        help="Model type: 'auto' for automobile, 'moto' for motorcycle",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to data directory (default: data/parsed)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output model directory (default: data/models)",
    )
    args = parser.parse_args()

    print(f"\n{'#' * 60}")
    print(f"# Training {'Automobile' if args.type == 'auto' else 'Motorcycle'} Valuation Model")
    print(f"{'#' * 60}")

    train_model(
        vehicle_type=args.type,
        data_path=args.data,
        model_save_path=args.output,
    )


if __name__ == "__main__":
    main()
