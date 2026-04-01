#!/usr/bin/env python3
"""
Train per-property regressors for nanoparticle/material properties.
Usage:
  python train_models.py --data data.csv --outdir models_out
"""
import os, json, argparse
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to CSV dataset")
    ap.add_argument("--outdir", type=str, default="models_out", help="Directory to save models")
    ap.add_argument("--targets", nargs="*", default=[
        "Band_Gap_eV","Quantum_Yield_%","Electron_Mobility_cm2Vs","Hole_Mobility_cm2Vs",
        "Dielectric_Constant","Emission_Peak_nm","Radiative_Lifetime_ns","Conductivity_Scm",
        "Aspect_Ratio"
    ], help="List of target columns to model")
    ap.add_argument("--n_estimators", type=int, default=150)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.data)

    categorical_cols = [c for c in df.columns if df[c].dtype == "object"]
    numeric_cols = [c for c in df.columns if c not in categorical_cols]

    input_features = [c for c in [
        "Nanoparticle","Material_Type","Crystal_Structure",
        "Temperature_K","Synthesis_Temperature_K","Precursor_Concentration_M","pH","Reaction_Time_h",
        "Quantum_Confinement_Size_nm"
    ] if c in df.columns]

    cat_feats = [c for c in input_features if c in categorical_cols]
    num_feats = [c for c in input_features if c in numeric_cols]

    preprocessor = ColumnTransformer(transformers=[
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_feats),
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), num_feats)
    ])

    metrics = []
    registry = {"models": {}, "input_features": input_features}

    for target in args.targets:
        if target not in df.columns:
            print(f"[skip] {target} not in dataset")
            continue
        mask = df[target].notna()
        if mask.sum() < 30:
            print(f"[skip] {target} has too few labeled samples ({mask.sum()})")
            continue

        X = df.loc[mask, input_features].copy()
        y = df.loc[mask, target].copy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

        model = Pipeline(steps=[("pre", preprocessor),
                                ("rf", RandomForestRegressor(n_estimators=args.n_estimators, random_state=0, n_jobs=-1))])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        model_path = os.path.join(args.outdir, f"model_{target}.joblib")
        joblib.dump({"pipeline": model, "input_features": input_features, "target": target}, model_path)
        registry["models"][target] = model_path
        metrics.append({"target": target, "r2": float(r2), "mae": float(mae), "n_samples": int(mask.sum()), "model_path": model_path})
        print(f"[trained] {target}: R2={r2:.3f}, MAE={mae:.4g}, samples={mask.sum()}")

    with open(os.path.join(args.outdir, "registry.json"), "w") as f:
        json.dump(registry, f, indent=2)

    pd.DataFrame(metrics).to_csv(os.path.join(args.outdir, "training_metrics.csv"), index=False)
    print(f"\nSaved registry & metrics to {args.outdir}")

if __name__ == "__main__":
    main()
