# Nanomaterial Property Predictor (Conference Demo)

This package trains ML regressors on your dataset (`data.csv`) to predict key properties from **material identity** and **synthesis conditions**, and produces a simple **simulation** visualization (radar profile + nanoparticle shape).

## 1) Environment
```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -U scikit-learn pandas matplotlib joblib
```

## 2) Train models
```bash
python train_models.py --data data.csv --outdir models_out
```
- Adjust `--targets` to include any columns you want to predict.
- Artifacts: `models_out/registry.json`, `models_out/model_<TARGET>.joblib`, `models_out/training_metrics.csv`.

## 3) Predict + simulate
```bash
python predict_cli.py --registry models_out/registry.json   --Nanoparticle "Gold" --Material_Type "Metal" --Crystal_Structure "FCC"   --Temperature_K 300 --pH 7 --Quantum_Confinement_Size_nm 10
```
Outputs in `pred_out/`:
- `predictions.json` and `predictions.csv`
- `predicted_profile_radar.png` (normalized radar chart of predicted properties)
- `simulated_shape.png` (ellipse based on predicted Aspect_Ratio & size)

## Notes
- Missing target labels are handled by training **one model per target** using only rows where that target is present.
- Categorical features are one-hot encoded; numeric features are median-imputed and standardized.
- The default estimator is `RandomForestRegressor` for robustness with minimal tuning.

## Recommended for publication-quality results
- Add **cross-validation** and hyperparameter tuning (e.g., `RandomizedSearchCV`).
- Consider domain-specific featurization (e.g., `matminer`) if you add elemental composition columns.
- Report performance metrics (R², MAE) from `training_metrics.csv` in your paper/poster.
