#!/usr/bin/env python3
"""
Predict properties from trained models and generate a simple "simulation" visualization.
Usage:
  python predict_cli.py --registry models_out/registry.json --Nanoparticle "Gold" --Material_Type "Metal" --pH 7 --Temperature_K 300 --Quantum_Confinement_Size_nm 10

Outputs:
  - predictions.json / predictions.csv
  - predicted_profile_radar.png
  - simulated_shape.png
"""
import os, json, argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=str, required=True, help="Path to registry.json")
    ap.add_argument("--outdir", type=str, default="pred_out", help="Output directory")
    # Accept arbitrary known inputs as optional args
    known_inputs = ["Nanoparticle","Material_Type","Crystal_Structure",
                    "Temperature_K","Synthesis_Temperature_K","Precursor_Concentration_M","pH","Reaction_Time_h",
                    "Quantum_Confinement_Size_nm"]
    for k in known_inputs:
        ap.add_argument(f"--{k}", type=str)  # parse as str, we'll cast later
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.registry) as f:
        registry = json.load(f)

    # build input row
    feats = registry["input_features"]
    provided = {}
    for ftr in feats:
        val = getattr(args, ftr)
        if val is None:
            provided[ftr] = np.nan
        else:
            # cast numeric where possible
            try:
                provided[ftr] = float(val)
            except:
                provided[ftr] = val

    # run predictions
    preds = {}
    for target, path in registry["models"].items():
        bundle = joblib.load(path)
        pipe = bundle["pipeline"]
        row = {f: provided.get(f, np.nan) for f in feats}
        X = pd.DataFrame([row])
        try:
            pred = float(pipe.predict(X)[0])
            preds[target] = pred
        except Exception as e:
            continue

    # save predictions
    with open(os.path.join(args.outdir, "predictions.json"), "w") as f:
        json.dump(preds, f, indent=2)
    pd.DataFrame([preds]).to_csv(os.path.join(args.outdir, "predictions.csv"), index=False)

    # Radar chart
    radar_keys = [k for k in ["Band_Gap_eV","Quantum_Yield_%","Electron_Mobility_cm2Vs","Hole_Mobility_cm2Vs",
                              "Dielectric_Constant","Conductivity_Scm","Aspect_Ratio"] if k in preds]
    if radar_keys:
        # normalize 0-1 using min/max seen in training (approximate by re-reading original CSV if available)
        # fallback: use min/max of predicted values only
        # For simplicity: use predicted value scaling with simple min/max fallback (not ideal but avoids extra I/O).
        vals = [preds[k] for k in radar_keys]
        vmin = min(vals); vmax = max(vals) if max(vals) > min(vals) else min(vals)+1e-6
        norm = [(v-vmin)/(vmax-vmin) for v in vals]

        import numpy as np
        angles = np.linspace(0, 2*np.pi, len(radar_keys), endpoint=False).tolist()
        norm += norm[:1]; angles += angles[:1]

        plt.figure(figsize=(6,6))
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, norm, marker="o")
        ax.fill(angles, norm, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), radar_keys)
        plt.title(f"Predicted profile: {provided.get('Nanoparticle','(material)')}")
        plt.savefig(os.path.join(args.outdir, "predicted_profile_radar.png"), bbox_inches="tight")
        plt.close()

    # Shape visualization
    size_nm = preds.get("Quantum_Confinement_Size_nm", provided.get("Quantum_Confinement_Size_nm", 10.0))
    ar = preds.get("Aspect_Ratio", 1.0)
    major = size_nm * (ar if ar >= 1 else 1.0)
    minor = size_nm * (1.0 if ar >= 1 else 1.0/ar)

    fig, ax = plt.subplots(figsize=(6,6))
    e = Ellipse(xy=(0.5,0.5), width=minor/max(major,minor), height=major/max(major,minor), angle=0)
    ax.add_patch(e)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Simulated nanoparticle shape (AR≈{ar:.2f}, size≈{size_nm:.1f} nm)")
    plt.savefig(os.path.join(args.outdir, "simulated_shape.png"), bbox_inches="tight")
    plt.close()

    print("Saved predictions and plots to:", args.outdir)

if __name__ == "__main__":
    main()
