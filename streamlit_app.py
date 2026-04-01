import json, os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

st.set_page_config(page_title="Nanomaterial Property Predictor", layout="centered")

st.title("🔬 Nanomaterial Property Predictor")
st.caption("Predict key properties from material identity & synthesis conditions, then visualize a simple simulation.")

registry_path = st.text_input("Path to registry.json", value="models_out/registry.json")

inputs = {}
default_inputs = {
    "Nanoparticle": "",
    "Material_Type": "",
    "Crystal_Structure": "",
    "Temperature_K": 300,
    "Synthesis_Temperature_K": 500,
    "Precursor_Concentration_M": 0.1,
    "pH": 7.0,
    "Reaction_Time_h": 10.0,
    "Quantum_Confinement_Size_nm": 10.0
}
cols = st.columns(2)
with cols[0]:
    inputs["Nanoparticle"] = st.text_input("Nanoparticle (e.g., Gold, CdSe)",
                                           value=str(default_inputs["Nanoparticle"]))
    inputs["Material_Type"] = st.text_input("Material_Type (e.g., Metal, Semiconductor)",
                                            value=str(default_inputs["Material_Type"]))
    inputs["Crystal_Structure"] = st.text_input("Crystal_Structure (e.g., FCC, Wurtzite)",
                                                value=str(default_inputs["Crystal_Structure"]))
    inputs["Temperature_K"] = st.number_input("Temperature_K", value=float(default_inputs["Temperature_K"]))
    inputs["pH"] = st.number_input("pH", value=float(default_inputs["pH"]))
with cols[1]:
    inputs["Synthesis_Temperature_K"] = st.number_input("Synthesis_Temperature_K", value=float(default_inputs["Synthesis_Temperature_K"]))
    inputs["Precursor_Concentration_M"] = st.number_input("Precursor_Concentration_M", value=float(default_inputs["Precursor_Concentration_M"]))
    inputs["Reaction_Time_h"] = st.number_input("Reaction_Time_h", value=float(default_inputs["Reaction_Time_h"]))
    inputs["Quantum_Confinement_Size_nm"] = st.number_input("Quantum_Confinement_Size_nm", value=float(default_inputs["Quantum_Confinement_Size_nm"]))

go = st.button("Predict")
if go:
    if not os.path.exists(registry_path):
        st.error("Registry not found. Train models first and set the correct path.")
        st.stop()

    with open(registry_path) as f:
        registry = json.load(f)

    feats = registry["input_features"]
    preds = {}
    for target, path in registry["models"].items():
        try:
            bundle = joblib.load(path)
            pipe = bundle["pipeline"]
            row = {f: inputs.get(f, np.nan) for f in feats}
            X = pd.DataFrame([row])
            preds[target] = float(pipe.predict(X)[0])
        except Exception as e:
            continue

    st.subheader("Predicted properties")
    st.json(preds)

    # Radar chart
    radar_keys = [k for k in ["Band_Gap_eV","Quantum_Yield_%","Electron_Mobility_cm2Vs",
                              "Hole_Mobility_cm2Vs","Dielectric_Constant","Conductivity_Scm","Aspect_Ratio"] if k in preds]
    if radar_keys:
        vals = [preds[k] for k in radar_keys]
        vmin = min(vals); vmax = max(vals) if max(vals) > min(vals) else min(vals)+1e-6
        norm = [(v-vmin)/(vmax-vmin) for v in vals]

        import numpy as np
        angles = np.linspace(0, 2*np.pi, len(radar_keys), endpoint=False).tolist()
        norm += norm[:1]; angles += angles[:1]

        fig = plt.figure(figsize=(6,6))
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, norm, marker="o")
        ax.fill(angles, norm, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), radar_keys)
        plt.title(f"Predicted profile: {inputs.get('Nanoparticle','(material)')}")
        st.pyplot(fig)

    # Shape visualization
    size_nm = preds.get("Quantum_Confinement_Size_nm", inputs.get("Quantum_Confinement_Size_nm", 10.0))
    ar = preds.get("Aspect_Ratio", 1.0)
    major = size_nm * (ar if ar >= 1 else 1.0)
    minor = size_nm * (1.0 if ar >= 1 else 1.0/ar)

    fig2, ax2 = plt.subplots(figsize=(5,5))
    e = Ellipse(xy=(0.5,0.5), width=minor/max(major,minor), height=major/max(major,minor), angle=0)
    ax2.add_patch(e)
    ax2.set_xlim(0,1); ax2.set_ylim(0,1); ax2.set_aspect("equal")
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title(f"Simulated nanoparticle shape (AR≈{ar:.2f}, size≈{size_nm:.1f} nm)")
    st.pyplot(fig2)
