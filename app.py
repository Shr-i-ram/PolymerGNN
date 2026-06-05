import streamlit as st
import pandas as pd

from predictor import predict
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
import py3Dmol
import streamlit.components.v1 as components
from rdkit.Chem import AllChem
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
import plotly.graph_objects as go

def create_radar(results):

    props = [
        "YM",
        "Tg",
        "Td",
        "rho",
        "LOI",
        "permCO2"
    ]

    values = [results[p] for p in props]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=props,
            fill="toself"
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True
            )
        ),
        showlegend=False
    )

    return fig

def render_molecule(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    AllChem.EmbedMolecule(
        mol,
        randomSeed=42
    )

    AllChem.MMFFOptimizeMolecule(mol)

    mol_block = Chem.MolToMolBlock(mol)

    viewer = py3Dmol.view(
        width=600,
        height=500
    )

    viewer.addModel(
        mol_block,
        "mol"
    )

    viewer.setStyle(
        {"stick": {},
        "sphere": {"scale": 0.25}
        }
    )

    viewer.zoomTo()

    return viewer._make_html()

st.set_page_config(
    page_title="PolymerGNN: Materials Informatics Platform",
    page_icon="🧪",
    layout="wide"
)
with st.sidebar:
    st.title("PolymerGNN")

    st.markdown("""
    ### Features

    - Property Prediction
    - Molecule Comparison
    - 3D Visualization
    - CSV Export
    """)

st.title("🧪 PolymerGNN: Materials Informatics Platform")
st.subheader("Multi-Property Polymer Prediction using Graph Neural Networks")
b1, b2, b3, b4 = st.columns(4)
PROPERTY_INFO = {
    # Thermal
    "Tg": "Glass Transition Temperature",
    "Tm": "Melting Temperature",
    "Td": "Thermal Decomposition Temperature",
    "Cp": "Specific Heat Capacity",

    # Mechanical
    "YM": "Young's Modulus",
    "TSb": "Tensile Strength at Break",
    "TSy": "Tensile Strength at Yield",

    # Gas Transport
    "permCH4": "Methane Permeability",
    "permCO2": "Carbon Dioxide Permeability",
    "permH2": "Hydrogen Permeability",
    "permO2": "Oxygen Permeability",
    "permN2": "Nitrogen Permeability",
    "permHe": "Helium Permeability",

    # Electronic
    "Egc": "Conduction Band Energy Gap",
    "Egb": "Band Gap Energy",
    "Ei": "Ionization Energy",
    "Eib": "Ionization Energy (Variant)",
    "Eea": "Electron Affinity",
    "Eat": "Total Electronic Energy",

    # Dielectric
    "epsc": "Static Dielectric Constant",
    "epsb": "Breakdown Dielectric Constant",
    "epse_1.78": "Dielectric Constant @ 1.78",
    "epse_2.0": "Dielectric Constant @ 2.0",
    "epse_3.0": "Dielectric Constant @ 3.0",
    "epse_4.0": "Dielectric Constant @ 4.0",
    "epse_5.0": "Dielectric Constant @ 5.0",
    "epse_6.0": "Dielectric Constant @ 6.0",
    "epse_7.0": "Dielectric Constant @ 7.0",
    "epse_9.0": "Dielectric Constant @ 9.0",
    "epse_15.0": "Dielectric Constant @ 15.0",

    # Structural
    "rho": "Density",
    "Xc": "Crystallinity Fraction",
    "Xe": "Experimental Crystallinity",
    "LOI": "Limiting Oxygen Index",
    "nc": "Refractive Index (Calculated)",
    "ne": "Refractive Index (Experimental)"
}
if b1.button("Ethanol"):
    st.session_state["smiles"] = "CCO"

if b2.button("Benzene"):
    st.session_state["smiles"] = "c1ccccc1"

if b3.button("Acetone"):
    st.session_state["smiles"] = "CC(=O)C"

if b4.button("Toluene"):
    st.session_state["smiles"] = "Cc1ccccc1"

smiles = st.text_input(
    "Enter SMILES",
    value=st.session_state.get("smiles", "CCO")
)

if st.button("Predict"):

    try:
        results = predict(smiles)
        st.success("Prediction completed!")

        mol = Chem.MolFromSmiles(smiles)
        num_atoms = mol.GetNumAtoms()
        formula = rdMolDescriptors.CalcMolFormula(mol)
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)

        rings = rdMolDescriptors.CalcNumRings(mol)

        donors = Lipinski.NumHDonors(mol)

        acceptors = Lipinski.NumHAcceptors(mol)
        if num_atoms < 3:
            st.warning(
                "⚠️ This molecule is much smaller than a typical polymer repeat unit. Predictions may be less reliable."
            )

        elif num_atoms < 5:
            st.info(
                "ℹ️ Small molecule detected. The model was trained primarily on polymer-related structures."
            )
        st.caption(
            f"Molecule contains {num_atoms} atoms."
        )
        left, right = st.columns([2,1])

        with left:

            st.subheader("3D Molecular Structure")

            html = render_molecule(smiles)

            if html:
                components.html(
                    html,
                    height=550
                )

        with right:

            st.subheader("Molecular Information")

            st.metric("Formula", formula)

            st.metric(
                "Mol Weight",
                f"{mw:.2f}"
            )

            st.metric(
                "Atoms",
                mol.GetNumAtoms()
            )

            st.metric(
                "Bonds",
                mol.GetNumBonds()
            )

            st.metric(
                "LogP",
                f"{logp:.2f}"
            )

            st.metric(
                "Rings",
                rings
            )

            st.metric(
                "H-Donors",
                donors
            )

            st.metric(
                "H-Acceptors",
                acceptors
            )

        thermal = {
            k: results[k]
            for k in ["Tg", "Tm", "Td", "Cp"]
        }

        mechanical = {
            k: results[k]
            for k in ["YM", "TSb", "TSy"]
        }

        gas = {
            k: results[k]
            for k in [
                "permCH4",
                "permCO2",
                "permH2",
                "permO2",
                "permN2",
                "permHe"
            ]
        }
        dielectric = {
            k: results[k]
            for k in [
                "epsc",
                "epsb",
                "epse_1.78",
                "epse_2.0",
                "epse_3.0",
                "epse_4.0",
                "epse_5.0",
                "epse_6.0",
                "epse_7.0",
                "epse_9.0",
                "epse_15.0"
            ]
        }

        electronic = {
            k: results[k]
            for k in [
                "Egc",
                "Egb",
                "Ei",
                "Eib",
                "Eea",
                "Eat"
            ]
        }

        structural = {
            k: results[k]
            for k in [
                "rho",
                "Xc",
                "Xe",
                "LOI",
                "nc",
                "ne"
            ]
        }

        st.subheader("Material Profile")

        st.plotly_chart(
            create_radar(results),
            use_container_width=True
        )

        row1_col1, row1_col2, row1_col3 = st.columns(3)

        with row1_col1:
            st.subheader("Thermal Properties")

            thermal = {
                "Tg": results["Tg"],
                "Tm": results["Tm"],
                "Td": results["Td"],
                "Cp": results["Cp"]
            }

            st.dataframe(
                pd.DataFrame(
                    [
                        [prop, PROPERTY_INFO[prop], value]
                        for prop, value in thermal.items()
                    ],
                    columns=["Property", "Description", "Value"]
                ),
                use_container_width=True
            )

        with row1_col2:
            st.subheader("Mechanical")

            st.dataframe(
                pd.DataFrame(
                    [
                        [prop, PROPERTY_INFO[prop], value]
                        for prop, value in mechanical.items()
                    ],
                    columns=["Property", "Description", "Value"]
                ),
                use_container_width=True
            )

        with row1_col3:
            st.subheader("Gas Transport")

            st.dataframe(
                pd.DataFrame(
                    [
                        [prop, PROPERTY_INFO[prop], value]
                        for prop, value in gas.items()
                    ],
                    columns=["Property", "Description", "Value"]
                ),
                use_container_width=True
            )

        row2_col1, row2_col2, row2_col3 = st.columns(3)

        with row2_col1:
            st.subheader("Dielectric Properties")

            st.dataframe(
                pd.DataFrame(
                    [
                        [prop, PROPERTY_INFO[prop], value]
                        for prop, value in dielectric.items()
                    ],
                    columns=["Property", "Description", "Value"]
                ),
                use_container_width=True
            )

        with row2_col2:
            st.subheader("Electronic Properties")

            st.dataframe(
                pd.DataFrame(
                    [
                        [prop, PROPERTY_INFO[prop], value]
                        for prop, value in electronic.items()
                    ],
                    columns=["Property", "Description", "Value"]
                ),
                use_container_width=True
            )

        with row2_col3:
            st.subheader("Structural Properties")

            st.dataframe(
                pd.DataFrame(
                    [
                        [prop, PROPERTY_INFO[prop], value]
                        for prop, value in structural.items()
                    ],
                    columns=["Property", "Description", "Value"]
                ),
                use_container_width=True
            )
        df_export = pd.DataFrame(
            results.items(),
            columns=["Property", "Value"]
        )

        csv = df_export.to_csv(index=False)

        st.download_button(
            label="📥 Download Predictions",
            data=csv,
            file_name="polymer_predictions.csv",
            mime="text/csv"
        )
        st.subheader("All Properties")

        st.dataframe(
            pd.DataFrame(
                results.items(),
                columns=["Property", "Value"]
            ),
            use_container_width=True
        )
        with st.expander("📖 Property Reference Guide"):

            ref_df = pd.DataFrame(
                PROPERTY_INFO.items(),
                columns=[
                    "Property",
                    "Description"
                ]
            )

            st.dataframe(
                ref_df,
                use_container_width=True
            )

    except Exception as e:
        st.error(str(e))
st.divider()

st.header("⚖️ Molecule Comparison")
compare_smiles_1 = st.text_input(
    "Molecule A",
    value="CCO",
    key="compare_a"
)

compare_smiles_2 = st.text_input(
    "Molecule B",
    value="c1ccccc1",
    key="compare_b"
)

if st.button("Compare Molecules"):
    results_a = predict(compare_smiles_1)
    results_b = predict(compare_smiles_2)
    comparison_props = [
        "Tg",
        "Tm",
        "Td",
        "YM",
        "rho",
        "LOI",
        "permCO2"
    ]
    left, right = st.columns(2)

    with left:
        st.subheader("Molecule A")
        html_a = render_molecule(compare_smiles_1)
        if html_a:
            components.html(html_a, height=400)

    with right:
        st.subheader("Molecule B")
        html_b = render_molecule(compare_smiles_2)
        if html_b:
            components.html(html_b, height=400)
    comparison_df = pd.DataFrame({
        "Property": comparison_props,
        "Molecule A": [results_a[p] for p in comparison_props],
        "Molecule B": [results_b[p] for p in comparison_props],
        "Difference": [
            results_a[p] - results_b[p]
            for p in comparison_props
        ]
    })
    st.subheader("Material Profile")

    st.plotly_chart(
        create_radar(results_a),
        use_container_width=True
    )
    st.plotly_chart(
        create_radar(results_b),
        use_container_width=True
    )
    st.dataframe(
        comparison_df,
        use_container_width=True
    )
st.divider()

st.caption(
    "PolymerGNN | Graph Neural Network for Polymer Property Prediction"
)
c1, c2, c3 = st.columns(3)

c1.metric("Properties", "37")
c2.metric("Model", "GIN")
c3.metric("Framework", "PyG")
