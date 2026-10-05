from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "planetary_defense_models"
EVALUATION_FILE = BASE_DIR / "model_evaluation.json"


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Planetary Defense Analytics",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# FEATURES
# =========================================================

CLASSIFICATION_FEATURES = [
    "e", "a", "q", "i", "om", "w", "ma", "ad", "n", "per",
    "per_y", "tp", "sigma_e", "sigma_a", "sigma_q", "sigma_i",
    "sigma_om", "sigma_w", "sigma_ma", "sigma_ad", "sigma_n",
    "sigma_tp", "sigma_per", "rms"
]

REGRESSION_FEATURES = [
    "e", "a", "q", "i", "om", "w", "ma", "ad", "n", "per",
    "per_y", "moid", "sigma_e", "sigma_a", "sigma_q", "sigma_i",
    "sigma_om", "sigma_w", "sigma_ma", "sigma_ad", "sigma_n",
    "sigma_tp", "sigma_per", "rms"
]

CLUSTER_FEATURES = [
    "e", "a", "q", "i", "moid", "per", "n"
]


# =========================================================
# DEFAULT INPUT VALUES
# =========================================================

DEFAULTS = {
    "e": 0.15,
    "a": 2.5,
    "q": 2.1,
    "i": 10.0,
    "om": 80.0,
    "w": 150.0,
    "ma": 180.0,
    "ad": 2.9,
    "n": 0.25,
    "per": 1440.0,
    "per_y": 3.94,
    "tp": 2459000.0,
    "moid": 0.5,
    "sigma_e": 0.0001,
    "sigma_a": 0.0001,
    "sigma_q": 0.0001,
    "sigma_i": 0.001,
    "sigma_om": 0.001,
    "sigma_w": 0.001,
    "sigma_ma": 180.0,
    "sigma_ad": 0.0001,
    "sigma_n": 0.000001,
    "sigma_tp": 0.001,
    "sigma_per": 0.001,
    "rms": 0.5
}


# =========================================================
# DARK THEME AND DASHBOARD STYLING
# =========================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #08111f, #101b30, #111827);
    color: #FFFFFF;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: #0B1220;
    border-right: 1px solid #293C56;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] button {
    background-color: #17263D !important;
    color: #FFFFFF !important;
    border: 1px solid #405674 !important;
}

[data-testid="stSidebar"] button p {
    color: #FFFFFF !important;
}

[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[aria-label="Close sidebar"],
button[aria-label="Open sidebar"] {
    color: #FFFFFF !important;
    visibility: visible !important;
}

[data-testid="collapsedControl"] svg,
[data-testid="stSidebarCollapseButton"] svg,
button[aria-label="Close sidebar"] svg,
button[aria-label="Open sidebar"] svg {
    fill: #FFFFFF !important;
    stroke: #FFFFFF !important;
    filter: brightness(0) invert(1) !important;
}

[data-testid="stHeaderActionElements"],
a.anchor-link,
.stMarkdown a.anchor-link,
h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
    display: none !important;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

h1, h2, h3, h4, h5, h6, p, li, label {
    color: #F1F5F9;
}

.stMarkdown p,
.stMarkdown li {
    color: #D5DEEB;
}

/* Metric cards */

.metric-card {
    background: #111E31;
    border: 1px solid #344966;
    border-radius: 14px;
    padding: 1.1rem;
    min-height: 105px;
}

.metric-label {
    color: #CBD5E1 !important;
    font-size: 0.88rem;
}

.metric-value {
    color: #FFFFFF !important;
    font-size: 1.45rem;
    font-weight: 700;
    margin-top: 0.4rem;
    overflow-wrap: anywhere;
}

/* Consistent dashboard card content height */

.dashboard-item {
    min-height: 120px;
}

.dashboard-item h4 {
    min-height: 52px;
    margin-top: 0;
    margin-bottom: 10px;
}

.dashboard-item p {
    min-height: 48px;
    margin-bottom: 12px;
    color: #D5DEEB !important;
}

/* Number input styling */

[data-testid="stWidgetLabel"] p,
[data-testid="stCaptionContainer"] p {
    color: #D5DEEB !important;
}

[data-testid="stNumberInput"] input {
    background-color: #F8FAFC !important;
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}

[data-testid="stNumberInput"] button {
    background-color: #E2E8F0 !important;
    color: #111827 !important;
}

/* Buttons */

.stButton > button,
.stFormSubmitButton > button {
    background: #2563EB !important;
    color: #FFFFFF !important;
    border: 1px solid #3B82F6 !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    min-height: 2.7rem;
}

.stButton > button *,
.stFormSubmitButton > button * {
    color: #FFFFFF !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #1D4ED8 !important;
}

[data-testid="stAlert"] p {
    color: #FFFFFF !important;
}

[data-testid="stExpander"] {
    border: 1px solid #344966;
    border-radius: 12px;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: #FFFFFF !important;
}

hr {
    border-color: #344966 !important;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD SAVED MODELS
# =========================================================

@st.cache_resource
def load_models():
    filenames = {
        "classifier": "pha_classifier.pkl",
        "diameter": "diameter_regressor.pkl",
        "kmeans": "kmeans_model.pkl",
        "imputer": "cluster_imputer.pkl",
        "scaler": "cluster_scaler.pkl"
    }

    loaded_models = {}

    for name, filename in filenames.items():
        model_path = MODEL_DIR / filename

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        loaded_models[name] = joblib.load(model_path)

    return loaded_models


@st.cache_data
def load_evaluation_results():
    if not EVALUATION_FILE.is_file():
        return None

    with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def feature_inputs(features, prefix, columns=3):
    values = {}
    layout = st.columns(columns)

    for index, feature in enumerate(features):
        with layout[index % columns]:
            values[feature] = st.number_input(
                feature,
                value=float(DEFAULTS.get(feature, 0.0)),
                format="%.6f",
                key=f"{prefix}_{feature}"
            )

    return values


def make_input_frame(values, features):
    frame = pd.DataFrame(
        [[values[feature] for feature in features]],
        columns=features
    )

    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Please enter valid numeric values.")

    return frame


def go_to_page(page_name):
    st.session_state["next_page"] = page_name


# =========================================================
# NAVIGATION STATE
# =========================================================

if "next_page" in st.session_state:
    st.session_state["navigation"] = st.session_state.pop("next_page")


# =========================================================
# LOAD MODELS SAFELY
# =========================================================

try:
    models = load_models()

except Exception as error:
    st.error("Could not load the saved models.")
    st.code(str(error))
    st.warning(
        "Saved model files may be incompatible with the current "
        "Python or scikit-learn environment. Use a compatible "
        "training environment or retrain and export the models."
    )
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.title("Planetary Defense")
    st.caption("ASTEROID ANALYTICS")
    st.divider()

    selected_page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "PHA Classification",
            "Diameter Prediction",
            "K-Means Clustering",
            "Model Evaluation"
        ],
        key="navigation"
    )

    st.divider()
    st.markdown("**Model Status**")
    st.success("Saved models loaded")


# =========================================================
# PAGE HEADER
# =========================================================

page = selected_page

if page != "Overview":
    title_col, back_col = st.columns(
        [6, 1.5],
        gap="small",
        vertical_alignment="top"
    )

    with title_col:
        st.title("Planetary Defense Analytics")
        st.caption("Asteroid orbital analysis using machine learning")

    with back_col:
        st.markdown(
            "<div style='height: 8px;'></div>",
            unsafe_allow_html=True
        )

        if st.button(
            "← Back",
            key=f"small_back_{page}",
            use_container_width=True
        ):
            go_to_page("Overview")
            st.rerun()

else:
    st.title("Planetary Defense Analytics")
    st.caption("Asteroid orbital analysis using machine learning")


# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":
    st.subheader("Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card("ANALYTICS MODULES", "03")

    with col2:
        metric_card("CLASSIFICATION", "Random Forest")

    with col3:
        metric_card("CLUSTERING", "K-Means")

    st.markdown("### Explore the Dashboard")

    dashboard_columns = st.columns(4, gap="medium")

    dashboard_items = [
        (
            "PHA Classification",
            "Classify asteroid features using the trained model.",
            "Open Classification",
            "PHA Classification",
            "open_pha"
        ),
        (
            "Diameter Prediction",
            "Estimate asteroid diameter in kilometres.",
            "Open Diameter Prediction",
            "Diameter Prediction",
            "open_diameter"
        ),
        (
            "K-Means Clustering",
            "Group asteroids by orbital characteristics.",
            "Open Clustering",
            "K-Means Clustering",
            "open_cluster"
        ),
        (
            "Model Evaluation",
            "View performance metrics and the confusion matrix.",
            "Open Model Evaluation",
            "Model Evaluation",
            "open_evaluation"
        )
    ]

    for column, item in zip(dashboard_columns, dashboard_items):
        title, description, button_text, page_name, button_key = item

        with column:
            st.markdown(
                f"""
                <div class="dashboard-item">
                    <h4>{title}</h4>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                button_text,
                key=button_key,
                use_container_width=True
            ):
                go_to_page(page_name)
                st.rerun()


# =========================================================
# PHA CLASSIFICATION
# =========================================================

elif page == "PHA Classification":
    st.subheader("PHA Classification")
    st.write("Enter orbital features to run the trained classifier.")

    threshold = st.slider(
        "Classification threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.80,
        step=0.05
    )

    with st.form("pha_form"):
        values = feature_inputs(
            CLASSIFICATION_FEATURES,
            "pha",
            columns=3
        )

        submitted = st.form_submit_button(
            "Run PHA Classification",
            use_container_width=True
        )

    if submitted:
        try:
            frame = make_input_frame(
                values,
                CLASSIFICATION_FEATURES
            )

            classifier = models["classifier"]

            probabilities = classifier.predict_proba(frame)[0]
            classes = list(classifier.classes_)

            # Display the saved model's actual class labels.
            st.write("Model class labels:", classes)

            # Identify the positive PHA label.
            positive_labels = {
                "1", "Y", "YES", "TRUE", "P", "PHA"
            }

            pha_index = next(
                (
                    index
                    for index, label in enumerate(classes)
                    if str(label).strip().upper() in positive_labels
                ),
                None
            )

            if pha_index is None:
                st.error(
                    "The PHA class could not be identified automatically. "
                    "Check the displayed class labels against the "
                    "original training target before interpreting "
                    "the probability."
                )
                st.stop()

            probability = float(probabilities[pha_index])
            is_pha = probability >= threshold

            col1, col2, col3 = st.columns(3)

            with col1:
                metric_card(
                    "PHA PROBABILITY",
                    f"{probability:.2%}"
                )

            with col2:
                metric_card(
                    "THRESHOLD",
                    f"{threshold:.0%}"
                )

            with col3:
                metric_card(
                    "MODEL OUTPUT",
                    "PHA class" if is_pha else "Non-PHA class"
                )

            st.progress(min(1.0, max(0.0, probability)))

            st.caption(
                "Change input values and submit again to compare "
                "the model probability."
            )

            st.info(
                "This is a model estimate, not an official asteroid "
                "hazard assessment."
            )

        except Exception as error:
            st.error(f"Classification failed: {error}")


# =========================================================
# DIAMETER PREDICTION
# =========================================================

elif page == "Diameter Prediction":
    st.subheader("Diameter Prediction")
    st.write("Estimate asteroid diameter in kilometres.")

    with st.form("diameter_form"):
        values = feature_inputs(
            REGRESSION_FEATURES,
            "diameter",
            columns=3
        )

        submitted = st.form_submit_button(
            "Predict Diameter",
            use_container_width=True
        )

    if submitted:
        try:
            frame = make_input_frame(
                values,
                REGRESSION_FEATURES
            )

            prediction = float(
                models["diameter"].predict(frame)[0]
            )

            if not np.isfinite(prediction):
                raise ValueError(
                    "The model returned an invalid value."
                )

            col1, col2 = st.columns(2)

            with col1:
                metric_card(
                    "ESTIMATED DIAMETER",
                    f"{max(0.0, prediction):,.3f} km"
                )

            with col2:
                metric_card(
                    "MODEL",
                    "Random Forest Regressor"
                )

            if prediction < 0:
                st.warning(
                    "The raw model output was negative. "
                    "The displayed diameter has been set to zero."
                )

        except Exception as error:
            st.error(f"Diameter prediction failed: {error}")


# =========================================================
# K-MEANS CLUSTERING
# =========================================================

elif page == "K-Means Clustering":
    st.subheader("K-Means Clustering")
    st.write(
        "Assign an asteroid to a learned orbital-feature cluster."
    )

    with st.form("cluster_form"):
        values = feature_inputs(
            CLUSTER_FEATURES,
            "cluster",
            columns=2
        )

        submitted = st.form_submit_button(
            "Assign Cluster",
            use_container_width=True
        )

    if submitted:
        try:
            frame = make_input_frame(
                values,
                CLUSTER_FEATURES
            )

            scaler = models["scaler"]
            kmeans = models["kmeans"]

            scaler_features = getattr(
                scaler, "feature_names_in_", None
            )

            if scaler_features is not None:
                expected = list(scaler_features)

                if expected != CLUSTER_FEATURES:
                    raise ValueError(
                        f"Scaler feature order mismatch. "
                        f"Expected: {expected}; "
                        f"App uses: {CLUSTER_FEATURES}"
                    )

            scaled = scaler.transform(frame)

            if not np.isfinite(np.asarray(scaled)).all():
                raise ValueError(
                    "The scaler returned invalid feature values."
                )

            cluster = int(kmeans.predict(scaled)[0])

            st.markdown("### Clustering Result")
            metric_card("ASSIGNED CLUSTER", str(cluster))

            st.caption(
                "Cluster labels are group identifiers, not hazard ratings."
            )

            # Static table: no interactive table toolbar.
            with st.expander("View submitted input values"):
                st.table(frame)

            st.markdown("### Clustering Diagnostics")
            st.write("Number of clusters:", int(kmeans.n_clusters))

            centers = np.asarray(kmeans.cluster_centers_)
            point = np.asarray(scaled[0])

            if centers.shape[1] != point.shape[0]:
                raise ValueError(
                    "The scaler output and K-Means centres have "
                    "different feature counts."
                )

            distances = np.linalg.norm(centers - point, axis=1)

            distance_table = pd.DataFrame({
                "Cluster": np.arange(len(distances)),
                "Distance": distances
            }).sort_values(
                "Distance"
            ).reset_index(drop=True)

            st.write(
                "Distance to each cluster centre "
                "(smaller means closer):"
            )

            # Static table with four decimal places.
            st.table(
                distance_table.style.format({
                    "Distance": "{:.4f}"
                })
            )

            closest_cluster = int(np.argmin(distances))

            if closest_cluster == cluster:
                st.success(
                    f"Nearest centre matches the prediction: "
                    f"Cluster {closest_cluster}."
                )
            else:
                st.warning(
                    "The nearest-centre calculation differs from the "
                    "predicted cluster. Check the scaler and K-Means "
                    "training pipeline."
                )

            st.info(
                "Different inputs can belong to the same cluster. "
                "Clusters are learned groups, not hazard ratings."
            )

        except Exception as error:
            st.error(f"Clustering failed: {error}")
            st.info(
                "If the saved models are incompatible, verify that "
                "the scaler and K-Means model were trained and exported "
                "together using the same feature order."
            )


# =========================================================
# MODEL EVALUATION
# =========================================================

elif page == "Model Evaluation":
    st.subheader("Model Evaluation")

    st.write(
        "Evaluation metrics calculated from your saved Colab test results."
    )

    results = load_evaluation_results()

    if results is None:
        st.error(
            "model_evaluation.json was not found. Place it in the "
            "same folder as app.py and refresh the app."
        )
        st.stop()

    try:
        classification = results["classification"]
        regression = results["regression"]

        st.markdown("### PHA Classification — Random Forest")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            metric_card(
                "ACCURACY",
                f'{classification["accuracy"]:.2%}'
            )

        with col2:
            metric_card(
                "PRECISION",
                f'{classification["precision"]:.2%}'
            )

        with col3:
            metric_card(
                "RECALL",
                f'{classification["recall"]:.2%}'
            )

        with col4:
            metric_card(
                "F1-SCORE",
                f'{classification["f1_score"]:.2%}'
            )

        st.markdown("#### Confusion Matrix")

        cm = np.asarray(
            classification["confusion_matrix"]
        )

        if cm.shape != (2, 2):
            st.error(
                "The saved confusion matrix does not have 2 × 2 values."
            )
        else:
            fig, ax = plt.subplots(figsize=(4, 3), dpi=100)

            fig.patch.set_facecolor("#111E31")
            ax.set_facecolor("#111E31")

            display = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["Non-PHA", "PHA"]
            )

            display.plot(
                ax=ax,
                cmap="Blues",
                colorbar=False,
                values_format="d"
            )

            ax.set_title(
                "PHA Classification Confusion Matrix",
                color="white",
                fontsize=10
            )

            ax.set_xlabel("Predicted label", color="white", fontsize=9)
            ax.set_ylabel("Actual label", color="white", fontsize=9)
            ax.tick_params(colors="white", labelsize=8)

            fig.tight_layout()

            # Keep the chart small and horizontally centred.
            left_space, chart_space, right_space = st.columns([1, 2, 1])

            with chart_space:
                st.pyplot(fig, use_container_width=False)

            plt.close(fig)

        st.info(
            "High accuracy alone does not guarantee good PHA detection. "
            "Recall measures the proportion of actual PHA examples "
            "detected by the model."
        )

        st.markdown("### Diameter Prediction — Random Forest")

        col1, col2, col3 = st.columns(3)

        with col1:
            metric_card("MAE", f'{regression["mae"]:.3f} km')

        with col2:
            metric_card("RMSE", f'{regression["rmse"]:.3f} km')

        with col3:
            metric_card("R² SCORE", f'{regression["r2"]:.4f}')

        st.caption(
            "MAE and RMSE measure prediction error in kilometres. "
            "R² measures how much variation in the test-set diameters "
            "is explained by the model."
        )

    except (KeyError, TypeError, ValueError) as error:
        st.error(
            "The evaluation JSON file has an unexpected structure. "
            f"Details: {error}"
        )