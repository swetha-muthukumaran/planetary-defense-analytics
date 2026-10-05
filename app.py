
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

from sklearn.metrics import ConfusionMatrixDisplay


# =========================================================
# PATHS AND PAGE CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "planetary_defense_models"
EVALUATION_FILE = BASE_DIR / "model_evaluation.json"

st.set_page_config(
    page_title="Planetary Defense Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# FEATURES
# =========================================================

CLASSIFICATION_FEATURES = [
    "e", "a", "q", "i", "om", "w", "ma", "ad", "n", "per",
    "per_y", "tp", "sigma_e", "sigma_a", "sigma_q", "sigma_i",
    "sigma_om", "sigma_w", "sigma_ma", "sigma_ad", "sigma_n",
    "sigma_tp", "sigma_per", "rms",
]

REGRESSION_FEATURES = [
    "e", "a", "q", "i", "om", "w", "ma", "ad", "n", "per",
    "per_y", "moid", "sigma_e", "sigma_a", "sigma_q", "sigma_i",
    "sigma_om", "sigma_w", "sigma_ma", "sigma_ad", "sigma_n",
    "sigma_tp", "sigma_per", "rms",
]

CLUSTER_FEATURES = ["e", "a", "q", "i", "moid", "per", "n"]


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
    "rms": 0.5,
}


# =========================================================
# DARK THEME AND UI STYLES
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            135deg, #08111f, #101b30, #111827
        );
        color: #FFFFFF;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    /* Hide heading link icons */
    [data-testid="stHeaderActionElements"],
    .stMarkdown h1 a,
    .stMarkdown h2 a,
    .stMarkdown h3 a,
    .stMarkdown h4 a {
        display: none !important;
    }

    [data-testid="stSidebar"] {
        background: #0B1220;
        border-right: 1px solid #293C56;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #FFFFFF !important;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1600px;
    }

    h1, h2, h3, h4, p, li, label {
        color: #F1F5F9;
    }

    .stMarkdown p,
    .stMarkdown li {
        color: #D5DEEB;
    }

    .metric-card {
        background: #111E31;
        border: 1px solid #344966;
        border-radius: 14px;
        padding: 1rem;
        min-height: 100px;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        color: #CBD5E1 !important;
        font-size: 0.82rem;
    }

    .metric-value {
        color: #FFFFFF !important;
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 0.4rem;
        overflow-wrap: anywhere;
    }

    [data-testid="stWidgetLabel"] p,
    [data-testid="stCaptionContainer"] p {
        color: #D5DEEB !important;
    }

    /* Number input appearance */
    [data-testid="stNumberInput"] input {
        background-color: #F8FAFC !important;
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
    }

    /* Hide manual plus and minus stepper buttons */
    [data-testid="stNumberInput"] button {
        display: none !important;
    }

    /* Consistent button sizing */
    .stButton > button,
    .stFormSubmitButton > button {
        background: #2563EB !important;
        color: #FFFFFF !important;
        border: 1px solid #3B82F6 !important;
        border-radius: 9px !important;
        font-weight: 600 !important;
        width: 100%;
        min-height: 2.8rem;
        height: auto;
        white-space: normal;
        overflow-wrap: anywhere;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background: #1D4ED8 !important;
    }

    [data-testid="stAlert"] p {
        color: #FFFFFF !important;
    }

    hr {
        border-color: #344966 !important;
    }

    /* Uniform overview card titles and descriptions */
    .home-card-title {
        min-height: 3.8rem;
        display: flex;
        align-items: flex-start;
    }

    .home-card-title h3 {
        margin: 0;
        font-size: 1.15rem;
        line-height: 1.5rem;
        color: #F1F5F9;
    }

    .home-card-description {
        min-height: 4.5rem;
        color: #D5DEEB;
        line-height: 1.5rem;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD MODELS AND EVALUATION
# =========================================================

@st.cache_resource
def load_models():
    filenames = {
        "classifier": "pha_classifier.pkl",
        "diameter": "diameter_regressor.pkl",
        "kmeans": "kmeans_model.pkl",
        "dbscan": "dbscan_model.pkl",
        "imputer": "cluster_imputer.pkl",
        "scaler": "cluster_scaler.pkl",
    }

    loaded = {}

    for name, filename in filenames.items():
        path = MODEL_DIR / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Model file not found: {path}"
            )

        loaded[name] = joblib.load(path)

    return loaded


@st.cache_data
def load_evaluation_results():
    if not EVALUATION_FILE.is_file():
        return None

    with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


try:
    models = load_models()

except Exception as error:
    st.error("Could not load the saved models.")
    st.code(str(error))

    st.warning(
        "Check that all model files exist and that your Python "
        "and scikit-learn versions are compatible with the saved models."
    )

    st.stop()


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
        unsafe_allow_html=True,
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
                key=f"{prefix}_{feature}",
            )

    return values


def make_input_frame(values, features):
    frame = pd.DataFrame(
        [[values[feature] for feature in features]],
        columns=features,
    )

    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Please enter valid numeric values.")

    return frame


def go_to_page(page_name):
    st.session_state["next_page"] = page_name


def scale_cluster_input(frame, scaler):
    scaler_features = getattr(
        scaler, "feature_names_in_", None
    )

    if scaler_features is not None:
        expected = list(scaler_features)

        if expected != list(frame.columns):
            raise ValueError(
                "The saved scaler expects different features. "
                f"Expected: {expected}; supplied: {list(frame.columns)}."
            )

    expected_count = getattr(
        scaler, "n_features_in_", None
    )

    if (
        expected_count is not None
        and expected_count != frame.shape[1]
    ):
        raise ValueError(
            f"The saved scaler expects {expected_count} features, "
            f"but the app supplies {frame.shape[1]}."
        )

    return scaler.transform(frame)


def show_shap_explanation(
    model,
    frame,
    title,
    positive_class_index=None,
):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(frame)

        if isinstance(shap_values, list):
            class_index = (
                positive_class_index
                if positive_class_index is not None
                else 0
            )

            values = np.asarray(
                shap_values[class_index]
            )[0]

        else:
            shap_values = np.asarray(shap_values)

            if shap_values.ndim == 3:
                class_index = (
                    positive_class_index
                    if positive_class_index is not None
                    else 0
                )

                values = shap_values[0, :, class_index]

            elif shap_values.ndim == 2:
                values = shap_values[0]

            else:
                st.warning(
                    "SHAP returned an unsupported output shape."
                )
                return

        if len(values) != len(frame.columns):
            st.warning(
                "SHAP values do not match the input features."
            )
            return

        importance = pd.DataFrame({
            "Feature": frame.columns,
            "Impact": np.abs(values),
        }).sort_values(
            "Impact",
            ascending=False,
        ).head(10)

        st.markdown(f"### {title}")

        st.caption(
            "Features are ranked by absolute SHAP impact for this input. "
            "The chart shows impact magnitude, not direction."
        )

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.barh(
            importance["Feature"].iloc[::-1],
            importance["Impact"].iloc[::-1],
            color="#3182F6",
        )

        ax.set_xlabel("Absolute SHAP value")
        ax.set_ylabel("Feature")
        ax.set_title(title)

        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    except Exception as error:
        st.warning(
            f"SHAP explanation could not be generated: {error}"
        )


# =========================================================
# NAVIGATION
# =========================================================

if "next_page" in st.session_state:
    st.session_state["navigation"] = (
        st.session_state.pop("next_page")
    )

PAGES = [
    "Overview",
    "PHA Classification",
    "Diameter Prediction",
    "K-Means Clustering",
    "DBSCAN Clustering",
    "Model Evaluation",
]

with st.sidebar:
    st.title("Planetary Defense")
    st.caption("ASTEROID ANALYTICS")
    st.divider()

    selected_page = st.radio(
        "NAVIGATION",
        PAGES,
        key="navigation",
    )

    st.divider()
    st.markdown("**Model Status**")
    st.success("Saved models loaded")

    dbscan_model = models["dbscan"]
    scaler_model = models["scaler"]

    st.subheader("DBSCAN Diagnostics")

    st.write(
        "DBSCAN eps:",
        getattr(dbscan_model, "eps", "Unknown"),
    )

    st.write(
        "DBSCAN metric:",
        getattr(dbscan_model, "metric", "Unknown"),
    )

    components = getattr(
        dbscan_model, "components_", None
    )

    st.write(
        "Core samples shape:",
        components.shape if components is not None else "Not available",
    )

    st.write(
        "Scaler feature count:",
        getattr(scaler_model, "n_features_in_", "Unknown"),
    )

    st.write(
        "Scaler feature names:",
        list(getattr(scaler_model, "feature_names_in_", []))
        or "Not saved",
    )


# =========================================================
# PAGE HEADER
# =========================================================

page = selected_page

st.title("Planetary Defense Analytics")
st.caption("Asteroid orbital analysis using machine learning")

if page != "Overview":
    if st.button(
        "← Back to Overview",
        key=f"back_{page}",
    ):
        go_to_page("Overview")
        st.rerun()


# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":
    st.subheader("Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card("ANALYTICS MODULES", "04")

    with col2:
        metric_card("CLASSIFICATION", "Random Forest")

    with col3:
        metric_card("CLUSTERING METHODS", "K-Means + DBSCAN")

    st.markdown("### Explore the Dashboard")

    items = [
        (
            "PHA Classification",
            "Classify asteroid features using the trained model.",
        ),
        (
            "Diameter Prediction",
            "Estimate asteroid diameter in kilometres.",
        ),
        (
            "K-Means Clustering",
            "Group asteroids by orbital characteristics.",
        ),
        (
            "DBSCAN Clustering",
            "Explore density-based clusters and noise detection.",
        ),
        (
            "Model Evaluation",
            "Review classification and regression metrics.",
        ),
    ]

    columns = st.columns(3, gap="medium")

    for index, (title, description) in enumerate(items):
        with columns[index % 3]:
            st.markdown(
                f"""
                <div class="home-card-title">
                    <h3>{title}</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="home-card-description">
                    {description}
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(
                f"Open {title}",
                key=f"open_{index}",
                use_container_width=True,
            ):
                go_to_page(title)
                st.rerun()


# =========================================================
# PHA CLASSIFICATION
# =========================================================

elif page == "PHA Classification":
    st.subheader("PHA Classification")

    st.caption(
        "Enter asteroid orbital features and view the model explanation."
    )

    left_col, right_col = st.columns([1, 1.15], gap="large")

    with left_col:
        st.markdown("### Orbital Features")

        threshold = st.slider(
            "Classification threshold",
            min_value=0.10,
            max_value=0.95,
            value=0.80,
            step=0.05,
            key="pha_threshold",
        )

        with st.form("pha_form"):
            values = feature_inputs(
                CLASSIFICATION_FEATURES,
                "pha",
            )

            submitted = st.form_submit_button(
                "Run PHA Classification",
                use_container_width=True,
            )

    with right_col:
        st.markdown("### Prediction Results")

        if submitted:
            try:
                frame = make_input_frame(
                    values,
                    CLASSIFICATION_FEATURES,
                )

                classifier = models["classifier"]
                probabilities = classifier.predict_proba(frame)[0]
                classes = list(classifier.classes_)

                positive_labels = {
                    "1", "Y", "YES", "TRUE", "P", "PHA"
                }

                pha_index = next(
                    (
                        index
                        for index, label in enumerate(classes)
                        if str(label).strip().upper()
                        in positive_labels
                    ),
                    None,
                )

                if pha_index is None:
                    st.error(
                        "Could not identify the PHA class. "
                        f"Model labels: {classes}. "
                        "Check the original training target labels."
                    )

                else:
                    probability = float(
                        probabilities[pha_index]
                    )

                    is_pha = probability >= threshold

                    c1, c2 = st.columns(2)

                    with c1:
                        metric_card(
                            "PHA PROBABILITY",
                            f"{probability:.2%}",
                        )

                    with c2:
                        metric_card(
                            "MODEL OUTPUT",
                            "PHA class" if is_pha else "Non-PHA class",
                        )

                    st.caption(
                        f"Classification threshold: {threshold:.0%}"
                    )

                    st.progress(
                        min(1.0, max(0.0, probability))
                    )

                    show_shap_explanation(
                        classifier,
                        frame,
                        "SHAP Feature Impact",
                        positive_class_index=pha_index,
                    )

                    st.info(
                        "This is a model estimate, not an official "
                        "asteroid hazard assessment."
                    )

            except Exception as error:
                st.error(f"Classification failed: {error}")

        else:
            st.info(
                "Enter the features on the left and click "
                "'Run PHA Classification' to view the prediction."
            )


# =========================================================
# DIAMETER PREDICTION
# =========================================================

elif page == "Diameter Prediction":
    st.subheader("Diameter Prediction")
    st.caption("Estimate asteroid diameter in kilometres.")

    left_col, right_col = st.columns([1, 1.15], gap="large")

    with left_col:
        st.markdown("### Orbital Features")

        with st.form("diameter_form"):
            values = feature_inputs(
                REGRESSION_FEATURES,
                "diameter",
            )

            submitted = st.form_submit_button(
                "Predict Diameter",
                use_container_width=True,
            )

    with right_col:
        st.markdown("### Prediction Results")

        if submitted:
            try:
                frame = make_input_frame(
                    values,
                    REGRESSION_FEATURES,
                )

                model = models["diameter"]
                prediction = float(model.predict(frame)[0])

                if not np.isfinite(prediction):
                    raise ValueError(
                        "The model returned an invalid value."
                    )

                c1, c2 = st.columns(2)

                with c1:
                    metric_card(
                        "ESTIMATED DIAMETER",
                        f"{max(0.0, prediction):,.3f} km",
                    )

                with c2:
                    metric_card(
                        "MODEL",
                        "Random Forest Regressor",
                    )

                if prediction < 0:
                    st.warning(
                        "The raw model output was negative. "
                        "The displayed diameter has been set to zero."
                    )

                show_shap_explanation(
                    model,
                    frame,
                    "Diameter Prediction — SHAP Feature Impact",
                )

            except Exception as error:
                st.error(f"Diameter prediction failed: {error}")

        else:
            st.info(
                "Enter the features and click 'Predict Diameter' "
                "to view the prediction and explanation."
            )


# =========================================================
# K-MEANS CLUSTERING
# =========================================================

elif page == "K-Means Clustering":
    st.subheader("K-Means Clustering")

    st.caption(
        "Group asteroid inputs using the saved K-Means model."
    )

    with st.form("cluster_form"):
        values = feature_inputs(
            CLUSTER_FEATURES,
            "cluster",
            columns=2,
        )

        submitted = st.form_submit_button(
            "Assign Cluster",
            use_container_width=True,
        )

    if submitted:
        try:
            frame = make_input_frame(
                values,
                CLUSTER_FEATURES,
            )

            scaled = scale_cluster_input(
                frame,
                models["scaler"],
            )

            kmeans = models["kmeans"]
            cluster = int(kmeans.predict(scaled)[0])

            metric_card("ASSIGNED CLUSTER", str(cluster))

            st.caption(
                "Cluster labels are group identifiers, not hazard ratings."
            )

            centers = np.asarray(kmeans.cluster_centers_)
            point = np.asarray(scaled[0])

            if centers.shape[1] != point.shape[0]:
                raise ValueError(
                    "Input feature count differs from the "
                    "K-Means cluster-centre feature count."
                )

            distances = np.linalg.norm(
                centers - point,
                axis=1,
            )

            distance_table = pd.DataFrame({
                "Cluster": np.arange(len(distances)),
                "Distance": distances,
            }).sort_values(
                "Distance"
            ).reset_index(drop=True)

            st.markdown("### Cluster Distances")

            st.dataframe(
                distance_table.style.format({
                    "Distance": "{:.4f}"
                }),
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("View submitted input values"):
                st.dataframe(
                    frame,
                    use_container_width=True,
                )

        except Exception as error:
            st.error(f"K-Means clustering failed: {error}")


# =========================================================
# DBSCAN CLUSTERING
# =========================================================

elif page == "DBSCAN Clustering":
    st.subheader("DBSCAN Clustering")

    st.caption(
        "Check whether an input is near a saved DBSCAN core sample."
    )

    st.info(
        "DBSCAN does not provide a standard predict() method for new "
        "samples. This page estimates the nearest saved core sample. "
        "It is valid only when the input uses the same features, "
        "preprocessing and distance metric as the original DBSCAN "
        "training process."
    )

    with st.form("dbscan_form"):
        values = feature_inputs(
            CLUSTER_FEATURES,
            "dbscan",
            columns=2,
        )

        submitted = st.form_submit_button(
            "Check DBSCAN Cluster",
            use_container_width=True,
        )

    if submitted:
        try:
            frame = make_input_frame(
                values,
                CLUSTER_FEATURES,
            )

            scaled = scale_cluster_input(
                frame,
                models["scaler"],
            )

            dbscan = models["dbscan"]

            if not hasattr(dbscan, "components_"):
                raise ValueError(
                    "The saved DBSCAN model has no core samples. "
                    "Check the original training code."
                )

            if not hasattr(dbscan, "core_sample_indices_"):
                raise ValueError(
                    "The saved DBSCAN model has no core sample indices."
                )

            if not hasattr(dbscan, "labels_"):
                raise ValueError(
                    "The saved DBSCAN model has no training labels."
                )

            core_samples = np.asarray(dbscan.components_)
            core_indices = np.asarray(dbscan.core_sample_indices_)
            training_labels = np.asarray(dbscan.labels_)

            if core_samples.size == 0 or core_indices.size == 0:
                raise ValueError(
                    "The saved DBSCAN model has no core samples."
                )

            if (
                core_samples.ndim != 2
                or core_samples.shape[1] != scaled.shape[1]
            ):
                raise ValueError(
                    "DBSCAN core samples and the app input have "
                    "different feature dimensions. The original DBSCAN "
                    "feature list and preprocessing must be checked."
                )

            if len(training_labels) <= int(core_indices.max()):
                raise ValueError(
                    "DBSCAN labels and core sample indices are inconsistent."
                )

            metric = getattr(dbscan, "metric", "euclidean")

            if metric not in ("euclidean", "l2"):
                raise ValueError(
                    "This approximation currently supports Euclidean "
                    f"distance only. Saved DBSCAN metric: {metric}."
                )

            distances = np.linalg.norm(
                core_samples - np.asarray(scaled[0]),
                axis=1,
            )

            nearest_position = int(np.argmin(distances))
            nearest_distance = float(distances[nearest_position])
            nearest_core_index = int(core_indices[nearest_position])
            nearest_label = int(training_labels[nearest_core_index])
            eps = float(dbscan.eps)

            st.markdown("### DBSCAN Result")

            c1, c2 = st.columns(2)

            with c1:
                metric_card(
                    "NEAREST CORE CLUSTER",
                    str(nearest_label),
                )

            with c2:
                metric_card(
                    "DISTANCE TO CORE SAMPLE",
                    f"{nearest_distance:.4f}",
                )

            if nearest_label == -1:
                st.warning(
                    "The nearest saved core index has a noise label. "
                    "Check the saved model and training artifacts."
                )

            elif nearest_distance <= eps:
                st.success(
                    "The input is within eps of a saved core sample. "
                    f"Nearest cluster: {nearest_label}."
                )

            else:
                st.warning(
                    "The input is farther than eps from every saved "
                    "core sample. This nearest-core approximation "
                    "treats it as unassigned/noise."
                )

            st.caption(
                "This is not a full DBSCAN prediction or refit. "
                "The training feature list and preprocessing must match "
                "before interpreting the result."
            )

            with st.expander("View submitted input values"):
                st.dataframe(
                    frame,
                    use_container_width=True,
                )

        except Exception as error:
            st.error(f"DBSCAN clustering failed: {error}")


# =========================================================
# MODEL EVALUATION
# =========================================================

elif page == "Model Evaluation":
    st.subheader("Model Evaluation")

    results = load_evaluation_results()

    if results is None:
        st.error(
            "model_evaluation.json was not found. "
            "Place it in the same folder as app.py."
        )
        st.stop()

    try:
        classification = results["classification"]
        regression = results["regression"]

        st.markdown("### PHA Classification — Random Forest")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card(
                "ACCURACY",
                f'{classification["accuracy"]:.2%}',
            )

        with c2:
            metric_card(
                "PRECISION",
                f'{classification["precision"]:.2%}',
            )

        with c3:
            metric_card(
                "RECALL",
                f'{classification["recall"]:.2%}',
            )

        with c4:
            metric_card(
                "F1-SCORE",
                f'{classification["f1_score"]:.2%}',
            )

        st.markdown("#### Confusion Matrix")

        cm = np.asarray(
            classification["confusion_matrix"]
        )

        if cm.shape != (2, 2):
            st.error(
                "The saved confusion matrix must have a 2 × 2 shape."
            )

        else:
            fig, ax = plt.subplots(figsize=(5, 4))

            fig.patch.set_facecolor("#111E31")
            ax.set_facecolor("#111E31")

            display = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["Non-PHA", "PHA"],
            )

            display.plot(
                ax=ax,
                cmap="Blues",
                colorbar=False,
                values_format="d",
            )

            ax.set_title(
                "PHA Classification Confusion Matrix",
                color="white",
            )

            ax.set_xlabel("Predicted label", color="white")
            ax.set_ylabel("Actual label", color="white")
            ax.tick_params(colors="white")

            fig.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True,
            )

            plt.close(fig)

        st.info(
            "Accuracy alone does not guarantee good PHA detection. "
            "Recall measures the proportion of actual PHA examples detected."
        )

        st.markdown("### Diameter Prediction — Random Forest")

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card(
                "MAE",
                f'{regression["mae"]:.3f} km',
            )

        with c2:
            metric_card(
                "RMSE",
                f'{regression["rmse"]:.3f} km',
            )

        with c3:
            metric_card(
                "R² SCORE",
                f'{regression["r2"]:.4f}',
            )

        st.caption(
            "MAE and RMSE measure prediction error in kilometres. "
            "R² measures the variation explained by the model on "
            "the evaluation dataset."
        )

    except (KeyError, TypeError, ValueError) as error:
        st.error(
            "The evaluation JSON file has an unexpected structure. "
            f"Details: {error}"
        )
