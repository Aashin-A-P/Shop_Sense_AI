"""ShopSense AI Streamlit application."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import FEATURE_COLUMNS, get_test_sample, load_dataset
from utils.predictions import DISPLAY_NAMES, load_metadata, load_model, predict_session

st.set_page_config(
    page_title="ShopSense AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f6f8fc; color: #172033; }
    h1, h2, h3 { color: #102a43; letter-spacing: -0.02em; }
    [data-testid="stSidebar"] { background: #102a43; }
    [data-testid="stSidebar"] * { color: #f7fafc; }
    [data-testid="stMetric"] {
        background: white; border: 1px solid #e7edf5; border-radius: 14px;
        padding: 18px; box-shadow: 0 5px 18px rgba(16,42,67,.05);
    }
    .hero { background: linear-gradient(120deg,#102a43,#1769aa); color: white;
        padding: 28px 32px; border-radius: 18px; margin-bottom: 22px; }
    .hero h1 { color: white; margin: 0; }
    .hero p { margin: 8px 0 0; color: #dbeafe; }
    .result-card { padding: 22px; border-radius: 16px; margin: 16px 0; }
    .positive { background: #e8f7ee; border: 1px solid #9bd5ae; color: #155b35; }
    .negative { background: #fff3e6; border: 1px solid #f3bd7b; color: #8a4700; }
    .result-card h2 { margin: 0 0 8px; color: inherit; }
    .small-note { color: #5d6b7c; font-size: .9rem; }
    div.stButton > button { border-radius: 9px; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_dataset()


@st.cache_data(show_spinner=False)
def get_metadata() -> dict:
    return load_metadata()


@st.cache_resource(show_spinner=False)
def get_model(model_key: str):
    return load_model(model_key)


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def dataset_defaults(df: pd.DataFrame) -> dict:
    integer_features = {
        "Administrative", "Informational", "ProductRelated", "OperatingSystems",
        "Browser", "Region", "TrafficType",
    }
    defaults = {}
    for column in FEATURE_COLUMNS:
        if column == "Weekend":
            defaults[column] = bool(df[column].mode().iloc[0])
        elif column in {"Month", "VisitorType"}:
            defaults[column] = str(df[column].mode().iloc[0])
        elif column in integer_features:
            defaults[column] = int(round(df[column].median()))
        else:
            defaults[column] = float(df[column].median())
    return defaults


def clear_stale_result() -> None:
    st.session_state.prediction_result = None
    if st.session_state.get("loaded_sample") is not None:
        st.session_state.loaded_sample = None
        st.session_state.sample_actual = None
        st.session_state.sample_message = None


def initialize_prediction_state(df: pd.DataFrame) -> None:
    defaults = dataset_defaults(df)
    for feature, value in defaults.items():
        st.session_state.setdefault(feature, value)
    st.session_state.setdefault("prediction_model", "Best Performing Model")
    st.session_state.setdefault("prediction_result", None)
    st.session_state.setdefault("loaded_sample", None)
    st.session_state.setdefault("sample_actual", None)
    st.session_state.setdefault("sample_message", None)


def load_sample(df: pd.DataFrame, purchase: bool) -> None:
    sample, actual = get_test_sample(df, purchase)
    sample_values = {}
    for feature in FEATURE_COLUMNS:
        value = sample[feature]
        if feature == "Weekend":
            value = bool(value)
        elif feature in {"Month", "VisitorType"}:
            value = str(value)
        elif feature in {
            "Administrative", "Informational", "ProductRelated", "OperatingSystems",
            "Browser", "Region", "TrafficType",
        }:
            value = int(value)
        else:
            value = float(value)
        st.session_state[feature] = value
        sample_values[feature] = value
    st.session_state.loaded_sample = sample_values
    st.session_state.sample_actual = actual
    st.session_state.prediction_result = None
    st.session_state.sample_message = (
        "Purchase example loaded. Click Predict to analyze."
        if purchase else "No-purchase example loaded. Click Predict to analyze."
    )


def reset_prediction(df: pd.DataFrame) -> None:
    for feature, value in dataset_defaults(df).items():
        st.session_state[feature] = value
    st.session_state.prediction_model = "Best Performing Model"
    st.session_state.prediction_result = None
    st.session_state.loaded_sample = None
    st.session_state.sample_actual = None
    st.session_state.sample_message = None


def input_frame() -> pd.DataFrame:
    values = {feature: st.session_state[feature] for feature in FEATURE_COLUMNS}
    values["Weekend"] = bool(values["Weekend"])
    return pd.DataFrame([values], columns=FEATURE_COLUMNS)


def dashboard_page(df: pd.DataFrame) -> None:
    hero("Customer Purchase Analytics", "Explore online shopping behaviour and purchase patterns.")
    purchases = int(df["Revenue"].astype(bool).sum())
    total = len(df)
    columns = st.columns(4)
    columns[0].metric("Total Shopping Sessions", f"{total:,}")
    columns[1].metric("Successful Purchases", f"{purchases:,}")
    columns[2].metric("Non-Purchase Sessions", f"{total - purchases:,}")
    columns[3].metric("Conversion Rate", format_percent(purchases / total))

    st.subheader("Purchase Distribution")
    distribution = pd.DataFrame({
        "Outcome": ["Purchase", "No Purchase"],
        "Sessions": [purchases, total - purchases],
    })
    figure = px.pie(
        distribution, values="Sessions", names="Outcome", hole=.62,
        color="Outcome", color_discrete_map={"Purchase": "#22a06b", "No Purchase": "#f59e42"},
    )
    figure.update_traces(textposition="outside", textinfo="label+value+percent")
    figure.update_layout(height=390, margin=dict(t=20, b=20, l=20, r=20), legend_title_text="")
    st.plotly_chart(figure, use_container_width=True)

    left, right = st.columns(2)
    visitor = (
        df.assign(Revenue=df["Revenue"].astype(int))
        .groupby("VisitorType", as_index=False)["Revenue"].mean()
        .rename(columns={"Revenue": "Conversion Rate", "VisitorType": "Visitor Type"})
    )
    visitor["Visitor Type"] = visitor["Visitor Type"].replace({
        "New_Visitor": "New Visitor", "Returning_Visitor": "Returning Visitor",
    })
    visitor_chart = px.bar(visitor, x="Visitor Type", y="Conversion Rate", text_auto=".1%", color="Visitor Type")
    visitor_chart.update_yaxes(tickformat=".0%", title="Conversion Rate")
    visitor_chart.update_layout(showlegend=False, title="Visitor Type Analysis", height=400)
    left.plotly_chart(visitor_chart, use_container_width=True)

    month_order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly = (
        df.assign(Revenue=df["Revenue"].astype(int))
        .groupby("Month", as_index=False)["Revenue"].mean()
        .rename(columns={"Revenue": "Conversion Rate"})
    )
    monthly["Month"] = pd.Categorical(monthly["Month"], month_order, ordered=True)
    monthly = monthly.sort_values("Month")
    month_chart = px.line(monthly, x="Month", y="Conversion Rate", markers=True, text="Conversion Rate")
    month_chart.update_traces(texttemplate="%{text:.1%}", textposition="top center", line_color="#1769aa")
    month_chart.update_yaxes(tickformat=".0%")
    month_chart.update_layout(title="Monthly Purchase Analysis", height=400)
    right.plotly_chart(month_chart, use_container_width=True)

    st.subheader("Browsing Behaviour")
    behaviour = df[["ProductRelated_Duration", "Revenue"]].copy()
    behaviour["Outcome"] = np.where(behaviour["Revenue"].astype(bool), "Purchase", "No Purchase")
    box = px.box(
        behaviour, x="Outcome", y="ProductRelated_Duration", color="Outcome",
        color_discrete_map={"Purchase": "#22a06b", "No Purchase": "#f59e42"},
        points=False, labels={"ProductRelated_Duration": "Time on Product Pages (seconds)"},
    )
    box.update_layout(showlegend=False, height=430)
    st.plotly_chart(box, use_container_width=True)

    st.subheader("Dataset Information")
    info = st.columns(4)
    info[0].markdown("**Dataset**  \nOnline Shoppers Purchasing Intention")
    info[1].markdown(f"**Records**  \n{len(df):,}")
    info[2].markdown(f"**Input Features**  \n{len(FEATURE_COLUMNS)}")
    info[3].markdown("**Target Variable**  \nRevenue")
    st.markdown("Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)")


def prediction_page(df: pd.DataFrame, metadata: dict) -> None:
    initialize_prediction_state(df)
    hero(
        "Predict Customer Purchase Intention",
        "Enter shopping-session information to estimate the likelihood of a purchase.",
    )
    st.subheader("Try a Sample Shopping Session")
    st.write("Explore how the model behaves using actual shopping sessions from the held-out test set.")
    sample_left, sample_right, reset_col = st.columns([1, 1, 1])
    if sample_left.button("LOAD PURCHASE EXAMPLE", use_container_width=True):
        load_sample(df, True)
        st.rerun()
    if sample_right.button("LOAD NO-PURCHASE EXAMPLE", use_container_width=True):
        load_sample(df, False)
        st.rerun()
    if reset_col.button("RESET PREDICTION", use_container_width=True):
        reset_prediction(df)
        st.rerun()
    if st.session_state.sample_message:
        st.info(st.session_state.sample_message)

    st.subheader("Basic Session Details")
    a, b = st.columns(2)
    a.number_input("Product Pages Visited", min_value=0, step=1, key="ProductRelated", on_change=clear_stale_result)
    b.number_input("Time Spent on Product Pages (seconds)", min_value=0.0, step=1.0, key="ProductRelated_Duration", on_change=clear_stale_result)
    a.number_input("Informational Pages Visited", min_value=0, step=1, key="Informational", on_change=clear_stale_result)
    b.number_input("Time Spent on Informational Pages (seconds)", min_value=0.0, step=1.0, key="Informational_Duration", on_change=clear_stale_result)
    a.number_input("Bounce Rate", min_value=0.0, max_value=1.0, step=0.001, format="%.4f", key="BounceRates", on_change=clear_stale_result)
    b.number_input("Exit Rate", min_value=0.0, max_value=1.0, step=0.001, format="%.4f", key="ExitRates", on_change=clear_stale_result)

    visitor_options = sorted(df["VisitorType"].astype(str).unique().tolist())
    a.selectbox("Visitor Type", visitor_options, key="VisitorType", on_change=clear_stale_result)
    b.selectbox(
        "Weekend", [False, True], key="Weekend", format_func=lambda value: "Yes" if value else "No",
        on_change=clear_stale_result,
    )

    with st.expander("Advanced Session Details", expanded=False):
        c, d = st.columns(2)
        c.number_input("Administrative Pages", min_value=0, step=1, key="Administrative", on_change=clear_stale_result)
        d.number_input("Administrative Duration (seconds)", min_value=0.0, step=1.0, key="Administrative_Duration", on_change=clear_stale_result)
        c.number_input("Page Value", min_value=0.0, step=0.1, key="PageValues", help="Average value of pages viewed before completing a transaction.", on_change=clear_stale_result)
        d.number_input("Special Day Proximity", min_value=0.0, max_value=1.0, step=0.1, key="SpecialDay", help="Closeness to a special shopping day, from 0 to 1.", on_change=clear_stale_result)
        c.selectbox("Month", sorted(df["Month"].astype(str).unique()), key="Month", on_change=clear_stale_result)
        for feature, label, column in [
            ("OperatingSystems", "Operating System", d), ("Browser", "Browser", c),
            ("Region", "Region", d), ("TrafficType", "Traffic Type", c),
        ]:
            column.selectbox(label, sorted(int(v) for v in df[feature].unique()), key=feature, on_change=clear_stale_result)

    model_options = ["Best Performing Model", *DISPLAY_NAMES.values()]
    st.selectbox(
        "SELECT PREDICTION MODEL", model_options, key="prediction_model",
        on_change=lambda: st.session_state.update(prediction_result=None),
    )
    if st.button("PREDICT PURCHASE INTENTION", type="primary", use_container_width=True):
        try:
            selection = st.session_state.prediction_model
            reverse_names = {name: key for key, name in DISPLAY_NAMES.items()}
            model_key = metadata["best_model"] if selection == "Best Performing Model" else reverse_names[selection]
            frame = input_frame()
            prediction, probability = predict_session(get_model(model_key), frame)
            st.session_state.prediction_result = {
                "prediction": prediction,
                "probability": probability,
                "model_key": model_key,
                "actual": st.session_state.sample_actual,
            }
        except (ValueError, TypeError) as exc:
            st.error(f"Invalid prediction input: {exc}")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    result = st.session_state.prediction_result
    if result:
        positive = result["prediction"] == 1
        title = "LIKELY TO PURCHASE" if positive else "UNLIKELY TO PURCHASE"
        card_class = "positive" if positive else "negative"
        outcome = "Purchase" if positive else "No Purchase"
        st.markdown(
            f'<div class="result-card {card_class}"><h2>{title}</h2>'
            f'<b>Purchase Probability:</b> {result["probability"]:.1%}<br>'
            f'<b>Selected Model:</b> {DISPLAY_NAMES[result["model_key"]]}<br>'
            f'<b>Predicted Outcome:</b> {outcome}</div>',
            unsafe_allow_html=True,
        )
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=result["probability"] * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Estimated Purchase Probability"},
            gauge={
                "axis": {"range": [0, 100]}, "bar": {"color": "#22a06b" if positive else "#f59e42"},
                "steps": [{"range": [0, 50], "color": "#fff3e6"}, {"range": [50, 100], "color": "#e8f7ee"}],
            },
        ))
        gauge.update_layout(height=300, margin=dict(t=55, b=15, l=30, r=30))
        st.plotly_chart(gauge, use_container_width=True)
        if result["actual"] is not None:
            actual_label = "Purchase" if result["actual"] else "No Purchase"
            correct = result["prediction"] == result["actual"]
            st.markdown(f"**Actual Recorded Outcome:** {actual_label}  \n\n**Result:** {'Correct Prediction' if correct else 'Incorrect Prediction'}")
            (st.success if correct else st.warning)(
                "The prediction matches the recorded outcome." if correct else "The prediction differs from the recorded outcome."
            )
        st.caption("This is an estimate based on historical shopping behaviour, not a certainty or causal explanation.")


def comparison_page(metadata: dict) -> None:
    hero("Machine Learning Model Comparison", "Compare the performance of supervised learning algorithms.")
    rows = []
    for key, label in DISPLAY_NAMES.items():
        metric = metadata["evaluation_metrics"][key]
        rows.append({
            "Model": label, "Accuracy": metric["accuracy"], "Precision": metric["precision"],
            "Recall": metric["recall"], "F1-Score": metric["f1_score"], "ROC-AUC": metric["roc_auc"],
        })
    table = pd.DataFrame(rows)
    st.subheader("Performance Table")
    st.dataframe(
        table.style.format({column: "{:.4f}" for column in table.columns if column != "Model"})
        .highlight_max(subset=["F1-Score"], color="#b9e7c8"),
        hide_index=True, use_container_width=True,
    )
    st.caption(f"Best held-out positive-class F1-score: {metadata['best_model_display_name']}")

    chart = px.bar(
        table, x="Model", y="F1-Score", color="Model", text_auto=".4f",
        color_discrete_sequence=["#1769aa", "#7c5ce5", "#22a06b"],
    )
    chart.update_layout(showlegend=False, yaxis_range=[0, 1], height=420, title="F1-Score by Model")
    st.plotly_chart(chart, use_container_width=True)

    st.subheader("Confusion Matrix")
    selected_name = st.selectbox("Select a model", list(DISPLAY_NAMES.values()), key="matrix_model")
    selected_key = {name: key for key, name in DISPLAY_NAMES.items()}[selected_name]
    matrix = np.array(metadata["evaluation_metrics"][selected_key]["confusion_matrix"])
    labels = [[f"True Negative<br>{matrix[0,0]}", f"False Positive<br>{matrix[0,1]}"],
              [f"False Negative<br>{matrix[1,0]}", f"True Positive<br>{matrix[1,1]}"]]
    heatmap = go.Figure(go.Heatmap(
        z=matrix, x=["Predicted No Purchase", "Predicted Purchase"],
        y=["Actual No Purchase", "Actual Purchase"], colorscale="Blues", showscale=False,
        text=labels, texttemplate="%{text}", textfont={"size": 15},
    ))
    heatmap.update_layout(height=430, yaxis_autorange="reversed", margin=dict(t=20, b=20))
    st.plotly_chart(heatmap, use_container_width=True)

    st.subheader("Decision Tree Feature Importance")
    tree = get_model("decision_tree")
    names = tree.named_steps["preprocessor"].get_feature_names_out()
    importance = tree.named_steps["classifier"].feature_importances_
    importance_df = pd.DataFrame({"Feature": names, "Importance": importance})
    importance_df["Feature"] = importance_df["Feature"].str.replace(r"^[^_]+__", "", regex=True)
    importance_df = importance_df.nlargest(10, "Importance").sort_values("Importance")
    importance_chart = px.bar(importance_df, x="Importance", y="Feature", orientation="h", text_auto=".3f")
    importance_chart.update_traces(marker_color="#1769aa")
    importance_chart.update_layout(height=480)
    st.plotly_chart(importance_chart, use_container_width=True)
    st.caption("Feature importance shows predictive contribution within this fitted tree; it does not prove causation.")

    st.subheader("Algorithm Explanations")
    with st.expander("Logistic Regression"):
        st.write("A supervised classification algorithm that uses the logistic function to estimate the probability of a binary outcome.")
    with st.expander("K-Nearest Neighbors"):
        st.write("A supervised learning algorithm that classifies a sample based on nearby training examples after feature scaling.")
    with st.expander("Decision Tree"):
        st.write("A supervised classification algorithm that learns decision rules by recursively splitting the data based on feature values.")


def main() -> None:
    st.sidebar.title("🛍️ ShopSense AI")
    st.sidebar.caption("Understand Customer Behaviour. Predict Purchase Intent.")
    page = st.sidebar.radio("Navigation", ["Dashboard", "Purchase Prediction", "Model Comparison"])
    st.sidebar.markdown("---")
    st.sidebar.caption("Powered by Scikit-learn  \nMachine Learning Mini Project")
    try:
        df = get_data()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()
    if page == "Dashboard":
        dashboard_page(df)
        return
    try:
        metadata = get_metadata()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.info("Trained models not found. Run python train_model.py first.")
        st.stop()
    if page == "Purchase Prediction":
        prediction_page(df, metadata)
    else:
        comparison_page(metadata)


if __name__ == "__main__":
    main()
