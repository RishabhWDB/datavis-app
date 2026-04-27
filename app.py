import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from components.uploader import render_uploader
from components.visualizer import render_visualizer
from utils.data_utils import drop_duplicates, cast_dtype, DTYPE_OPTIONS, DTYPE_DISPLAY, dtype_to_str


# ── Cleaning tab ──────────────────────────────────────────────────────────────

def render_cleaning_tab():
    df = st.session_state["df"]

    st.header("Clean & Edit Data Types")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    c4.metric("Duplicate Rows", int(df.duplicated().sum()))

    # ── Actions ──
    st.subheader("Quick Actions")
    btn1, btn2 = st.columns(2)

    with btn1:
        if st.button("Drop Duplicate Rows", use_container_width=True):
            before = len(df)
            df = drop_duplicates(df)
            st.session_state["df"] = df
            st.success(f"Removed {before - len(df)} duplicate rows.")
            st.rerun()

    with btn2:
        if st.button("Reset to Original Dataset", use_container_width=True):
            st.session_state["df"] = st.session_state["df_original"].copy()
            st.success("Reset to original.")
            st.rerun()

    # ── Missing values ──
    missing_cols = [c for c in df.columns if df[c].isnull().any()]

    st.subheader("Handle Missing Values")
    if missing_cols:
        strategies = {}
        for col in missing_cols:
            n = int(df[col].isnull().sum())
            pct = n / len(df) * 100
            is_num = pd.api.types.is_numeric_dtype(df[col])
            opts = ["Keep as-is", "Drop rows", "Fill with mode"]
            if is_num:
                opts = ["Keep as-is", "Drop rows", "Fill with mean", "Fill with median", "Fill with mode"]

            strategies[col] = st.selectbox(
                f"`{col}` — {n} missing ({pct:.1f}%) | dtype: {df[col].dtype}",
                opts,
                key=f"null_{col}",
            )

        if st.button("Apply Missing Value Strategies", type="primary", use_container_width=True):
            df = df.copy()
            for col, strategy in strategies.items():
                if strategy == "Drop rows":
                    df = df.dropna(subset=[col])
                elif strategy == "Fill with mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == "Fill with median":
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == "Fill with mode":
                    df[col] = df[col].fillna(df[col].mode()[0])
            df = df.reset_index(drop=True)
            st.session_state["df"] = df
            st.success("Applied missing value strategies.")
            st.rerun()
    else:
        st.success("No missing values — dataset is clean.")

    # ── Dtype editor ──
    st.subheader("Column Data Types")
    st.caption("Select a new type for any column, then click Apply.")

    dtype_changes = {}
    cols_per_row = 3
    col_list = df.columns.tolist()

    for i in range(0, len(col_list), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for j, col in enumerate(col_list[i : i + cols_per_row]):
            current = dtype_to_str(df[col].dtype)
            idx = DTYPE_OPTIONS.index(current) if current in DTYPE_OPTIONS else 0
            chosen = row_cols[j].selectbox(
                f"`{col}`",
                DTYPE_OPTIONS,
                index=idx,
                key=f"dtype_{col}",
                format_func=lambda x: DTYPE_DISPLAY.get(x, x),
            )
            if chosen != current:
                dtype_changes[col] = chosen

    if st.button("Apply Data Type Changes", type="primary", use_container_width=True):
        if not dtype_changes:
            st.info("No type changes detected.")
        else:
            errors = []
            for col, new_dtype in dtype_changes.items():
                df, err = cast_dtype(df, col, new_dtype)
                if err:
                    errors.append(f"**{col}**: {err}")
            st.session_state["df"] = df
            if errors:
                for e in errors:
                    st.error(e)
            else:
                st.success(f"Updated: {', '.join(dtype_changes.keys())}")
            st.rerun()

    st.subheader("Preview (first 50 rows)")
    st.dataframe(st.session_state["df"].head(50), use_container_width=True)


# ── ML tab ────────────────────────────────────────────────────────────────────

def render_ml_tab():
    df = st.session_state["df"]

    st.header("Machine Learning")

    all_cols = df.columns.tolist()
    if len(all_cols) < 2:
        st.warning("Need at least 2 columns to run ML.")
        return

    st.subheader("Setup")

    target = st.selectbox("Target column (what to predict)", all_cols)
    feature_options = [c for c in all_cols if c != target]
    features = st.multiselect("Feature columns", feature_options, default=feature_options)

    if not features:
        st.warning("Select at least one feature column.")
        return

    # Auto-detect task type
    is_numeric_target = pd.api.types.is_numeric_dtype(df[target])
    n_unique = df[target].nunique()
    auto_task = "Regression" if (is_numeric_target and n_unique > 10) else "Classification"

    task = st.radio(
        "Task type",
        ["Classification", "Regression"],
        index=0 if auto_task == "Classification" else 1,
        horizontal=True,
    )
    st.caption(f"Auto-detected: **{auto_task}**")

    test_size = st.slider("Test set size", 10, 40, 20, format="%d%%") / 100

    # Model picker
    clf_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "K-Nearest Neighbors": KNeighborsClassifier(),
    }
    reg_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(n_estimators=100),
        "K-Nearest Neighbors": KNeighborsRegressor(),
    }

    model_options = clf_models if task == "Classification" else reg_models
    model_name = st.selectbox("Model", list(model_options.keys()))
    model = model_options[model_name]

    if not st.button("Train Model", type="primary", use_container_width=True):
        return

    try:
        X = df[features].copy()
        y = df[target].copy()

        # Encode categorical features
        for col in X.select_dtypes(exclude="number").columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        # Drop rows with any NaN in X or y
        mask = X.notna().all(axis=1) & y.notna()
        X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

        if len(X) < 10:
            st.error("Not enough clean rows to train. Check missing values in the Clean tab.")
            return

        le_target = None
        if task == "Classification":
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        with st.spinner("Training…"):
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("Results")

        if task == "Classification":
            acc = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{acc:.4f}")

            report = pd.DataFrame(
                classification_report(y_test, y_pred, output_dict=True)
            ).T.round(4)
            st.dataframe(report, use_container_width=True)

            cm = confusion_matrix(y_test, y_pred)
            labels = [str(c) for c in le_target.classes_]
            fig_cm = px.imshow(
                cm, text_auto=True, x=labels, y=labels,
                color_continuous_scale="Blues",
                title="Confusion Matrix",
                template="plotly_white",
            )
            fig_cm.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(fig_cm, use_container_width=True)

        else:
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("R² Score", f"{r2:.4f}")
            mc2.metric("MAE", f"{mae:.4f}")
            mc3.metric("RMSE", f"{mse ** 0.5:.4f}")

            fig_ap = px.scatter(
                x=y_test, y=y_pred,
                labels={"x": "Actual", "y": "Predicted"},
                title="Actual vs Predicted",
                template="plotly_white",
                opacity=0.6,
            )
            min_val = float(min(y_test.min(), y_pred.min()))
            max_val = float(max(y_test.max(), y_pred.max()))
            fig_ap.add_shape(
                type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                line=dict(color="red", dash="dash"),
            )
            st.plotly_chart(fig_ap, use_container_width=True)

        # Feature importance / coefficients
        if hasattr(model, "feature_importances_"):
            imp = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
            imp = imp.sort_values("Importance", ascending=True)
            fig_imp = px.bar(
                imp, x="Importance", y="Feature", orientation="h",
                title="Feature Importance", template="plotly_white",
            )
            st.plotly_chart(fig_imp, use_container_width=True)

        elif hasattr(model, "coef_"):
            coef = model.coef_
            if coef.ndim > 1:
                coef = np.abs(coef).mean(axis=0)
            if len(coef) == len(features):
                imp = pd.DataFrame({"Feature": features, "Coefficient": coef})
                imp = imp.sort_values("Coefficient", ascending=True)
                fig_coef = px.bar(
                    imp, x="Coefficient", y="Feature", orientation="h",
                    title="Feature Coefficients", template="plotly_white",
                )
                st.plotly_chart(fig_coef, use_container_width=True)

    except Exception as e:
        st.error(f"Training failed: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="DataVis App",
        page_icon="📊",
        layout="wide",
    )

    st.title("Dataset Visualizer & ML Studio")
    st.caption("Upload a CSV → clean it → visualize → run ML models, all in one place.")

    tabs = st.tabs(["Upload", "Clean & Edit Types", "Visualize", "Machine Learning"])

    with tabs[0]:
        render_uploader()

    df = st.session_state.get("df")

    if df is None:
        for tab in tabs[1:]:
            with tab:
                st.info("Upload a CSV file in the **Upload** tab to get started.")
        return

    with tabs[1]:
        render_cleaning_tab()

    with tabs[2]:
        render_visualizer(st.session_state["df"])

    with tabs[3]:
        render_ml_tab()


if __name__ == "__main__":
    main()
