import streamlit as st
from utils.data_utils import load_csv


def render_uploader():
    st.header("Upload Dataset")
    st.markdown("Upload any CSV file to get started. Your data will carry through all tabs.")

    file = st.file_uploader("Choose a CSV file", type=["csv"])

    if file:
        if st.session_state.get("filename") != file.name:
            df = load_csv(file)
            st.session_state["df_original"] = df.copy()
            st.session_state["df"] = df.copy()
            st.session_state["filename"] = file.name

        df = st.session_state["df"]
        st.success(f"Loaded **{file.name}** — {len(df):,} rows × {len(df.columns)} columns")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", len(df.columns))
        c3.metric("Missing Values", int(df.isnull().sum().sum()))
        c4.metric("Duplicate Rows", int(df.duplicated().sum()))

        st.subheader("Preview")
        st.dataframe(df.head(50), use_container_width=True)

        st.subheader("Column Summary")
        summary = df.describe(include="all").T
        st.dataframe(summary, use_container_width=True)
    else:
        st.info("No file uploaded yet. Upload a CSV above to begin.")
