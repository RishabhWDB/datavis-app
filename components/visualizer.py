import streamlit as st
from utils.chart_builder import (
    histogram, bar_chart, scatter, box_plot,
    correlation_heatmap, line_chart, pair_plot
)


def render_visualizer(df):
    st.header("Visualize Data")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    all_cols = df.columns.tolist()

    chart_types = ["Histogram", "Bar Chart", "Scatter Plot", "Box Plot",
                   "Correlation Heatmap", "Line Chart", "Pair Plot"]

    chart_type = st.selectbox("Chart Type", chart_types)
    fig = None

    if chart_type == "Histogram":
        if not numeric_cols:
            st.warning("No numeric columns available for histogram.")
            return
        col = st.selectbox("Column", numeric_cols)
        color = st.selectbox("Color by (optional)", ["None"] + categorical_cols)
        fig = histogram(df, col, color=None if color == "None" else color)

    elif chart_type == "Bar Chart":
        options = categorical_cols if categorical_cols else all_cols
        col = st.selectbox("Column", options)
        top_n = st.slider("Show top N values", 5, 50, 20)
        plot_df = df[col].value_counts().nlargest(top_n).reset_index()
        plot_df.columns = [col, "count"]
        import plotly.express as px
        fig = px.bar(plot_df, x=col, y="count", title=f"Top {top_n} Values: {col}",
                     template="plotly_white")

    elif chart_type == "Scatter Plot":
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for a scatter plot.")
            return
        x = st.selectbox("X axis", numeric_cols, index=0)
        y = st.selectbox("Y axis", numeric_cols, index=1)
        color = st.selectbox("Color by (optional)", ["None"] + all_cols)
        fig = scatter(df, x, y, color=None if color == "None" else color)

    elif chart_type == "Box Plot":
        if not numeric_cols:
            st.warning("No numeric columns available for box plot.")
            return
        y = st.selectbox("Value column", numeric_cols)
        x = st.selectbox("Group by (optional)", ["None"] + categorical_cols)
        fig = box_plot(df, y=y, x=None if x == "None" else x)

    elif chart_type == "Correlation Heatmap":
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for a correlation heatmap.")
            return
        fig = correlation_heatmap(df)

    elif chart_type == "Line Chart":
        if not numeric_cols:
            st.warning("No numeric columns available for line chart.")
            return
        x = st.selectbox("X axis", all_cols)
        y = st.selectbox("Y axis", numeric_cols)
        fig = line_chart(df, x, y)

    elif chart_type == "Pair Plot":
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for a pair plot.")
            return
        default = numeric_cols[:min(4, len(numeric_cols))]
        selected = st.multiselect("Select columns (2–5 recommended)", numeric_cols, default=default)
        color = st.selectbox("Color by (optional)", ["None"] + categorical_cols)
        if len(selected) < 2:
            st.info("Select at least 2 columns.")
            return
        fig = pair_plot(df, selected, color=None if color == "None" else color)

    if fig:
        st.plotly_chart(fig, use_container_width=True)
