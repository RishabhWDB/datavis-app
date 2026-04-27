import plotly.express as px
import plotly.graph_objects as go

def histogram(df, col, color=None):
    return px.histogram(
        df, x=col, color=color, marginal="box",
        title=f"Distribution of {col}",
        template="plotly_white"
    )

def bar_chart(df, col):
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "count"]
    return px.bar(counts, x=col, y="count", title=f"Value Counts: {col}", template="plotly_white")

def scatter(df, x, y, color=None):
    return px.scatter(
        df, x=x, y=y, color=color,
        title=f"{x} vs {y}", template="plotly_white",
        opacity=0.7
    )

def box_plot(df, y, x=None):
    return px.box(df, x=x, y=y, title=f"Box Plot: {y}", template="plotly_white")

def correlation_heatmap(df):
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return None
    corr = numeric.corr().round(2)
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=corr.values,
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(title="Correlation Heatmap", template="plotly_white")
    return fig

def line_chart(df, x, y):
    return px.line(df, x=x, y=y, title=f"{y} over {x}", template="plotly_white")

def pair_plot(df, cols, color=None):
    return px.scatter_matrix(
        df, dimensions=cols, color=color,
        title="Pair Plot", template="plotly_white",
        opacity=0.6
    )
