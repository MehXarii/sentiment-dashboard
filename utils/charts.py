import plotly.express as px

# ── Color Palette (Refreshing Light Theme) ─────────────────────────────────────
COLOR_MAP = {
    "POSITIVE": "#0d9488",  # Teal
    "NEGATIVE": "#e11d48",  # Rose
    "NEUTRAL": "#d97706",   # Amber
}

BG_COLOR = "#ffffff"
TEXT_COLOR = "#0f172a"
SUBTEXT_COLOR = "#64748b"
GRID_COLOR = "#f1f5f9"


def style_figure(fig, title_text: str, show_legend: bool = False):
    """Applies clean light styling and explicitly removes dark defaults."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        showlegend=show_legend,
        title=dict(
            text=f"<b>{title_text}</b>",
            font=dict(family="Plus Jakarta Sans, sans-serif", size=16, color=TEXT_COLOR),
            x=0.02,
            y=0.95,
        ),
        font=dict(family="Inter, sans-serif", color=TEXT_COLOR),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    fig.update_xaxes(
        showgrid=False,
        tickfont=dict(color=SUBTEXT_COLOR, size=12),
        title=None,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        tickfont=dict(color=SUBTEXT_COLOR, size=12),
    )
    return fig


def sentiment_bar_chart(df):
    counts = df["label"].value_counts().reindex(["POSITIVE", "NEGATIVE", "NEUTRAL"], fill_value=0)
    
    fig = px.bar(
        x=counts.index,
        y=counts.values,
        labels={"x": "Sentiment", "y": "Count"},
        color=counts.index,
        color_discrete_map=COLOR_MAP,
        text=counts.values,
    )

    fig.update_traces(
        textposition="outside",
        textfont=dict(color=TEXT_COLOR, size=13),
        marker=dict(line=dict(width=0)),
    )
    return style_figure(fig, "Sentiment Distribution", show_legend=False)


def sentiment_pie_chart(df):
    counts = df["label"].value_counts().reset_index()
    counts.columns = ["label", "count"]

    fig = px.pie(
        counts,
        names="label",
        values="count",
        color="label",
        color_discrete_map=COLOR_MAP,
        hole=0.45,
    )

    fig.update_traces(
        textinfo="percent+label",
        textfont=dict(size=12),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}",
    )
    
    style_figure(fig, "Sentiment Share", show_legend=True)
    fig.update_layout(
        legend=dict(
            font=dict(color=TEXT_COLOR, size=12),
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
        )
    )
    return fig


def confidence_chart(df):
    fig = px.histogram(
        df,
        x="confidence",
        color="label",
        color_discrete_map=COLOR_MAP,
        nbins=15,
        opacity=0.85,
        barmode="overlay",
        labels={"confidence": "Confidence Score"},
    )

    style_figure(fig, "Model Confidence Distribution", show_legend=True)
    fig.update_layout(
        legend=dict(
            font=dict(color=TEXT_COLOR, size=12),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        )
    )
    fig.update_xaxes(showgrid=True, tickformat=".0%")
    return fig