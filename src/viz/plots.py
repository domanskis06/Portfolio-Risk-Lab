"""Plotly charts for the Streamlit dashboard (dark theme)."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Shared dark styling
_PAPER = "#0B1220"
_PLOT = "#111827"
_GRID = "#1F2937"
_TEXT = "#E8EDF5"
_MUTED = "#9CA3AF"
_BLUE = "#5B8DEF"
_TEAL = "#2DD4BF"
_AMBER = "#FBBF24"
_ROSE = "#FB7185"


def _layout(
    fig: go.Figure,
    *,
    height: int = 360,
    title: str | None = None,
    show_legend: bool = True,
) -> go.Figure:
    # Legend sits *below* the plot so it never collides with the title.
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_PAPER,
        plot_bgcolor=_PLOT,
        font=dict(color=_TEXT, size=12, family="Inter, Segoe UI, sans-serif"),
        title=(
            dict(
                text=title,
                x=0.0,
                xanchor="left",
                y=0.99,
                yanchor="top",
                font=dict(size=15, color=_TEXT),
                pad=dict(b=8),
            )
            if title
            else None
        ),
        margin=dict(l=58, r=24, t=56 if title else 28, b=72 if show_legend else 44),
        height=height,
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            x=0,
            xanchor="left",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=_MUTED),
        ),
        hovermode="x unified",
    )
    fig.update_xaxes(
        gridcolor=_GRID,
        zeroline=False,
        color=_MUTED,
        tickfont=dict(size=11),
        title_font=dict(size=12, color=_MUTED),
    )
    fig.update_yaxes(
        gridcolor=_GRID,
        zeroline=False,
        color=_MUTED,
        tickfont=dict(size=11),
        title_font=dict(size=12, color=_MUTED),
    )
    return fig


def nav_drawdown_figure(perf: pd.DataFrame) -> go.Figure:
    """Two-panel NAV + drawdown chart (no overlapping titles/legend)."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.64, 0.36],
        vertical_spacing=0.10,
    )
    fig.add_trace(
        go.Scatter(
            x=perf.index,
            y=perf["nav"],
            name="NAV",
            line=dict(color=_BLUE, width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>NAV=%{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=perf.index,
            y=perf["drawdown"],
            name="Drawdown",
            fill="tozeroy",
            line=dict(color=_ROSE, width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>DD=%{y:.2%}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="NAV", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    return _layout(fig, height=500, title="Net asset value & drawdown")


def correlation_heatmap(returns: pd.DataFrame) -> go.Figure:
    """Asset return correlation matrix heatmap."""
    corr = returns.corr()
    n = len(corr.columns)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig.update_traces(
        xgap=2,
        ygap=2,
        hovertemplate="%{x} x %{y}<br>rho=%{z:.3f}<extra></extra>",
        textfont=dict(size=13 if n <= 12 else 10),
    )
    fig = _layout(
        fig, height=max(680, 140 + 62 * n), title="Return correlation", show_legend=False
    )
    fig.update_layout(
        coloraxis_colorbar=dict(
            title=dict(text="rho", side="right"),
            thickness=14,
            len=0.72,
            x=1.02,
            tickfont=dict(size=11),
        ),
        margin=dict(l=70, r=80, t=56, b=90),
    )
    fig.update_xaxes(side="bottom", tickangle=-35, title_text="", tickfont=dict(size=13))
    fig.update_yaxes(autorange="reversed", title_text="", tickfont=dict(size=13))
    return fig


def var_comparison_figure(
    risk_df: pd.DataFrame,
    *,
    bootstrap_point: float | None = None,
    bootstrap_low: float | None = None,
    bootstrap_high: float | None = None,
    source_note: str | None = None,
) -> go.Figure:
    """Grouped bars: VaR / ES by method, optional bootstrap CI on historical VaR."""
    title = "VaR & ES by method"
    if source_note:
        title = f"{title}  |  {source_note}"

    if risk_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No risk figures available",
            showarrow=False,
            font=dict(color=_MUTED, size=14),
        )
        return _layout(fig, height=360, title=title)

    order = ["historical", "parametric", "monte_carlo"]
    df = risk_df.copy()
    df["method"] = pd.Categorical(df["method"], categories=order, ordered=True)
    df = df.sort_values("method")
    labels = [str(m).replace("_", " ") for m in df["method"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="VaR",
            x=labels,
            y=df["var_value"],
            marker_color=_BLUE,
            text=[f"{v:.2%}" for v in df["var_value"]],
            textposition="inside",
            textfont=dict(size=11, color="#0B1220"),
            hovertemplate="%{x}<br>VaR=%{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="ES",
            x=labels,
            y=df["es_value"],
            marker_color=_TEAL,
            text=[f"{v:.2%}" for v in df["es_value"]],
            textposition="inside",
            textfont=dict(size=11, color="#0B1220"),
            hovertemplate="%{x}<br>ES=%{y:.4f}<extra></extra>",
        )
    )

    if (
        bootstrap_point is not None
        and bootstrap_low is not None
        and bootstrap_high is not None
    ):
        fig.add_trace(
            go.Scatter(
                name="Hist VaR · bootstrap 95% CI",
                x=["historical"],
                y=[bootstrap_point],
                mode="markers",
                marker=dict(symbol="diamond", size=11, color=_AMBER),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[bootstrap_high - bootstrap_point],
                    arrayminus=[bootstrap_point - bootstrap_low],
                    color=_AMBER,
                    thickness=2,
                    width=10,
                ),
                hovertemplate=(
                    f"point={bootstrap_point:.4f}<br>"
                    f"CI=[{bootstrap_low:.4f}, {bootstrap_high:.4f}]<extra></extra>"
                ),
            )
        )

    ymax = float(df[["var_value", "es_value"]].to_numpy().max()) * 1.18
    fig.update_layout(
        barmode="group",
        bargap=0.28,
        yaxis=dict(tickformat=".2%", range=[0, ymax], title="Loss"),
        uniformtext_minsize=10,
        uniformtext_mode="hide",
    )
    return _layout(fig, height=420, title=title)


def _runs_of_ones(flag: pd.Series) -> list[tuple[int, int]]:
    """Positional (start, end) index pairs of consecutive True values."""
    arr = flag.to_numpy(dtype=bool)
    out, start = [], None
    for i, v in enumerate(arr):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(arr) - 1))
    return out


def ml_regime_figure(pred: pd.DataFrame) -> go.Figure:
    """Top: model P(high vol). Bottom: trailing vol. Rose bands = true high-vol regime."""
    if pred.empty:
        fig = go.Figure()
        fig.add_annotation(text="No predictions for this run", showarrow=False)
        return _layout(fig, height=380, title="OOS regime forecast")

    pred = pred.sort_values("dt").reset_index(drop=True)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.5],
        vertical_spacing=0.08,
    )

    for i0, i1 in _runs_of_ones(pred["y_true"] >= 0.5):
        for r in (1, 2):
            fig.add_vrect(
                x0=pred["dt"].iloc[i0] - pd.Timedelta(hours=12),
                x1=pred["dt"].iloc[i1] + pd.Timedelta(hours=12),
                fillcolor=_ROSE,
                opacity=0.16,
                line_width=0,
                layer="below",
                row=r,
                col=1,
                exclude_empty_subplots=False,
            )
    fig.add_trace(
        go.Scatter(
            x=[pred["dt"].iloc[0]],
            y=[None],
            mode="markers",
            marker=dict(symbol="square", size=11, color=_ROSE, opacity=0.4),
            name="True high-vol regime",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=pred["dt"],
            y=pred["y_pred"],
            name="Predicted P(high vol)",
            line=dict(color=_BLUE, width=1.6),
            fill="tozeroy",
            fillcolor="rgba(91,141,239,0.18)",
            hovertemplate="%{x|%Y-%m-%d}<br>P=%{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    if "vol_21d" in pred.columns and pred["vol_21d"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=pred["dt"],
                y=pred["vol_21d"],
                name="Trailing 21d vol",
                line=dict(color=_AMBER, width=1.5),
                hovertemplate="%{x|%Y-%m-%d}<br>vol=%{y:.1%}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    fig.update_yaxes(title_text="P(high vol)", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="Ann. vol", tickformat=".0%", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    return _layout(
        fig, height=560, title="Out-of-sample regime forecast vs. realized volatility"
    )


def weights_bar_figure(weights: pd.Series) -> go.Figure:
    """Vertical bars: ticker on x, portfolio weight (%) on y."""
    s = weights[weights > 0].sort_values(ascending=False)
    fig = go.Figure(
        go.Bar(
            x=s.index.astype(str),
            y=s.values,
            marker_color=_BLUE,
            text=[f"{v:.1%}" for v in s.values],
            textposition="outside",
            textfont=dict(size=12, color=_TEXT),
            cliponaxis=False,
            hovertemplate="%{x}: %{y:.2%}<extra></extra>",
        )
    )
    fig.update_yaxes(
        tickformat=".0%",
        range=[0, float(s.max()) * 1.25],
        title_text="Weight",
    )
    fig.update_xaxes(title_text="", type="category")
    fig.update_layout(bargap=0.35, hovermode="closest")
    return _layout(fig, height=380, title="Portfolio weights", show_legend=False)
