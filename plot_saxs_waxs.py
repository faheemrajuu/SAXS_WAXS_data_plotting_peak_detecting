# plot_saxs_waxs.py
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

# Optional (only used if SHOW_PEAKS=True)
from scipy.signal import find_peaks, peak_widths


# ==========================================================
# USER HANDLES
# ==========================================================
FILE = Path("YourExcelFileName.xlsx")
SAMPLE = "SampleName"         # name or 0-based index

X_SPLIT = 1.0
X_MIN = None
X_MAX = None

# SAXS axes
X_SCALE = "log"        # "log" or "linear"
Y_SCALE = "log"        # "log" or "linear"

# WAXS axes (defaults like your nice code)
WAXS_X_SCALE = "linear"
WAXS_Y_SCALE = "linear"

# Output
SAVE_IMAGE = False
OUT_DIR = Path("./plots_out")
DPI_SCALE = 2

# Remove extra whitespace
TIGHTEN = True
XPAD = 0.02
YPAD = 0.05

# Peak markers (optional)
SHOW_PEAKS = True
SAXS_PROM_FRAC = 2e-7
WAXS_PROM_FRAC = 0.02


# ==========================================================
# NICE STYLE (your favorite look)
# ==========================================================
def style_common(fig: go.Figure) -> None:
    fig.update_layout(
        xaxis_title=dict(
            text="q (Å⁻¹)",
            font=dict(size=34, family="Arial Black", color="black"),
        ),
        yaxis_title=dict(
            text="Intensity (a.u.)",
            font=dict(size=34, family="Arial Black", color="black"),
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        template="plotly_white",
        width=900,
        height=600,
        margin=dict(l=80, r=40, t=40, b=80),
        showlegend=True,
        legend=dict(
            x=0.65,
            y=0.95,
            bgcolor="rgba(0, 0, 0, 0)",
            font=dict(size=24, family="Arial Black", color="black"),
        ),
        title=None,
    )


def finalize_axes(fig: go.Figure, xscale: str, yscale: str, hide_y_ticklabels: bool = False) -> None:
    fig.update_xaxes(
        type=("log" if xscale == "log" else "linear"),
        showline=True,
        mirror=True,
        linewidth=3,
        linecolor="black",
        tickfont=dict(size=36, family="Arial Black", color="black"),
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(
        type=("log" if yscale == "log" else "linear"),
        showline=True,
        mirror=True,
        linewidth=3,
        linecolor="black",
        tickfont=dict(size=36, family="Arial Black", color="black"),
        showgrid=False,
        zeroline=False,
        showticklabels=(not hide_y_ticklabels),
    )


def tighten_view(fig: go.Figure, x, y, xscale="log", yscale="log", xpad=None, ypad=None) -> None:
    x = np.asarray(x)
    y = np.asarray(y)

    m = np.isfinite(x) & np.isfinite(y)
    if xscale == "log":
        m &= x > 0
    if yscale == "log":
        m &= y > 0

    x = x[m]
    y = y[m]
    if x.size == 0:
        return

    # If None -> no padding
    xpad = 0.0 if xpad is None else float(xpad)
    ypad = 0.0 if ypad is None else float(ypad)

    if xscale == "log":
        lx0, lx1 = np.log10(x.min()), np.log10(x.max())
        dx = lx1 - lx0 if lx1 > lx0 else 1.0
        fig.update_xaxes(range=[lx0 - xpad * dx, lx1 + xpad * dx])
    else:
        x0, x1 = x.min(), x.max()
        dx = x1 - x0 if x1 > x0 else 1.0
        fig.update_xaxes(range=[x0 - xpad * dx, x1 + xpad * dx])

    if yscale == "log":
        ly0, ly1 = np.log10(y.min()), np.log10(y.max())
        dy = ly1 - ly0 if ly1 > ly0 else 1.0
        fig.update_yaxes(range=[ly0 - ypad * dy, ly1 + ypad * dy])
    else:
        y0, y1 = y.min(), y.max()
        dy = y1 - y0 if y1 > y0 else 1.0
        fig.update_yaxes(range=[y0 - ypad * dy, y1 + ypad * dy])


# ==========================================================
# LOADER: col0=q shared, then (I, dI) pairs
# ==========================================================
def load_excel_series(file: Path) -> tuple[list[str], list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    df = pd.read_excel(file, header=None)
    arr = df.to_numpy()

    q_all = pd.to_numeric(pd.Series(arr[1:, 0]), errors="coerce").to_numpy()

    names: list[str] = []
    series: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    ncols = arr.shape[1]
    for icol in range(1, ncols, 2):     # I columns: 1,3,5,...
        dicol = icol + 1                # dI columns: 2,4,6,...
        if dicol >= ncols:
            break

        name = arr[0, icol]
        if pd.isna(name) or str(name).strip() == "":
            name = f"sample_{len(names)+1}"
        name = str(name).strip()

        I = pd.to_numeric(pd.Series(arr[1:, icol]), errors="coerce").to_numpy()
        dI = pd.to_numeric(pd.Series(arr[1:, dicol]), errors="coerce").to_numpy()

        m = np.isfinite(q_all) & np.isfinite(I)
        q = q_all[m]
        I = I[m]
        dI = dI[m]
        dI = np.where(np.isfinite(dI), dI, np.nan)

        if q.size:
            # IMPORTANT: sort by q so lines don't "tilt"
            order = np.argsort(q)
            q, I, dI = q[order], I[order], dI[order]

            names.append(name)
            series.append((q, I, dI))

    if not series:
        raise ValueError("No series found. Check Excel format.")
    return names, series


def pick_sample(names, series, sample: str | int):
    if isinstance(sample, int):
        idx = sample
        if idx < 0 or idx >= len(names):
            raise IndexError(f"Sample index {idx} out of range (0..{len(names)-1}).")
    else:
        if sample in names:
            idx = names.index(sample)
        else:
            lower_map = {n.lower(): i for i, n in enumerate(names)}
            key = str(sample).strip().lower()
            if key not in lower_map:
                raise ValueError("Sample not found. Available:\n" + "\n".join(names))
            idx = lower_map[key]
    return names[idx], *series[idx]


# ==========================================================
# Optional peak markers (same vibe as your code)
# ==========================================================
def peak_diagnostics_q(q, y, prominence_frac: float):
    if q.size == 0:
        return []

    # use d-space for width calculation like your code
    d = 2 * np.pi / q
    idx = np.argsort(d)
    d = d[idx]
    y = y[idx]

    prom_abs = prominence_frac * np.max(y)
    peaks, props = find_peaks(y, prominence=prom_abs)
    if len(peaks) == 0:
        return []

    widths = peak_widths(y, peaks, rel_height=0.5)
    dd = np.mean(np.diff(d))
    fwhm_d = widths[0] * dd

    out = []
    for i, p in enumerate(peaks):
        d_peak = float(d[p])
        q_peak = float(2 * np.pi / d_peak)
        out.append(
            dict(
                d_peak=d_peak,
                q_peak=q_peak,
                I_peak=float(y[p]),
                prominence=float(props["prominences"][i]),
                FWHM_d=float(fwhm_d[i]),
            )
        )
    return out


# ==========================================================
# PLOT
# ==========================================================
def plot_one_sample_split() -> None:
    names, series = load_excel_series(FILE)
    sample_name, q, I, dI = pick_sample(names, series, SAMPLE)

    qmin = float(np.nanmin(q)) if X_MIN is None else float(X_MIN)
    qmax = float(np.nanmax(q)) if X_MAX is None else float(X_MAX)

    m = (q >= qmin) & (q <= qmax)
    q, I, dI = q[m], I[m], dI[m]

    # split
    msaxs = q <= X_SPLIT
    mwaxs = q >= X_SPLIT

    q_saxs, I_saxs, dI_saxs = q[msaxs], I[msaxs], dI[msaxs]
    q_waxs, I_waxs, dI_waxs = q[mwaxs], I[mwaxs], dI[mwaxs]

    # optional peaks
    saxs_peaks = peak_diagnostics_q(q_saxs, I_saxs, SAXS_PROM_FRAC) if SHOW_PEAKS else []
    waxs_peaks = peak_diagnostics_q(q_waxs, I_waxs, WAXS_PROM_FRAC) if SHOW_PEAKS else []

    # figures
    fig_saxs = go.Figure()
    fig_waxs = go.Figure()

    has_err_saxs = bool(np.any(np.isfinite(dI_saxs)))
    has_err_waxs = bool(np.any(np.isfinite(dI_waxs)))

    if q_saxs.size:
        fig_saxs.add_trace(
            go.Scatter(
                x=q_saxs,
                y=I_saxs,
                mode="lines",
                name=f"{sample_name}_SAXS",
                line=dict(width=4, color="#0072B2"),
                error_y=dict(type="data", array=dI_saxs, visible=has_err_saxs),
            )
        )

    if q_waxs.size:
        fig_waxs.add_trace(
            go.Scatter(
                x=q_waxs,
                y=I_waxs,
                mode="lines",
                name=f"{sample_name}_WAXS",
                line=dict(width=4, color="#D55E00"),
                error_y=dict(type="data", array=dI_waxs, visible=has_err_waxs),
            )
        )

    # peak markers (triangle-down)
    if SHOW_PEAKS:
        for p in saxs_peaks:
            fig_saxs.add_trace(
                go.Scatter(
                    x=[p["q_peak"]],
                    y=[p["I_peak"]],
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=14, color="black"),
                    showlegend=False,
                )
            )
        for p in waxs_peaks:
            fig_waxs.add_trace(
                go.Scatter(
                    x=[p["q_peak"]],
                    y=[p["I_peak"]],
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=14, color="black"),
                    showlegend=False,
                )
            )

    # style + axes
    style_common(fig_saxs)
    finalize_axes(fig_saxs, X_SCALE, Y_SCALE)
    if TIGHTEN:
        tighten_view(fig_saxs, q_saxs, I_saxs, xscale=X_SCALE, yscale=Y_SCALE, xpad=None, ypad=None)

    style_common(fig_waxs)
    finalize_axes(fig_waxs, WAXS_X_SCALE, WAXS_Y_SCALE)
    if TIGHTEN:
        tighten_view(fig_waxs, q_waxs, I_waxs, xscale=WAXS_X_SCALE, yscale=WAXS_Y_SCALE, xpad=None, ypad=None)
        
    fig_saxs.show()
    fig_waxs.show()

    if SAVE_IMAGE:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        png1 = OUT_DIR / f"{sample_name}_SAXS_{qmin:g}_to_{min(X_SPLIT, qmax):g}.png"
        png2 = OUT_DIR / f"{sample_name}_WAXS_{max(X_SPLIT, qmin):g}_to_{qmax:g}.png"
        fig_saxs.write_image(str(png1), scale=DPI_SCALE)
        fig_waxs.write_image(str(png2), scale=DPI_SCALE)
        print("Saved:")
        print(png1)
        print(png2)


# ==========================================================
# JUPYTER / SCRIPT ENTRY
# ==========================================================
if __name__ == "__main__":
    plot_one_sample_split()
