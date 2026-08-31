from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _build_peak_labels(protein_name: str, rt: np.ndarray) -> np.ndarray:
    n = rt.size
    labels = np.empty(n, dtype=object)
    order = np.argsort(rt)
    for rank, idx in enumerate(order, start=1):
        labels[idx] = f"{rank:03d}"
    return labels


def _plot_same_diff_judgement(
    rt1: np.ndarray,
    rt2: np.ndarray,
    protein1_name: str,
    protein2_name: str,
    labels1: np.ndarray,
    labels2: np.ndarray,
    same_mask1: np.ndarray,
    same_mask2: np.ndarray,
    same_pairs: list[tuple[int, int]],
    final_percent: float,
    save_path: str,
) -> None:
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 20
    plt.rcParams["font.weight"] = "bold"

    same_color = "#5B84B1"
    diff_color = "#C27D52"
    line_color = "#9DB3C1"
    text_color = "#2F3E46"

    fig, ax = plt.subplots(figsize=(20, 10))

    y_top = 0.64
    y_bottom = 0.36
    y1 = np.full(rt1.size, y_top)
    y2 = np.full(rt2.size, y_bottom)

    c1 = np.where(same_mask1, same_color, diff_color)
    c2 = np.where(same_mask2, same_color, diff_color)
    ax.scatter(rt1, y1, c=c1, s=200, edgecolors="white", linewidths=0.9, zorder=3)
    ax.scatter(rt2, y2, c=c2, s=200, edgecolors="white", linewidths=0.9, zorder=3)

    max_lines = 300
    for k, (i, j) in enumerate(same_pairs):
        if k >= max_lines:
            break
        ax.plot([rt1[i], rt2[j]], [y_top, y_bottom], color=line_color, linewidth=0.8, alpha=0.6, zorder=1)

    x_min = float(min(np.min(rt1), np.min(rt2))) if (rt1.size and rt2.size) else 0.0
    x_max = float(max(np.max(rt1), np.max(rt2))) if (rt1.size and rt2.size) else 1.0
    x_span = max(1e-6, x_max - x_min)
    min_dx = 0.010 * x_span

    def _spread_x(xs: np.ndarray, gap: float) -> np.ndarray:
        if xs.size == 0:
            return xs
        order = np.argsort(xs)
        arr = xs[order].astype(float).copy()
        for m in range(1, arr.size):
            if arr[m] - arr[m - 1] < gap:
                arr[m] = arr[m - 1] + gap
        over = arr[-1] - x_max
        if over > 0:
            arr -= over
        for m in range(arr.size - 2, -1, -1):
            if arr[m + 1] - arr[m] < gap:
                arr[m] = arr[m + 1] - gap
        if arr[0] < x_min:
            arr += (x_min - arr[0])
        out = np.empty_like(arr)
        out[np.argsort(order)] = arr
        return out

    same_total = int(np.sum(same_mask1) + np.sum(same_mask2))
    diff_total = int((rt1.size - np.sum(same_mask1)) + (rt2.size - np.sum(same_mask2)))
    label_same = same_total < diff_total

    idx1_to_label = np.where(same_mask1)[0] if label_same else np.where(~same_mask1)[0]
    idx2_to_label = np.where(same_mask2)[0] if label_same else np.where(~same_mask2)[0]
    x1_adj = _spread_x(rt1[idx1_to_label], min_dx)
    x2_adj = _spread_x(rt2[idx2_to_label], min_dx)

    for m, i in enumerate(idx1_to_label.tolist()):
        up = (m % 2 == 0)
        y_lab = y_top + (0.042 if up else -0.042)
        va = "bottom" if up else "top"
        ax.plot([rt1[i], x1_adj[m]], [y_top, y_lab], color="#C7CDD2", linewidth=0.6, alpha=0.9, zorder=2)
        ax.text(
            x1_adj[m],
            y_lab,
            str(labels1[i]),
            fontsize=20,
            ha="center",
            va=va,
            color=text_color,
            bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.88),
            clip_on=False,
            zorder=4,
        )

    for m, j in enumerate(idx2_to_label.tolist()):
        up = (m % 2 == 1)
        y_lab = y_bottom + (0.042 if up else -0.042)
        va = "bottom" if up else "top"
        ax.plot([rt2[j], x2_adj[m]], [y_bottom, y_lab], color="#C7CDD2", linewidth=0.6, alpha=0.9, zorder=2)
        ax.text(
            x2_adj[m],
            y_lab,
            str(labels2[j]),
            fontsize=20,
            ha="center",
            va=va,
            color=text_color,
            bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.88),
            clip_on=False,
            zorder=4,
        )

    ax.set_yticks([y_top, y_bottom])
    ax.set_yticklabels([protein1_name, protein2_name], fontsize=20, rotation=90, va='center')
    ax.set_xlabel("Retention Time (min)", fontsize=20, fontweight="bold")
    ax.tick_params(axis="x", labelsize=20)
    ax.set_ylim(0.26, 0.74)
    ax.grid(axis="x", color="#D8DEE3", linestyle="--", linewidth=0.9, alpha=0.9)
    ax.set_axisbelow(True)

    diff_p1_count = int(rt1.size - np.sum(same_mask1))
    diff_p2_count = int(rt2.size - np.sum(same_mask2))

    fig.subplots_adjust(right=0.78)
    panel_x, panel_y, panel_w, panel_h = 0.63, 0.38, 0.13, 0.23
    panel = fig.add_axes([panel_x, panel_y, panel_w, panel_h])
    panel.set_facecolor("#F7F7F5")
    for spine in panel.spines.values():
        spine.set_edgecolor("#C9CED3")
        spine.set_linewidth(0.9)
    panel.set_xticks([])
    panel.set_yticks([])
    panel.set_xlim(0, 1)
    panel.set_ylim(0, 1)

    panel.scatter([0.10], [0.87], s=120, c=[same_color], edgecolors="white", linewidths=0.8, zorder=3)
    panel.text(0.18, 0.86, "SAME", ha="left", va="center", fontsize=17, fontweight="bold", color=text_color)
    panel.scatter([0.10], [0.75], s=120, c=[diff_color], edgecolors="white", linewidths=0.8, zorder=3)
    panel.text(0.18, 0.74, "DIFF", ha="left", va="center", fontsize=17, fontweight="bold", color=text_color)

    panel.text(0.08, 0.57, f"Similarity: {final_percent:.2f}%", ha="left", va="center", fontsize=17.5, fontweight="bold", color=text_color)
    panel.text(0.08, 0.4, "Number of DIFF:", ha="left", va="center", fontsize=17.5, fontweight="bold", color=text_color)
    panel.text(0.12, 0.26, f"{protein1_name}: {diff_p1_count}", ha="left", va="center", fontsize=17, color=text_color)
    panel.text(0.12, 0.12, f"{protein2_name}: {diff_p2_count}", ha="left", va="center", fontsize=17, color=text_color)

    fig.savefig(save_path, dpi=800, bbox_inches="tight")
    plt.close(fig)


def _plot_high_similarity_ratio_comparison(
    rt1: np.ndarray,
    rt2: np.ndarray,
    ratio1: np.ndarray,
    ratio2: np.ndarray,
    protein1_name: str,
    protein2_name: str,
    labels1: np.ndarray,
    labels2: np.ndarray,
    same_pairs: list[tuple[int, int]],
    final_percent: float,
    ratio_diff_threshold: float,
    save_path: str,
) -> int:
    if ratio_diff_threshold < 0:
        raise ValueError("--significant-ratio-diff must be >= 0")

    unique_pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for i, j in same_pairs:
        key = (int(i), int(j))
        if key not in seen:
            seen.add(key)
            unique_pairs.append(key)

    if not unique_pairs:
        return 0

    order = np.argsort([min(float(rt1[i]), float(rt2[j])) for i, j in unique_pairs])
    pairs = [unique_pairs[int(k)] for k in order]
    x = np.arange(len(pairs))
    r1 = np.asarray([ratio1[i] for i, _ in pairs], dtype=float)
    r2 = np.asarray([ratio2[j] for _, j in pairs], dtype=float)
    diffs = np.abs(r2 - r1)
    sig = diffs >= ratio_diff_threshold
    sig_count = int(np.sum(sig))

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 20
    plt.rcParams["font.weight"] = "bold"
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(max(12, 0.34 * len(pairs)), 9),
        gridspec_kw={"height_ratios": [2.2, 1.2]},
        sharex=True,
    )

    width = 0.38
    normal1 = "#5B84B1"
    normal2 = "#7D9D6C"
    sig_color = "#D62728"
    edge_colors = [sig_color if s else "white" for s in sig]
    line_widths = [2.2 if s else 0.6 for s in sig]

    ax1.bar(x - width / 2, r1, width, label=protein1_name, color=normal1, alpha=0.88, edgecolor=edge_colors, linewidth=line_widths)
    ax1.bar(x + width / 2, r2, width, label=protein2_name, color=normal2, alpha=0.88, edgecolor=edge_colors, linewidth=line_widths)
    for idx, (i, j) in enumerate(pairs):
        if sig[idx]:
            y = max(r1[idx], r2[idx])
            ax1.scatter([idx], [y * 1.04 if y > 0 else 0.02], s=150, marker="*", color=sig_color, zorder=5)
            ax1.text(idx, y * 1.10 if y > 0 else 0.05, f"Δ={diffs[idx]:.2f}", ha="center", va="bottom",
                     fontsize=18, color=sig_color, fontweight="bold", rotation=45)

    ax1.set_ylabel("Normalized ratio (%)", fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(axis="y", alpha=0.25, linestyle="--")

    colors = np.where(sig, sig_color, "#9DB3C1")
    ax2.bar(x, diffs, color=colors, alpha=0.90)
    ax2.axhline(ratio_diff_threshold, color=sig_color, linestyle="--", linewidth=2.5,
                label=f"significant threshold = {ratio_diff_threshold:g}")
    ax2.set_ylabel("|Δratio|", fontweight="bold")
    ax2.set_xlabel("Matched SAME peaks ordered by RT", fontweight="bold")
    ax2.grid(axis="y", alpha=0.25, linestyle="--")
    ax2.legend(loc="upper right")

    tick_labels = [f"{labels1[i]}/{labels2[j]}" for i, j in pairs]
    max_ticks = 80
    if len(tick_labels) > max_ticks:
        step = int(np.ceil(len(tick_labels) / max_ticks))
        show = np.arange(0, len(tick_labels), step)
        ax2.set_xticks(show)
        ax2.set_xticklabels([tick_labels[k] for k in show], rotation=90, fontsize=13)
    else:
        ax2.set_xticks(x)
        ax2.set_xticklabels(tick_labels, rotation=90, fontsize=13)

    info = (
        f"Significant matched peaks: {sig_count}/{len(pairs)}\n"
        f"Criterion: |ratio_sample - ratio_standard| >= {ratio_diff_threshold:g}"
    )
    ax1.text(0.012, 0.965, info, transform=ax1.transAxes, ha="left", va="top", fontsize=18,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       edgecolor="#D62728" if sig_count else "#C9CED3", alpha=0.92))

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return sig_count


def _read_table(path: str, sheet_name: str = "Sheet1") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    suffix = p.suffix.lower()
    if suffix in [".csv", ".tsv", ".txt"]:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(p, sep=sep)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(p, sheet_name=sheet_name)

    raise ValueError(f"Unsupported file type: {suffix}. Please use csv/tsv/xlsx.")


def _extract_rt_ratio(
    df: pd.DataFrame, rt_col: str, ratio_col: str, dropna: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    if rt_col not in df.columns:
        raise ValueError(f"rt_col '{rt_col}' not found. Available: {list(df.columns)}")
    if ratio_col not in df.columns:
        raise ValueError(f"ratio_col '{ratio_col}' not found. Available: {list(df.columns)}")

    out = df[[rt_col, ratio_col]].copy()
    out[rt_col] = pd.to_numeric(out[rt_col], errors="coerce")
    out[ratio_col] = pd.to_numeric(out[ratio_col], errors="coerce")
    if dropna:
        out = out.dropna(subset=[rt_col, ratio_col])

    return out[rt_col].to_numpy(dtype=float), out[ratio_col].to_numpy(dtype=float)


def _filter_and_normalize_ratios(
    rt: np.ndarray,
    ratio: np.ndarray,
    min_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if min_ratio < 0:
        raise ValueError("--min-ratio must be >= 0")

    keep_mask = ratio >= min_ratio
    rt_clean = rt[keep_mask]
    ratio_clean = ratio[keep_mask]

    if ratio_clean.size == 0:
        return rt_clean, ratio_clean

    total = float(np.sum(ratio_clean))
    if total <= 0:
        raise ValueError("Remaining ratio values must sum to a positive number after filtering.")

    ratio_normalized = ratio_clean / total * 100.0
    return rt_clean, ratio_normalized


def _write_diff_peaks_txt(
    save_path: str,
    protein1_name: str,
    protein2_name: str,
    labels1: np.ndarray,
    labels2: np.ndarray,
    rt1: np.ndarray,
    rt2: np.ndarray,
    ratio1: np.ndarray,
    ratio2: np.ndarray,
    same_mask1: np.ndarray,
    same_mask2: np.ndarray,
) -> None:
    lines: list[str] = []
    lines.append(f"{protein1_name} DIFF peaks")
    lines.append("peak_id\trt\tratio")
    for i in np.where(~same_mask1)[0].tolist():
        lines.append(f"{labels1[i]}\t{rt1[i]:.6f}\t{ratio1[i]:.6f}")

    lines.append("")
    lines.append(f"{protein2_name} DIFF peaks")
    lines.append("peak_id\trt\tratio")
    for j in np.where(~same_mask2)[0].tolist():
        lines.append(f"{labels2[j]}\t{rt2[j]:.6f}\t{ratio2[j]:.6f}")

    Path(save_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _match_same_pairs_hungarian(
    rt1: np.ndarray,
    rt2: np.ndarray,
    delta_min: float,
    big_m: float = 1e6,
) -> Tuple[np.ndarray, np.ndarray]:
    n1 = rt1.size
    n2 = rt2.size
    if n1 == 0 or n2 == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    try:
        from scipy.optimize import linear_sum_assignment
    except Exception as exc:
        raise ImportError(
            "SciPy is required. Please install: pip install scipy"
        ) from exc

    diff = np.abs(rt1[:, None] - rt2[None, :])
    allowed = diff <= delta_min

    n = max(n1, n2)
    cost = np.zeros((n, n), dtype=float)
    unmatched_penalty = float(delta_min + 1.0)
    cost[:n1, n2:] = unmatched_penalty
    cost[n1:, :n2] = unmatched_penalty
    cost[:n1, :n2] = np.where(allowed, diff, big_m)

    row_ind, col_ind = linear_sum_assignment(cost)

    out_i: list[int] = []
    out_j: list[int] = []
    for r, c in zip(row_ind.tolist(), col_ind.tolist()):
        if r < n1 and c < n2 and allowed[r, c]:
            out_i.append(r)
            out_j.append(c)

    return np.asarray(out_i, dtype=int), np.asarray(out_j, dtype=int)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare two proteins by peak RT and ratio using the Hungarian algorithm."
    )
    p.add_argument("protein1_file", help="CSV/TSV/XLSX of protein1 peaks (rt + ratio).")
    p.add_argument("protein2_file", help="CSV/TSV/XLSX of protein2 peaks (rt + ratio).")
    p.add_argument("--sheet", default="Sheet1", help="Excel sheet name (for xlsx/xls). Default: Sheet1.")
    p.add_argument("--rt-col", default="rt", help="Column name for retention time (minutes). Default: rt")
    p.add_argument("--ratio-col", default="ratio", help="Column name for peak area proportion. Default: ratio")
    p.add_argument("--delta-min", type=float, required=True, help="Retention-time tolerance (min)")
    p.add_argument("--print-details", action="store_true", help="Print pair-level details for all matched SAME pairs.")
    p.add_argument("--save-plot", default="same_diff_judgement.png", help="Output path for SAME/DIFF visualization figure.")
    p.add_argument("--save-diff-txt", default="diff_peaks.txt", help="Output path for DIFF peak summary text file.")
    p.add_argument(
        "--high-similarity-threshold",
        type=float,
        default=80.0,
        help="Generate significant ratio-difference plot only when similarity is above this threshold. Default: 80.0",
    )
    p.add_argument(
        "--significant-ratio-diff",
        type=float,
        default=None,
        help="Ratio difference threshold for significant matched peaks. If omitted, the extra plot is disabled.",
    )
    p.add_argument(
        "--save-significant-plot",
        default="significant_ratio_comparison.png",
        help="Output path for high-similarity significant ratio-difference figure.",
    )
    p.add_argument(
        "--min-ratio",
        type=float,
        default=0.0,
        help="Remove peaks with ratio below this threshold, then renormalize to sum to 100. Default: 0.0",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    df1 = _read_table(args.protein1_file, sheet_name=args.sheet)
    df2 = _read_table(args.protein2_file, sheet_name=args.sheet)

    rt1, ratio1 = _extract_rt_ratio(df1, args.rt_col, args.ratio_col)
    rt2, ratio2 = _extract_rt_ratio(df2, args.rt_col, args.ratio_col)

    original_n1 = rt1.size
    original_n2 = rt2.size
    rt1, ratio1 = _filter_and_normalize_ratios(rt1, ratio1, args.min_ratio)
    rt2, ratio2 = _filter_and_normalize_ratios(rt2, ratio2, args.min_ratio)

    print(f"min_ratio = {args.min_ratio}")
    print(f"protein1 preprocessing: kept {rt1.size}/{original_n1} peaks, normalized_ratio_sum = {np.sum(ratio1):.6f}")
    print(f"protein2 preprocessing: kept {rt2.size}/{original_n2} peaks, normalized_ratio_sum = {np.sum(ratio2):.6f}")

    protein1_name = Path(args.protein1_file).stem
    protein2_name = Path(args.protein2_file).stem
    labels1 = _build_peak_labels(protein1_name, rt1)
    labels2 = _build_peak_labels(protein2_name, rt2)

    i_idx, j_idx = _match_same_pairs_hungarian(rt1=rt1, rt2=rt2, delta_min=args.delta_min)
    same_contrib = float(np.sum(np.abs(ratio1[i_idx] - ratio2[j_idx]))) if i_idx.size else 0.0
    n_same = int(i_idx.size)

    used1 = np.zeros(rt1.size, dtype=bool)
    used2 = np.zeros(rt2.size, dtype=bool)
    used1[i_idx] = True
    used2[j_idx] = True
    unmatched = float(np.sum(ratio1[~used1]) + np.sum(ratio2[~used2]))

    result = same_contrib + unmatched
    final = 100.0 - result

    print(f"delta_min = {args.delta_min}")
    print(f"hungarian_same_pairs = {n_same}")
    print(f"unmatched_ratio_sum = {unmatched:.6f}")
    print(f"result = {result:.6f}")
    print(f"final = {final:.6f}")

    same_mask1 = used1.copy()
    same_mask2 = used2.copy()
    same_pairs_plot: list[tuple[int, int]] = [
        (int(i), int(j)) for i, j in zip(i_idx.tolist(), j_idx.tolist())
    ]

    if args.print_details:
        for k in range(i_idx.size):
            i = int(i_idx[k])
            j = int(j_idx[k])
            diff = abs(rt1[i] - rt2[j])
            contrib = abs(ratio1[i] - ratio2[j])
            print(
                f"[{k}] SAME diff={diff:.6f} "
                f"rt1={rt1[i]:.6f} rt2={rt2[j]:.6f} "
                f"ratio1={ratio1[i]:.6f} ratio2={ratio2[j]:.6f} contrib={contrib:.6f}"
            )

    _write_diff_peaks_txt(
        save_path=args.save_diff_txt,
        protein1_name=protein1_name,
        protein2_name=protein2_name,
        labels1=labels1,
        labels2=labels2,
        rt1=rt1,
        rt2=rt2,
        ratio1=ratio1,
        ratio2=ratio2,
        same_mask1=same_mask1,
        same_mask2=same_mask2,
    )

    _plot_same_diff_judgement(
        rt1=rt1,
        rt2=rt2,
        protein1_name=protein1_name,
        protein2_name=protein2_name,
        labels1=labels1,
        labels2=labels2,
        same_mask1=same_mask1,
        same_mask2=same_mask2,
        same_pairs=same_pairs_plot,
        final_percent=final,
        save_path=args.save_plot,
    )
    print(f"saved_diff_txt = {args.save_diff_txt}")
    print(f"saved_plot = {args.save_plot}")

    if args.significant_ratio_diff is not None:
        if final >= args.high_similarity_threshold:
            sig_count = _plot_high_similarity_ratio_comparison(
                rt1=rt1,
                rt2=rt2,
                ratio1=ratio1,
                ratio2=ratio2,
                protein1_name=protein1_name,
                protein2_name=protein2_name,
                labels1=labels1,
                labels2=labels2,
                same_pairs=same_pairs_plot,
                final_percent=final,
                ratio_diff_threshold=args.significant_ratio_diff,
                save_path=args.save_significant_plot,
            )
            print(f"significant_ratio_diff_threshold = {args.significant_ratio_diff}")
            print(f"significant_ratio_peak_count = {sig_count}")
            print(f"saved_significant_plot = {args.save_significant_plot}")
        else:
            print(
                f"skip_significant_plot = similarity {final:.6f} < "
                f"high_similarity_threshold {args.high_similarity_threshold:.6f}"
            )


if __name__ == "__main__":
    main()
