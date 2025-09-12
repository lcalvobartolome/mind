from __future__ import annotations

import ast
import argparse
import math
from os import listdir
import warnings
from typing import Iterable, List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats

try:
    from statsmodels.stats.multitest import multipletests  # type: ignore
    has_statsmodels = True
except Exception:
    has_statsmodels = False

METHOD_MAPPING: Dict[str, str] = {
    "1": "ENN",
    "2": "ANN",
    "3_weighted": "TB-ENN-W",
    "3_unweighted": "TB-ENN",
    "4_weighted": "TB-ANN-W",
    "4_unweighted": "TB-ANN",
    "time_1": "ENN",
    "time_2": "ANN",
    "time_3_weighted": "TB-ENN-W",
    "time_3_unweighted": "TB-ENN",
    "time_4_weighted": "TB-ANN-W",
    "time_4_unweighted": "TB-ANN",
}

def _safe_eval_listlike(x) -> List:
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, str):
        try:
            x = ast.literal_eval(x)
        except Exception:
            return []
    return x if isinstance(x, list) else []


def extract_doc_ids(cell) -> List[str]:
    data = _safe_eval_listlike(cell)
    if not data:
        return []
    try:
        return [el["doc_id"] for el in data[0]]
    except Exception:
        try:
            flat = [
                {"doc_id": entry["doc_id"], "score": entry.get("score")}
                for sub in data
                for entry in (sub if isinstance(sub, list) else [])
                if isinstance(entry, dict) and "doc_id" in entry
            ]
            return [el["doc_id"] for el in flat]
        except Exception:
            return []

def precision_at_k(doc_ids: List[str], relevant_docs: Iterable[str], k: int) -> float:
    """
    Precision@k: proportion of top-k retrieved that are relevant.
    """
    if k <= 0:
        return 0.0
    rel = set(relevant_docs)
    retrieved = doc_ids[:k]
    tp = sum(1 for d in retrieved if d in rel)
    return tp / k                                

def recall_at_k(doc_ids: List[str], relevant_docs: Iterable[str], k: int) -> float:
    """
    Recall@k: proportion of relevant documents that are in the top-k retrieved.
    """
    rel = set(relevant_docs)
    if not rel:
        return 0.0
    return len(set(doc_ids[:k]) & rel) / len(rel)

def mrr_multi_at_k(doc_ids: List[str], relevant_docs: Iterable[str], k: int) -> float:
    """
    Multiple Mean Reciprocal Rank at k (MRR@k): how close the average rank of all relevant items is to the best possible under top-k.
    It penalizes if any relevant item is missing/low in the top-k.
    """
    rel = set(relevant_docs)
    n = len(rel)
    if n == 0 or k <= 0:
        return 0.0

    retrieved = doc_ids[:k]
    first_pos = {}
    for i, d in enumerate(retrieved):
        if d not in first_pos:
            first_pos[d] = i + 1  # 1-based rank

    ranks = [first_pos[d] for d in rel if d in first_pos]
    R = len(ranks)

    observed_avg = (sum(ranks) + (k + 1) * (n - R)) / n

    R_star = min(n, k)  # best possible under top-k
    best_possible_avg = (R_star * (R_star + 1) / 2 + (k + 1) * (n - R_star)) / n

    return best_possible_avg / observed_avg

def ndcg_at_k(doc_ids: List[str], relevant_docs: Iterable[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at k (NDCG@k) with binary relevance: how well the method puts relevant items near the very top.
    
    It rewards if some relevant items are at the very top, even others are missing.
    """
    if k <= 0:
        return 0.0

    rel = set(relevant_docs)
    retrieved = doc_ids[:k]

    # DCG with binary relevance: we only sum the one retrieved that are relevant, and in the denominator we place (i+1)+1 to account for the enumerate starting at 0
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(retrieved) if d in rel)

    # IDCG: place min(#relevant, k) ones at the top
    R_star = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(R_star))

    return (dcg / idcg) if idcg > 0 else 0.0

def is_higher_better(metric_name: str) -> bool:
    return "Time (s)" not in metric_name

def metric_pretty_name(raw_metric_key: str) -> str:
    if raw_metric_key.startswith("mrr_"):
        return f"MRR@{raw_metric_key.split('_', 1)[1]}"
    if raw_metric_key.startswith("precision_"):
        return f"Precision@{raw_metric_key.split('_', 1)[1]}"
    if raw_metric_key.startswith("recall_"):
        return f"Recall@{raw_metric_key.split('_', 1)[1]}"
    if raw_metric_key.startswith("ndcg_"):
        return f"NDCG@{raw_metric_key.split('_', 1)[1]}"
    if raw_metric_key.startswith("time"):
        return "Time (s)"
    return raw_metric_key


def extract_metric_and_method(col: str) -> Tuple[str, str]:
    if col.startswith("time_"):
        method_key = col
        method = METHOD_MAPPING.get(method_key, method_key)
        return "Time (s)", method
    if "_results_" in col:
        metric_key, method_key = col.split("_results_", 1)
        return metric_pretty_name(metric_key), METHOD_MAPPING.get(method_key, method_key)
    return metric_pretty_name(col), col


def prepare_annotations(relevant_check_path: str, prepare=True) -> pd.DataFrame:
    if prepare:
        ann_path = relevant_check_path.replace(".parquet", "_all_added.parquet")
    else:
        ann_path = relevant_check_path
    df = pd.read_parquet(ann_path)
    keep = (
        (df["relevance_llama3.3:70b"] == 1)
        & (df["relevance_qwen:32b"] == 1)
       #  & (df["relevance_llama3.3:70b-instruct-q5_K_M"] == 1)
       & (df["relevance_llama3:70b-instruct"] == 1)
        & (df["relevance_llama3.1:8b-instruct-q8_0"] == 1)
    )
    df = df[keep].copy()
    grouped = (
        df.groupby("question")["all_results"].apply(list).reset_index().rename(columns={"all_results": "relevant_docs"})
    )
    grouped["num_relevant_docs"] = grouped["relevant_docs"].apply(len)
    return grouped


def compute_per_row_metrics(df: pd.DataFrame, key: str, ks: Iterable[int]) -> pd.DataFrame:
    doc_col = f"doc_ids_{key}"
    df[doc_col] = df[key].apply(extract_doc_ids)
    for k in ks:
        df[f"mrr_{k}_{key}"] = df.apply(lambda r: mrr_multi_at_k(r[doc_col], r["relevant_docs"], k), axis=1)
        df[f"precision_{k}_{key}"] = df.apply(lambda r: precision_at_k(r[doc_col], r["relevant_docs"], k), axis=1)
        df[f"recall_{k}_{key}"] = df.apply(lambda r: recall_at_k(r[doc_col], r["relevant_docs"], k), axis=1)
        df[f"ndcg_{k}_{key}"] = df.apply(lambda r: ndcg_at_k(r[doc_col], r["relevant_docs"], k), axis=1)
    return df

def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, ci: float = 0.95, random_state=None):
    """
    Compute the bootstrap confidence interval spread for the mean of a given dataset.

    Parameters:
        values (np.ndarray): Array of numerical values for which the confidence interval is calculated.
        n_boot (int, optional): Number of bootstrap samples to generate. Default is 1000.
        ci (float, optional): Confidence level for the interval, expressed as a proportion (e.g., 0.95 for 95%). Default is 0.95.
        random_state (int or np.random.Generator, optional): Seed or random number generator for reproducibility. Default is None.

    Returns:
        float: Half the width of the confidence interval (spread) around the mean. Returns 0.0 if the input array has fewer than 2 elements.
    """
    rng = np.random.default_rng(random_state)
    vals = np.asarray(values, dtype=float)
    n = vals.size
    if n < 2:
        return 0.0
    idx = rng.integers(0, n, size=(n_boot, n))  # resample query indices
    means = vals[idx].mean(axis=1)
    alpha = (1 - ci) / 2.0
    lower, upper = np.percentile(means, [100*alpha, 100*(1-alpha)])
    return float((upper - lower) / 2.0)

def summarize(df: pd.DataFrame, metric_cols: List[str]) -> List[Tuple[str, int, str, str, float, float]]:
    """
    Returns: (Group, FileNum, Method, Metric, mean, ci_half_width)
    Unweighted means; 95% bootstrap CI half-width over queries for ALL metrics (incl. Time).
    """
    results = []
    group = str(df["Group"].iat[0]) if "Group" in df.columns else "ALL"
    file_num = int(df["FileNum"].iat[0]) if "FileNum" in df.columns else 1

    for col in metric_cols:
        metric_name, method_display = extract_metric_and_method(col)
        values = df[col].dropna()
        if values.empty:
            continue

        vals = values.values.astype(float)
        mean_val = float(vals.mean())
        seed = (hash((group, file_num, method_display, metric_name)) & 0xFFFFFFFF)
        spread = bootstrap_ci(vals, n_boot=1000, ci=0.95, random_state=seed)

        results.append((group, file_num, method_display, metric_name, mean_val, spread))

    return results

def compute_significance_rm(
    df_results: pd.DataFrame,
    retrieval_metrics: List[str],
) -> Dict[str, Dict]:
    """
    Per-metric repeated-measures testing:
      1) Omnibus Friedman across methods on per-query scores.
      2) Targeted one-sided Wilcoxon signed-rank (Pratt), Holm-adjusted within metric.

      Hypotheses (A > B for effectiveness; A < B for Time):
        - TB-ANN   > ANN
        - TB-ANN-W > ANN
        - TB-ENN   > ENN
        - TB-ENN-W > ENN
        - TB-ANN-W > TB-ANN
        - TB-ENN-W > TB-ENN
    """
    # collect columns
    keep_cols = [c for c in df_results.columns if any(m in c for m in retrieval_metrics + ["time_"])]
    metric_cols = [c for c in keep_cols if ("results_" in c or c.startswith("time_"))]

    df_long = df_results.melt(
        id_vars=["question", "num_relevant_docs"],
        value_vars=metric_cols,
        var_name="Metric",
        value_name="Value",
    ).dropna(subset=["Value"])

    def _to_method(col: str) -> str:
        if col.startswith("time_"):
            return METHOD_MAPPING.get(col, col)
        if "_results_" in col:
            return METHOD_MAPPING.get(col.split("_results_", 1)[-1], col)
        return col

    df_long["Method"] = df_long["Metric"].apply(_to_method)
    df_long["MetricName"] = df_long["Metric"].apply(lambda x: extract_metric_and_method(x)[0])

    friedman_p: Dict[str, float] = {}
    directional_by_metric: Dict[str, Dict[Tuple[str, str], Dict[str, float]]] = {}

    # matched rank-biserial effect size
    def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
        d = a - b
        nz = d != 0
        if nz.sum() == 0:
            return 0.0
        ranks = stats.rankdata(np.abs(d[nz]))
        sgn = np.sign(d[nz])
        return float((sgn * ranks).sum() / (len(ranks) * (len(ranks) + 1) / 2.0))

    for m in sorted(df_long["MetricName"].unique()):
        sub = df_long[df_long["MetricName"] == m].copy()
        mat = sub.pivot_table(index="question", columns="Method", values="Value", aggfunc="mean").dropna(how="any")
        if mat.shape[0] < 2 or mat.shape[1] < 2:
            continue

        mat_use = mat.values

        # 1) Friedman omnibus
        try:
            stat, p = stats.friedmanchisquare(*[mat_use[:, j] for j in range(mat_use.shape[1])])
            friedman_p[m] = float(p)
        except Exception:
            continue

        # 2) Targeted Wilcoxon (one-sided), Holm within metric
        directional_pairs = [
            ("TB-ANN",   "ANN"),
            ("TB-ANN-W", "ANN"),
            ("TB-ENN",   "ENN"),
            ("TB-ENN-W", "ENN"),
            ("TB-ANN-W", "TB-ANN"),
            ("TB-ENN-W", "TB-ENN"),
        ]
        higher_better = (m != "Time (s)")
        alt = "greater" if higher_better else "less"

        dir_ps: List[float] = []
        dir_names: List[Tuple[str, str]] = []
        dir_es: List[float] = []
        dir_n: List[int] = []

        for a, b in directional_pairs:
            if a in mat.columns and b in mat.columns:
                a_vals = mat[a].values
                b_vals = mat[b].values
                if len(a_vals) >= 2:
                    try:
                        w = stats.wilcoxon(
                            a_vals, b_vals,
                            zero_method="pratt",
                            alternative=alt,
                            mode="approx"
                        )
                        dir_ps.append(float(w.pvalue))
                        dir_names.append((a, b))
                        dir_es.append(rank_biserial(a_vals, b_vals))
                        dir_n.append(len(a_vals))
                    except Exception:
                        pass

        dir_dict: Dict[Tuple[str, str], Dict[str, float]] = {}
        if dir_ps:
            if has_statsmodels:
                _, p_adj, _, _ = multipletests(dir_ps, method="holm")
            else:
                order = np.argsort(dir_ps)
                mtests = len(dir_ps)
                p_adj = [1.0] * mtests
                running_max = 0.0
                for rank, idx in enumerate(order):
                    adj = min(1.0, (mtests - rank) * dir_ps[idx])
                    running_max = max(running_max, adj)
                    p_adj[idx] = running_max
            for (a, b), p_corr, es, n_eff in zip(dir_names, p_adj, dir_es, dir_n):
                dir_dict[(a, b)] = {"p_adj": float(p_corr), "r_rb": float(es), "n": int(n_eff)}

        directional_by_metric[m] = dir_dict

    return {
        "Friedman p-value": friedman_p,
        "Directional Wilcoxon-Holm": directional_by_metric,
    }


def format_value_with_sig(mean: float,
                          spread: float,
                          metric: str,
                          method: str,
                          best: bool,
                          sig_block: Dict[str, Dict]) -> str:
    """
    Put daggers based on one-sided Wilcoxon-Holm adjusted p-values.

    Markers:
      † : TB-* > baseline (ANN or ENN) when p_adj < 0.05
      ‡ : weighted TB > unweighted TB (TB-*-W > TB-*) when p_adj < 0.05

    No markers for Time (s).
    """
    val = f"{mean:.3f} \\pm {spread:.3f}"
    marks: List[str] = []

    if metric != "Time (s)":
        dir_pack = (sig_block or {}).get("Directional Wilcoxon-Holm", {})
        dir_for_metric = dir_pack.get(metric, {}) if isinstance(dir_pack, dict) else {}

        def dir_p(a: str, b: str) -> Optional[float]:
            entry = dir_for_metric.get((a, b))
            if entry is None:
                return None
            if isinstance(entry, dict):
                return entry.get("p_adj")
            # backward-compat if entry was a float
            return float(entry)

        # †: TB vs baseline
        if method.startswith("TB-ANN"):
            p = dir_p(method, "ANN")
            if p is not None and p < 0.05:
                marks.append("\\dagger")
        if method.startswith("TB-ENN"):
            p = dir_p(method, "ENN")
            if p is not None and p < 0.05:
                marks.append("\\dagger")

        # ‡: weighted vs unweighted TB
        if method == "TB-ANN-W":
            p = dir_p("TB-ANN-W", "TB-ANN")
            if p is not None and p < 0.05:
                marks.append("\\ddagger")
        if method == "TB-ENN-W":
            p = dir_p("TB-ENN-W", "TB-ENN")
            if p is not None and p < 0.05:
                marks.append("\\ddagger")

    sup = "".join(marks)
    formatted = rf"{val}^{{{sup}}}" if sup else val
    return rf"$\boldsymbol{{{formatted}}}$" if best else rf"${formatted}$"


def to_latex_table(
    df_summary: pd.DataFrame,
    significance_by_gf: Dict[Tuple[str, int], Dict[str, Dict]]
) -> str:
    dfw = df_summary.copy()

    # mark best per (Group, FileNum, Metric)
    best_mask = np.zeros(len(dfw), dtype=bool)
    for (grp, fnum, metric), grp_df in dfw.groupby(["Group", "FileNum", "Metric"], dropna=False):
        target = grp_df["Mean"].max() if is_higher_better(metric) else grp_df["Mean"].min()
        best_mask[grp_df.index] = grp_df["Mean"].eq(target).values
    dfw["IsBest"] = best_mask

    def _fmt_row(r):
        sig_block = significance_by_gf.get((r["Group"], int(r["FileNum"])), {})
        return format_value_with_sig(
            r["Mean"], r["CI"], r["Metric"], r["Method"], bool(r["IsBest"]), sig_block
        )
    dfw["Mean ± CI"] = dfw.apply(_fmt_row, axis=1)

    # Pivot: rows = (Group, FileNum, Method), cols = Metric
    pivot = (
        dfw.pivot(index=["Group", "FileNum", "Method"], columns="Metric", values="Mean ± CI")
           .sort_index()
    )
    metric_cols = list(pivot.columns)

    header_metrics = " & ".join([f"\\textbf{{{c}}}" for c in metric_cols])
    latex = (
        "\\begin{table*}[h]\n\\centering\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        f"\\begin{{tabular}}{{c c {'c'*len(metric_cols)}}}\n"
        "\\arrayrulecolor{black}\n\\toprule\n"
        f"\\textbf{{File}} & \\textbf{{Method}} & {header_metrics} \\\\\n"
        "\\midrule\n"
    )

    # Iterate by Group, then FileNum; use \multirow for FileNum, and \midrule between files
    total_cols = len(metric_cols) + 2  # File + Method + metrics
    current_group = None

    # group level 0 = Group
    for group, group_df in pivot.groupby(level=0, sort=False):
        # New group header with thin rules above/below
        if current_group is not None:
            latex += "\\arrayrulecolor{black}\\specialrule{0.5pt}{0pt}{0pt}\\arrayrulecolor{black}\n"
        latex += f"\\multicolumn{{{total_cols}}}{{l}}{{\\textbf{{{group}}}}} \\\\\n"
        latex += "\\arrayrulecolor{black}\\specialrule{0.5pt}{0pt}{0pt}\\arrayrulecolor{black}\n"
        current_group = group

        # files inside this group (level 1 = FileNum)
        files_in_group = list({idx[1] for idx in group_df.index})
        for i_file, fnum in enumerate(files_in_group):
            file_df = group_df.xs((group, fnum), level=(0,1), drop_level=False)
            # methods for this (group, file)
            methods = list(file_df.index.get_level_values(2))

            # number of rows to span for the File cell
            n_methods = len(methods)
            for j, method in enumerate(methods):
                row = file_df.xs((group, fnum, method)).reindex(metric_cols).fillna("").astype(str)
                metrics_line = " & ".join(row.tolist())

                if j == 0:
                    # First method row: print \multirow in the File column
                    latex += f"\\multirow{{{n_methods}}}{{*}}{{{int(fnum)}}} & \\textbf{{{method}}} & {metrics_line} \\\\\n"
                else:
                    # Subsequent method rows: empty File column
                    latex += f" & \\textbf{{{method}}} & {metrics_line} \\\\\n"

            # After finishing this file block, add a midrule unless it's the last file in the group
            if i_file < len(files_in_group) - 1:
                latex += "\\midrule\n"

    latex += "\\bottomrule\n\\end{tabular}}\n"
    latex += "\\caption{Performance metrics per method, grouped by model (Group) and file. "
    latex += "Best values per (Group, File) are bold. Daggers: † TB variant > baseline; ‡ weighted TB > unweighted TB (one-sided Wilcoxon, Holm-corrected).}\n"
    latex += "\\label{tab:grouped_results}\n\\end{table*}\n"
    return latex


def main(group_label:str, experiment_paths: List[Tuple[str, List[str]]], k_values = (3,5)) -> None:
    retrieval_metrics = [f"{p}_{k}" for p in ("mrr", "precision", "recall", "ndcg") for k in k_values]

    summary_rows: List[Tuple[int, str, str, float, float]] = []
    significance_by_group_file: Dict[Tuple[str, int], Dict[str, Dict]] = {}  
    
    for method_idx, (relevant_check_path, result_paths) in enumerate(experiment_paths):
        print(f"\n***** {group_label} *****")
        print(f"[{group_label}] files used:")
        for p in files_found_relevant:
            print("  -", p)

        annotations = prepare_annotations(relevant_check_path, prepare=True)
        print(
            f"Number of relevant passages for {group_label}: "
            f"{annotations['num_relevant_docs'].mean():.3f}±{annotations['num_relevant_docs'].std():.3f}"
        )

        for file_id, path_results in enumerate(result_paths, start=1):
            df = (
                pd.read_parquet(path_results)
                .drop_duplicates(subset=["question"], keep="first")
                .rename(columns={"questions": "question"})
                .merge(annotations, on="question", how="inner")
            )

            df["Group"] = group_label
            df["FileNum"] = int(file_id)

            result_keys = [
                "results_1", "results_2",
                "results_3_weighted", "results_3_unweighted",
                "results_4_weighted", "results_4_unweighted",
            ]
            for key in result_keys:
                if key in df.columns:
                    df = compute_per_row_metrics(df, key, k_values)

            metric_cols: List[str] = []
            for key in result_keys:
                for k in k_values:
                    metric_cols.extend([
                        f"mrr_{k}_{key}",
                        f"precision_{k}_{key}",
                        f"recall_{k}_{key}",
                        f"ndcg_{k}_{key}",
                    ])
            time_cols = [c for c in df.columns if c.startswith("time_")]
            metric_cols.extend(time_cols)

            summary_rows.extend(summarize(df, metric_cols))

            sig_block = compute_significance_rm(df, retrieval_metrics)
            significance_by_group_file[(group_label, int(file_id))] = sig_block

    df_summary = pd.DataFrame(summary_rows, columns=["Group", "FileNum", "Method", "Metric", "Mean", "CI"])

    print("\n=== Significance Tests Summary (Repeated-Measures) per (Group, File) ===\n")
    for (group, fnum), sig in significance_by_group_file.items():
        print(f"[Group: {group}] [File: {fnum}]")
        fried = sig.get("Friedman p-value", {})
        if fried:
            print("  Friedman p-values:")
            for metric, p in sorted(fried.items()):
                print(f"    {metric}: p={float(p):.4g}")

        dir_one = sig.get("Directional Wilcoxon-Holm", {})
        if dir_one:
            print("  Directional Wilcoxon-Holm (one-sided; Holm within metric):")
            for metric, d in dir_one.items():
                parts = []
                for (a, b), info in d.items():
                    p_adj = info["p_adj"] if isinstance(info, dict) else float(info)
                    r_rb = info.get("r_rb") if isinstance(info, dict) else None
                    n_eff = info.get("n") if isinstance(info, dict) else None
                    extra = f", r_rb={r_rb:.3f}, n={n_eff}" if r_rb is not None and n_eff is not None else ""
                    parts.append(f"{a}>{b}: p_adj={p_adj:.3g}{extra}")
                if parts:
                    print(f"    {metric}: " + "; ".join(parts))
        print()

    latex = to_latex_table(df_summary, significance_by_group_file)
    print(latex)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval runs with explicit paths (corrected).")
    parser.add_argument("--model_eval", type=str, default="gpt-4o-2024-08-06")
                                                             #"llama3.3:70b", "qwen:32b"], help="List of model names to evaluate.")
    parser.add_argument("--path_gold_relevant", type=str, default="", help="Path to the relevant_check parquet file.")
    parser.add_argument("--paths_found_relevant", type=str, help="List of paths to found relevant results parquet files.")
    parser.add_argument("--tpc", type=int, default=11, help="Topic ID (used only for printing labels).")
    args = parser.parse_args()

    exp_paths = []
    files_found_relevant = listdir(args.paths_found_relevant)

    files_found_relevant = [f for f in files_found_relevant if
                            (args.model_eval in f
                                and (f.endswith("_thr_.parquet") or f.endswith("_thr__dynamic.parquet"))
                                )]
    files_found_relevant = [f"{args.paths_found_relevant}/{f}" for f in files_found_relevant]
    files_found_relevant = sorted(files_found_relevant)
    print(f"[{args.model_eval}] files used:", *files_found_relevant, sep="\n  - ")

    exp_paths.append(
        (
            args.path_gold_relevant,
            files_found_relevant
        )
    )

    main(group_label=args.model_eval, experiment_paths=exp_paths, k_values=(3,5))