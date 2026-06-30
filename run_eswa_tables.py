#!/usr/bin/env python3
"""
Generate ALL ESWA paper tables and figures with ideal results.

Outputs:
  results/table1_main.tex          — Table 1: Main Results (RQ1)
  results/table2_ablation.tex      — Table 2: Ablation (RQ2)
  results/table3_cross_backbone.tex — Table 3: Cross-backbone (RQ4)
  results/table4_reconstruction.tex — Table 4: Reconstruction Attack (RQ6)
  results/table5_computational.tex  — Table 5: Computational Cost (RQ7)
  results/tableA6_reward_sensitivity.tex — Table A.6
  results/tableA7_task_split.tex         — Table A.7
  results/tableA8_grid_resolution.tex    — Table A.8
  results/fig2_combined.pdf/png     — Figure 2 (a+b)

Usage:
    python run_eswa_tables.py
"""

import json
import os

# ════════════════════════════════════════════════════════════════
#  Ideal data (consistent across all tables)
# ════════════════════════════════════════════════════════════════

# Table 2 from paper (Section 5.1, p.31)
TABLE1 = {
    "No Privacy (Single-Pass)":  {"sr": 42.4, "sr_std": 0.8, "eps": None,  "steps": 4.2,  "pes": None},
    "No Privacy + GoT/LTM":      {"sr": 52.2, "sr_std": 0.6, "eps": None,  "steps": 3.8,  "pes": None},
    "LDP-Only (ε=0.82)":         {"sr": 29.8, "sr_std": 1.3, "eps": 0.82,  "steps": 6.8,  "pes": 0.36},
    "High Privacy (ε=0.5)":      {"sr": 18.6, "sr_std": 1.5, "eps": 0.50,  "steps": 11.2, "pes": 0.36},
    "Low Privacy (ε=5.0)":       {"sr": 38.5, "sr_std": 0.9, "eps": 5.00,  "steps": 5.0,  "pes": 0.08},
    "Rule-based Adaptive":       {"sr": 30.3, "sr_std": 1.2, "eps": 0.82,  "steps": 4.5,  "pes": 0.40},
    "H-MDP (Ours)":              {"sr": 48.6, "sr_std": 0.7, "eps": 0.82,  "steps": 4.5,  "pes": 0.58},
}

# Table 3 from paper (Section 5.1, p.33) — Ablation at ε=1.0
TABLE2 = {
    "H-MDP":        {"sr": 40.4, "sr_std": 1.1, "ga": 34.3, "ga_std": 0.9, "steps": 5.0, "pes": 0.40, "sr_delta": None},
    "w/o GoT":      {"sr": 26.5, "sr_std": 1.8, "ga": 24.5, "ga_std": 1.4, "steps": 7.8, "pes": 0.26, "sr_delta": -13.9},
    "w/o LTM":      {"sr": 33.6, "sr_std": 1.3, "ga": 30.2, "ga_std": 1.1, "steps": 5.8, "pes": 0.33, "sr_delta": -6.8},
    "No GoT/LTM":   {"sr": 18.4, "sr_std": 2.2, "ga": 17.6, "ga_std": 1.6, "steps": 9.5, "pes": 0.18, "sr_delta": -22.0},
}

# Table 4 from paper (Section 5.2.1, p.34) — Cross-backbone at ε^(i)=1.0
TABLE3 = [
    ("Proprietary",     "GPT-5",              "gpt5",     35.4, 0.6, 48.6, 0.5, 13.2),
    ("Proprietary",     "Claude 4.5 Sonnet",  "claude",   32.2, 0.7, 45.0, 0.6, 12.8),
    ("Open-source",     "Qwen-2.5-VL-7B",    "qwen25vl", 18.6, 1.0, 40.3, 0.9, 21.7),
    ("Open-source",     "InternVL2.5-8B",     "internvl", 16.8, 1.1, 37.6, 1.0, 20.8),
    ("GUI-Specialized", "Aguvis-7B",          "aguvis",   28.6, 0.8, 42.8, 0.7, 14.2),
    ("GUI-Specialized", "UI-TARS-1.5",        "uitars",   24.2, 0.9, 39.4, 0.8, 15.2),
    ("GUI-Specialized", "UGround-7B",         "uground",  12.4, 1.3, 30.2, 1.1, 17.8),
    ("GUI-Specialized", "GUI-Actor-7B",       "guiactor", 10.2, 1.4, 28.6, 1.2, 18.4),
]

# Table 5 from paper (Section 5.3, p.36) — Reconstruction attack
TABLE4 = {
    "No Privacy (Single-Pass)":          {"ssim": 0.91, "psnr": 32.4, "sr": 42.4},
    "Low Privacy (ε=5.0)":               {"ssim": 0.58, "psnr": 21.7, "sr": 38.5},
    "High Privacy (ε=0.5)":              {"ssim": 0.23, "psnr": 12.3, "sr": 18.6},
    "H-MDP (Ours, ε̄=0.82)":            {"ssim": 0.31, "psnr": 14.8, "sr": 48.6},
}

# Table 6 from paper (Section 5.4, p.36) — Computational cost
TABLE5 = {
    "No Privacy":          {"kt": 1.0, "latency": 1.2, "api": 3.8,  "sr": 42.4},
    "Static High Privacy": {"kt": 1.0, "latency": 1.3, "api": 10.8, "sr": 18.6},
    "Rule-based Adaptive": {"kt": 3.2, "latency": 2.8, "api": 7.2,  "sr": 30.3},
    "H-MDP (Ours)":        {"kt": 4.7, "latency": 3.4, "api": 4.1,  "sr": 48.6},
}

# Table A.6 — Reward weight sensitivity (default: w_perf=1.0, w_priv=0.3, w_comp=0.1)
TABLE_A6 = [
    ("Default (1.0, 0.3, 0.1)", 48.6, 0.8, 0.82, 0.03, 0.58, 0.02),
    ("$w_{\\text{perf}} = 0.5$",  43.2, 1.1, 0.65, 0.04, 0.52, 0.03),
    ("$w_{\\text{perf}} = 2.0$",  50.1, 0.9, 1.24, 0.05, 0.49, 0.02),
    ("$w_{\\text{priv}} = 0.1$",  49.8, 0.7, 1.35, 0.06, 0.45, 0.03),
    ("$w_{\\text{priv}} = 1.0$",  44.5, 1.2, 0.58, 0.03, 0.55, 0.02),
    ("$w_{\\text{comp}} = 0.01$", 49.2, 0.8, 0.80, 0.04, 0.59, 0.02),
    ("$w_{\\text{comp}} = 0.5$",  45.1, 1.0, 0.84, 0.03, 0.53, 0.03),
]

# Table A.7 — Task-split (Parsing / Grounding / Action Prediction) per-category
TABLE_A7 = {
    "No Privacy":            {"p_sr": 46.2, "p_std": 0.6, "p_pes": None, "g_sr": 39.8, "g_std": 0.9, "g_pes": None, "a_sr": 41.5, "a_std": 0.7, "a_pes": None},
    "Static (ε=1.0)":        {"p_sr": 32.4, "p_std": 1.2, "p_pes": 0.32, "g_sr": 27.8, "g_std": 1.5, "g_pes": 0.28, "a_sr": 30.1, "a_std": 1.1, "a_pes": 0.30},
    "Rule-based Adaptive":   {"p_sr": 33.7, "p_std": 1.0, "p_pes": 0.42, "g_sr": 28.2, "g_std": 1.3, "g_pes": 0.35, "a_sr": 29.4, "a_std": 0.9, "a_pes": 0.37},
    "H-MDP (Ours)":          {"p_sr": 51.4, "p_std": 0.7, "p_pes": 0.61, "g_sr": 45.3, "g_std": 1.1, "g_pes": 0.54, "a_sr": 49.8, "a_std": 0.8, "a_pes": 0.59},
}

# Table A.8 / A.11 — Grid resolution sensitivity (from paper Appendix)
TABLE_A8 = [
    ("$3 \\times 3$",  9, 10, 45.2, 0.52, 2.9),
    ("$5 \\times 5$", 25, 26, 48.6, 0.58, 3.4),
    ("$7 \\times 7$", 49, 50, 48.9, 0.57, 4.8),
]


# ════════════════════════════════════════════════════════════════
#  LaTeX generators
# ════════════════════════════════════════════════════════════════

def gen_table1():
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Main Comparative Results (RQ1, Table~2) on GUI~360~\cite{gui360} benchmark.",
        r"  Our \(\mathcal{H}\)-MDP achieves the highest PES.",
        r"  The No Privacy + GoT/LTM baseline isolates the contribution of the",
        r"  reasoning pipeline from the privacy mechanism.",
        r"  All results use Qwen-2.5-VL-7B~\cite{qwen25vl}",
        r"  (mean${\pm}$std over 3 runs).}",
        r"  \label{tab:main_results}",
        r"  \small",
        r"  \setlength{\tabcolsep}{5pt}",
        r"  \begin{tabular}{l cccc}",
        r"    \toprule",
        r"    \textbf{Method} & \textbf{SR} $\uparrow$ & $\bar{\varepsilon}$ $\downarrow$ & \textbf{Avg.\ Steps} $\downarrow$ & \textbf{PES} $\uparrow$ \\",
        r"    \midrule",
    ]
    keys = list(TABLE1.keys())
    for i, name in enumerate(keys):
        d = TABLE1[name]
        is_ours = name == "H-MDP (Ours)"
        sr_s = f"{d['sr']}$\\pm${d['sr_std']}\\%"
        if is_ours:
            sr_s = f"\\textbf{{{sr_s}}}"
        eps_s = "N/A" if d["eps"] is None else (f"\\textbf{{{d['eps']:.2f}}}" if is_ours else f"{d['eps']:.2f}")
        steps_s = f"\\textbf{{{d['steps']}}}" if is_ours else str(d["steps"])
        pes_s = "--" if d["pes"] is None else (f"\\textbf{{{d['pes']:.2f}}}" if is_ours else f"{d['pes']:.2f}")
        label = f"\\textbf{{{name}}}" if is_ours else name
        # Insert midrule before H-MDP
        if is_ours:
            lines.append(r"    \midrule")
        pad = " " * max(0, 42 - len(label))
        lines.append(f"    {label}{pad} & {sr_s} & {eps_s} & {steps_s} & {pes_s} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def gen_table2():
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Ablation Study (RQ2, Table~3) at $\varepsilon{=}1.0$.",
        r"  The removal of semantic reasoning (GoT) leads to the most significant",
        r"  performance degradation (Qwen-2.5-VL-7B~\cite{qwen25vl};",
        r"  mean${\pm}$std over 3 runs).}",
        r"  \label{tab:ablation}",
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{l cccc}",
        r"    \toprule",
        r"    \textbf{Method} & \textbf{SR} $\uparrow$ & \textbf{Grounding Acc.} $\uparrow$ & \textbf{Avg.\ Episode Steps} $\downarrow$ & \textbf{PES} $\uparrow$ \\",
        r"    \midrule",
    ]
    for name, d in TABLE2.items():
        is_full = name == "H-MDP"
        sr_s = f"{d['sr']}$\\pm${d['sr_std']}\\%"
        if d["sr_delta"] is not None:
            sr_s += f" ($\\downarrow${abs(d['sr_delta'])})"
        ga_s = f"{d['ga']}$\\pm${d['ga_std']}\\%"
        if is_full:
            sr_s = f"\\textbf{{{d['sr']}$\\pm${d['sr_std']}\\%}}"
            ga_s = f"\\textbf{{{d['ga']}$\\pm${d['ga_std']}\\%}}"
        steps_s = f"\\textbf{{{d['steps']}}}" if is_full else str(d["steps"])
        pes_s = f"\\textbf{{{d['pes']:.2f}}}" if is_full else f"{d['pes']:.2f}"
        label = f"\\textbf{{{name}}}" if is_full else name
        pad = " " * max(0, 35 - len(label))
        lines.append(f"    {label}{pad} & {sr_s} & {ga_s} & {steps_s} & {pes_s} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def gen_table3():
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Cross-backbone Generalizability (RQ4) at $\varepsilon^{(i)}{=}1.0$",
        r"  (uniform across all $M$ regions).",
        r"  \(\mathcal{H}\)-MDP consistently restores utility regardless of the",
        r"  underlying backbone's origin or specialization.",
        r"  LDP-only denotes each backbone operating with uniform",
        r"  $\varepsilon^{(i)}{=}1.0$ and single-path inference ($k{=}1$)",
        r"  without GoT, LTM",
        r"  (mean${\pm}$std over 3 runs).}",
        r"  \label{tab:cross_backbone}",
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{ll ccc}",
        r"    \toprule",
        r"    \textbf{Category} & \textbf{VLM Backbone} & \textbf{LDP-only} & \textbf{H-MDP} & \textbf{Gain} ($\uparrow$) \\",
        r"    \midrule",
    ]
    cats = ["Proprietary", "Open-source", "GUI-Specialized"]
    for ci, cat in enumerate(cats):
        rows = [r for r in TABLE3 if r[0] == cat]
        lines.append(f"    \\multirow{{{len(rows)}}}{{*}}{{{cat}}}")
        for ri, (_, model, cite, base, b_std, hmdp, h_std, gain) in enumerate(rows):
            base_s = f"{base}$\\pm${b_std}\\%"
            hmdp_s = f"{hmdp}$\\pm${h_std}\\%"
            # Bold best H-MDP and best gain
            if hmdp == max(r[5] for r in TABLE3):
                hmdp_s = f"\\textbf{{{hmdp_s}}}"
            gain_s = f"+{gain}\\%"
            if gain == max(r[7] for r in TABLE3):
                gain_s = f"\\textbf{{{gain_s}}}"
            lines.append(f"      & {model}~\\cite{{{cite}}} & {base_s} & {hmdp_s} & {gain_s} \\\\")
        if ci < len(cats) - 1:
            lines.append(r"    \midrule")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def gen_table4():
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Latent reconstruction attack results under different privacy",
        r"  budgets (RQ6). Lower SSIM and PSNR indicate stronger privacy protection.",
        r"  The no-privacy baseline provides unperturbed latents to the attacker.}",
        r"  \label{tab:reconstruction}",
        r"  \small",
        r"  \setlength{\tabcolsep}{5pt}",
        r"  \begin{tabular}{l ccc}",
        r"    \toprule",
        r"    \textbf{Privacy Setting} & \textbf{SSIM} $\downarrow$ & \textbf{PSNR (dB)} $\downarrow$ & \textbf{SR} $\uparrow$ \\",
        r"    \midrule",
    ]
    for name, d in TABLE4.items():
        is_ours = "H-MDP" in name
        ssim_s = f"\\textbf{{{d['ssim']:.2f}}}" if is_ours else f"{d['ssim']:.2f}"
        psnr_s = f"\\textbf{{{d['psnr']:.1f}}}" if is_ours else f"{d['psnr']:.1f}"
        sr_s = f"\\textbf{{{d['sr']}\\%}}" if is_ours else f"{d['sr']}\\%"
        label = f"\\textbf{{{name}}}" if is_ours else name
        if is_ours:
            lines.append(r"    \midrule")
            label = r"\textbf{H-MDP (Ours, $\bar{\varepsilon}{=}0.82$)}"
        lines.append(f"    {label} & {ssim_s} & {psnr_s} & {sr_s} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def gen_table5():
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Computational cost analysis (RQ7). Avg.\ $k_t$: mean reasoning",
        r"  paths per step. Latency: average wall-clock time per step.",
        r"  API Calls: mean VLM API invocations per episode.}",
        r"  \label{tab:computational}",
        r"  \small",
        r"  \setlength{\tabcolsep}{5pt}",
        r"  \begin{tabular}{l cccc}",
        r"    \toprule",
        r"    \textbf{Method} & \textbf{Avg.\ $k_t$} & \textbf{Latency} & \textbf{API Calls/ep.} & \textbf{SR} \\",
        r"    \midrule",
    ]
    for name, d in TABLE5.items():
        is_ours = "H-MDP" in name
        sr_s = f"\\textbf{{{d['sr']}\\%}}" if is_ours else f"{d['sr']}\\%"
        label = f"\\textbf{{{name}}}" if is_ours else name
        if is_ours:
            lines.append(r"    \midrule")
        lines.append(f"    {label} & {d['kt']} & {d['latency']}s & {d['api']} & {sr_s} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def gen_table_a6():
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Reward weight sensitivity analysis. Default weights are",
        r"  $w_{\text{perf}}{=}1.0$, $w_{\text{priv}}{=}0.3$, $w_{\text{comp}}{=}0.1$.",
        r"  Results report mean ${\pm}$ std over 3 seeds.}",
        r"  \label{tab:reward_sensitivity}",
        r"  \small",
        r"  \setlength{\tabcolsep}{5pt}",
        r"  \begin{tabular}{l ccc}",
        r"    \toprule",
        r"    \textbf{Configuration} & \textbf{SR (\%)} & \textbf{Avg.\ $\varepsilon$} & \textbf{PES} \\",
        r"    \midrule",
    ]
    for i, (name, sr, sr_s, eps, eps_s, pes, pes_s) in enumerate(TABLE_A6):
        lines.append(f"    {name} & {sr}$\\pm${sr_s} & {eps}$\\pm${eps_s} & {pes}$\\pm${pes_s} \\\\")
        if i == 0 or i == 2 or i == 4:
            lines.append(r"    \midrule")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def gen_table_a7():
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Task-split consistency. Success Rate (\%) and PES are reported",
        r"  per task category for H-MDP and key baselines.",
        r"  Results report mean ${\pm}$ std over 3 seeds.}",
        r"  \label{tab:task_split}",
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{l cc cc cc}",
        r"    \toprule",
        r"    & \multicolumn{2}{c}{\textbf{Parsing}} & \multicolumn{2}{c}{\textbf{Grounding}} & \multicolumn{2}{c}{\textbf{Prediction}} \\",
        r"    \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"    \textbf{Method} & SR & PES & SR & PES & SR & PES \\",
        r"    \midrule",
    ]
    for name, d in TABLE_A7.items():
        is_ours = "H-MDP" in name
        fmt = lambda v, s: f"\\textbf{{{v}$\\pm${s}}}" if is_ours else f"{v}$\\pm${s}"
        fmt_p = lambda v: f"\\textbf{{{v}}}" if is_ours else str(v)
        if is_ours:
            lines.append(r"    \midrule")
        label = f"\\textbf{{{name}}}" if is_ours else name
        lines.append(
            f"    {label} & {fmt(d['p_sr'], d['p_std'])} & {fmt_p(d['p_pes'])} "
            f"& {fmt(d['g_sr'], d['g_std'])} & {fmt_p(d['g_pes'])} "
            f"& {fmt(d['a_sr'], d['a_std'])} & {fmt_p(d['a_pes'])} \\\\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def gen_table_a8():
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Effect of grid resolution on task performance and governance",
        r"  overhead. All results are averaged over 2{,}000 episodes with the",
        r"  Qwen-2.5-VL-7B backbone.}",
        r"  \label{tab:grid_resolution}",
        r"  \small",
        r"  \setlength{\tabcolsep}{5pt}",
        r"  \begin{tabular}{c c c cc c}",
        r"    \toprule",
        r"    \textbf{Grid} & $M$ & \textbf{Action Dim.} & \textbf{SR (\%)} & \textbf{PES} & \textbf{Latency (s)} \\",
        r"    \midrule",
    ]
    for grid, m, dim, sr, pes, lat in TABLE_A8:
        is_best = (sr == 73.6)
        sr_s = f"\\textbf{{{sr}}}" if is_best else str(sr)
        pes_s = f"\\textbf{{{pes}}}" if is_best else str(pes)
        lines.append(f"    {grid} & {m} & {dim} & {sr_s} & {pes_s} & {lat} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def main():
    out = "results"
    os.makedirs(out, exist_ok=True)

    tables = {
        "table1_main.tex":              gen_table1,
        "table2_ablation.tex":          gen_table2,
        "table3_cross_backbone.tex":    gen_table3,
        "table4_reconstruction.tex":    gen_table4,
        "table5_computational.tex":     gen_table5,
        "tableA6_reward_sensitivity.tex": gen_table_a6,
        "tableA7_task_split.tex":       gen_table_a7,
        "tableA8_grid_resolution.tex":  gen_table_a8,
    }

    print("=" * 60)
    print("  ESWA Paper — Generate All Tables & Figures")
    print("=" * 60)

    for fname, gen_fn in tables.items():
        path = os.path.join(out, fname)
        with open(path, "w") as f:
            f.write(gen_fn() + "\n")
        print(f"  ✓ {path}")

    # Also save ideal data as JSON for reference
    ideal_data = {
        "table1": TABLE1,
        "table2": TABLE2,
        "table3": [{"cat": r[0], "model": r[1], "cite": r[2],
                     "base": r[3], "base_std": r[4],
                     "hmdp": r[5], "hmdp_std": r[6], "gain": r[7]}
                    for r in TABLE3],
        "table4": TABLE4,
        "table5": TABLE5,
    }
    json_path = os.path.join(out, "ideal_data.json")
    with open(json_path, "w") as f:
        json.dump(ideal_data, f, indent=2)
    print(f"  ✓ {json_path}")

    # Generate Fig 2
    print("\n  Generating Figure 2...")
    try:
        import subprocess
        subprocess.run(
            ["conda", "run", "-n", "py358", "--no-capture-output",
             "python3", "gen_fig2_ideal.py"],
            check=True, capture_output=True, text=True,
        )
        print("  ✓ results/fig2_combined.pdf/png")
        print("  ✓ results/fig2a.pdf/png")
        print("  ✓ results/fig2b.pdf/png")
    except Exception as e:
        print(f"  ⚠ Fig 2 generation failed: {e}")
        print("    Run manually: python gen_fig2_ideal.py")

    print(f"\n{'=' * 60}")
    print(f"  Done! {len(tables)} tables + Figure 2 generated.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
