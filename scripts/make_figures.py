import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, itertools
from scipy import stats

SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#8a8984"; GRID="#e3e2dd"
S1="#2a78d6"; S2="#eb6834"; S3="#1baf7a"; S4="#eda100"; NEUTRAL="#b8b7b1"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": ["DejaVu Sans"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 1.0,
    "xtick.major.size": 0, "ytick.major.size": 0,
})

df = pd.read_csv("cascade_benchmark_results_2026-08-22_gpt-oss-120b.csv")
ORDER = ["baseline","cascade","multi_call","multi_call_full_context"]
LBL = {"baseline":"baseline","cascade":"cascade","multi_call":"multi_call","multi_call_full_context":"multi_call\nfull_context"}
COL = {"baseline":S4,"cascade":S1,"multi_call":S2,"multi_call_full_context":S3}
piv = {c: g.set_index("example_idx").sort_index() for c,g in df.groupby("condition")}

def figtitle(fig, t, sub=None, top=0.86):
    """Flush-left title block in figure coordinates, drawn after tight_layout."""
    fig.subplots_adjust(top=top)
    fig.text(0.012, 0.975, t, fontsize=13, fontweight="bold", color=INK, va="top")
    if sub:
        fig.text(0.012, 0.905, sub, fontsize=9.5, color=INK2, va="top")

def title(ax, t, sub=None):
    if sub:
        ax.set_title(t, loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=30)
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")
    else:
        ax.set_title(t, loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=10)

# ---------- FIG 1: cost/quality frontier ----------
fig, ax = plt.subplots(figsize=(8.2,5.0))
ax.grid(axis="both", zorder=0)
ax.set_axisbelow(True)
POS = {"baseline":(0,16,"left"), "cascade":(0,16,"center"),
       "multi_call":(0,-40,"center"), "multi_call_full_context":(12,10,"left")}
for c in ORDER:
    g = piv[c]
    x = g.total_tokens.median(); y = g.hotpot_em.mean()
    ax.scatter([x],[y], s=200 if c=="cascade" else 130, color=COL[c], zorder=5,
               edgecolor=SURF, linewidth=2)
    dx,dy,ha = POS[c]
    ax.annotate(f"{LBL[c].replace(chr(10),'_')}\nEM {y:.3f} · {x:,.0f} tok",
                (x,y), textcoords="offset points", xytext=(dx,dy), ha=ha,
                fontsize=9.5, color=INK2, linespacing=1.4, zorder=6)
xc, yc = piv["cascade"].total_tokens.median(), piv["cascade"].hotpot_em.mean()
xf, yf = piv["multi_call_full_context"].total_tokens.median(), piv["multi_call_full_context"].hotpot_em.mean()
ax.annotate("", xy=(xf-60,yf), xytext=(xc+60,yc),
            arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.4, ls=(0,(4,3))), zorder=3)
ax.text((xc+xf)/2, yc-0.016, "identical exact match · 2.56× the tokens · 1.87× the wall clock",
        fontsize=9, color=MUTED, ha="center", va="top")
ax.set_xlabel("median total tokens per question"); ax.set_ylabel("HotpotQA exact match")
ax.set_xlim(1150, 7600); ax.set_ylim(0.06, 0.80)
ax.set_yticks(np.arange(0.1,0.75,0.1))
fig.tight_layout(rect=[0,0,1,0.99])
figtitle(fig, "Cost against quality",
         "n = 1,000 HotpotQA questions · gpt-oss-120b · every condition answers the same questions.", top=0.83)
fig.savefig("figures/fig1-cost-quality-frontier.png", dpi=200); plt.close(fig)

# ---------- FIG 2: token composition ----------
fig, ax = plt.subplots(figsize=(9.0,4.4))
ax.grid(axis="x", zorder=0); ax.set_axisbelow(True)
ys = np.arange(len(ORDER))[::-1]
for y,c in zip(ys, ORDER):
    g = piv[c]; i = g.input_tokens.median(); o = g.output_tokens.median()
    ax.barh(y, i, height=0.5, color=S1, zorder=3, label="input (prompt) tokens" if c=="baseline" else None)
    ax.barh(y, o, left=i+50, height=0.5, color=S2, zorder=3, label="output (generated) tokens" if c=="baseline" else None)
    ax.text(i+o+240, y, f"{i:,.0f} in + {o:,.0f} out", va="center", fontsize=9.5, color=INK2)
ax.set_yticks(ys); ax.set_yticklabels([c.replace("multi_call_full_context","mc_full_context") for c in ORDER], fontsize=10, color=INK)
ax.set_xlabel("median tokens per question"); ax.set_xlim(0, 7800)
ax.legend(frameon=False, loc="upper center", fontsize=9.5, ncols=2, bbox_to_anchor=(0.5,-0.20))
fig.tight_layout(rect=[0,0,1,0.99])
figtitle(fig, "Where the multi-call tax is actually paid",
         "Median per-question token split. The extra calls buy almost no extra generation — they re-send the prompt.", top=0.80)
fig.savefig("figures/fig2-token-composition.png", dpi=200); plt.close(fig)

# ---------- FIG 3: McNemar discordant pairs ----------
pairs = [("cascade","multi_call_full_context"),("cascade","multi_call"),("cascade","baseline")]
fig, ax = plt.subplots(figsize=(9.0,4.4))
ax.grid(axis="x", zorder=0); ax.set_axisbelow(True)
ys = np.arange(len(pairs))[::-1]
for y,(a,b) in zip(ys,pairs):
    A = piv[a].hotpot_em==1; B = piv[b].hotpot_em==1
    n_ab = int((A&~B).sum()); n_ba = int((~A&B).sum())
    p = stats.binomtest(min(n_ab,n_ba), n_ab+n_ba, 0.5).pvalue
    ax.barh(y, n_ab, height=0.42, color=S1, zorder=3, label="cascade right, other wrong" if y==ys[0] else None)
    ax.barh(y, -n_ba, left=-4, height=0.42, color=S2, zorder=3, label="other right, cascade wrong" if y==ys[0] else None)
    ax.text(n_ab+14, y, str(n_ab), va="center", fontsize=9.5, color=INK2)
    ax.text(-n_ba-14, y, str(n_ba), va="center", ha="right", fontsize=9.5, color=INK2)
    ax.text(-548, y+0.30, f"cascade  vs  {b}", ha="left", va="bottom", fontsize=10, color=INK, zorder=6)
    ax.text(548, y+0.30, f"p = {p:.3g}", ha="right", va="bottom", fontsize=9.5, color=INK2, zorder=6)
ax.axvline(0, color=MUTED, lw=1.2, zorder=4)
ax.set_yticks([]); ax.set_xlim(-560, 560); ax.set_ylim(-0.55, 2.65)
ax.set_xlabel("discordant pairs — questions where the two conditions disagree")
ax.legend(frameon=False, loc="upper center", fontsize=9.5, ncols=2, bbox_to_anchor=(0.5,-0.20))
fig.tight_layout(rect=[0,0,1,0.99])
figtitle(fig, "What McNemar's test actually looks at",
         "Only the questions where the two conditions disagree. Concordant pairs carry no signal and are discarded.", top=0.80)
fig.savefig("figures/fig3-mcnemar-discordant.png", dpi=200); plt.close(fig)

# ---------- FIG 4: one outlier eats the mean ----------
fig, axes = plt.subplots(1, 2, figsize=(9.4,4.4), gridspec_kw={"width_ratios":[1.25,1]})
ax = axes[0]
ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)
for i,c in enumerate(ORDER):
    v = piv[c].latency_s.values
    xs = np.random.default_rng(0).normal(i, 0.075, len(v))
    ax.scatter(xs, v, s=5, color=COL[c], alpha=0.28, zorder=3, linewidths=0)
    ax.scatter([i],[np.median(v)], marker="_", s=900, color=INK, zorder=5, linewidths=2)
ax.set_yscale("log"); ax.set_xticks(range(4))
ax.set_xticklabels([LBL[c] for c in ORDER], fontsize=9)
ax.set_ylabel("latency, seconds (log scale)")
ax.annotate("one baseline run: 6,068 s\n129k output tokens, degenerate loop",
            (0.06, 6067), textcoords="offset points", xytext=(26,-30), fontsize=9,
            color=INK2, linespacing=1.35, va="center",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.0,
                            connectionstyle="angle,angleA=0,angleB=90,rad=3"))
ax.text(0.02, 0.02, "— = median", transform=ax.transAxes, fontsize=9, color=MUTED)
title(ax, "One run eats the mean", "Per-question latency, all 1,000 questions")

ax = axes[1]
ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)
b = piv["baseline"].latency_s
vals = [b.mean(), b[b<100].mean(), b.median()]
names = ["mean\n(all runs)","mean\n(outlier dropped)","median"]
ax.bar(range(3), vals, width=0.55, color=[S4,NEUTRAL,NEUTRAL], zorder=3)
for i,v in enumerate(vals):
    ax.text(i, v+0.35, f"{v:.2f}s", ha="center", fontsize=9.5, color=INK2)
ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("baseline latency, seconds"); ax.set_ylim(0,17)
title(ax, "…and why the t-test missed it", "p = 0.84 (paired t) vs 4.3e-161 (Wilcoxon)")
fig.tight_layout(); fig.savefig("figures/fig4-outlier-effect.png", dpi=200); plt.close(fig)

# ---------- FIG 5: layer-by-layer EM ----------
from cascade_prompting.metrics.hotpot import hotpot_answer_metrics
layers = ["draft","fact_checked","final"]
conds = ["cascade","multi_call","multi_call_full_context"]
data = {}
for c in conds:
    x = df[df.condition==c]
    data[c] = [np.mean([hotpot_answer_metrics(str(p),str(g))["hotpot_em"] for p,g in zip(x[f"layer_{L}_data"], x.gold_answer)]) for L in layers]
fig, ax = plt.subplots(figsize=(9.0,4.6))
ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)
w=0.235; xs=np.arange(3)
for j,c in enumerate(conds):
    off=(j-1)*(w+0.03)
    ax.bar(xs+off, data[c], width=w, color=COL[c], zorder=3, label=LBL[c].replace("\n","_"))
    for xi,v in zip(xs+off, data[c]):
        ax.text(xi, v+0.012, f"{v:.2f}", ha="center", fontsize=9, color=INK2)
bl = df[df.condition=="baseline"].hotpot_em.mean()
ax.axhline(bl, color=MUTED, lw=1.4, ls=(0,(4,3)), zorder=4)
ax.text(2.47, bl, f"baseline\n{bl:.3f}", fontsize=9, color=MUTED, ha="left", va="center", linespacing=1.3)
ax.set_xlim(-0.5, 3.05)
ax.set_xticks(xs); ax.set_xticklabels(["layer 1 — draft","layer 2 — fact-checked","layer 3 — final"], fontsize=10)
ax.set_ylabel("HotpotQA exact match"); ax.set_ylim(0,0.72)
ax.legend(frameon=False, fontsize=9.5, loc="upper left")
fig.tight_layout(rect=[0,0,1,0.99])
figtitle(fig, "Exact match, scored at every layer",
         "The jump lives entirely in layer 3, where a prose answer becomes a short one. Layers 1–2 sit at baseline.", top=0.82)
fig.savefig("figures/fig5-layer-em.png", dpi=200); plt.close(fig)
print("ok")
