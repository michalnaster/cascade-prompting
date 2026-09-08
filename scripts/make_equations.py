"""Render the article's display equations to PNG.

Medium does not support LaTeX, so every display formula ships as an image.
Simple inline math (p, n, alpha, subscripts) is written as Unicode in the
prose instead — rendering those as images would break the line flow.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURF = "#fcfcfb"
INK = "#0b0b0b"

EQUATIONS = {
    "eq1-f1": (
        r"$\mathrm{precision} = \frac{c}{|P|}"
        r"\qquad \mathrm{recall} = \frac{c}{|G|}"
        r"\qquad F_1 = \frac{2 \cdot \mathrm{precision} \cdot \mathrm{recall}}"
        r"{\mathrm{precision} + \mathrm{recall}}$"
    ),
    "eq2-mcnemar": (
        r"$n_{10} \mid (n_{10} + n_{01}) \sim "
        r"\mathrm{Binomial}\left(n_{10} + n_{01},\ \frac{1}{2}\right)$"
    ),
    "eq3-paired-t": r"$t = \frac{\bar{d}}{s_d / \sqrt{n}} \qquad \mathrm{df} = n - 1$",
    "eq4-effect-sizes": (
        r"$d_z = \frac{\bar{d}}{s_d} \qquad\qquad "
        r"r = \frac{2W^+}{n(n+1)/2} - 1$"
    ),
}

for name, tex in EQUATIONS.items():
    fig = plt.figure(figsize=(0.01, 0.01), facecolor=SURF)
    fig.text(0, 0, tex, fontsize=19, color=INK)
    fig.savefig(
        f"figures/{name}.png",
        dpi=200,
        facecolor=SURF,
        bbox_inches="tight",
        pad_inches=0.28,
    )
    plt.close(fig)
    print(f"figures/{name}.png")
