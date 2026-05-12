import re
from pathlib import Path

target = Path(r"C:/Users/bilgu/var_project/backtesting/plot_backtest.py")

new_func = '''def plot_lr_statistics(
    results: list,
    save: bool = True,
) -> tuple:
    """
    Three-panel dot plot: LR_UC, LR_IND, LR_CC for every method with
    chi-squared critical value lines.  Green = not rejected, red = rejected.

    X-axis is capped at cv * 8 per panel so that methods close to the
    critical value remain distinguishable.  Off-scale values are pinned to
    the right edge and their actual value printed in italics to the left
    of the dot.

    Parameters
    ----------
    results : list of BacktestResult
    save    : bool
    """
    _apply_style()

    methods = [r.method_name for r in results]
    y       = np.arange(len(methods))

    panels = [
        ([r.lr_uc  for r in results], [r.reject_uc  for r in results],
         _CV1, "LR$_{UC}$",  "Kupiec  (UC)",           f"chi2(1) cv = {_CV1:.2f}"),
        ([r.lr_ind for r in results], [r.reject_ind for r in results],
         _CV1, "LR$_{IND}$", "Christoffersen  (IND)",  f"chi2(1) cv = {_CV1:.2f}"),
        ([r.lr_cc  for r in results], [r.reject_cc  for r in results],
         _CV2, "LR$_{CC}$",  "Christoffersen  (CC)",   f"chi2(2) cv = {_CV2:.2f}"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=_TALL, sharey=True)

    for ax, (values, rejected, cv, xlabel, title, cv_label) in zip(axes, panels):
        cap_factor  = 8
        max_display = cv * cap_factor

        dot_colors = [_pass_fail_color(r) for r in rejected]

        ax.axvline(0,  color="#cccccc", linewidth=0.6, zorder=0)
        ax.axvline(cv, color=C["crit"], linewidth=1.3, linestyle="--",
                   label=cv_label, zorder=1)

        plot_vals = [min(v, max_display * 0.97) for v in values]
        ax.scatter(plot_vals, y, color=dot_colors, s=65, zorder=3,
                   edgecolors="white", linewidths=0.5)

        x_offset = max_display * 0.03
        for val, plot_val, yi, rej in zip(values, plot_vals, y, rejected):
            col = _pass_fail_color(rej)
            if val > max_display:
                ax.text(max_display * 0.93, yi,
                        f"{val:.2f}", va="center", ha="right",
                        fontsize=7.5, color=col, style="italic")
            else:
                ax.text(val + x_offset, yi,
                        f"{val:.2f}", va="center", ha="left",
                        fontsize=7.5, color=col)

        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_xlim(left=-0.5, right=max_display)
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(axis="x")
        ax.grid(axis="y", alpha=0.35)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(methods, fontsize=9)
    axes[0].invert_yaxis()

    fig.suptitle(
        "Likelihood Ratio Test Statistics  |  5% Significance",
        fontsize=11, fontweight="bold", x=0.02, ha="left", y=1.01
    )
    fig.tight_layout()

    if save:
        _save(fig, _fig_path(13, "lr_statistics"))

    return fig, axes
'''

content = target.read_text(encoding="utf-8")

# Replace from def plot_lr_statistics up to (but not including) the next top-level def/class
pattern = r'(def plot_lr_statistics\(.*?)(?=\n# ----|^def |^class )'
new_content = re.sub(pattern, new_func, content, flags=re.DOTALL)

if new_content == content:
    print("ERROR: pattern did not match — file unchanged")
else:
    target.write_text(new_content, encoding="utf-8")
    print("SUCCESS: plot_lr_statistics replaced")