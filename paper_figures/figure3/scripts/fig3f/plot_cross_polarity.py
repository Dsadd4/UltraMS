"""Publication-size vector figure, loaded only from the frozen local plot tables."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source_data"
OUT = HERE / "outputs"
MM = 1 / 25.4
W, H = 210, 62
ULTRA, DREAMS = "#3965BB", "#E9A0AA"
DREAMS_STROKE = "#AE6473"


def main():
    OUT.mkdir(exist_ok=True)
    assert json.loads((SOURCE / "SOURCE_BINDING.json").read_text())["status"] == "pass"
    points = pd.read_csv(SOURCE / "recall_points.csv", float_precision="round_trip")
    paired = pd.read_csv(SOURCE / "paired_points.csv", float_precision="round_trip")
    plt.rcParams.update({"font.family": "Arial", "font.size": 6.5, "axes.labelsize": 6.5,
        "axes.titlesize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black", "axes.edgecolor": "black",
        "axes.linewidth": .55, "xtick.major.width": .55, "ytick.major.width": .55,
        "xtick.major.size": 2.2, "ytick.major.size": 2.2, "xtick.minor.visible": False,
        "ytick.minor.visible": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "savefig.facecolor": "white", "figure.facecolor": "white"})
    fig = plt.figure(figsize=(W * MM, H * MM))
    boxes = [[14, 13, 42, 34], [64, 13, 42, 34], [120, 13, 40, 34], [168, 13, 40, 34]]
    axes = [fig.add_axes([x/W, y/H, w/W, h/H]) for x, y, w, h in boxes]
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out", pad=2)
        ax.grid(False)
    directions = ["pos_to_neg", "neg_to_pos"]
    titles = ["Positive → negative", "Negative → positive"]
    for index, (direction, title) in enumerate(zip(directions, titles)):
        ax = axes[index]
        for model, color, stroke, marker, line in (("ultra", ULTRA, ULTRA, "o", "-"),
                                                  ("dreams", DREAMS, DREAMS_STROKE, "s", (0, (2.2, 1.4)))):
            frame = points[(points.direction == direction) & (points.model == model)].sort_values("k")
            assert frame.k.tolist() == [1, 5, 10]
            ax.plot(frame.k, frame.recall_percent, color=stroke, marker=marker,
                    markersize=3.5, markerfacecolor=color, markeredgecolor=stroke,
                    markeredgewidth=.55, linewidth=1.05, linestyle=line, clip_on=False)
            last = frame.iloc[-1]
            ax.annotate(f"{last.recall_percent:.1f}", (10, last.recall_percent), xytext=(4, 0),
                        textcoords="offset points", ha="left", va="center", color="black")
        ax.set(xlim=(.3, 12.3), ylim=(0, 75), xticks=[1, 5, 10], yticks=[0, 25, 50, 75])
        ax.set_xlabel("Top-k candidates", labelpad=3)
        if index == 0:
            ax.set_ylabel("Reference-supported\nrecall (%)", labelpad=4)
        ax.set_title(title, pad=5)
    for index, (direction, title) in enumerate(zip(directions, titles)):
        ax = axes[index + 2]
        frame = paired[paired.direction == direction].sort_values("k")
        assert frame.k.tolist() == [1, 5, 10]
        for y, row in enumerate(frame.itertuples(index=False)):
            u, d = row.dreams_wrong_ultra_correct, row.dreams_correct_ultra_wrong
            assert u - d == row.net_new_correct
            ax.barh(y, u, left=0, height=.34, facecolor=ULTRA, edgecolor=ULTRA,
                    linewidth=.55, zorder=2)
            ax.barh(y, -d, left=0, height=.34, facecolor=DREAMS, edgecolor=DREAMS_STROKE,
                    linewidth=.55, zorder=2)
            ax.annotate(f"+{u}", (u, y), xytext=(3, 0), textcoords="offset points",
                        ha="left", va="center", color="black")
            ax.annotate(f"−{d}", (-d, y), xytext=(-3, 0), textcoords="offset points",
                        ha="right", va="center", color="black")
            ax.annotate(f"Net +{row.net_new_correct}", (0, y), xytext=(5, 8.5),
                        textcoords="offset points", ha="left", va="bottom", color="black")
        ax.axvline(0, color="black", linewidth=.55, zorder=3)
        ax.set(xlim=(-550, 1050), ylim=(2.55, -.55), xticks=[-400, 0, 400, 800], yticks=[0, 1, 2], yticklabels=[1, 5, 10])
        ax.set_xlabel("Paired hit change", labelpad=3)
        if index == 0:
            ax.set_ylabel("Top-k cutoff", labelpad=4)
        ax.set_title(title, pad=5)
    common = [Line2D([], [], color=ULTRA, marker="o", markersize=3.5, linewidth=1.05, label="UltraMS"),
              Line2D([], [], color=DREAMS_STROKE, marker="s", markerfacecolor=DREAMS,
                     markersize=3.5, linewidth=1.05, linestyle=(0, (2.2, 1.4)), label="DreaMS")]
    exclusive = [Patch(facecolor=ULTRA, edgecolor=ULTRA, linewidth=.55, label="Gained with UltraMS"),
                 Patch(facecolor=DREAMS, edgecolor=DREAMS_STROKE, linewidth=.55, label="Lost with UltraMS")]
    fig.legend(handles=common, loc="center", bbox_to_anchor=(60/W, 57/H), ncol=2,
               frameon=False, handlelength=1.8, handletextpad=.5, columnspacing=1.5, borderaxespad=0)
    fig.legend(handles=exclusive, loc="center", bbox_to_anchor=(162/W, 57/H), ncol=2,
               frameon=False, handlelength=1.1, handletextpad=.4, columnspacing=1.0, borderaxespad=0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = []
    for obj in fig.findobj(matplotlib.text.Text):
        if obj.get_visible() and obj.get_text():
            box = obj.get_window_extent(renderer)
            texts.append(dict(text=obj.get_text(), bbox_px=[float(v) for v in box.bounds], fontsize=float(obj.get_fontsize())))
    (OUT / "layout_text_boxes.json").write_text(json.dumps(dict(canvas_px=list(fig.canvas.get_width_height()), texts=texts), indent=2) + "\n")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"cross_polarity_retrieval.{ext}", dpi=500, facecolor="white", bbox_inches=None,
                    metadata={"Creator": "UltraMS publication figure"} if ext == "pdf" else None)
    settings = dict(size_mm=[W, H], axes_mm=boxes, font="Arial", font_pt=6.5, png_dpi=500,
        palette=dict(UltraMS=ULTRA, DreaMS_fill=DREAMS, DreaMS_stroke=DREAMS_STROKE),
        recall_ylim=[0, 75], exclusive_xlim=[-550, 1050], input_mass_filter=False,
        paired_chart="zero-centered horizontal gain/loss bars; no dumbbell or lollipop marks",
        point_estimates_only=True, fixed_k=[1, 5, 10], panel_letters=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT / "PLOT_SETTINGS.json").write_text(json.dumps(settings, indent=2) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
