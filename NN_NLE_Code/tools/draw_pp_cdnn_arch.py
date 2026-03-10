from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def add_block(ax, x, y, w, h, title, subtitle="", fc="#F7FAFF", ec="#3A4A6B", lw=1.6, fs=11):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=fs, weight="bold", color="#1C2A44")
    if subtitle:
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center", fontsize=fs - 1, color="#32415F")
    return box


def arrow(ax, x1, y1, x2, y2, color="#2D3B55", lw=1.8, style="-|>"):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=2, shrinkB=2),
    )


def main():
    out_dir = Path(r"D:\paperwork\cursor\figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(19, 8), dpi=220)
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.92])
    ax.set_xlim(0, 1.18)
    ax.set_ylim(0, 1.0)
    ax.axis("off")

    ax.text(
        0.62,
        0.965,
        "PP-CDNN: Physics-Prior Cascaded Dual-Head Architecture",
        ha="center",
        va="center",
        fontsize=19,
        weight="bold",
        color="#10233C",
    )
    ax.text(
        0.62,
        0.928,
        "Model: PP-CDNN.py  |  Block-wise alignment (stride=G) + PCM-prior cascaded PAM decision",
        ha="center",
        va="center",
        fontsize=11,
        color="#3B4E6D",
    )

    # Block-wise preprocessing + Backbone row
    add_block(
        ax, 0.03, 0.37, 0.14, 0.28, "Rx 1-SPS Sequence",
        "dnn_*_input", fc="#FFF8E8", ec="#9E6A00"
    )
    add_block(
        ax, 0.20, 0.37, 0.16, 0.28, "Block-wise Windowing",
        "window=2*seq_len+1\nstride=G (=quant/2)", fc="#FFF1D6", ec="#B57900"
    )
    add_block(ax, 0.40, 0.30, 0.09, 0.42, "Shared Block 1", "Linear -> 256\nBN + GELU", fc="#DDF8E8", ec="#1F7A4A")
    add_block(ax, 0.53, 0.30, 0.09, 0.42, "Shared Block 2", "Linear -> 128\nBN + GELU", fc="#DDF8E8", ec="#1F7A4A")
    add_block(ax, 0.66, 0.30, 0.09, 0.42, "Shared Block 3", "Linear -> 64\nBN + GELU", fc="#DDF8E8", ec="#1F7A4A")
    ax.text(0.705, 0.75, "x_shared (64)", fontsize=10, color="#1F7A4A", ha="center")

    # Upper branch: PCM
    add_block(ax, 0.82, 0.64, 0.14, 0.16, "PCM Head", "Linear 64->1 + Tanh", fc="#DDEEFF", ec="#20639B")
    add_block(ax, 1.00, 0.64, 0.16, 0.16, "PCM Output", "pcm_out  [B,1]\n(one per block)", fc="#DDEEFF", ec="#20639B")
    add_block(ax, 1.00, 0.47, 0.15, 0.10, "L_pcm", "MSE(pcm_out, pcm_ref)", fc="#EEF6FF", ec="#5E84AB", fs=10)

    # Lower branch: gated fusion + PAM
    add_block(ax, 0.82, 0.38, 0.14, 0.12, "Gating", "g = x_shared * pcm_out", fc="#EEE6FF", ec="#6A49A3")
    add_block(ax, 0.82, 0.16, 0.22, 0.16, "PAM Fusion", "concat[x_shared, pcm_out, g]\nLinear 129->96->G", fc="#FFDDE4", ec="#B23A48")
    add_block(ax, 1.07, 0.16, 0.10, 0.16, "PAM Output", "pam_out  [B,G]\n(G symbols/block)", fc="#FFDDE4", ec="#B23A48", fs=9)
    add_block(ax, 1.01, 0.05, 0.15, 0.09, "L_pam", "MSE(pam_out, pam_label)", fc="#FFF0F4", ec="#B26673", fs=10)

    # Adaptive weighting block
    add_block(
        ax,
        0.84,
        0.84,
        0.26,
        0.10,
        "Adaptive Multi-Task Weighting",
        "auto_uncertainty + EMA norm + PAM protection",
        fc="#F3F7FB",
        ec="#4E6078",
        fs=9,
    )

    # Main flow arrows
    arrow(ax, 0.17, 0.51, 0.20, 0.51)
    arrow(ax, 0.36, 0.51, 0.40, 0.51)
    arrow(ax, 0.49, 0.51, 0.53, 0.51)
    arrow(ax, 0.62, 0.51, 0.66, 0.51)

    # Branch arrows from shared feature
    arrow(ax, 0.75, 0.57, 0.82, 0.72)  # to PCM head
    arrow(ax, 0.75, 0.47, 0.82, 0.44)  # to gating
    arrow(ax, 0.75, 0.43, 0.82, 0.24)  # to PAM fusion direct

    # PCM to output and downstream
    arrow(ax, 0.96, 0.72, 1.00, 0.72)
    arrow(ax, 1.075, 0.64, 1.075, 0.57, color="#5E84AB", lw=1.3)  # to L_pcm
    arrow(ax, 1.00, 0.66, 0.96, 0.44)  # pcm_out to gating
    arrow(ax, 1.00, 0.66, 0.95, 0.24)  # pcm_out to fusion

    # Gating to fusion and pam output
    arrow(ax, 0.89, 0.38, 0.89, 0.32)
    arrow(ax, 1.04, 0.24, 1.07, 0.24)
    arrow(ax, 1.12, 0.16, 1.095, 0.14, color="#B26673", lw=1.3)  # to L_pam

    # Loss to weighting
    arrow(ax, 1.03, 0.57, 0.97, 0.89, color="#4E6078", lw=1.2)
    arrow(ax, 1.03, 0.14, 0.97, 0.86, color="#4E6078", lw=1.2)

    # Branch labels
    ax.text(0.93, 0.82, "Upper branch: PCM prior estimation", fontsize=10, color="#20639B", ha="center")
    ax.text(0.96, 0.33, "Lower branch: PAM decision with cascaded prior", fontsize=10, color="#B23A48", ha="center")

    # Bottom note
    ax.text(
        0.04,
        0.015,
        "Block-level semantics are explicit: each sample corresponds to one physical block (G PAM symbols + 1 PCM target).",
        fontsize=9.5,
        color="#344D6E",
    )

    png_path = out_dir / "PP-CDNN_architecture.png"
    svg_path = out_dir / "PP-CDNN_architecture.svg"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:\n- {png_path}\n- {svg_path}")


if __name__ == "__main__":
    main()
