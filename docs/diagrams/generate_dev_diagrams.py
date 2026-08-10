#!/usr/bin/env python3
"""Generate dependency-free SVG diagrams for the current DEV architecture.

The diagrams intentionally cover only:
  1. StyleContentAttentionFusion (including its outer residual handoff, no G).
  2. StrokePatchD and its matched adaptive crop input (no global D).

Run from any directory:
    python /path/to/dev/docs/diagrams/generate_dev_diagrams.py

The implementation uses only Python's standard library and writes editable SVG.
"""

from __future__ import annotations

from html import escape
from pathlib import Path


FONT = "Inter,Segoe UI,Arial,sans-serif"
INK = "#172033"
MUTED = "#526078"
LINE = "#475569"
PANEL = "#F8FAFC"
PANEL_BORDER = "#CBD5E1"
CONTENT = "#DBEAFE"
CONTENT_BORDER = "#2563EB"
STYLE = "#F3E8FF"
STYLE_BORDER = "#9333EA"
STABLE = "#DCFCE7"
STABLE_BORDER = "#16A34A"
ALLOGRAPH = "#FFEDD5"
ALLOGRAPH_BORDER = "#EA580C"
CRITIC = "#FEE2E2"
CRITIC_BORDER = "#DC2626"
LOSS = "#FEF3C7"
LOSS_BORDER = "#D97706"
WHITE = "#FFFFFF"


class SvgDiagram:
    def __init__(self, width: int, height: int, title: str, subtitle: str):
        self.width = width
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            f'aria-labelledby="diagram-title diagram-desc">',
            f'<title id="diagram-title">{escape(title)}</title>',
            f'<desc id="diagram-desc">{escape(subtitle)}</desc>',
            "<defs>",
            '<marker id="arrow" markerWidth="10" markerHeight="10" '
            'refX="8" refY="5" orient="auto" markerUnits="strokeWidth">',
            f'<path d="M0,0 L10,5 L0,10 z" fill="{LINE}"/>',
            "</marker>",
            '<filter id="soft-shadow" x="-15%" y="-15%" width="130%" height="140%">',
            '<feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0F172A" flood-opacity="0.10"/>',
            "</filter>",
            "</defs>",
            f'<rect width="{width}" height="{height}" fill="{WHITE}"/>',
        ]
        self.text(42, 54, [title], size=30, weight=600, color=INK)
        self.text(42, 86, [subtitle], size=16, color=MUTED)

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: str,
        stroke: str,
        radius: int = 14,
        stroke_width: float = 1.5,
        dashed: bool = False,
        shadow: bool = False,
    ) -> None:
        dash = ' stroke-dasharray="8 6"' if dashed else ""
        filt = ' filter="url(#soft-shadow)"' if shadow else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'rx="{radius}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{stroke_width}"{dash}{filt}/>'
        )

    def text(
        self,
        x: float,
        y: float,
        lines: list[str] | tuple[str, ...],
        *,
        size: int = 16,
        weight: int = 400,
        color: str = INK,
        anchor: str = "start",
        line_height: int | None = None,
    ) -> None:
        line_height = line_height or int(size * 1.32)
        spans = []
        for index, line in enumerate(lines):
            dy = 0 if index == 0 else line_height
            spans.append(
                f'<tspan x="{x}" dy="{dy}">{escape(str(line))}</tspan>'
            )
        self.parts.append(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
            f'font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
            f'fill="{color}">{"".join(spans)}</text>'
        )

    def box(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        title: str,
        lines: list[str] | tuple[str, ...],
        *,
        fill: str,
        stroke: str,
        title_size: int = 18,
        text_size: int = 15,
        shadow: bool = True,
    ) -> None:
        self.rect(
            x, y, width, height, fill=fill, stroke=stroke, shadow=shadow
        )
        self.text(x + 18, y + 30, [title], size=title_size, weight=600)
        self.text(
            x + 18,
            y + 57,
            list(lines),
            size=text_size,
            color=MUTED,
            line_height=int(text_size * 1.45),
        )

    def panel(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        title: str,
        subtitle: str = "",
        *,
        stroke: str = PANEL_BORDER,
    ) -> None:
        self.rect(
            x,
            y,
            width,
            height,
            fill=PANEL,
            stroke=stroke,
            radius=18,
            stroke_width=1.3,
        )
        self.text(x + 20, y + 31, [title], size=19, weight=600)
        if subtitle:
            self.text(x + 20, y + 55, [subtitle], size=14, color=MUTED)

    def arrow(
        self,
        points: list[tuple[float, float]],
        *,
        label: str = "",
        dashed: bool = False,
        color: str = LINE,
        width: float = 2.0,
    ) -> None:
        coords = " ".join(f"{x},{y}" for x, y in points)
        dash = ' stroke-dasharray="7 6"' if dashed else ""
        self.parts.append(
            f'<polyline points="{coords}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linecap="round" '
            f'stroke-linejoin="round" marker-end="url(#arrow)"{dash}/>'
        )
        if label:
            middle = points[len(points) // 2]
            label_width = max(74, len(label) * 7.3)
            self.rect(
                middle[0] - label_width / 2,
                middle[1] - 19,
                label_width,
                25,
                fill=WHITE,
                stroke=WHITE,
                radius=4,
                stroke_width=0,
            )
            self.text(
                middle[0],
                middle[1],
                [label],
                size=13,
                weight=600,
                color=color,
                anchor="middle",
            )

    def footer(self, lines: list[str] | tuple[str, ...]) -> None:
        self.parts.append(
            f'<line x1="42" y1="{self.height - 70}" '
            f'x2="{self.width - 42}" y2="{self.height - 70}" '
            f'stroke="{PANEL_BORDER}" stroke-width="1"/>'
        )
        self.text(
            42,
            self.height - 43,
            list(lines),
            size=13,
            color=MUTED,
            line_height=18,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.parts + ["</svg>"]), encoding="utf-8")


def draw_fusion(path: Path) -> None:
    d = SvgDiagram(
        2040,
        1160,
        "DEV content–style fusion",
        "StyleContentAttentionFusion only — the diagram stops before filter_linear and GBlocks.",
    )

    # Top-level style inputs and split.
    d.box(
        42, 128, 205, 104, "Style code z", ["B × 8 × 32", "token 0 + tokens 1–7"],
        fill=STYLE, stroke=STYLE_BORDER,
    )
    d.box(
        290, 128, 190, 104, "Split tokens", ["global: B × 32", "local: B × 7 × 32"],
        fill=STYLE, stroke=STYLE_BORDER,
    )
    d.box(
        540, 120, 286, 120, "Global style contract",
        ["Conditions content context", "Never used as local K/V"],
        fill=STYLE, stroke=STYLE_BORDER,
    )
    d.box(
        885, 112, 350, 136, "Local-style projection",
        ["LN(32) → bias-free 32→120", "orthogonal base + gated MLP", "projection residual init = 0.10"],
        fill=STYLE, stroke=STYLE_BORDER,
    )
    d.arrow([(247, 180), (290, 180)])
    d.arrow([(480, 161), (540, 161)])
    d.arrow([(480, 207), (700, 207), (700, 270), (885, 270), (885, 206)], label="tokens 1–7")

    # Content inputs.
    d.box(
        42, 445, 205, 110, "Character IDs y", ["B × L", "vocabulary = 80"],
        fill=CONTENT, stroke=CONTENT_BORDER,
    )
    d.box(
        42, 592, 205, 100, "Valid lengths", ["y_lens → token mask"],
        fill=STABLE, stroke=STABLE_BORDER,
    )
    d.box(
        290, 445, 190, 110, "Text embedding", ["80 → 120", "content: B × L × 120"],
        fill=CONTENT, stroke=CONTENT_BORDER,
    )
    d.arrow([(247, 500), (290, 500)])

    # Stage 1.
    d.panel(
        525, 315, 370, 500, "1  Global content context",
        "Style-conditioned self-attention; no local-style K/V.",
        stroke=CONTENT_BORDER,
    )
    d.box(
        558, 392, 304, 102, "Bounded style modulation",
        ["global style → shift/scale", "conditioning limit = ±0.50"],
        fill=STYLE, stroke=STYLE_BORDER, shadow=False,
    )
    d.box(
        558, 520, 304, 112, "4-head self-attention",
        ["affine-free LN + relative bias", "attention residual init = 0.25"],
        fill=CONTENT, stroke=CONTENT_BORDER, shadow=False,
    )
    d.box(
        558, 658, 304, 112, "Gated FFN residual",
        ["Linear → SiLU gate → Linear", "FFN residual init = 0.25"],
        fill=CONTENT, stroke=CONTENT_BORDER, shadow=False,
    )
    d.arrow([(710, 494), (710, 520)])
    d.arrow([(710, 632), (710, 658)])
    d.arrow([(480, 500), (525, 500)])
    d.arrow([(683, 240), (683, 315)], label="global style")

    # Stage 2.
    d.panel(
        930, 380, 260, 372, "2  Local continuity",
        "Character-neighborhood smoothing.",
        stroke=STABLE_BORDER,
    )
    d.box(
        960, 452, 200, 88, "Depthwise Conv1D", ["kernel 5, 120 groups"],
        fill=STABLE, stroke=STABLE_BORDER, shadow=False,
    )
    d.box(
        960, 564, 200, 96, "Pointwise + selector", ["SiLU projection", "content-dependent gate"],
        fill=STABLE, stroke=STABLE_BORDER, shadow=False,
    )
    d.box(
        960, 684, 200, 48, "Residual init = 0.25", [],
        fill=STABLE, stroke=STABLE_BORDER, title_size=15, shadow=False,
    )
    d.arrow([(895, 576), (930, 576)])
    d.arrow([(1060, 540), (1060, 564)])
    d.arrow([(1060, 660), (1060, 684)])

    # Stage 3 with actual routing and bounded modulation.
    d.panel(
        1225, 286, 500, 624, "3  Allograph routing and modulation",
        "The only content-to-local-style attention in DEV fusion.",
        stroke=ALLOGRAPH_BORDER,
    )
    d.box(
        1255, 372, 205, 100, "Content query",
        ["bias-free Q projection", "cosine normalization"],
        fill=CONTENT, stroke=CONTENT_BORDER, shadow=False,
    )
    d.box(
        1485, 372, 205, 100, "Centered style keys",
        ["local − mean(local)", "bias-free cosine K"],
        fill=STYLE, stroke=STYLE_BORDER, shadow=False,
    )
    d.box(
        1255, 505, 205, 108, "Character prior",
        ["character embedding", "+ content-context query"],
        fill=ALLOGRAPH, stroke=ALLOGRAPH_BORDER, shadow=False,
    )
    d.box(
        1485, 505, 205, 108, "Routing softmax",
        ["content score + char score", "temperature = 0.50"],
        fill=ALLOGRAPH, stroke=ALLOGRAPH_BORDER, shadow=False,
    )
    d.box(
        1255, 648, 205, 110, "Uncentered values",
        ["weighted local-style sum", "retain writer-common evidence"],
        fill=STYLE, stroke=STYLE_BORDER, shadow=False,
    )
    d.box(
        1485, 648, 205, 110, "Bounded modulation",
        ["char gain ±0.25", "RMS cap 1.0; tanh ±0.30"],
        fill=STABLE, stroke=STABLE_BORDER, shadow=False,
    )
    d.box(
        1370, 793, 205, 82, "Residual modulation",
        ["initial strength = 0.50"],
        fill=STABLE, stroke=STABLE_BORDER, shadow=False,
    )
    d.arrow([(1460, 422), (1485, 422)])
    d.arrow([(1357, 472), (1357, 505)])
    d.arrow([(1460, 559), (1485, 559)])
    d.arrow([(1587, 613), (1587, 648)])
    d.arrow([(1460, 703), (1485, 703)])
    d.arrow([(1587, 758), (1587, 777), (1472, 777), (1472, 793)])
    d.arrow([(1190, 576), (1225, 576)])
    d.arrow([(1060, 248), (1060, 272), (1587, 272), (1587, 372)], label="7 local tokens")
    d.arrow([(247, 475), (265, 475), (265, 558), (1255, 558)], label="character IDs")

    # Outer fusion handoff (part of Generator.forward, but no generator shown).
    d.panel(
        1760, 400, 238, 330, "Fusion handoff",
        "Stops before the generator.",
        stroke=STABLE_BORDER,
    )
    d.box(
        1783, 476, 192, 126, "Outer residual gate",
        ["content + σ(g)·", "(fused − content)", "initial gate = 0.25"],
        fill=STABLE, stroke=STABLE_BORDER, shadow=False,
    )
    d.box(
        1783, 632, 192, 72, "Output", ["B × L × 120"],
        fill=LOSS, stroke=LOSS_BORDER, shadow=False,
    )
    d.arrow([(1575, 834), (1738, 834), (1738, 539), (1783, 539)], label="fused")
    d.arrow([(480, 500), (500, 500), (500, 945), (1748, 945), (1748, 572), (1783, 572)], label="raw content")
    d.arrow([(1879, 602), (1879, 632)])

    # Mask lane and legend.
    d.arrow([(247, 642), (505, 642), (505, 840), (1880, 840), (1880, 730)], label="padding mask", dashed=True, color=STABLE_BORDER)

    d.box(
        525, 888, 315, 92, "Global/local separation",
        ["global style: Stage 1 only", "local style: Stage 3 only"],
        fill=PANEL, stroke=PANEL_BORDER, shadow=False,
    )
    d.box(
        880, 888, 315, 92, "Identity-safe initialization",
        ["all new branches start as", "small non-zero residuals"],
        fill=PANEL, stroke=PANEL_BORDER, shadow=False,
    )

    d.footer([
        "Source: dev/networks/fusion.py::StyleContentAttentionFusion and AllographicModulation;",
        "outer residual handoff: dev/networks/BigGAN_networks.py::Generator.forward. No GBlock is included.",
    ])
    d.save(path)


def draw_stroke_patch_d(path: Path) -> None:
    d = SvgDiagram(
        2040,
        1130,
        "DEV StrokePatchD",
        "Matched adaptive 32×32 crops and the lightweight stroke-specialized local critic. Global D is omitted.",
    )

    d.box(
        42, 135, 245, 196, "Patch sources",
        ["real word", "augmented real", "random fake", "style-transfer fake", "reconstruction fake"],
        fill=CONTENT, stroke=CONTENT_BORDER,
    )
    d.box(
        335, 132, 310, 202, "Matched adaptive sampler",
        ["patch size = 32×32", "count = ceil(valid_width/32)", "clamped to 4–8 per image", "stratified horizontal coverage", "alternating upper/lower bands"],
        fill=STABLE, stroke=STABLE_BORDER,
    )
    d.box(
        695, 150, 238, 166, "Optional local mask",
        ["masking_mode = combined", "light patch-only occlusion", "never applied to style input"],
        fill=STABLE, stroke=STABLE_BORDER,
    )
    d.box(
        980, 165, 180, 134, "Patch tensor",
        ["N × 1 × 32 × 32", "same policy for real/fake"],
        fill=STYLE, stroke=STYLE_BORDER,
    )
    d.arrow([(287, 232), (335, 232)])
    d.arrow([(645, 232), (695, 232)])
    d.arrow([(933, 232), (980, 232)])

    # Main PatchD pipeline.
    d.panel(
        1200, 114, 798, 266, "StrokePatchD forward path",
        "Spectral normalization is used on every convolution.",
        stroke=CRITIC_BORDER,
    )
    d.box(
        1230, 191, 135, 112, "Stem", ["3×3 SNConv", "1 → 32"],
        fill=CRITIC, stroke=CRITIC_BORDER, shadow=False,
    )
    d.box(
        1392, 191, 160, 112, "Block 1", ["32 → 64", "32² → 16²"],
        fill=CRITIC, stroke=CRITIC_BORDER, shadow=False,
    )
    d.box(
        1578, 191, 160, 112, "Block 2", ["64 → 128", "16² → 8²"],
        fill=CRITIC, stroke=CRITIC_BORDER, shadow=False,
    )
    d.box(
        1764, 191, 160, 112, "Block 3", ["128 → 192", "8² → 4²"],
        fill=CRITIC, stroke=CRITIC_BORDER, shadow=False,
    )
    d.box(
        1796, 326, 128, 40, "1×1 logits", [],
        fill=LOSS, stroke=LOSS_BORDER, title_size=14, shadow=False,
    )
    d.arrow([(1160, 232), (1200, 232)])
    d.arrow([(1365, 247), (1392, 247)])
    d.arrow([(1552, 247), (1578, 247)])
    d.arrow([(1738, 247), (1764, 247)])
    d.arrow([(1844, 303), (1844, 326)])

    # Detailed internal block.
    d.panel(
        310, 425, 1215, 530, "Inside each StrokePatchBlock",
        "Anisotropic depthwise paths explicitly model horizontal and vertical handwriting strokes.",
        stroke=STYLE_BORDER,
    )
    d.box(
        350, 535, 155, 90, "Input x", ["Cin × H × W"],
        fill=CONTENT, stroke=CONTENT_BORDER, shadow=False,
    )
    d.box(
        560, 492, 190, 106, "Main projection", ["LeakyReLU", "3×3 SNConv", "Cin → Cout"],
        fill=CRITIC, stroke=CRITIC_BORDER, shadow=False,
    )
    d.box(
        800, 462, 200, 94, "Horizontal branch", ["depthwise 1×5", "stroke flow / joins"],
        fill=STYLE, stroke=STYLE_BORDER, shadow=False,
    )
    d.box(
        800, 585, 200, 94, "Vertical branch", ["depthwise 5×1", "ascender / descender"],
        fill=STYLE, stroke=STYLE_BORDER, shadow=False,
    )
    d.box(
        1050, 522, 190, 112, "Oriented fusion", ["horizontal + vertical", "LeakyReLU", "1×1 SNConv"],
        fill=STYLE, stroke=STYLE_BORDER, shadow=False,
    )
    d.box(
        1280, 522, 205, 112, "Main residual", ["projected h +", "oriented correction", "AvgPool2d(2)"],
        fill=STABLE, stroke=STABLE_BORDER, shadow=False,
    )

    d.box(
        560, 725, 190, 104, "Shortcut", ["1×1 SNConv", "Cin → Cout", "AvgPool2d(2)"],
        fill=STABLE, stroke=STABLE_BORDER, shadow=False,
    )
    d.box(
        1050, 725, 190, 104, "Residual sum", ["downsampled main", "+ downsampled shortcut"],
        fill=STABLE, stroke=STABLE_BORDER, shadow=False,
    )
    d.box(
        1280, 725, 205, 104, "Block output", ["Cout × H/2 × W/2"],
        fill=LOSS, stroke=LOSS_BORDER, shadow=False,
    )

    d.arrow([(505, 566), (560, 545)])
    d.arrow([(505, 596), (535, 596), (535, 777), (560, 777)])
    d.arrow([(750, 545), (775, 545), (775, 509), (800, 509)])
    d.arrow([(750, 565), (775, 565), (775, 632), (800, 632)])
    d.arrow([(1000, 509), (1025, 509), (1025, 558), (1050, 558)])
    d.arrow([(1000, 632), (1025, 632), (1025, 598), (1050, 598)])
    d.arrow([(1240, 578), (1280, 578)])
    d.arrow([(750, 777), (1050, 777)])
    d.arrow([(1240, 777), (1280, 777)])
    d.arrow([(1382, 634), (1382, 690), (1150, 690), (1150, 725)], label="main")

    # Patch-only hinge objective.
    d.panel(
        1570, 425, 428, 530, "Patch adversarial objective",
        "The local critic returns spatial 4×4 logits.",
        stroke=LOSS_BORDER,
    )
    d.box(
        1605, 512, 358, 105, "Real patch hinge",
        ["mean ReLU(1 − P(real))", "real + augmented-real groups"],
        fill=LOSS, stroke=LOSS_BORDER, shadow=False,
    )
    d.box(
        1605, 648, 358, 120, "Fake patch hinge",
        ["mean ReLU(1 + P(fake))", "random + transfer + reconstruction", "groups receive equal semantic weight"],
        fill=LOSS, stroke=LOSS_BORDER, shadow=False,
    )
    d.box(
        1605, 800, 358, 105, "Contribution to training",
        ["D: 0.75·(real_patch + fake_patch)", "G: −0.75·mean P(fake)"],
        fill=STABLE, stroke=STABLE_BORDER, shadow=False,
    )
    d.arrow([(1924, 346), (1978, 346), (1978, 475), (1784, 475), (1784, 512)], label="4×4 logits")
    d.arrow([(1784, 617), (1784, 648)])
    d.arrow([(1784, 768), (1784, 800)])

    d.footer([
        "Source: dev/networks/BigGAN_networks.py::StrokePatchBlock and PatchDiscriminator;",
        "crop policy: dev/networks/utils.py::sample_adaptive_patches; patch weight: dev/configs/gan_iam.yml. Global D is omitted.",
    ])
    d.save(path)


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    fusion_path = output_dir / "dev_fusion.svg"
    patch_path = output_dir / "dev_strokepatchd.svg"
    draw_fusion(fusion_path)
    draw_stroke_patch_d(patch_path)
    print(f"wrote {fusion_path}")
    print(f"wrote {patch_path}")


if __name__ == "__main__":
    main()
