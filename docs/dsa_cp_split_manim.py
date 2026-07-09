"""
DSA Prefill CP Split Visualization — Manim Community Edition

Renders 3 scenes as MP4:
  Scene 1 (RoundRobinSplit): token-level stride interleave animation
  Scene 2 (InSeqZigzagSplit): block-level zigzag head-tail pairing animation
  Scene 3 (SplitComparison): side-by-side comparison with full detail

Parameters: seq_len=16, cp_size=4 (matching dsa_prefill_cp_split_visualization.ipynb)

Usage:
  manim -qm --media_dir docs/media docs/dsa_cp_split_manim.py RoundRobinSplit
  manim -qm --media_dir docs/media docs/dsa_cp_split_manim.py InSeqZigzagSplit
  manim -qm --media_dir docs/media docs/dsa_cp_split_manim.py SplitComparison
"""

from manim import *
import numpy as np
import html as _html

_ManimText = Text

def Text(text, font_size=14, color=WHITE, weight=NORMAL, **kwargs):
    escaped = _html.escape(str(text))
    is_bold = weight not in (NORMAL, None, "normal", "NORMAL")
    w = "bold" if is_bold else "normal"
    markup = (f'<span font_family="Times New Roman" '
              f'letter_spacing="-400" weight="{w}">{escaped}</span>')
    return MarkupText(markup, font_size=font_size, color=color, **kwargs)

# ─── Constants ───────────────────────────────────────────────
SEQ_LEN = 16
CP_SIZE = 4
TOKENS_PER_RANK = SEQ_LEN // CP_SIZE  # 4
NUM_BLOCKS = 2 * CP_SIZE              # 8
BLOCK_SIZE = SEQ_LEN // NUM_BLOCKS    # 2

RANK_COLORS = [
    "#7A95AE",  # rank 0 — muted steel blue
    "#B89A88",  # rank 1 — muted terracotta
    "#7BAA8E",  # rank 2 — muted sage
    "#A87A7C",  # rank 3 — muted rose
]
RANK_COLORS_LIGHT = ["#A0B5C6", "#D4BCA6", "#A0C4AC", "#C4A0A2"]
NEUTRAL = "#555577"
BG = "#1a1a2e"

FRAME_W = config.frame_width
FRAME_H = config.frame_height
SAFE_W = FRAME_W - 1.2
SAFE_H = FRAME_H - 0.8

# ─── Round-Robin data ────────────────────────────────────────
rr_ranks = {r: [t for t in range(SEQ_LEN) if t % CP_SIZE == r]
            for r in range(CP_SIZE)}

# ─── In-Seq (zigzag) data ───────────────────────────────────
zigzag_pairs = {r: (r, 2 * CP_SIZE - 1 - r) for r in range(CP_SIZE)}
inseq_token_rank = {}
for r in range(CP_SIZE):
    bp, bn = zigzag_pairs[r]
    for t in range(bp * BLOCK_SIZE, (bp + 1) * BLOCK_SIZE):
        inseq_token_rank[t] = r
    for t in range(bn * BLOCK_SIZE, (bn + 1) * BLOCK_SIZE):
        inseq_token_rank[t] = r
inseq_ranks = {r: sorted(t for t, rk in inseq_token_rank.items() if rk == r)
               for r in range(CP_SIZE)}


# ─── Helpers ─────────────────────────────────────────────────

_TEX_TEMPLATE = TexTemplate()
_TEX_TEMPLATE.body = (
    r"\documentclass[preview]{article}" "\n"
    r"\usepackage{amsmath}" "\n"
    r"\usepackage{amssymb}" "\n"
    r"\begin{document}" "\n"
    r"YourTextHere" "\n"
    r"\end{document}"
)


def latex_label(tex_str, font_size=24, color="#88aacc"):
    return MathTex(tex_str, font_size=font_size, color=color,
                   tex_template=_TEX_TEMPLATE)


def make_token_cell(label, color=NEUTRAL, w=0.45, h=0.45):
    rect = RoundedRectangle(
        width=w, height=h, corner_radius=0.07,
        fill_color=color, fill_opacity=1,
        stroke_color=WHITE, stroke_width=1.2,
    )
    txt = Text(str(label), font_size=16, color=WHITE, weight=BOLD)
    txt.move_to(rect.get_center())
    return VGroup(rect, txt)


def make_token_row(tokens, color_fn=None, w=0.45, h=0.45, gap=0.06):
    cells = VGroup()
    for i, t in enumerate(tokens):
        c = color_fn(t) if color_fn else NEUTRAL
        cell = make_token_cell(t, color=c, w=w, h=h)
        cells.add(cell)
    cells.arrange(RIGHT, buff=gap)
    return cells


def fit_to_safe(mob, max_w=None, max_h=None):
    mw = max_w or SAFE_W
    mh = max_h or SAFE_H
    if mob.width > mw:
        mob.scale_to_fit_width(mw)
    if mob.height > mh:
        mob.scale_to_fit_height(mh)
    return mob


# ═════════════════════════════════════════════════════════════
# Scene 1: Round-Robin Split
# ═════════════════════════════════════════════════════════════

class RoundRobinSplit(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BG

        # ── Title ──
        title = Text("DSA Context Parallel: Round-Robin Split",
                      font_size=32, color=WHITE, weight=BOLD)
        subtitle = Text(
            f"seq_len={SEQ_LEN}   cp_size={CP_SIZE}   rule: token[i] → rank(i % {CP_SIZE})",
            font_size=18, color="#aaaacc")
        hdr = VGroup(title, subtitle).arrange(DOWN, buff=0.1)
        hdr.to_edge(UP, buff=0.3)
        fit_to_safe(hdr)
        self.play(Write(title), FadeIn(subtitle), run_time=1)

        # ── Original sequence ──
        orig_label = Text("Original Sequence (input_ids)", font_size=18, color="#ccccdd")
        orig_row = make_token_row(range(SEQ_LEN), w=0.42, h=0.42, gap=0.04)
        fit_to_safe(orig_row)
        orig_row.next_to(subtitle, DOWN, buff=0.4)
        orig_label.next_to(orig_row, UP, buff=0.12)

        self.play(FadeIn(orig_label), LaggedStart(
            *[FadeIn(c, shift=UP * 0.2) for c in orig_row], lag_ratio=0.04),
            run_time=1.5)
        self.wait(0.5)

        # ── Color tokens by rank & show formula ──
        color_anims = []
        for idx, cell in enumerate(orig_row):
            rank = idx % CP_SIZE
            color_anims.append(cell[0].animate.set_fill(RANK_COLORS[rank]))
        formula = latex_label(
            r"\text{view}(-1,\ \text{cp\_size},\ *\text{shape}[1:])[:, \text{cp\_rank}]",
            font_size=22)
        formula.next_to(orig_row, DOWN, buff=0.2)
        self.play(*color_anims, FadeIn(formula), run_time=1)
        self.wait(0.5)

        # ── Build rank rows (targets) ──
        rank_groups = VGroup()
        rank_labels = VGroup()
        for r in range(CP_SIZE):
            row = make_token_row(rr_ranks[r],
                                  color_fn=lambda t, r=r: RANK_COLORS[r],
                                  w=0.42, h=0.42, gap=0.04)
            lbl = Text(f"Rank {r}", font_size=16, color=RANK_COLORS[r], weight=BOLD)
            rank_groups.add(row)
            rank_labels.add(lbl)

        rank_groups.arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        rank_groups.next_to(formula, DOWN, buff=0.45)
        for lbl, row in zip(rank_labels, rank_groups):
            lbl.next_to(row, LEFT, buff=0.25)

        # Check total content height, auto-zoom if needed
        all_content = VGroup(hdr, orig_label, orig_row, formula,
                             rank_groups, rank_labels)
        if all_content.height > FRAME_H - 0.5:
            self.play(self.camera.auto_zoom(all_content, margin=0.4), run_time=0.5)

        # ── Animate tokens flying to rank rows ──
        self.play(FadeOut(formula), run_time=0.3)

        fly_anims = []
        for r in range(CP_SIZE):
            for j, t in enumerate(rr_ranks[r]):
                src = orig_row[t]
                tgt = rank_groups[r][j]
                fly_anims.append(
                    AnimationGroup(
                        src.animate.move_to(tgt.get_center()).set_opacity(0.25),
                        FadeIn(tgt, shift=DOWN * 0.1),
                        lag_ratio=0.3,
                    )
                )

        self.play(
            LaggedStart(*[FadeIn(lbl) for lbl in rank_labels], lag_ratio=0.1),
            run_time=0.5)
        self.play(
            LaggedStart(*fly_anims, lag_ratio=0.06), run_time=3)
        self.wait(0.3)

        # ── Annotations ──
        annotations = VGroup()
        for r in range(CP_SIZE):
            tokens_str = ", ".join(str(t) for t in rr_ranks[r])
            ann = Text(f"idx%4={r}: [{tokens_str}]",
                       font_size=12, color=RANK_COLORS_LIGHT[r])
            ann.next_to(rank_groups[r], RIGHT, buff=0.3)
            annotations.add(ann)
        self.play(LaggedStart(
            *[FadeIn(a) for a in annotations], lag_ratio=0.1), run_time=1)

        # ── Bottom summary box ──
        box_text = Text(
            "Pattern: stride interleave — token i → rank(i mod cp_size)",
            font_size=18, color="#aaddff")
        box = SurroundingRectangle(box_text, color="#4466aa", fill_color="#222244",
                                    fill_opacity=0.85, corner_radius=0.12, buff=0.2)
        box_group = VGroup(box, box_text)
        fn_text = latex_label(
            r"\text{Rerange: view} \to \text{transpose} \to \text{reshape}",
            font_size=20, color="#88aacc")
        bottom = VGroup(box_group, fn_text).arrange(DOWN, buff=0.1)
        bottom.next_to(rank_groups, DOWN, buff=0.35)
        fit_to_safe(bottom)

        # Re-zoom to fit everything
        all_final = VGroup(hdr, orig_label, orig_row, rank_groups, rank_labels,
                           annotations, bottom)
        self.play(self.camera.auto_zoom(all_final, margin=0.3), run_time=0.5)
        self.play(FadeIn(box_group), FadeIn(fn_text), run_time=1)
        self.wait(2)


# ═════════════════════════════════════════════════════════════
# Scene 2: In-Seq Zigzag Split
# ═════════════════════════════════════════════════════════════

class InSeqZigzagSplit(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BG

        # ── Title ──
        title = Text("DSA Context Parallel: In-Seq Split (Zigzag)",
                      font_size=32, color=WHITE, weight=BOLD)
        subtitle = Text(
            f"seq_len={SEQ_LEN}   cp_size={CP_SIZE}   "
            f"blocks=2×cp_size={NUM_BLOCKS}   block_size={BLOCK_SIZE}",
            font_size=18, color="#aaaacc")
        hdr = VGroup(title, subtitle).arrange(DOWN, buff=0.1)
        hdr.to_edge(UP, buff=0.3)
        fit_to_safe(hdr)
        self.play(Write(title), FadeIn(subtitle), run_time=1)

        # ── Original sequence ──
        orig_label = Text("Original Sequence (input_ids)", font_size=18, color="#ccccdd")
        orig_row = make_token_row(range(SEQ_LEN), w=0.42, h=0.42, gap=0.04)
        fit_to_safe(orig_row)
        orig_row.next_to(subtitle, DOWN, buff=0.4)
        orig_label.next_to(orig_row, UP, buff=0.12)
        self.play(FadeIn(orig_label), LaggedStart(
            *[FadeIn(c, shift=UP * 0.2) for c in orig_row], lag_ratio=0.04),
            run_time=1.5)
        self.wait(0.3)

        # ── Show block boundaries ──
        block_labels = VGroup()
        block_lines = VGroup()
        for b in range(NUM_BLOCKS):
            if BLOCK_SIZE > 1:
                mid_pos = (orig_row[b * BLOCK_SIZE].get_center() +
                           orig_row[b * BLOCK_SIZE + BLOCK_SIZE - 1].get_center()) / 2
            else:
                mid_pos = orig_row[b * BLOCK_SIZE].get_center()
            bl = Text(f"b{b}", font_size=12, color="#ccccee")
            bl.next_to(mid_pos, DOWN, buff=0.35)
            block_labels.add(bl)

            if b > 0:
                left_edge = (orig_row[b * BLOCK_SIZE - 1].get_right() +
                             orig_row[b * BLOCK_SIZE].get_left()) / 2
                line = DashedLine(
                    left_edge + UP * 0.25, left_edge + DOWN * 0.25,
                    color=WHITE, stroke_opacity=0.5, dash_length=0.06)
                block_lines.add(line)

        split_label = Text(f"Split into {NUM_BLOCKS} blocks (2×cp_size)",
                           font_size=18, color="#ccccdd")
        split_label.move_to(orig_label.get_center())

        self.play(
            Transform(orig_label, split_label),
            LaggedStart(*[FadeIn(bl) for bl in block_labels], lag_ratio=0.05),
            LaggedStart(*[Create(ln) for ln in block_lines], lag_ratio=0.05),
            run_time=1.2)
        self.wait(0.3)

        # ── Color by rank ──
        color_anims = []
        for i in range(SEQ_LEN):
            rank = inseq_token_rank[i]
            color_anims.append(orig_row[i][0].animate.set_fill(RANK_COLORS[rank]))
        self.play(*color_anims, run_time=1)

        # ── Show zigzag pairing arrows ──
        arrows = VGroup()
        pair_labels = VGroup()
        for r in range(CP_SIZE):
            bp, bn = zigzag_pairs[r]
            if BLOCK_SIZE > 1:
                pos_prev = (orig_row[bp * BLOCK_SIZE].get_center() +
                            orig_row[bp * BLOCK_SIZE + BLOCK_SIZE - 1].get_center()) / 2
                pos_next = (orig_row[bn * BLOCK_SIZE].get_center() +
                            orig_row[bn * BLOCK_SIZE + BLOCK_SIZE - 1].get_center()) / 2
            else:
                pos_prev = orig_row[bp * BLOCK_SIZE].get_center()
                pos_next = orig_row[bn * BLOCK_SIZE].get_center()

            arrow = CurvedDoubleArrow(
                pos_prev + DOWN * 0.45,
                pos_next + DOWN * 0.45,
                color=RANK_COLORS[r],
                stroke_width=2,
                angle=-TAU / 8 - r * TAU / 30,
            )
            arrows.add(arrow)
            pl = Text(f"r{r}: b{bp}+b{bn}", font_size=11,
                       color=RANK_COLORS_LIGHT[r])
            pl.next_to(arrow, DOWN, buff=0.04)
            pair_labels.add(pl)

        self.play(
            LaggedStart(*[Create(a) for a in arrows], lag_ratio=0.15),
            LaggedStart(*[FadeIn(pl) for pl in pair_labels], lag_ratio=0.15),
            run_time=2)
        self.wait(0.5)

        # ── Fade arrows, build rank rows ──
        self.play(
            FadeOut(arrows), FadeOut(pair_labels),
            FadeOut(block_labels), FadeOut(block_lines),
            run_time=0.5)

        rank_groups = VGroup()
        rank_labels_g = VGroup()
        block_dividers = VGroup()
        block_annots = VGroup()
        for r in range(CP_SIZE):
            row = make_token_row(inseq_ranks[r],
                                  color_fn=lambda t, r=r: RANK_COLORS[r],
                                  w=0.42, h=0.42, gap=0.04)
            lbl = Text(f"Rank {r}", font_size=16, color=RANK_COLORS[r], weight=BOLD)
            rank_groups.add(row)
            rank_labels_g.add(lbl)

            bp, bn = zigzag_pairs[r]
            bl_prev = Text(f"b{bp}", font_size=11, color=RANK_COLORS_LIGHT[r])
            bl_next = Text(f"b{bn}", font_size=11, color=RANK_COLORS_LIGHT[r])
            block_annots.add(VGroup(bl_prev, bl_next))

        rank_groups.arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        rank_groups.next_to(orig_row, DOWN, buff=0.55)
        for lbl, row in zip(rank_labels_g, rank_groups):
            lbl.next_to(row, LEFT, buff=0.25)
        for r, (annot, row) in enumerate(zip(block_annots, rank_groups)):
            mid = row[BLOCK_SIZE - 1].get_right()
            mid2 = row[BLOCK_SIZE].get_left()
            div_x = (mid + mid2) / 2
            div = Line(div_x + UP * 0.24, div_x + DOWN * 0.24,
                       color=WHITE, stroke_width=2, stroke_opacity=0.6)
            block_dividers.add(div)
            prev_center = VGroup(*row[:BLOCK_SIZE]).get_center()
            next_center = VGroup(*row[BLOCK_SIZE:]).get_center()
            annot[0].next_to(prev_center, DOWN, buff=0.28)
            annot[1].next_to(next_center, DOWN, buff=0.28)

        # Auto-zoom to fit
        all_pre_fly = VGroup(hdr, orig_label, orig_row, rank_groups, rank_labels_g)
        self.play(self.camera.auto_zoom(all_pre_fly, margin=0.4), run_time=0.5)

        # ── Fly tokens ──
        fly_anims = []
        for r in range(CP_SIZE):
            for j, t in enumerate(inseq_ranks[r]):
                src = orig_row[t]
                tgt = rank_groups[r][j]
                fly_anims.append(
                    AnimationGroup(
                        src.animate.move_to(tgt.get_center()).set_opacity(0.25),
                        FadeIn(tgt, shift=DOWN * 0.1),
                        lag_ratio=0.3,
                    )
                )
        self.play(
            LaggedStart(*[FadeIn(lbl) for lbl in rank_labels_g], lag_ratio=0.1),
            run_time=0.5)
        self.play(
            LaggedStart(*fly_anims, lag_ratio=0.06), run_time=3)
        self.play(
            LaggedStart(*[Create(d) for d in block_dividers], lag_ratio=0.1),
            LaggedStart(*[FadeIn(a) for a in block_annots], lag_ratio=0.1),
            run_time=0.8)

        # ── Right annotations ──
        annotations = VGroup()
        for r in range(CP_SIZE):
            bp, bn = zigzag_pairs[r]
            tokens_str = ", ".join(str(t) for t in inseq_ranks[r])
            ann = Text(f"b{bp}+b{bn}: [{tokens_str}]",
                       font_size=12, color=RANK_COLORS_LIGHT[r])
            ann.next_to(rank_groups[r], RIGHT, buff=0.3)
            annotations.add(ann)
        self.play(LaggedStart(
            *[FadeIn(a) for a in annotations], lag_ratio=0.1), run_time=1)

        # ── Bottom summary ──
        box_text = Text(
            "Pattern: zigzag — rank r gets block_r (head) + block_{2N-1-r} (tail)",
            font_size=17, color="#aaddff")
        box = SurroundingRectangle(box_text, color="#4466aa", fill_color="#222244",
                                    fill_opacity=0.85, corner_radius=0.12, buff=0.2)
        box_group = VGroup(box, box_text)
        fn_text = Text(
            "Function: cp_split_and_rebuild_data()",
            font_size=12, color="#88aacc")
        balance_text = Text(
            "Head-tail pairing balances load: each rank sees both early and late context",
            font_size=12, color="#aaaacc")
        bottom = VGroup(box_group, fn_text, balance_text).arrange(DOWN, buff=0.06)
        bottom.next_to(rank_groups, DOWN, buff=0.35)
        fit_to_safe(bottom)

        all_final = VGroup(hdr, orig_label, orig_row, rank_groups, rank_labels_g,
                           annotations, block_dividers, block_annots, bottom)
        self.play(self.camera.auto_zoom(all_final, margin=0.3), run_time=0.5)
        self.play(FadeIn(box_group), FadeIn(fn_text), FadeIn(balance_text), run_time=1)
        self.wait(2)


# ═════════════════════════════════════════════════════════════
# Scene 3: Side-by-side Comparison
# ═════════════════════════════════════════════════════════════

class SplitComparison(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("DSA CP Split: Round-Robin vs In-Seq (Zigzag)",
                      font_size=28, color=WHITE, weight=BOLD)
        title.to_edge(UP, buff=0.3)
        fit_to_safe(title)
        self.play(Write(title), run_time=0.8)

        # ── Dividing line ──
        divider = DashedLine(UP * 3, DOWN * 3.5, color="#444466",
                              dash_length=0.1, stroke_width=1.5)
        divider.move_to(ORIGIN)
        self.play(Create(divider), run_time=0.4)

        CW, CH = 0.28, 0.28

        # ════ LEFT: Round-Robin ════
        left_title = Text("Round-Robin Split", font_size=20,
                           color="#aaddff", weight=BOLD)
        left_formula = latex_label(
            r"\text{token}[i] \to \text{rank}(i \bmod \text{cp\_size})",
            font_size=22, color="#8899bb")
        left_title.move_to(LEFT * 3.5 + UP * 2.4)
        left_formula.next_to(left_title, DOWN, buff=0.08)

        left_orig = make_token_row(range(SEQ_LEN),
                                    color_fn=lambda t: RANK_COLORS[t % CP_SIZE],
                                    w=CW, h=CH, gap=0.02)
        fit_to_safe(left_orig, max_w=SAFE_W/2 - 0.5)
        left_orig.next_to(left_formula, DOWN, buff=0.2)
        left_orig.set_opacity(0.4)

        left_rank_rows = VGroup()
        left_rank_labels = VGroup()
        for r in range(CP_SIZE):
            row = make_token_row(rr_ranks[r],
                                  color_fn=lambda t, r=r: RANK_COLORS[r],
                                  w=0.32, h=0.32, gap=0.04)
            lbl = Text(f"R{r}", font_size=12, color=RANK_COLORS[r], weight=BOLD)
            left_rank_rows.add(row)
            left_rank_labels.add(lbl)
        left_rank_rows.arrange(DOWN, buff=0.14, aligned_edge=LEFT)
        left_rank_rows.next_to(left_orig, DOWN, buff=0.25)
        for lbl, row in zip(left_rank_labels, left_rank_rows):
            lbl.next_to(row, LEFT, buff=0.15)

        # ════ RIGHT: In-Seq (Zigzag) ════
        right_title = Text("In-Seq Split (Zigzag)", font_size=20,
                            color="#ffddaa", weight=BOLD)
        right_formula = latex_label(
            r"\text{rank}\ r \to \text{block}_r + \text{block}_{2N\!-\!1\!-\!r}",
            font_size=22, color="#bb9988")
        right_title.move_to(RIGHT * 3.5 + UP * 2.4)
        right_formula.next_to(right_title, DOWN, buff=0.08)

        right_orig = make_token_row(range(SEQ_LEN),
                                     color_fn=lambda t: RANK_COLORS[inseq_token_rank[t]],
                                     w=CW, h=CH, gap=0.02)
        fit_to_safe(right_orig, max_w=SAFE_W/2 - 0.5)
        right_orig.next_to(right_formula, DOWN, buff=0.2)
        right_orig.set_opacity(0.4)

        right_rank_rows = VGroup()
        right_rank_labels = VGroup()
        right_dividers = VGroup()
        right_block_labels = VGroup()
        for r in range(CP_SIZE):
            row = make_token_row(inseq_ranks[r],
                                  color_fn=lambda t, r=r: RANK_COLORS[r],
                                  w=0.32, h=0.32, gap=0.04)
            lbl = Text(f"R{r}", font_size=12, color=RANK_COLORS[r], weight=BOLD)
            right_rank_rows.add(row)
            right_rank_labels.add(lbl)
        right_rank_rows.arrange(DOWN, buff=0.14, aligned_edge=LEFT)
        right_rank_rows.next_to(right_orig, DOWN, buff=0.25)
        for lbl, row in zip(right_rank_labels, right_rank_rows):
            lbl.next_to(row, LEFT, buff=0.15)

        for r in range(CP_SIZE):
            row = right_rank_rows[r]
            if len(inseq_ranks[r]) > BLOCK_SIZE:
                mid = (row[BLOCK_SIZE - 1].get_right() + row[BLOCK_SIZE].get_left()) / 2
                dv = Line(mid + UP * 0.18, mid + DOWN * 0.18,
                          color=WHITE, stroke_width=1.5, stroke_opacity=0.6)
                right_dividers.add(dv)
                bp, bn = zigzag_pairs[r]
                bl_p = Text(f"b{bp}", font_size=9, color=RANK_COLORS_LIGHT[r])
                bl_n = Text(f"b{bn}", font_size=9, color=RANK_COLORS_LIGHT[r])
                prev_c = VGroup(*row[:BLOCK_SIZE]).get_center()
                next_c = VGroup(*row[BLOCK_SIZE:]).get_center()
                bl_p.next_to(prev_c, DOWN, buff=0.2)
                bl_n.next_to(next_c, DOWN, buff=0.2)
                right_block_labels.add(bl_p, bl_n)

        # ── Animate both sides ──
        self.play(
            FadeIn(left_title), FadeIn(left_formula),
            FadeIn(right_title), FadeIn(right_formula),
            run_time=0.8)
        self.play(
            FadeIn(left_orig), FadeIn(right_orig),
            run_time=0.5)
        self.play(
            LaggedStart(*[FadeIn(lbl) for lbl in left_rank_labels], lag_ratio=0.05),
            LaggedStart(*[FadeIn(row) for row in left_rank_rows], lag_ratio=0.08),
            LaggedStart(*[FadeIn(lbl) for lbl in right_rank_labels], lag_ratio=0.05),
            LaggedStart(*[FadeIn(row) for row in right_rank_rows], lag_ratio=0.08),
            run_time=1.5)
        self.play(
            LaggedStart(*[Create(d) for d in right_dividers], lag_ratio=0.1),
            LaggedStart(*[FadeIn(bl) for bl in right_block_labels], lag_ratio=0.1),
            run_time=0.6)

        # ── Feature comparison bullets ──
        left_bullets = [
            "Granularity: single token",
            "Split: dsa_cp_round_robin_split_data",
            "Rerange: view → transpose → reshape",
            "MLP: gather / reduce-scatter",
            "MoE: TP-only path",
            "Metadata: none (pure stride)",
        ]
        right_bullets = [
            "Granularity: block (2×cp blocks)",
            "Split: cp_split_and_rebuild_data",
            "Rerange: cp_reverse_index",
            "MLP: bypass via DeepEP (a2a)",
            "MoE: deepep, ep=tp_size",
            "Metadata: split_list/zigzag/rev",
        ]

        left_items = VGroup()
        for line in left_bullets:
            t = Text(f"• {line}", font_size=11, color="#aabbcc")
            left_items.add(t)
        left_items.arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        left_items.next_to(left_rank_rows, DOWN, buff=0.25)
        left_items.align_to(left_rank_rows, LEFT)

        right_items = VGroup()
        for line in right_bullets:
            t = Text(f"• {line}", font_size=11, color="#ccbbaa")
            right_items.add(t)
        right_items.arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        right_items.next_to(right_rank_rows, DOWN, buff=0.25)
        right_items.align_to(right_rank_rows, LEFT)

        # Auto-zoom to fit everything
        all_content = VGroup(title, divider,
                             left_title, left_formula, left_orig,
                             left_rank_rows, left_rank_labels, left_items,
                             right_title, right_formula, right_orig,
                             right_rank_rows, right_rank_labels, right_items,
                             right_dividers, right_block_labels)
        self.play(self.camera.auto_zoom(all_content, margin=0.3), run_time=0.5)

        self.play(
            LaggedStart(*[FadeIn(item) for item in left_items], lag_ratio=0.08),
            LaggedStart(*[FadeIn(item) for item in right_items], lag_ratio=0.08),
            run_time=2)

        # ── Legend ──
        legend = VGroup()
        for r in range(CP_SIZE):
            sq = Square(side_length=0.18, fill_color=RANK_COLORS[r],
                         fill_opacity=1, stroke_color=WHITE, stroke_width=1)
            lb = Text(f"Rank {r}", font_size=11, color="#e0e0e0")
            lb.next_to(sq, RIGHT, buff=0.08)
            legend.add(VGroup(sq, lb))
        legend.arrange(RIGHT, buff=0.35)
        legend.next_to(all_content, DOWN, buff=0.2)
        self.play(FadeIn(legend), run_time=0.5)

        self.wait(3)
