"""
DSA CP Split — Actual Tensor Operations Visualization (Manim CE)

Shows the real PyTorch operations used in SGLang's CP split/rerange code,
step by step with matrix shapes and intermediate states.

All scenes use MovingCameraScene + auto_zoom to fit content in one frame.

Usage:
  manim -qm --media_dir docs/media docs/dsa_cp_tensor_ops_manim.py RoundRobinViewSlice
  manim -qm --media_dir docs/media docs/dsa_cp_tensor_ops_manim.py RoundRobinInputIds
  manim -qm --media_dir docs/media docs/dsa_cp_tensor_ops_manim.py AllGatherRerange
  manim -qm --media_dir docs/media docs/dsa_cp_tensor_ops_manim.py InSeqSplitRebuild
  manim -qm --media_dir docs/media docs/dsa_cp_tensor_ops_manim.py InSeqInputIds
  manim -qm --media_dir docs/media docs/dsa_cp_tensor_ops_manim.py InSeqRerange
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

SEQ_LEN = 16
CP_SIZE = 4
HIDDEN = 4
TOKENS_PER_RANK = SEQ_LEN // CP_SIZE
NUM_BLOCKS = 2 * CP_SIZE
BLOCK_SIZE = SEQ_LEN // NUM_BLOCKS

RANK_COLORS = ["#7A95AE", "#B89A88", "#7BAA8E", "#A87A7C"]
RANK_COLORS_LIGHT = ["#A0B5C6", "#D4BCA6", "#A0C4AC", "#C4A0A2"]
NEUTRAL = "#555577"
BG = "#1a1a2e"
CODE_COLOR = "#88aacc"
DIM_LABEL = "#aaaacc"
HIGHLIGHT_YELLOW = "#FFD700"


def make_cell(label, color=NEUTRAL, w=0.35, h=0.3, font_size=11):
    rect = RoundedRectangle(
        width=w, height=h, corner_radius=0.04,
        fill_color=color, fill_opacity=1,
        stroke_color=WHITE, stroke_width=0.8,
    )
    txt = Text(str(label), font_size=font_size, color=WHITE, weight=BOLD)
    txt.move_to(rect.get_center())
    return VGroup(rect, txt)


def make_matrix(rows, cols, data, color_fn=None, w=0.35, h=0.3, gap=0.02,
                font_size=11):
    matrix = VGroup()
    for r in range(rows):
        row_group = VGroup()
        for c in range(cols):
            label = data[r][c]
            color = color_fn(r, c, label) if color_fn else NEUTRAL
            cell = make_cell(label, color=color, w=w, h=h, font_size=font_size)
            row_group.add(cell)
        row_group.arrange(RIGHT, buff=gap)
        matrix.add(row_group)
    matrix.arrange(DOWN, buff=gap)
    return matrix


def code_text(text, font_size=13):
    return Text(text, font_size=font_size, color=CODE_COLOR)


_TEX_TEMPLATE = TexTemplate()
_TEX_TEMPLATE.body = (
    r"\documentclass[preview]{article}" "\n"
    r"\usepackage{amsmath}" "\n"
    r"\usepackage{amssymb}" "\n"
    r"\begin{document}" "\n"
    r"YourTextHere" "\n"
    r"\end{document}"
)


def latex_label(tex_str, font_size=24, color=CODE_COLOR):
    return MathTex(tex_str, font_size=font_size, color=color,
                   tex_template=_TEX_TEMPLATE)


def dim_text(text, font_size=10):
    return Text(text, font_size=font_size, color=DIM_LABEL)


# ═════════════════════════════════════════════════════════════
# Scene 4: Round-Robin — view(-1, cp_size, D)[:, cp_rank]
# One page: left=original [16,4], middle=4 groups [4,4,4], right=slice result
# ═════════════════════════════════════════════════════════════

class RoundRobinViewSlice(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BG
        CW, CH, FS = 0.28, 0.22, 9

        # Title
        title = Text("Round-Robin: view(-1, cp_size, D)[:, cp_rank]",
                      font_size=26, color=WHITE, weight=BOLD)
        source = Text("dsa/utils.py:126 — dsa_cp_round_robin_split_data()",
                       font_size=12, color=CODE_COLOR)
        hdr = VGroup(title, source).arrange(DOWN, buff=0.06)

        # ── LEFT: Original [16, 4] ──
        code1 = latex_label(r"\text{input\_}\ [16, 4]")
        data_16x4 = [[f"t{r}" for _ in range(HIDDEN)] for r in range(SEQ_LEN)]
        mat1 = make_matrix(SEQ_LEN, HIDDEN, data_16x4,
                           color_fn=lambda r, c, l: RANK_COLORS[r % CP_SIZE],
                           w=CW, h=CH, gap=0.02, font_size=FS)
        brace_r = Brace(mat1, LEFT, color=DIM_LABEL, buff=0.06)
        dim_r = dim_text("16")
        dim_r.next_to(brace_r, LEFT, buff=0.04)
        left_panel = VGroup(code1, VGroup(dim_r, brace_r, mat1)).arrange(DOWN, buff=0.1)

        # ── MIDDLE: view → 4 groups [4,4,4] ──
        code2 = latex_label(r".\text{view}(-1, 4, 4)")
        groups = VGroup()
        for g in range(CP_SIZE):
            data_g = [[f"t{g*CP_SIZE+r}" for _ in range(HIDDEN)]
                      for r in range(CP_SIZE)]
            mat_g = make_matrix(CP_SIZE, HIDDEN, data_g,
                                color_fn=lambda r, c, l, gg=g: RANK_COLORS[r],
                                w=CW, h=CH, gap=0.02, font_size=FS)
            gl = dim_text(f"g{g}")
            gl.next_to(mat_g, UP, buff=0.04)
            groups.add(VGroup(gl, mat_g))
        groups.arrange(DOWN, buff=0.12)
        mid_panel = VGroup(code2, groups).arrange(DOWN, buff=0.06)

        # ── RIGHT: [:, cp_rank] → result per rank ──
        code3 = latex_label(r"[:, \text{rank}]")
        result_rows = VGroup()
        for r in range(CP_SIZE):
            tokens = [r + g * CP_SIZE for g in range(CP_SIZE)]
            tok_str = ", ".join([f"t{t}" for t in tokens])
            lbl = Text(f"R{r}: [{tok_str}]", font_size=11,
                       color=RANK_COLORS_LIGHT[r])
            # small row of cells
            row = VGroup()
            for t in tokens:
                cell = make_cell(f"t{t}", color=RANK_COLORS[r], w=CW, h=CH, font_size=FS)
                row.add(cell)
            row.arrange(RIGHT, buff=0.02)
            rank_lbl = Text(f"R{r}", font_size=9, color=RANK_COLORS[r], weight=BOLD)
            rank_lbl.next_to(row, LEFT, buff=0.08)
            result_rows.add(VGroup(rank_lbl, row))
        result_rows.arrange(DOWN, buff=0.08)
        note = latex_label(r"\text{stride} = \text{cp\_size}", font_size=20,
                           color="#aaddff")
        right_panel = VGroup(code3, result_rows, note).arrange(DOWN, buff=0.1)

        # Arrows
        arrow1 = Arrow(LEFT*0.3, RIGHT*0.3, color=HIGHLIGHT_YELLOW,
                        stroke_width=2, buff=0.05)
        arr1_lbl = latex_label(r"\text{view}", font_size=18, color=HIGHLIGHT_YELLOW)
        arr1_lbl.next_to(arrow1, UP, buff=0.03)
        arr1_grp = VGroup(arrow1, arr1_lbl)

        arrow2 = Arrow(LEFT*0.3, RIGHT*0.3, color=HIGHLIGHT_YELLOW,
                        stroke_width=2, buff=0.05)
        arr2_lbl = latex_label(r"[:, r]", font_size=18, color=HIGHLIGHT_YELLOW)
        arr2_lbl.next_to(arrow2, UP, buff=0.03)
        arr2_grp = VGroup(arrow2, arr2_lbl)

        # Layout: left → arrow → middle → arrow → right
        full = VGroup(left_panel, arr1_grp, mid_panel, arr2_grp, right_panel)
        full.arrange(RIGHT, buff=0.3)
        all_content = VGroup(hdr, full).arrange(DOWN, buff=0.3)

        # Animate
        self.play(Write(title), FadeIn(source), run_time=0.8)
        self.play(self.camera.auto_zoom(all_content, margin=0.5), run_time=0.3)
        self.play(FadeIn(left_panel), run_time=1)
        self.wait(0.5)
        self.play(GrowArrow(arrow1), FadeIn(arr1_lbl), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(g) for g in groups], lag_ratio=0.1),
                  FadeIn(code2), run_time=1.5)
        self.wait(0.5)

        # Highlight row 0 in each group
        highlights = VGroup()
        for g in range(CP_SIZE):
            h = SurroundingRectangle(groups[g][1][0], color=RANK_COLORS[0],
                                     stroke_width=2, buff=0.02)
            highlights.add(h)
        self.play(LaggedStart(*[Create(h) for h in highlights], lag_ratio=0.08),
                  run_time=0.8)

        self.play(GrowArrow(arrow2), FadeIn(arr2_lbl), run_time=0.4)
        self.play(FadeIn(code3), FadeIn(result_rows), FadeIn(note), run_time=1)
        self.wait(2)



# ═════════════════════════════════════════════════════════════
# Scene 5: cp_round_robin_input_ids — reshape.T.flatten
# One page: top=original 1D, middle-left=reshape, middle-right=transpose,
# bottom=flatten result.  Cells move smoothly between arrangements.
# ═════════════════════════════════════════════════════════════

class RoundRobinInputIds(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BG
        CW, CH, FS = 0.34, 0.30, 11

        title = Text("Round-Robin Input IDs: reshape → T → flatten",
                      font_size=26, color=WHITE, weight=BOLD)
        source = Text("cp_utils.py:209 — cp_round_robin_input_ids()",
                       font_size=12, color=CODE_COLOR)
        hdr = VGroup(title, source).arrange(DOWN, buff=0.06)

        # ── Ghost layout (defines positions, never added to scene) ──

        code1 = latex_label(r"\text{input\_ids}\ [16]")
        g_orig = VGroup(*[make_cell(str(i), w=CW, h=CH, font_size=FS)
                          for i in range(SEQ_LEN)])
        g_orig.arrange(RIGHT, buff=0.03)
        row1 = VGroup(code1, g_orig).arrange(DOWN, buff=0.08)

        code2 = latex_label(r"\text{reshape}(-1, 4)")
        g_reshape = make_matrix(
            TOKENS_PER_RANK, CP_SIZE,
            [[str(r * CP_SIZE + c) for c in range(CP_SIZE)]
             for r in range(TOKENS_PER_RANK)],
            w=CW, h=CH, gap=0.03, font_size=FS)
        left_mat = VGroup(code2, g_reshape).arrange(DOWN, buff=0.1)

        code3 = latex_label(r"\cdot^\top")
        g_trans = make_matrix(
            CP_SIZE, TOKENS_PER_RANK,
            [[str(c * CP_SIZE + r) for c in range(TOKENS_PER_RANK)]
             for r in range(CP_SIZE)],
            w=CW, h=CH, gap=0.03, font_size=FS)
        right_mat = VGroup(code3, g_trans).arrange(DOWN, buff=0.1)

        arrow_t = Arrow(LEFT * 0.4, RIGHT * 0.4, color=HIGHLIGHT_YELLOW,
                        stroke_width=2, buff=0.05)
        arr_lbl = latex_label(r"^\top", font_size=22, color=HIGHLIGHT_YELLOW)
        arr_lbl.next_to(arrow_t, UP, buff=0.03)
        arr_grp = VGroup(arrow_t, arr_lbl)
        row2 = VGroup(left_mat, arr_grp, right_mat).arrange(RIGHT, buff=0.4)

        code4 = latex_label(r".\text{flatten}()")
        flat_order = []
        for r in range(CP_SIZE):
            for c in range(TOKENS_PER_RANK):
                flat_order.append(r + c * CP_SIZE)
        g_flat = VGroup(*[make_cell(str(v), w=CW, h=CH, font_size=FS)
                          for v in flat_order])
        g_flat.arrange(RIGHT, buff=0.03)
        rank_brackets = VGroup()
        for r in range(CP_SIZE):
            sub = VGroup(*g_flat[r * TOKENS_PER_RANK:(r + 1) * TOKENS_PER_RANK])
            br = Brace(sub, DOWN, color=RANK_COLORS[r], buff=0.04)
            bl = Text(f"R{r}", font_size=9, color=RANK_COLORS[r])
            bl.next_to(br, DOWN, buff=0.04)
            rank_brackets.add(VGroup(br, bl))
        row3 = VGroup(code4, g_flat, rank_brackets).arrange(DOWN, buff=0.06)

        all_content = VGroup(hdr, row1, row2, row3).arrange(DOWN, buff=0.3)

        col_labels = VGroup()
        for c in range(CP_SIZE):
            cl = Text(f"r{c}", font_size=9, color=RANK_COLORS[c], weight=BOLD)
            cl.next_to(g_reshape[0][c], UP, buff=0.06)
            col_labels.add(cl)
        row_labels = VGroup()
        for r in range(CP_SIZE):
            rl = Text(f"R{r}", font_size=9, color=RANK_COLORS[r], weight=BOLD)
            rl.next_to(g_trans[r], LEFT, buff=0.08)
            row_labels.add(rl)

        # ── Extract target positions ──
        orig_pos = [g_orig[i].get_center() for i in range(SEQ_LEN)]
        reshape_pos = {}
        for r in range(TOKENS_PER_RANK):
            for c in range(CP_SIZE):
                reshape_pos[r * CP_SIZE + c] = g_reshape[r][c].get_center()
        trans_pos = {}
        for r in range(CP_SIZE):
            for c in range(TOKENS_PER_RANK):
                trans_pos[c * CP_SIZE + r] = g_trans[r][c].get_center()
        flat_pos = {}
        for idx, v in enumerate(flat_order):
            flat_pos[v] = g_flat[idx].get_center()

        # ── 16 persistent cells ──
        cells = []
        for i in range(SEQ_LEN):
            cell = make_cell(str(i), color=RANK_COLORS[i % CP_SIZE],
                             w=CW, h=CH, font_size=FS)
            cell.move_to(orig_pos[i])
            cells.append(cell)

        # ── Animate ──
        self.play(Write(title), FadeIn(source), run_time=0.8)
        self.play(self.camera.auto_zoom(all_content, margin=0.4), run_time=0.3)

        self.play(FadeIn(code1), run_time=0.2)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.1) for c in cells],
                               lag_ratio=0.03), run_time=1)
        self.wait(0.5)

        # 1D → reshape (fold into grid)
        self.play(
            FadeIn(code2),
            LaggedStart(*[cells[i].animate.move_to(reshape_pos[i])
                          for i in range(SEQ_LEN)], lag_ratio=0.05),
            run_time=1.5)
        self.play(FadeIn(col_labels), run_time=0.3)
        self.wait(0.3)

        # reshape → transpose (row/col swap)
        self.play(GrowArrow(arrow_t), FadeIn(arr_lbl), run_time=0.4)
        self.play(
            FadeIn(code3),
            LaggedStart(*[cells[i].animate.move_to(trans_pos[i])
                          for i in range(SEQ_LEN)], lag_ratio=0.03),
            run_time=1.5)
        self.play(FadeIn(row_labels), run_time=0.3)
        self.wait(0.5)

        # transpose → flatten (unfold to 1D)
        self.play(FadeIn(code4), run_time=0.2)
        self.play(
            LaggedStart(*[cells[i].animate.move_to(flat_pos[i])
                          for i in range(SEQ_LEN)], lag_ratio=0.03),
            run_time=1.5)
        self.play(LaggedStart(*[FadeIn(rb) for rb in rank_brackets],
                               lag_ratio=0.08), run_time=0.6)
        self.wait(2)


# ═════════════════════════════════════════════════════════════
# Scene 6: AllGather Rerange — view.transpose.reshape
# One page: top=per-rank before AllGather,
# middle=view(4,-1,D) → .T(0,1) side by side,
# bottom=reshape result [16].  Cells move between stages.
# ═════════════════════════════════════════════════════════════

class AllGatherRerange(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BG
        CW, CH, FS = 0.32, 0.28, 10

        title = Text("AllGather + Rerange: Restore Original Order",
                      font_size=26, color=WHITE, weight=BOLD)
        source = Text("cp_utils.py:341-358 — round-robin path",
                       font_size=12, color=CODE_COLOR)
        hdr = VGroup(title, source).arrange(DOWN, buff=0.06)

        # ── Ghost row 1: per-rank data ──
        label1 = dim_text("Each rank holds 4 tokens:")
        g_rank_rows = VGroup()
        for r in range(CP_SIZE):
            tokens = [r + g * CP_SIZE for g in range(TOKENS_PER_RANK)]
            row = VGroup()
            for t in tokens:
                cell = make_cell(str(t), color=RANK_COLORS[r],
                                 w=CW, h=CH, font_size=FS)
                row.add(cell)
            row.arrange(RIGHT, buff=0.03)
            lbl = Text(f"R{r}", font_size=9, color=RANK_COLORS[r], weight=BOLD)
            lbl.next_to(row, LEFT, buff=0.1)
            g_rank_rows.add(VGroup(lbl, row))
        g_rank_rows.arrange(RIGHT, buff=0.3)
        row1 = VGroup(label1, g_rank_rows).arrange(DOWN, buff=0.08)

        # ── Ghost row 2 left: view(4,-1,D) ──
        code_v = latex_label(r"\text{AllGather} \to \text{view}(4, -1, D)")
        g_view = VGroup()
        for r in range(CP_SIZE):
            tokens = [r + g * CP_SIZE for g in range(TOKENS_PER_RANK)]
            row = VGroup()
            for t in tokens:
                cell = make_cell(str(t), color=RANK_COLORS[r],
                                 w=CW, h=CH, font_size=FS)
                row.add(cell)
            row.arrange(RIGHT, buff=0.03)
            gl = Text(f"[{r},:]", font_size=8, color=DIM_LABEL)
            gl.next_to(row, LEFT, buff=0.06)
            g_view.add(VGroup(gl, row))
        g_view.arrange(DOWN, buff=0.05)
        left_block = VGroup(code_v, g_view).arrange(DOWN, buff=0.08)

        # ── Ghost row 2 right: .transpose(0,1) ──
        code_t = latex_label(r".\text{transpose}(0, 1)")
        g_trans = VGroup()
        for g in range(TOKENS_PER_RANK):
            tokens = [g * CP_SIZE + r for r in range(CP_SIZE)]
            row = VGroup()
            for t in tokens:
                cell = make_cell(str(t), color=RANK_COLORS[t % CP_SIZE],
                                 w=CW, h=CH, font_size=FS)
                row.add(cell)
            row.arrange(RIGHT, buff=0.03)
            gl = Text(f"[{g},:]", font_size=8, color=DIM_LABEL)
            gl.next_to(row, LEFT, buff=0.06)
            g_trans.add(VGroup(gl, row))
        g_trans.arrange(DOWN, buff=0.05)
        right_block = VGroup(code_t, g_trans).arrange(DOWN, buff=0.08)

        arrow_t = Arrow(LEFT * 0.4, RIGHT * 0.4, color=HIGHLIGHT_YELLOW,
                        stroke_width=2, buff=0.05)
        arr_lbl = latex_label(r"^\top(0,1)", font_size=20, color=HIGHLIGHT_YELLOW)
        arr_lbl.next_to(arrow_t, UP, buff=0.03)
        arr_grp = VGroup(arrow_t, arr_lbl)
        row2 = VGroup(left_block, arr_grp, right_block).arrange(RIGHT, buff=0.35)

        # ── Ghost row 3: flatten ──
        code_r = latex_label(r".\text{reshape}(\text{shape}) \to [16, D]",
                              color="#55ff55")
        g_result = VGroup()
        for i in range(SEQ_LEN):
            cell = make_cell(str(i), color=RANK_COLORS[i % CP_SIZE],
                             w=CW, h=CH, font_size=FS)
            g_result.add(cell)
        g_result.arrange(RIGHT, buff=0.03)
        check = Text("0,1,2,...,15 — sequential order restored",
                      font_size=11, color="#aaddff")
        row3 = VGroup(code_r, g_result, check).arrange(DOWN, buff=0.06)

        all_content = VGroup(hdr, row1, row2, row3).arrange(DOWN, buff=0.25)

        # ── Extract positions ──
        rank_pos = {}
        for r in range(CP_SIZE):
            row_cells = g_rank_rows[r][1]
            tokens = [r + g * CP_SIZE for g in range(TOKENS_PER_RANK)]
            for j, t in enumerate(tokens):
                rank_pos[t] = row_cells[j].get_center()

        view_pos = {}
        for r in range(CP_SIZE):
            row_cells = g_view[r][1]
            tokens = [r + g * CP_SIZE for g in range(TOKENS_PER_RANK)]
            for j, t in enumerate(tokens):
                view_pos[t] = row_cells[j].get_center()

        trans_pos = {}
        for g in range(TOKENS_PER_RANK):
            row_cells = g_trans[g][1]
            tokens = [g * CP_SIZE + r for r in range(CP_SIZE)]
            for j, t in enumerate(tokens):
                trans_pos[t] = row_cells[j].get_center()

        flat_pos = {i: g_result[i].get_center() for i in range(SEQ_LEN)}

        # Ghost labels (use actual objects from ghost layout)
        rank_lbls = VGroup(*[g_rank_rows[r][0] for r in range(CP_SIZE)])
        view_lbls = VGroup(*[g_view[r][0] for r in range(CP_SIZE)])
        trans_lbls = VGroup(*[g_trans[g][0] for g in range(TOKENS_PER_RANK)])

        # ── 16 persistent cells ──
        cells = {}
        for t in range(SEQ_LEN):
            cell = make_cell(str(t), color=RANK_COLORS[t % CP_SIZE],
                             w=CW, h=CH, font_size=FS)
            cell.move_to(rank_pos[t])
            cells[t] = cell

        # ── Animate ──
        self.play(Write(title), FadeIn(source), run_time=0.8)
        self.play(self.camera.auto_zoom(all_content, margin=0.4), run_time=0.3)

        # Show per-rank data
        self.play(FadeIn(label1), run_time=0.2)
        self.play(
            FadeIn(rank_lbls),
            LaggedStart(*[FadeIn(cells[t]) for t in range(SEQ_LEN)],
                         lag_ratio=0.03),
            run_time=1)
        self.wait(0.5)

        # per-rank → view (horizontal groups → vertical stack)
        self.play(
            FadeIn(code_v), FadeIn(view_lbls),
            FadeOut(rank_lbls), FadeOut(label1),
            LaggedStart(*[cells[t].animate.move_to(view_pos[t])
                          for t in range(SEQ_LEN)], lag_ratio=0.03),
            run_time=1.5)
        self.wait(0.3)

        # view → transpose (cells cross to rearranged positions)
        self.play(GrowArrow(arrow_t), FadeIn(arr_lbl), run_time=0.4)
        self.play(
            FadeIn(code_t), FadeIn(trans_lbls),
            LaggedStart(*[cells[t].animate.move_to(trans_pos[t])
                          for t in range(SEQ_LEN)], lag_ratio=0.03),
            run_time=1.5)
        self.wait(0.5)

        # transpose → flatten
        self.play(FadeIn(code_r), run_time=0.3)
        self.play(
            LaggedStart(*[cells[t].animate.move_to(flat_pos[t])
                          for t in range(SEQ_LEN)], lag_ratio=0.02),
            run_time=1.5)
        self.play(FadeIn(check), run_time=0.5)
        self.wait(2)


# ═════════════════════════════════════════════════════════════
# Scene 7: In-Seq Split — split → zigzag_index → cat
# One page: top=original with block labels,
# bottom=per-rank results with zigzag pairing.
# Cells move from 1D row into rank rows.
# ═════════════════════════════════════════════════════════════

class InSeqSplitRebuild(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BG
        CW, CH, FS = 0.32, 0.28, 10

        zigzag_pairs = {r: (r, 2*CP_SIZE-1-r) for r in range(CP_SIZE)}
        block_rank = {}
        for r in range(CP_SIZE):
            bp, bn = zigzag_pairs[r]
            block_rank[bp] = r
            block_rank[bn] = r

        title = Text("In-Seq Split: split → zigzag select → cat",
                      font_size=26, color=WHITE, weight=BOLD)
        source = Text("cp_utils.py:145-164 — cp_split_and_rebuild_data()",
                       font_size=12, color=CODE_COLOR)
        hdr = VGroup(title, source).arrange(DOWN, buff=0.06)

        # ── Row 1: Original with block coloring ──
        code1 = latex_label(
            rf"\text{{torch.split}}(\text{{input\_}}, [{BLOCK_SIZE}] \times {NUM_BLOCKS})")
        orig_cells = VGroup()
        for i in range(SEQ_LEN):
            cell = make_cell(str(i), color=RANK_COLORS[block_rank[i // BLOCK_SIZE]],
                             w=CW, h=CH, font_size=FS)
            orig_cells.add(cell)
        orig_cells.arrange(RIGHT, buff=0.03)

        block_seps = VGroup()
        for b in range(1, NUM_BLOCKS):
            idx = b * BLOCK_SIZE
            pos = (orig_cells[idx-1].get_right() + orig_cells[idx].get_left()) / 2
            line = DashedLine(pos + UP*0.18, pos + DOWN*0.18,
                              color=WHITE, stroke_opacity=0.5, dash_length=0.04)
            block_seps.add(line)

        block_lbls = VGroup()
        for b in range(NUM_BLOCKS):
            start_idx = b * BLOCK_SIZE
            end_idx = start_idx + BLOCK_SIZE - 1
            sub = VGroup(*orig_cells[start_idx:end_idx+1])
            bl = Text(f"b{b}", font_size=8, color=DIM_LABEL)
            bl.next_to(sub, DOWN, buff=0.06)
            block_lbls.add(bl)

        row1 = VGroup(code1, VGroup(orig_cells, block_seps, block_lbls))
        row1.arrange(DOWN, buff=0.1)

        # ── Zigzag description ──
        zigzag_desc = latex_label(
            r"\text{zigzag: rank}\ r \to (\text{block}_r,\ \text{block}_{2N\!-\!1\!-\!r})",
            color=HIGHLIGHT_YELLOW)

        # ── Ghost per-rank results (for positioning) ──
        g_rank_results = VGroup()
        rank_result_lbls = VGroup()
        for r in range(CP_SIZE):
            bp, bn = zigzag_pairs[r]
            tokens = list(range(bp*BLOCK_SIZE, (bp+1)*BLOCK_SIZE)) + \
                     list(range(bn*BLOCK_SIZE, (bn+1)*BLOCK_SIZE))
            row = VGroup()
            for t in tokens:
                cell = make_cell(str(t), color=RANK_COLORS[r],
                                 w=CW, h=CH, font_size=FS)
                row.add(cell)
            row.arrange(RIGHT, buff=0.03)

            mid = (row[BLOCK_SIZE-1].get_right() + row[BLOCK_SIZE].get_left()) / 2
            div = Line(mid + UP*0.15, mid + DOWN*0.15,
                       color=WHITE, stroke_width=1.2, stroke_opacity=0.6)
            row.add(div)

            lbl = Text(f"R{r}: b{bp}+b{bn}", font_size=10,
                       color=RANK_COLORS[r], weight=BOLD)
            lbl.next_to(row, LEFT, buff=0.12)
            rank_result_lbls.add(lbl)
            g_rank_results.add(VGroup(lbl, row))
        g_rank_results.arrange(DOWN, buff=0.1)

        summary = Text(
            "Head-tail pairing: each rank sees early + late context",
            font_size=12, color="#aaddff")
        summary_box = SurroundingRectangle(summary, color="#4466aa",
                                            fill_color="#222244",
                                            fill_opacity=0.85,
                                            corner_radius=0.08, buff=0.12)
        summary_g = VGroup(summary_box, summary)

        # ── Ghost rebuild row (cells return to sequential order) ──
        rebuild_code = latex_label(
            r"\text{cp\_reverse\_index} \to \text{rebuild}",
            color="#55ff55")
        g_rebuild = VGroup(*[make_cell(str(i), w=CW, h=CH, font_size=FS)
                             for i in range(SEQ_LEN)])
        g_rebuild.arrange(RIGHT, buff=0.03)
        rebuild_check = Text("0,1,2,...,15 — sequential order restored",
                              font_size=11, color="#aaddff")
        row_rebuild = VGroup(rebuild_code, g_rebuild, rebuild_check).arrange(DOWN, buff=0.06)

        all_content = VGroup(hdr, row1, zigzag_desc, g_rank_results,
                             summary_g, row_rebuild)
        all_content.arrange(DOWN, buff=0.18)

        # ── Token → target position mappings ──
        token_target = {}
        for r in range(CP_SIZE):
            bp, bn = zigzag_pairs[r]
            tokens = list(range(bp*BLOCK_SIZE, (bp+1)*BLOCK_SIZE)) + \
                     list(range(bn*BLOCK_SIZE, (bn+1)*BLOCK_SIZE))
            row_cells = g_rank_results[r][1]
            for j, t in enumerate(tokens):
                token_target[t] = row_cells[j].get_center()

        rebuild_pos = {i: g_rebuild[i].get_center() for i in range(SEQ_LEN)}

        # ── Animate ──
        self.play(Write(title), FadeIn(source), run_time=0.8)
        self.play(self.camera.auto_zoom(all_content, margin=0.4), run_time=0.3)

        self.play(FadeIn(code1), run_time=0.2)
        self.play(LaggedStart(*[FadeIn(c) for c in orig_cells], lag_ratio=0.02),
                  run_time=1)
        self.play(LaggedStart(*[Create(s) for s in block_seps], lag_ratio=0.03),
                  LaggedStart(*[FadeIn(bl) for bl in block_lbls], lag_ratio=0.03),
                  run_time=0.6)
        self.wait(0.5)

        # Highlight block pairs
        highlights = VGroup()
        highlight_anims = []
        for r in range(CP_SIZE):
            bp, bn = zigzag_pairs[r]
            sub_p = VGroup(*orig_cells[bp*BLOCK_SIZE:(bp+1)*BLOCK_SIZE])
            sub_n = VGroup(*orig_cells[bn*BLOCK_SIZE:(bn+1)*BLOCK_SIZE])
            h1 = SurroundingRectangle(sub_p, color=RANK_COLORS[r],
                                       stroke_width=2, buff=0.03)
            h2 = SurroundingRectangle(sub_n, color=RANK_COLORS[r],
                                       stroke_width=2, buff=0.03)
            highlights.add(h1, h2)
            highlight_anims.extend([Create(h1), Create(h2)])

        self.play(FadeIn(zigzag_desc), run_time=0.3)
        self.play(LaggedStart(*highlight_anims, lag_ratio=0.06), run_time=1.2)
        self.wait(0.5)

        # Fade decorations, show labels, animate cells moving to rank rows
        self.play(
            FadeOut(highlights), FadeOut(block_seps), FadeOut(block_lbls),
            FadeOut(code1),
            run_time=0.4)

        # Show rank result labels
        lbls = VGroup(*[g_rank_results[r][0] for r in range(CP_SIZE)])
        self.play(FadeIn(lbls), run_time=0.3)

        # Move cells from 1D row to per-rank positions
        self.play(
            LaggedStart(*[orig_cells[t].animate.move_to(token_target[t])
                          for t in range(SEQ_LEN)], lag_ratio=0.04),
            run_time=2)

        # Show dividers between block pairs
        dividers = VGroup(*[g_rank_results[r][1][-1] for r in range(CP_SIZE)])
        self.play(LaggedStart(*[Create(d) for d in dividers], lag_ratio=0.1),
                  run_time=0.4)

        self.play(FadeIn(summary_g), run_time=0.5)
        self.wait(1)

        # ── Rebuild: cells return to sequential order ──
        self.play(FadeOut(summary_g), FadeOut(dividers), FadeOut(lbls),
                  run_time=0.4)
        self.play(
            FadeIn(rebuild_code),
            LaggedStart(*[orig_cells[t].animate.move_to(rebuild_pos[t])
                          for t in range(SEQ_LEN)], lag_ratio=0.03),
            run_time=2)
        self.play(FadeIn(rebuild_check), run_time=0.5)
        self.wait(2)


# ═════════════════════════════════════════════════════════════
# Scene 8: In-Seq Input IDs — split → zigzag select → cat
# Counterpart to RoundRobinInputIds.
# ═════════════════════════════════════════════════════════════

class InSeqInputIds(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BG
        CW, CH, FS = 0.34, 0.30, 11

        zigzag_pairs = {r: (r, 2 * CP_SIZE - 1 - r) for r in range(CP_SIZE)}

        title = Text("In-Seq Input IDs: split → zigzag select → cat",
                      font_size=26, color=WHITE, weight=BOLD)
        source = Text("cp_utils.py:158-163 — cp_split_and_rebuild_data()",
                       font_size=12, color=CODE_COLOR)
        hdr = VGroup(title, source).arrange(DOWN, buff=0.06)

        # ── Ghost row 1: original 1D ──
        code1 = latex_label(r"\text{input\_ids}\ [16]")
        g_orig = VGroup(*[make_cell(str(i), w=CW, h=CH, font_size=FS)
                          for i in range(SEQ_LEN)])
        g_orig.arrange(RIGHT, buff=0.03)
        row1 = VGroup(code1, g_orig).arrange(DOWN, buff=0.08)

        # ── Ghost row 2: split into blocks ──
        code2 = latex_label(
            rf"\text{{torch.split}}(\text{{ids}},\ [{BLOCK_SIZE}] \times {NUM_BLOCKS})")
        g_blocks = VGroup()
        for b in range(NUM_BLOCKS):
            blk = VGroup()
            for j in range(BLOCK_SIZE):
                cell = make_cell(str(b * BLOCK_SIZE + j), w=CW, h=CH, font_size=FS)
                blk.add(cell)
            blk.arrange(RIGHT, buff=0.02)
            g_blocks.add(blk)
        g_blocks.arrange(RIGHT, buff=0.12)

        block_lbls = VGroup()
        for b in range(NUM_BLOCKS):
            bl = dim_text(f"b{b}")
            bl.next_to(g_blocks[b], DOWN, buff=0.06)
            block_lbls.add(bl)
        row2 = VGroup(code2, VGroup(g_blocks, block_lbls)).arrange(DOWN, buff=0.1)

        # ── Ghost row 3: zigzag select per rank ──
        code3 = latex_label(
            r"\text{zigzag\_index:}\ r \to (b_r,\ b_{2N\!-\!1\!-\!r})")
        g_rank_results = VGroup()
        rank_result_lbls = VGroup()
        for r in range(CP_SIZE):
            bp, bn = zigzag_pairs[r]
            tokens = list(range(bp * BLOCK_SIZE, (bp + 1) * BLOCK_SIZE)) + \
                     list(range(bn * BLOCK_SIZE, (bn + 1) * BLOCK_SIZE))
            row = VGroup()
            for t in tokens:
                cell = make_cell(str(t), color=RANK_COLORS[r],
                                 w=CW, h=CH, font_size=FS)
                row.add(cell)
            row.arrange(RIGHT, buff=0.03)

            mid = (row[BLOCK_SIZE - 1].get_right() + row[BLOCK_SIZE].get_left()) / 2
            div = Line(mid + UP * 0.15, mid + DOWN * 0.15,
                       color=WHITE, stroke_width=1.2, stroke_opacity=0.6)
            row.add(div)

            lbl = Text(f"R{r}: b{bp}+b{bn}", font_size=10,
                       color=RANK_COLORS[r], weight=BOLD)
            lbl.next_to(row, LEFT, buff=0.12)
            rank_result_lbls.add(lbl)
            g_rank_results.add(VGroup(lbl, row))
        g_rank_results.arrange(DOWN, buff=0.1)
        row3 = VGroup(code3, g_rank_results).arrange(DOWN, buff=0.1)

        all_content = VGroup(hdr, row1, row2, row3).arrange(DOWN, buff=0.25)

        # ── Extract positions ──
        orig_pos = [g_orig[i].get_center() for i in range(SEQ_LEN)]

        block_pos = {}
        for b in range(NUM_BLOCKS):
            for j in range(BLOCK_SIZE):
                idx = b * BLOCK_SIZE + j
                block_pos[idx] = g_blocks[b][j].get_center()

        rank_pos = {}
        for r in range(CP_SIZE):
            bp, bn = zigzag_pairs[r]
            tokens = list(range(bp * BLOCK_SIZE, (bp + 1) * BLOCK_SIZE)) + \
                     list(range(bn * BLOCK_SIZE, (bn + 1) * BLOCK_SIZE))
            row_cells = g_rank_results[r][1]
            for j, t in enumerate(tokens):
                rank_pos[t] = row_cells[j].get_center()

        # ── Block-to-rank color map ──
        block_rank = {}
        for r in range(CP_SIZE):
            bp, bn = zigzag_pairs[r]
            block_rank[bp] = r
            block_rank[bn] = r

        # ── 16 persistent cells ──
        cells = []
        for i in range(SEQ_LEN):
            cell = make_cell(str(i), w=CW, h=CH, font_size=FS)
            cell.move_to(orig_pos[i])
            cells.append(cell)

        # ── Animate ──
        self.play(Write(title), FadeIn(source), run_time=0.8)
        self.play(self.camera.auto_zoom(all_content, margin=0.4), run_time=0.3)

        # Show original 1D
        self.play(FadeIn(code1), run_time=0.2)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.1) for c in cells],
                               lag_ratio=0.03), run_time=1)
        self.wait(0.5)

        # 1D → split into blocks
        self.play(
            FadeIn(code2),
            LaggedStart(*[cells[i].animate.move_to(block_pos[i])
                          for i in range(SEQ_LEN)], lag_ratio=0.03),
            run_time=1.5)
        self.play(FadeIn(block_lbls), run_time=0.3)
        self.wait(0.3)

        # Color cells by rank
        color_anims = []
        for i in range(SEQ_LEN):
            b = i // BLOCK_SIZE
            r = block_rank[b]
            color_anims.append(cells[i][0].animate.set_fill(RANK_COLORS[r]))
        self.play(*color_anims, run_time=0.8)
        self.wait(0.3)

        # Blocks → per-rank zigzag results
        lbls = VGroup(*[g_rank_results[r][0] for r in range(CP_SIZE)])
        self.play(FadeIn(code3), FadeIn(lbls), run_time=0.3)
        self.play(
            LaggedStart(*[cells[i].animate.move_to(rank_pos[i])
                          for i in range(SEQ_LEN)], lag_ratio=0.03),
            run_time=1.5)

        dividers = VGroup(*[g_rank_results[r][1][-1] for r in range(CP_SIZE)])
        self.play(LaggedStart(*[Create(d) for d in dividers], lag_ratio=0.1),
                  run_time=0.4)
        self.wait(2)


# ═════════════════════════════════════════════════════════════
# Scene 9: In-Seq AllGather Rerange — split → cp_reverse_index → cat
# Counterpart to AllGatherRerange.
# ═════════════════════════════════════════════════════════════

class InSeqRerange(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BG
        CW, CH, FS = 0.32, 0.28, 10

        zigzag_pairs = {r: (r, 2 * CP_SIZE - 1 - r) for r in range(CP_SIZE)}

        title = Text("In-Seq Rerange: AllGather → reverse_index → cat",
                      font_size=26, color=WHITE, weight=BOLD)
        source = Text("cp_utils.py:360-377 — in-seq path",
                       font_size=12, color=CODE_COLOR)
        hdr = VGroup(title, source).arrange(DOWN, buff=0.06)

        # ── Ghost row 1: per-rank data (each rank holds bp+bn) ──
        label1 = dim_text("After attention, each rank holds:")
        g_rank_rows = VGroup()
        for r in range(CP_SIZE):
            bp, bn = zigzag_pairs[r]
            tokens = list(range(bp * BLOCK_SIZE, (bp + 1) * BLOCK_SIZE)) + \
                     list(range(bn * BLOCK_SIZE, (bn + 1) * BLOCK_SIZE))
            row = VGroup()
            for t in tokens:
                cell = make_cell(str(t), color=RANK_COLORS[r],
                                 w=CW, h=CH, font_size=FS)
                row.add(cell)
            row.arrange(RIGHT, buff=0.03)
            lbl = Text(f"R{r}", font_size=9, color=RANK_COLORS[r], weight=BOLD)
            lbl.next_to(row, LEFT, buff=0.1)
            g_rank_rows.add(VGroup(lbl, row))
        g_rank_rows.arrange(RIGHT, buff=0.25)
        row1 = VGroup(label1, g_rank_rows).arrange(DOWN, buff=0.08)

        # ── Ghost row 2: after AllGather (concatenated) ──
        code_ag = latex_label(r"\text{AllGather} \to \text{concat}")
        allgather_order = []
        for r in range(CP_SIZE):
            bp, bn = zigzag_pairs[r]
            allgather_order.extend(range(bp * BLOCK_SIZE, (bp + 1) * BLOCK_SIZE))
            allgather_order.extend(range(bn * BLOCK_SIZE, (bn + 1) * BLOCK_SIZE))
        g_ag = VGroup()
        for t in allgather_order:
            cell = make_cell(str(t), color=RANK_COLORS[t % CP_SIZE
                             if any(t in range(bp * BLOCK_SIZE, (bp + 1) * BLOCK_SIZE)
                                    or t in range(bn * BLOCK_SIZE, (bn + 1) * BLOCK_SIZE)
                                    for bp, bn in [zigzag_pairs[0]]) else 0],
                             w=CW, h=CH, font_size=FS)
            g_ag.add(cell)
        # Fix colors — use block_rank mapping
        block_rank = {}
        for r in range(CP_SIZE):
            bp, bn = zigzag_pairs[r]
            block_rank[bp] = r
            block_rank[bn] = r
        g_ag = VGroup()
        for t in allgather_order:
            b = t // BLOCK_SIZE
            cell = make_cell(str(t), color=RANK_COLORS[block_rank[b]],
                             w=CW, h=CH, font_size=FS)
            g_ag.add(cell)
        g_ag.arrange(RIGHT, buff=0.02)

        ag_block_seps = VGroup()
        for i in range(1, len(allgather_order) // BLOCK_SIZE):
            idx = i * BLOCK_SIZE
            if idx < len(g_ag):
                pos = (g_ag[idx - 1].get_right() + g_ag[idx].get_left()) / 2
                line = DashedLine(pos + UP * 0.15, pos + DOWN * 0.15,
                                  color=WHITE, stroke_opacity=0.4, dash_length=0.04)
                ag_block_seps.add(line)

        ag_block_lbls = VGroup()
        for i in range(len(allgather_order) // BLOCK_SIZE):
            start = i * BLOCK_SIZE
            sub = VGroup(*g_ag[start:start + BLOCK_SIZE])
            b_idx = allgather_order[start] // BLOCK_SIZE
            bl = dim_text(f"b{b_idx}")
            bl.next_to(sub, DOWN, buff=0.05)
            ag_block_lbls.add(bl)

        row2 = VGroup(code_ag, VGroup(g_ag, ag_block_seps, ag_block_lbls))
        row2.arrange(DOWN, buff=0.08)

        # ── Ghost row 3: split + reorder by cp_reverse_index ──
        code_ri = latex_label(
            r"\text{split} \to \text{cp\_reverse\_index} \to \text{cat}")
        g_reorder = VGroup()
        reorder_block_lbls = VGroup()
        for b in range(NUM_BLOCKS):
            blk = VGroup()
            for j in range(BLOCK_SIZE):
                t = b * BLOCK_SIZE + j
                cell = make_cell(str(t), color=RANK_COLORS[block_rank[b]],
                                 w=CW, h=CH, font_size=FS)
                blk.add(cell)
            blk.arrange(RIGHT, buff=0.02)
            g_reorder.add(blk)
        g_reorder.arrange(RIGHT, buff=0.06)
        for b in range(NUM_BLOCKS):
            bl = dim_text(f"b{b}")
            bl.next_to(g_reorder[b], DOWN, buff=0.05)
            reorder_block_lbls.add(bl)
        row3 = VGroup(code_ri, VGroup(g_reorder, reorder_block_lbls))
        row3.arrange(DOWN, buff=0.08)

        # ── Ghost row 4: final sequential result ──
        code_r = latex_label(
            r"\text{result} \to [0, 1, 2, \ldots, 15]", color="#55ff55")
        g_result = VGroup()
        for i in range(SEQ_LEN):
            cell = make_cell(str(i), color=RANK_COLORS[block_rank[i // BLOCK_SIZE]],
                             w=CW, h=CH, font_size=FS)
            g_result.add(cell)
        g_result.arrange(RIGHT, buff=0.03)
        check = Text("sequential order restored",
                      font_size=11, color="#aaddff")
        row4 = VGroup(code_r, g_result, check).arrange(DOWN, buff=0.06)

        all_content = VGroup(hdr, row1, row2, row3, row4).arrange(DOWN, buff=0.22)

        # ── Extract positions ──
        rank_cell_pos = {}
        for r in range(CP_SIZE):
            bp, bn = zigzag_pairs[r]
            tokens = list(range(bp * BLOCK_SIZE, (bp + 1) * BLOCK_SIZE)) + \
                     list(range(bn * BLOCK_SIZE, (bn + 1) * BLOCK_SIZE))
            row_cells = g_rank_rows[r][1]
            for j, t in enumerate(tokens):
                rank_cell_pos[t] = row_cells[j].get_center()

        ag_pos = {}
        for idx, t in enumerate(allgather_order):
            ag_pos[t] = g_ag[idx].get_center()

        reorder_pos = {}
        for b in range(NUM_BLOCKS):
            for j in range(BLOCK_SIZE):
                t = b * BLOCK_SIZE + j
                reorder_pos[t] = g_reorder[b][j].get_center()

        final_pos = {i: g_result[i].get_center() for i in range(SEQ_LEN)}

        rank_lbls = VGroup(*[g_rank_rows[r][0] for r in range(CP_SIZE)])

        # ── 16 persistent cells ──
        cells = {}
        for t in range(SEQ_LEN):
            b = t // BLOCK_SIZE
            cell = make_cell(str(t), color=RANK_COLORS[block_rank[b]],
                             w=CW, h=CH, font_size=FS)
            cell.move_to(rank_cell_pos[t])
            cells[t] = cell

        # ── Animate ──
        self.play(Write(title), FadeIn(source), run_time=0.8)
        self.play(self.camera.auto_zoom(all_content, margin=0.4), run_time=0.3)

        # Show per-rank data
        self.play(FadeIn(label1), run_time=0.2)
        self.play(
            FadeIn(rank_lbls),
            LaggedStart(*[FadeIn(cells[t]) for t in range(SEQ_LEN)],
                         lag_ratio=0.03),
            run_time=1)
        self.wait(0.5)

        # per-rank → AllGather concatenation
        self.play(
            FadeIn(code_ag),
            FadeOut(rank_lbls), FadeOut(label1),
            LaggedStart(*[cells[t].animate.move_to(ag_pos[t])
                          for t in range(SEQ_LEN)], lag_ratio=0.03),
            run_time=1.5)
        self.play(FadeIn(ag_block_seps), FadeIn(ag_block_lbls), run_time=0.3)
        self.wait(0.3)

        # AllGather → reorder by cp_reverse_index
        self.play(
            FadeIn(code_ri),
            FadeOut(ag_block_seps), FadeOut(ag_block_lbls),
            LaggedStart(*[cells[t].animate.move_to(reorder_pos[t])
                          for t in range(SEQ_LEN)], lag_ratio=0.03),
            run_time=1.5)
        self.play(FadeIn(reorder_block_lbls), run_time=0.3)
        self.wait(0.3)

        # Reorder → final sequential
        self.play(FadeIn(code_r), run_time=0.2)
        self.play(
            FadeOut(reorder_block_lbls),
            LaggedStart(*[cells[t].animate.move_to(final_pos[t])
                          for t in range(SEQ_LEN)], lag_ratio=0.02),
            run_time=1.5)
        self.play(FadeIn(check), run_time=0.5)
        self.wait(2)
