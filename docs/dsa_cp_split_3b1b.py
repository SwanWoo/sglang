"""
DSA Prefill CP Split — 3Blue1Brown style matrix animations.

Shows round-robin and in-seq (zigzag) context-parallel split strategies
as tensor reshape/slice operations on actual matrices.

Usage:
  manim -qm --media_dir docs/media docs/dsa_cp_split_3b1b.py RoundRobinSplit
  manim -qm --media_dir docs/media docs/dsa_cp_split_3b1b.py InSeqZigzagSplit
  manim -qm --media_dir docs/media docs/dsa_cp_split_3b1b.py SplitComparison
"""

from manim import *
import numpy as np

# ─── 3b1b Palette ─────────────────────────────────────────────
BG = "#1C1C1C"
C_TXT = "#EAEAEA"
C_DIM = "#888888"
C_BLUE = "#58C4DD"
C_YELLOW = "#FFFF00"
C_GREEN = "#83C167"
C_TEAL = "#5CD0B3"
C_PURPLE = "#9A72AC"
C_ORANGE = "#FF862F"
C_RED = "#FC6255"
C_PINK = "#E48DBF"

RANK_C = [C_BLUE, C_YELLOW, C_GREEN, C_ORANGE]
RANK_C_DIM = ["#2E6A78", "#7A7A00", "#426330", "#7A4118"]

SEQ_LEN = 16
CP_SIZE = 4
TOKENS_PER_RANK = SEQ_LEN // CP_SIZE
NUM_BLOCKS = 2 * CP_SIZE
BLOCK_SIZE = SEQ_LEN // NUM_BLOCKS

_TEX = TexTemplate()
_TEX.body = (
    r"\documentclass[preview]{article}" "\n"
    r"\usepackage{amsmath}" "\n"
    r"\usepackage{amssymb}" "\n"
    r"\begin{document}" "\n"
    r"YourTextHere" "\n"
    r"\end{document}"
)
config.tex_template = _TEX


def colored_entry(val, color):
    return MathTex(str(val), font_size=32, color=color)


def make_vec_entries(vals, color_fn=None):
    entries = []
    for v in vals:
        c = color_fn(v) if color_fn else C_TXT
        entries.append(colored_entry(v, c))
    return entries


def highlight_rect(mob, color, buff=0.08):
    return SurroundingRectangle(mob, color=color, buff=buff, stroke_width=2.5)


# ═══════════════════════════════════════════════════════════════
# Scene 1: Round-Robin Split — view/transpose/slice on a matrix
# ═══════════════════════════════════════════════════════════════

class RoundRobinSplit(Scene):
    def construct(self):
        self.camera.background_color = BG

        # ── Title ──
        title = MathTex(
            r"\text{Round-Robin Split: }",
            r"\text{token}[i] \to \text{rank}(i \bmod 4)",
            font_size=36, color=C_TXT,
        )
        title[1].set_color(C_BLUE)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=1.2)

        # ── Step 1: Original 1D sequence ──
        seq = list(range(SEQ_LEN))
        seq_entries = [MathTex(str(s), font_size=26, color=C_TXT) for s in seq]

        seq_row = VGroup(*seq_entries).arrange(RIGHT, buff=0.22)
        seq_row.next_to(title, DOWN, buff=0.7)
        if seq_row.width > config.frame_width - 1:
            seq_row.scale_to_fit_width(config.frame_width - 1)

        lbracket = MathTex(r"[", font_size=36, color=C_DIM).next_to(seq_row, LEFT, buff=0.08)
        rbracket = MathTex(r"]", font_size=36, color=C_DIM).next_to(seq_row, RIGHT, buff=0.08)
        seq_label = MathTex(r"\text{input\_ids}", font_size=24, color=C_DIM)
        seq_label.next_to(lbracket, LEFT, buff=0.15)

        seq_group = VGroup(seq_label, lbracket, seq_row, rbracket)

        self.play(
            FadeIn(seq_label),
            FadeIn(lbracket), FadeIn(rbracket),
            LaggedStart(*[FadeIn(e, shift=UP * 0.15) for e in seq_entries], lag_ratio=0.03),
            run_time=1.5,
        )
        self.wait(0.5)

        # ── Color by rank ──
        color_anims = []
        for i, e in enumerate(seq_entries):
            rank = i % CP_SIZE
            color_anims.append(e.animate.set_color(RANK_C[rank]))
        self.play(*color_anims, run_time=0.8)
        self.wait(0.3)

        # ── Step 2: Reshape to (cp_size, tokens_per_rank) matrix ──
        reshape_tex = MathTex(
            r"\text{view}(",
            r"4",
            r",\ ",
            r"4",
            r")",
            font_size=28, color=C_TXT,
        )
        reshape_tex[1].set_color(C_BLUE)
        reshape_tex[3].set_color(C_TEAL)
        reshape_tex.next_to(seq_group, DOWN, buff=0.4)

        arrow_down = MathTex(r"\Downarrow", font_size=32, color=C_DIM)
        arrow_down.next_to(reshape_tex, LEFT, buff=0.15)

        self.play(FadeIn(arrow_down), Write(reshape_tex), run_time=0.8)
        self.wait(0.3)

        # Build the 4x4 matrix with colored entries
        mat_data = []
        for col in range(TOKENS_PER_RANK):
            row = []
            for r in range(CP_SIZE):
                row.append(col * CP_SIZE + r)
            mat_data.append(row)

        # Actually for round-robin: token i goes to rank (i % cp_size)
        # Row r of the reshaped matrix = tokens [r, r+4, r+8, r+12]
        # But view(4, 4) on [0..15] gives:
        #   row 0: [0, 1, 2, 3]
        #   row 1: [4, 5, 6, 7]
        #   row 2: [8, 9, 10, 11]
        #   row 3: [12, 13, 14, 15]
        # Then transpose gives:
        #   row 0: [0, 4, 8, 12]   <- rank 0
        #   row 1: [1, 5, 9, 13]   <- rank 1
        #   ...

        # Step 2a: view as 4x4 (row-major)
        view_data = [[r * CP_SIZE + c for c in range(TOKENS_PER_RANK)]
                     for r in range(CP_SIZE)]

        mat_entries = [[MathTex(str(view_data[r][c]), font_size=28,
                                color=RANK_C[view_data[r][c] % CP_SIZE])
                        for c in range(TOKENS_PER_RANK)]
                       for r in range(CP_SIZE)]

        mat = MobjectMatrix(
            mat_entries,
            h_buff=1.0, v_buff=0.8,
            bracket_h_buff=0.15,
            bracket_v_buff=0.15,
        )
        mat.next_to(reshape_tex, DOWN, buff=0.5)
        mat.scale(0.85)

        view_label = MathTex(
            r"\text{view}(4, 4)",
            font_size=22, color=C_DIM,
        )
        view_label.next_to(mat, LEFT, buff=0.3)

        self.play(
            FadeIn(mat),
            FadeIn(view_label),
            run_time=1.0,
        )
        self.wait(0.5)

        # ── Highlight columns = same rank ──
        col_rects = []
        for c in range(TOKENS_PER_RANK):
            col_entries = VGroup(*[mat_entries[r][c] for r in range(CP_SIZE)])
            rect = highlight_rect(col_entries, RANK_C[c], buff=0.1)
            col_rects.append(rect)

        # But wait — in view(4,4), columns don't correspond to ranks yet.
        # We need transpose. Let me show that.

        # Actually, let me show the correct reshape:
        # view(-1, cp_size) gives shape (tokens_per_rank, cp_size)
        # Then [:, rank] selects rank's tokens
        # Or equivalently: view(cp_size, -1).T

        # Simpler approach: show view(-1, cp_size) = (4, 4) where
        #   each ROW is [0,1,2,3], [4,5,6,7], etc.
        #   and each COLUMN is a rank
        # This is what the code actually does!

        # Highlight columns as ranks
        col_labels = []
        for c in range(CP_SIZE):
            col_mobs = VGroup(*[mat_entries[r][c] for r in range(TOKENS_PER_RANK)])
            rect = highlight_rect(col_mobs, RANK_C[c], buff=0.12)
            col_rects.append(rect)
            lbl = MathTex(rf"\text{{rank}}\ {c}", font_size=20, color=RANK_C[c])
            lbl.next_to(col_mobs, DOWN, buff=0.25)
            col_labels.append(lbl)

        # Oops, I added to col_rects twice. Let me fix.
        # The first 4 are empty placeholders. Let me redo.
        col_rects_final = []
        col_labels_final = []
        for c in range(CP_SIZE):
            col_mobs = VGroup(*[mat_entries[r][c] for r in range(TOKENS_PER_RANK)])
            rect = highlight_rect(col_mobs, RANK_C[c], buff=0.12)
            col_rects_final.append(rect)
            lbl = MathTex(rf"\text{{rank}}\ {c}", font_size=20, color=RANK_C[c])
            lbl.next_to(col_mobs, DOWN, buff=0.25)
            col_labels_final.append(lbl)

        slice_tex = MathTex(
            r"[:, \text{cp\_rank}]",
            font_size=24, color=C_TEAL,
        )
        slice_tex.next_to(mat, RIGHT, buff=0.4)

        self.play(
            LaggedStart(*[Create(r) for r in col_rects_final], lag_ratio=0.15),
            LaggedStart(*[FadeIn(l) for l in col_labels_final], lag_ratio=0.15),
            Write(slice_tex),
            run_time=1.5,
        )
        self.wait(0.8)

        # ── Step 3: Extract columns → rank sub-sequences ──
        self.play(
            FadeOut(mat), FadeOut(view_label),
            FadeOut(reshape_tex), FadeOut(arrow_down),
            FadeOut(VGroup(*col_rects_final)),
            FadeOut(VGroup(*col_labels_final)),
            FadeOut(slice_tex),
            seq_group.animate.shift(UP * 0.3).scale(0.8),
            run_time=0.6,
        )

        rank_rows = VGroup()
        rank_labels = VGroup()
        for r in range(CP_SIZE):
            tokens = [t for t in range(SEQ_LEN) if t % CP_SIZE == r]
            entries = VGroup(*[MathTex(str(t), font_size=28, color=RANK_C[r])
                               for t in tokens])
            entries.arrange(RIGHT, buff=0.4)
            lb = MathTex(r"[", font_size=34, color=RANK_C_DIM[r])
            rb = MathTex(r"]", font_size=34, color=RANK_C_DIM[r])
            lb.next_to(entries, LEFT, buff=0.08)
            rb.next_to(entries, RIGHT, buff=0.08)

            rank_lbl = MathTex(
                rf"\text{{rank}}\ {r}:",
                font_size=24, color=RANK_C[r],
            )
            row = VGroup(rank_lbl, lb, entries, rb).arrange(RIGHT, buff=0.15)
            rank_rows.add(row)

        rank_rows.arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        rank_rows.move_to(DOWN * 0.5)

        self.play(
            LaggedStart(*[FadeIn(row, shift=LEFT * 0.3) for row in rank_rows],
                         lag_ratio=0.2),
            run_time=2.0,
        )
        self.wait(0.5)

        # ── Summary formula ──
        summary = MathTex(
            r"\text{input\_ids}",
            r".\text{view}(-1,\ \text{cp\_size})",
            r"[:,\ \text{cp\_rank}]",
            font_size=26, color=C_TXT,
        )
        summary[0].set_color(C_DIM)
        summary[1].set_color(C_BLUE)
        summary[2].set_color(C_TEAL)
        summary.to_edge(DOWN, buff=0.5)

        underline = Underline(summary, color=C_BLUE, stroke_width=1.5)

        self.play(Write(summary), Create(underline), run_time=1.2)
        self.wait(0.3)

        # Pattern note
        pattern = MathTex(
            r"\text{stride interleave — every }",
            r"4",
            r"\text{-th token}",
            font_size=22, color=C_DIM,
        )
        pattern[1].set_color(C_BLUE)
        pattern.next_to(summary, UP, buff=0.25)
        self.play(FadeIn(pattern), run_time=0.8)
        self.wait(2)


# ═══════════════════════════════════════════════════════════════
# Scene 2: In-Seq Zigzag Split — block pairing
# ═══════════════════════════════════════════════════════════════

class InSeqZigzagSplit(Scene):
    def construct(self):
        self.camera.background_color = BG

        # ── Title ──
        title = MathTex(
            r"\text{In-Seq Split: }",
            r"\text{zigzag block pairing}",
            font_size=36, color=C_TXT,
        )
        title[1].set_color(C_TEAL)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=1.2)

        # ── Step 1: Show sequence as blocks ──
        params = MathTex(
            r"\text{blocks} = 2 \times \text{cp\_size} = ",
            r"8",
            r",\quad \text{block\_size} = ",
            r"2",
            font_size=24, color=C_DIM,
        )
        params[1].set_color(C_TEAL)
        params[3].set_color(C_TEAL)
        params.next_to(title, DOWN, buff=0.3)
        self.play(FadeIn(params), run_time=0.6)

        # Build blocks as small matrices
        blocks = VGroup()
        block_labels = VGroup()
        for b in range(NUM_BLOCKS):
            tokens = list(range(b * BLOCK_SIZE, (b + 1) * BLOCK_SIZE))
            entries = VGroup(*[MathTex(str(t), font_size=24, color=C_TXT)
                               for t in tokens])
            entries.arrange(RIGHT, buff=0.15)
            box = SurroundingRectangle(entries, color=C_DIM, buff=0.12,
                                       stroke_width=1.5, corner_radius=0.05)
            block = VGroup(box, entries)
            blocks.add(block)

            lbl = MathTex(rf"b_{b}", font_size=20, color=C_DIM)
            block_labels.add(lbl)

        blocks.arrange(RIGHT, buff=0.15)
        blocks.next_to(params, DOWN, buff=0.6)
        if blocks.width > config.frame_width - 1:
            blocks.scale_to_fit_width(config.frame_width - 1)

        for lbl, block in zip(block_labels, blocks):
            lbl.next_to(block, DOWN, buff=0.15)

        self.play(
            LaggedStart(*[FadeIn(b, shift=UP * 0.15) for b in blocks], lag_ratio=0.06),
            run_time=1.5,
        )
        self.play(
            LaggedStart(*[FadeIn(l) for l in block_labels], lag_ratio=0.04),
            run_time=0.8,
        )
        self.wait(0.5)

        # ── Step 2: Show zigzag pairing ──
        zigzag_formula = MathTex(
            r"\text{rank}\ r \gets b_r + b_{2N-1-r}",
            font_size=26, color=C_TEAL,
        )
        zigzag_formula.next_to(blocks, DOWN, buff=0.8)
        self.play(Write(zigzag_formula), run_time=0.8)

        # Draw curved arrows for each pair
        arrows = VGroup()
        pair_info = []
        for r in range(CP_SIZE):
            b_head = r
            b_tail = 2 * CP_SIZE - 1 - r
            pair_info.append((b_head, b_tail))

            start = blocks[b_head].get_bottom() + DOWN * 0.1
            end = blocks[b_tail].get_bottom() + DOWN * 0.1

            arc_height = 0.3 + r * 0.25
            arrow = CurvedArrow(
                start, end,
                color=RANK_C[r],
                stroke_width=2.5,
                angle=-TAU / (6 + r),
                tip_length=0.15,
            )
            arrows.add(arrow)

            # Color the paired blocks
            blocks[b_head][0].set_stroke(RANK_C[r], width=2.5)
            blocks[b_tail][0].set_stroke(RANK_C[r], width=2.5)
            for e in blocks[b_head][1]:
                e.set_color(RANK_C[r])
            for e in blocks[b_tail][1]:
                e.set_color(RANK_C[r])
            block_labels[b_head].set_color(RANK_C[r])
            block_labels[b_tail].set_color(RANK_C[r])

        self.play(
            LaggedStart(*[Create(a) for a in arrows], lag_ratio=0.2),
            run_time=2.0,
        )
        self.wait(0.8)

        # ── Step 3: Collapse into rank sub-sequences ──
        self.play(
            FadeOut(arrows),
            FadeOut(zigzag_formula),
            FadeOut(params),
            run_time=0.5,
        )

        # Move blocks+labels up
        top_group = VGroup(blocks, block_labels)
        self.play(top_group.animate.shift(UP * 0.5).scale(0.85), run_time=0.5)

        # Build rank rows
        rank_rows = VGroup()
        for r in range(CP_SIZE):
            b_head, b_tail = pair_info[r]
            head_tokens = list(range(b_head * BLOCK_SIZE, (b_head + 1) * BLOCK_SIZE))
            tail_tokens = list(range(b_tail * BLOCK_SIZE, (b_tail + 1) * BLOCK_SIZE))
            all_tokens = head_tokens + tail_tokens

            head_entries = VGroup(*[MathTex(str(t), font_size=28, color=RANK_C[r])
                                    for t in head_tokens])
            head_entries.arrange(RIGHT, buff=0.3)
            head_box = SurroundingRectangle(head_entries, color=RANK_C[r], buff=0.1,
                                             stroke_width=1.5, corner_radius=0.05)
            head_lbl = MathTex(rf"b_{b_head}", font_size=18, color=RANK_C_DIM[r])
            head_lbl.next_to(head_box, DOWN, buff=0.08)

            divider = MathTex(r"\mid", font_size=32, color=RANK_C_DIM[r])

            tail_entries = VGroup(*[MathTex(str(t), font_size=28, color=RANK_C[r])
                                    for t in tail_tokens])
            tail_entries.arrange(RIGHT, buff=0.3)
            tail_box = SurroundingRectangle(tail_entries, color=RANK_C[r], buff=0.1,
                                             stroke_width=1.5, corner_radius=0.05)
            tail_lbl = MathTex(rf"b_{b_tail}", font_size=18, color=RANK_C_DIM[r])
            tail_lbl.next_to(tail_box, DOWN, buff=0.08)

            rank_lbl = MathTex(
                rf"\text{{rank}}\ {r}:",
                font_size=24, color=RANK_C[r],
            )

            row_content = VGroup(
                VGroup(head_box, head_entries, head_lbl),
                divider,
                VGroup(tail_box, tail_entries, tail_lbl),
            ).arrange(RIGHT, buff=0.2)
            row = VGroup(rank_lbl, row_content).arrange(RIGHT, buff=0.3)
            rank_rows.add(row)

        rank_rows.arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        rank_rows.move_to(DOWN * 0.8)

        self.play(
            LaggedStart(*[FadeIn(row, shift=LEFT * 0.3) for row in rank_rows],
                         lag_ratio=0.2),
            run_time=2.0,
        )
        self.wait(0.5)

        # ── Summary ──
        summary = MathTex(
            r"\text{head-tail pairing} \Rightarrow \text{balanced load across ranks}",
            font_size=24, color=C_TEAL,
        )
        summary.to_edge(DOWN, buff=0.5)
        underline = Underline(summary, color=C_TEAL, stroke_width=1.5)
        self.play(Write(summary), Create(underline), run_time=1.0)
        self.wait(2)


# ═══════════════════════════════════════════════════════════════
# Scene 3: Side-by-side Comparison
# ═══════════════════════════════════════════════════════════════

class SplitComparison(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = MathTex(
            r"\text{Round-Robin}",
            r"\quad \text{vs} \quad",
            r"\text{In-Seq (Zigzag)}",
            font_size=36, color=C_TXT,
        )
        title[0].set_color(C_BLUE)
        title[2].set_color(C_TEAL)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=1.0)

        # Vertical divider
        divider = DashedLine(
            UP * 2.5, DOWN * 3,
            color=C_DIM, dash_length=0.1, stroke_width=1.2,
        )
        self.play(Create(divider), run_time=0.4)

        # ════ LEFT: Round-Robin ════
        rr_formula = MathTex(
            r"\text{token}[i] \to \text{rank}(i \bmod 4)",
            font_size=22, color=C_BLUE,
        )
        rr_formula.move_to(LEFT * 3.5 + UP * 2)

        # Build mini sequence colored by rank
        rr_seq = VGroup()
        for i in range(SEQ_LEN):
            e = MathTex(str(i), font_size=16, color=RANK_C[i % CP_SIZE])
            rr_seq.add(e)
        rr_seq.arrange(RIGHT, buff=0.08)
        rr_seq.scale_to_fit_width(3.2)
        rr_seq.next_to(rr_formula, DOWN, buff=0.3)

        # Rank rows
        rr_ranks = VGroup()
        for r in range(CP_SIZE):
            tokens = [t for t in range(SEQ_LEN) if t % CP_SIZE == r]
            entries = VGroup(*[MathTex(str(t), font_size=18, color=RANK_C[r])
                               for t in tokens])
            entries.arrange(RIGHT, buff=0.2)
            lbl = MathTex(rf"R{r}", font_size=18, color=RANK_C[r])
            row = VGroup(lbl, entries).arrange(RIGHT, buff=0.15)
            rr_ranks.add(row)
        rr_ranks.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        rr_ranks.next_to(rr_seq, DOWN, buff=0.35)

        # ════ RIGHT: In-Seq ════
        is_formula = MathTex(
            r"\text{rank}\ r \gets b_r + b_{7-r}",
            font_size=22, color=C_TEAL,
        )
        is_formula.move_to(RIGHT * 3.5 + UP * 2)

        # Build mini sequence colored by inseq rank
        inseq_token_rank = {}
        for r in range(CP_SIZE):
            b_head, b_tail = r, 2 * CP_SIZE - 1 - r
            for t in range(b_head * BLOCK_SIZE, (b_head + 1) * BLOCK_SIZE):
                inseq_token_rank[t] = r
            for t in range(b_tail * BLOCK_SIZE, (b_tail + 1) * BLOCK_SIZE):
                inseq_token_rank[t] = r

        is_seq = VGroup()
        for i in range(SEQ_LEN):
            e = MathTex(str(i), font_size=16, color=RANK_C[inseq_token_rank[i]])
            is_seq.add(e)
        is_seq.arrange(RIGHT, buff=0.08)
        is_seq.scale_to_fit_width(3.2)
        is_seq.next_to(is_formula, DOWN, buff=0.3)

        # Rank rows with block dividers
        is_ranks = VGroup()
        for r in range(CP_SIZE):
            b_head, b_tail = r, 2 * CP_SIZE - 1 - r
            head_tokens = list(range(b_head * BLOCK_SIZE, (b_head + 1) * BLOCK_SIZE))
            tail_tokens = list(range(b_tail * BLOCK_SIZE, (b_tail + 1) * BLOCK_SIZE))

            h_entries = VGroup(*[MathTex(str(t), font_size=18, color=RANK_C[r])
                                 for t in head_tokens])
            h_entries.arrange(RIGHT, buff=0.15)
            div = MathTex(r"|", font_size=22, color=RANK_C_DIM[r])
            t_entries = VGroup(*[MathTex(str(t), font_size=18, color=RANK_C[r])
                                 for t in tail_tokens])
            t_entries.arrange(RIGHT, buff=0.15)

            lbl = MathTex(rf"R{r}", font_size=18, color=RANK_C[r])
            row = VGroup(lbl, h_entries, div, t_entries).arrange(RIGHT, buff=0.12)
            is_ranks.add(row)
        is_ranks.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        is_ranks.next_to(is_seq, DOWN, buff=0.35)

        # Animate both sides
        self.play(
            Write(rr_formula), Write(is_formula),
            run_time=0.8,
        )
        self.play(
            FadeIn(rr_seq), FadeIn(is_seq),
            run_time=0.5,
        )
        self.play(
            LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rr_ranks], lag_ratio=0.1),
            LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in is_ranks], lag_ratio=0.1),
            run_time=1.5,
        )
        self.wait(0.5)

        # ── Feature comparison ──
        features_left = VGroup(
            MathTex(r"\bullet\ \text{stride interleave}", font_size=16, color=C_DIM),
            MathTex(r"\bullet\ \text{view} \to \text{transpose} \to \text{reshape}", font_size=16, color=C_DIM),
            MathTex(r"\bullet\ \text{gather / reduce-scatter}", font_size=16, color=C_DIM),
        )
        features_left.arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        features_left.next_to(rr_ranks, DOWN, buff=0.35)

        features_right = VGroup(
            MathTex(r"\bullet\ \text{block-level zigzag}", font_size=16, color=C_DIM),
            MathTex(r"\bullet\ \text{cp\_reverse\_index}", font_size=16, color=C_DIM),
            MathTex(r"\bullet\ \text{DeepEP all-to-all}", font_size=16, color=C_DIM),
        )
        features_right.arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        features_right.next_to(is_ranks, DOWN, buff=0.35)

        self.play(
            LaggedStart(*[FadeIn(f) for f in features_left], lag_ratio=0.1),
            LaggedStart(*[FadeIn(f) for f in features_right], lag_ratio=0.1),
            run_time=1.5,
        )

        # ── Legend ──
        legend = VGroup()
        for r in range(CP_SIZE):
            sq = Square(side_length=0.15, fill_color=RANK_C[r],
                         fill_opacity=1, stroke_width=0)
            lb = MathTex(rf"\text{{Rank}}\ {r}", font_size=16, color=C_TXT)
            lb.next_to(sq, RIGHT, buff=0.08)
            legend.add(VGroup(sq, lb))
        legend.arrange(RIGHT, buff=0.4)
        legend.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(legend), run_time=0.5)
        self.wait(3)
