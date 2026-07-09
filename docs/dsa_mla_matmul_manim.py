"""
MLA Matrix Multiplication — 3Blue1Brown style.
Simple, efficient matrix multiplication animations.
"""
from manim import *
import numpy as np

BG = "#1C1C1C"
C_TXT = "#EAEAEA"
C_BLUE = "#58C4DD"
C_YELLOW = "#FFFF00"
C_GREEN = "#83C167"
C_PURPLE = "#9A72AC"
C_ORANGE = "#FF862F"
C_GREY = "#888888"

_TEX_TEMPLATE = TexTemplate()
_TEX_TEMPLATE.body = (
    r"\documentclass[preview]{article}" "\n"
    r"\usepackage{amsmath}" "\n"
    r"\usepackage{amssymb}" "\n"
    r"\begin{document}" "\n"
    r"YourTextHere" "\n"
    r"\end{document}"
)

config.tex_template = _TEX_TEMPLATE


class MLAAbsorbMatMul(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BG

        # Small demo matrices
        A_data = [[1, 2, 0, 1],
                  [0, 1, 2, 1],
                  [2, 0, 1, 1]]
        B_data = [[1, 0, 2, 1, 1],
                  [0, 1, 1, 2, 0],
                  [2, 1, 0, 1, 1],
                  [1, 1, 1, 0, 2]]
        C_data = (np.array(A_data) @ np.array(B_data)).tolist()
        nr, nc = 3, 5

        A_mat = Matrix(A_data)
        B_mat = Matrix(B_data)
        C_mat = Matrix([["\\cdot"] * nc for _ in range(nr)])

        VGroup(A_mat, B_mat, C_mat).arrange(RIGHT, buff=0.9)

        A_name = MathTex(r"q_{\text{nope}}", font_size=28, color=C_BLUE)
        A_name.next_to(A_mat, UP, buff=0.15)
        B_name = MathTex(r"W_{kc}", font_size=28, color=C_YELLOW)
        B_name.next_to(B_mat, UP, buff=0.15)
        C_name = MathTex(r"q_{\text{abs}}", font_size=28, color=C_GREEN)
        C_name.next_to(C_mat, UP, buff=0.15)

        times = MathTex(r"\times", font_size=28)
        times.move_to((A_mat.get_right() + B_mat.get_left()) / 2)
        eq = MathTex(r"=", font_size=28)
        eq.move_to((B_mat.get_right() + C_mat.get_left()) / 2)

        title = MathTex(
            r"\text{MLA Weight Absorption: } q_{\text{nope}} \times W_{kc}",
            font_size=24, color=C_TXT).to_edge(UP, buff=0.2)

        self.play(Write(title))
        self.play(
            FadeIn(A_mat), FadeIn(A_name),
            FadeIn(B_mat), FadeIn(B_name),
            FadeIn(C_mat), FadeIn(C_name),
            Write(times), Write(eq),
            run_time=1.0
        )
        self.wait(0.3)

        # Animate dot products for selected cells
        full_cells = [(0, 0), (0, 1)]
        quick_cells = [(0, 2), (0, 3), (0, 4), (1, 0), (1, 1)]

        for i in range(nr):
            for j in range(nc):
                if (i, j) in full_cells:
                    self.fill_cell_full(i, j, A_mat, B_mat, C_mat, A_data, B_data, C_data, nc)
                elif (i, j) in quick_cells:
                    self.fill_cell_quick(i, j, A_mat, B_mat, C_mat, C_data, nc)
                else:
                    self.fill_cell_instant(i, j, C_mat, C_data, nc)

        caption = MathTex(
            r"W_{kc} \text{ absorbed into } q \Rightarrow "
            r"\text{attn uses } c_{kv} \text{ directly}",
            font_size=18, color=C_GREEN)
        caption.to_edge(DOWN, buff=0.3)
        self.play(Write(caption))
        self.wait(2)

    def fill_cell_full(self, i, j, A_mat, B_mat, C_mat, A_data, B_data, C_data, nc):
        r_rect = SurroundingRectangle(A_mat.get_rows()[i], color=C_BLUE, buff=0.05)
        c_rect = SurroundingRectangle(B_mat.get_columns()[j], color=C_YELLOW, buff=0.05)
        self.play(Create(r_rect), Create(c_rect), run_time=0.3)

        terms = " + ".join(f"{A_data[i][k]}\\cdot{B_data[k][j]}"
                           for k in range(len(A_data[i])))
        eq_str = f"{terms} = {C_data[i][j]}"
        eq_mob = MathTex(eq_str, font_size=20, color=C_TXT)
        eq_mob.to_edge(DOWN, buff=1.2)
        self.play(Write(eq_mob), run_time=0.8)
        self.wait(0.3)

        cell_entry = C_mat.get_entries()[i * nc + j]
        val = MathTex(str(C_data[i][j]), font_size=28, color=C_GREEN)
        val.move_to(cell_entry.get_center())
        self.play(FadeOut(cell_entry), FadeIn(val), run_time=0.4)
        self.play(FadeOut(r_rect), FadeOut(c_rect), FadeOut(eq_mob), run_time=0.3)

    def fill_cell_quick(self, i, j, A_mat, B_mat, C_mat, C_data, nc):
        r_rect = SurroundingRectangle(A_mat.get_rows()[i], color=C_BLUE, buff=0.05)
        c_rect = SurroundingRectangle(B_mat.get_columns()[j], color=C_YELLOW, buff=0.05)
        self.play(Create(r_rect), Create(c_rect), run_time=0.2)

        cell_entry = C_mat.get_entries()[i * nc + j]
        val = MathTex(str(C_data[i][j]), font_size=28, color=C_GREEN)
        val.move_to(cell_entry.get_center())
        self.play(FadeOut(cell_entry), FadeIn(val), run_time=0.3)
        self.play(FadeOut(r_rect), FadeOut(c_rect), run_time=0.2)

    def fill_cell_instant(self, i, j, C_mat, C_data, nc):
        cell_entry = C_mat.get_entries()[i * nc + j]
        val = MathTex(str(C_data[i][j]), font_size=28, color=C_GREEN)
        val.move_to(cell_entry.get_center())
        cell_entry.become(val)


class MLADownProjMatMul(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BG

        A_data = [[1, 2, 0, 1]]
        B_data = [[1, 0, 2, 1, 1],
                  [0, 1, 1, 2, 0],
                  [2, 1, 0, 1, 1],
                  [1, 1, 1, 0, 2]]
        C_data = (np.array(A_data) @ np.array(B_data)).tolist()
        nr, nc = 1, 5

        A_mat = Matrix(A_data)
        B_mat = Matrix(B_data)
        C_mat = Matrix([["\\cdot"] * nc])

        VGroup(A_mat, B_mat, C_mat).arrange(RIGHT, buff=0.9)

        A_name = MathTex(r"\mathbf{x}", font_size=28, color=C_BLUE)
        A_name.next_to(A_mat, UP, buff=0.15)
        B_name = MathTex(r"W_{\!A}", font_size=28, color=C_YELLOW)
        B_name.next_to(B_mat, UP, buff=0.15)
        C_name = MathTex(r"\text{qkv}", font_size=28, color=C_GREEN)
        C_name.next_to(C_mat, UP, buff=0.15)

        times = MathTex(r"\times", font_size=28)
        times.move_to((A_mat.get_right() + B_mat.get_left()) / 2)
        eq = MathTex(r"=", font_size=28)
        eq.move_to((B_mat.get_right() + C_mat.get_left()) / 2)

        title = MathTex(
            r"\text{MLA Down-Projection: } \mathbf{x} \times W_A",
            font_size=24, color=C_TXT).to_edge(UP, buff=0.2)

        self.play(Write(title))
        self.play(
            FadeIn(A_mat), FadeIn(A_name),
            FadeIn(B_mat), FadeIn(B_name),
            FadeIn(C_mat), FadeIn(C_name),
            Write(times), Write(eq),
            run_time=1.0
        )
        self.wait(0.3)

        i = 0
        for j in range(nc):
            r_rect = SurroundingRectangle(A_mat.get_rows()[i], color=C_BLUE, buff=0.05)
            c_rect = SurroundingRectangle(B_mat.get_columns()[j], color=C_YELLOW, buff=0.05)

            if j == 0:
                self.play(Create(r_rect), Create(c_rect), run_time=0.3)
                terms = " + ".join(f"{A_data[i][k]}\\cdot{B_data[k][j]}"
                                   for k in range(len(A_data[i])))
                eq_str = f"{terms} = {C_data[i][j]}"
                eq_mob = MathTex(eq_str, font_size=20, color=C_TXT)
                eq_mob.to_edge(DOWN, buff=1.2)
                self.play(Write(eq_mob), run_time=0.8)
                self.wait(0.3)
                cell_entry = C_mat.get_entries()[j]
                val = MathTex(str(C_data[i][j]), font_size=28, color=C_GREEN)
                val.move_to(cell_entry.get_center())
                self.play(FadeOut(cell_entry), FadeIn(val), run_time=0.4)
                self.play(FadeOut(r_rect), FadeOut(c_rect), FadeOut(eq_mob), run_time=0.3)
            else:
                self.play(Create(r_rect), Create(c_rect), run_time=0.2)
                cell_entry = C_mat.get_entries()[j]
                val = MathTex(str(C_data[i][j]), font_size=28, color=C_GREEN)
                val.move_to(cell_entry.get_center())
                self.play(FadeOut(cell_entry), FadeIn(val), run_time=0.3)
                self.play(FadeOut(r_rect), FadeOut(c_rect), run_time=0.2)

        caption = MathTex(
            r"\text{qkv} = [c_q(1536) \mid c_{kv}(512) \mid k_{pe}(64)]",
            font_size=18, color=C_GREEN)
        caption.to_edge(DOWN, buff=0.3)
        self.play(Write(caption))
        self.wait(2)


class MLAVAbsorbMatMul(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BG

        A_data = [[1, 2, 0, 1],
                  [0, 1, 2, 1],
                  [2, 0, 1, 1]]
        B_data = [[1, 0, 2],
                  [0, 1, 1],
                  [2, 1, 0],
                  [1, 1, 1]]
        C_data = (np.array(A_data) @ np.array(B_data)).tolist()
        nr, nc = 3, 3

        A_mat = Matrix(A_data)
        B_mat = Matrix(B_data)
        C_mat = Matrix([["\\cdot"] * nc for _ in range(nr)])

        VGroup(A_mat, B_mat, C_mat).arrange(RIGHT, buff=0.9)

        A_name = MathTex(r"\text{attn}", font_size=28, color=C_PURPLE)
        A_name.next_to(A_mat, UP, buff=0.15)
        B_name = MathTex(r"W_{vc}", font_size=28, color=C_ORANGE)
        B_name.next_to(B_mat, UP, buff=0.15)
        C_name = MathTex(r"\text{out}", font_size=28, color=C_GREEN)
        C_name.next_to(C_mat, UP, buff=0.15)

        times = MathTex(r"\times", font_size=28)
        times.move_to((A_mat.get_right() + B_mat.get_left()) / 2)
        eq = MathTex(r"=", font_size=28)
        eq.move_to((B_mat.get_right() + C_mat.get_left()) / 2)

        title = MathTex(
            r"\text{MLA V-Absorption: } \text{attn} \times W_{vc}",
            font_size=24, color=C_TXT).to_edge(UP, buff=0.2)

        self.play(Write(title))
        self.play(
            FadeIn(A_mat), FadeIn(A_name),
            FadeIn(B_mat), FadeIn(B_name),
            FadeIn(C_mat), FadeIn(C_name),
            Write(times), Write(eq),
            run_time=1.0
        )
        self.wait(0.3)

        full_cells = [(0, 0)]
        quick_cells = [(0, 1), (1, 0)]

        for i in range(nr):
            for j in range(nc):
                r_rect = SurroundingRectangle(A_mat.get_rows()[i], color=C_PURPLE, buff=0.05)
                c_rect = SurroundingRectangle(B_mat.get_columns()[j], color=C_ORANGE, buff=0.05)

                if (i, j) in full_cells:
                    self.play(Create(r_rect), Create(c_rect), run_time=0.3)
                    terms = " + ".join(f"{A_data[i][k]}\\cdot{B_data[k][j]}"
                                       for k in range(len(A_data[i])))
                    eq_str = f"{terms} = {C_data[i][j]}"
                    eq_mob = MathTex(eq_str, font_size=20, color=C_TXT)
                    eq_mob.to_edge(DOWN, buff=1.2)
                    self.play(Write(eq_mob), run_time=0.8)
                    self.wait(0.3)
                    cell_entry = C_mat.get_entries()[i * nc + j]
                    val = MathTex(str(C_data[i][j]), font_size=28, color=C_GREEN)
                    val.move_to(cell_entry.get_center())
                    self.play(FadeOut(cell_entry), FadeIn(val), run_time=0.4)
                    self.play(FadeOut(r_rect), FadeOut(c_rect), FadeOut(eq_mob), run_time=0.3)
                elif (i, j) in quick_cells:
                    self.play(Create(r_rect), Create(c_rect), run_time=0.2)
                    cell_entry = C_mat.get_entries()[i * nc + j]
                    val = MathTex(str(C_data[i][j]), font_size=28, color=C_GREEN)
                    val.move_to(cell_entry.get_center())
                    self.play(FadeOut(cell_entry), FadeIn(val), run_time=0.3)
                    self.play(FadeOut(r_rect), FadeOut(c_rect), run_time=0.2)
                else:
                    cell_entry = C_mat.get_entries()[i * nc + j]
                    val = MathTex(str(C_data[i][j]), font_size=28, color=C_GREEN)
                    val.move_to(cell_entry.get_center())
                    cell_entry.become(val)

        caption = MathTex(
            r"\text{output} = W_o \cdot \text{flatten}(\text{attn} \cdot W_{vc})",
            font_size=18, color=C_GREEN)
        caption.to_edge(DOWN, buff=0.3)
        self.play(Write(caption))
        self.wait(2)
