from manim import *
import numpy as np

BG = "#1C1C1C"
C_TXT = "#EAEAEA"
C_BLUE = "#58C4DD"
C_YELLOW = "#FFFF00"
C_GREEN = "#83C167"

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

class TestMatMul(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BG

        A_data = [[1, 2], [3, 4]]
        B_data = [[5, 6], [7, 8]]
        C_data = (np.array(A_data) @ np.array(B_data)).tolist()

        A = Matrix(A_data)
        B = Matrix(B_data)
        C = Matrix([["?", "?"], ["?", "?"]])

        VGroup(A, B, C).arrange(RIGHT, buff=1.0)

        times = MathTex(r"\times")
        times.move_to((A.get_right() + B.get_left()) / 2)
        eq = MathTex(r"=")
        eq.move_to((B.get_right() + C.get_left()) / 2)

        self.play(FadeIn(A), FadeIn(B), FadeIn(C), Write(times), Write(eq))
        self.wait(1)

        # Fill first cell
        i, j = 0, 0
        r_rect = SurroundingRectangle(A.get_rows()[i], color=C_BLUE, buff=0.06)
        c_rect = SurroundingRectangle(B.get_columns()[j], color=C_YELLOW, buff=0.06)
        self.play(Create(r_rect), Create(c_rect))

        result = MathTex(str(C_data[i][j]), color=C_GREEN)
        result.move_to(C.get_entries()[i * 2 + j])
        self.play(FadeOut(C.get_entries()[i * 2 + j]), FadeIn(result))
        self.play(FadeOut(r_rect), FadeOut(c_rect))

        self.wait(2)
