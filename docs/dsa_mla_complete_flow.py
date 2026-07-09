"""
MLA Complete Flow — Clean layout using positioning best practices

Uses next_to(), move_to(), to_corner() properly.
No complex animate chains. Each step: position first, then show.
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
C_RED = "#FF6B6B"
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


class MLACompleteFlow(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BG

        title = MathTex(
            r"\text{MLA: Complete Forward Pass}",
            font_size=36, color=C_TXT)
        title.to_edge(UP, buff=0.2)
        self.play(Write(title), run_time=0.8)
        self.wait(0.5)

        # ═══════════════════════════════════════════════════════
        # Step 1: x · W_A  (down-projection)
        # ═══════════════════════════════════════════════════════
        step1_label = MathTex(
            r"\textcircled{1}\ \mathbf{x} \times W_A\ \rightarrow \text{qkv}",
            font_size=28, color=C_BLUE)
        step1_label.next_to(title, DOWN, buff=0.3)

        x_data = [[1, 2, 0, 1]]
        WA_data = [[1, 0, 2],
                   [0, 1, 1],
                   [2, 1, 0],
                   [1, 1, 1]]
        qkv_data = (np.array(x_data) @ np.array(WA_data)).tolist()

        x_mat = Matrix(x_data, h_buff=0.5, v_buff=0.4).scale(0.85)
        WA_mat = Matrix(WA_data, h_buff=0.5, v_buff=0.4).scale(0.85)
        qkv_mat = Matrix([["\\cdot", "\\cdot", "\\cdot"]], h_buff=0.5, v_buff=0.4).scale(0.85)

        x_mat.move_to(ORIGIN)
        WA_mat.next_to(x_mat, RIGHT, buff=0.8)
        qkv_mat.next_to(WA_mat, RIGHT, buff=0.8)

        x_name = MathTex(r"\mathbf{x}", font_size=26, color=C_BLUE)
        x_name.next_to(x_mat, UP, buff=0.12)
        WA_name = MathTex(r"W_A", font_size=26, color=C_YELLOW)
        WA_name.next_to(WA_mat, UP, buff=0.12)
        qkv_name = MathTex(r"\text{qkv}", font_size=26, color=C_GREEN)
        qkv_name.next_to(qkv_mat, UP, buff=0.12)

        times1 = MathTex(r"\times", font_size=26)
        times1.move_to((x_mat.get_right() + WA_mat.get_left()) / 2)
        eq1 = MathTex(r"=", font_size=26)
        eq1.move_to((WA_mat.get_right() + qkv_mat.get_left()) / 2)

        self.play(FadeIn(step1_label), run_time=0.4)
        self.play(
            FadeIn(x_mat), FadeIn(x_name),
            FadeIn(WA_mat), FadeIn(WA_name),
            FadeIn(qkv_mat), FadeIn(qkv_name),
            Write(times1), Write(eq1),
            run_time=1.0
        )

        # Show one detailed computation
        r_rect = SurroundingRectangle(x_mat.get_rows()[0], color=C_BLUE, buff=0.06)
        c_rect = SurroundingRectangle(WA_mat.get_columns()[0], color=C_YELLOW, buff=0.06)
        self.play(Create(r_rect), Create(c_rect), run_time=0.35)

        eq_str = f"1\\cdot1 + 2\\cdot0 + 0\\cdot2 + 1\\cdot1 = {qkv_data[0][0]}"
        eq_mob = MathTex(eq_str, font_size=22, color=C_TXT)
        eq_mob.to_edge(DOWN, buff=1.0)
        self.play(Write(eq_mob), run_time=0.7)

        # Fill result
        for j, val in enumerate(qkv_data[0]):
            cell = qkv_mat.get_entries()[j]
            val_mob = MathTex(str(val), font_size=24, color=C_GREEN)
            val_mob.move_to(cell.get_center())
            if j == 0:
                self.play(FadeOut(cell), FadeIn(val_mob), run_time=0.4)
            else:
                cell.become(val_mob)

        self.play(FadeOut(r_rect), FadeOut(c_rect), FadeOut(eq_mob), run_time=0.3)
        self.wait(0.6)

        # Clean up, keep only qkv
        self.play(
            FadeOut(x_mat), FadeOut(x_name),
            FadeOut(WA_mat), FadeOut(WA_name),
            FadeOut(times1), FadeOut(eq1),
            qkv_mat.animate.move_to(UP * 0.8),
            qkv_name.animate.next_to(qkv_mat.target, UP, buff=0.12),
            run_time=0.7
        )

        # ═══════════════════════════════════════════════════════
        # Step 2: Split qkv
        # ═══════════════════════════════════════════════════════
        step2_label = MathTex(
            r"\textcircled{2}\ \text{split: } [c_q \mid c_{kv} \mid k_{pe}]",
            font_size=28, color=C_PURPLE)
        step2_label.move_to(step1_label.get_center())

        self.play(FadeOut(step1_label), FadeIn(step2_label), run_time=0.4)

        # Create split boxes
        cq_box = Rectangle(width=1.2, height=0.9, color=C_BLUE, fill_opacity=0.25)
        cq_val = MathTex(str(qkv_data[0][0]), font_size=30, color=C_BLUE)
        cq_val.move_to(cq_box.get_center())
        cq_label = MathTex(r"c_q", font_size=24, color=C_BLUE)
        cq_label.next_to(cq_box, DOWN, buff=0.15)
        cq_grp = VGroup(cq_box, cq_val, cq_label)

        ckv_box = Rectangle(width=1.2, height=0.9, color=C_PURPLE, fill_opacity=0.25)
        ckv_val = MathTex(str(qkv_data[0][1]), font_size=30, color=C_PURPLE)
        ckv_val.move_to(ckv_box.get_center())
        ckv_label = MathTex(r"c_{kv}", font_size=24, color=C_PURPLE)
        ckv_label.next_to(ckv_box, DOWN, buff=0.15)
        ckv_grp = VGroup(ckv_box, ckv_val, ckv_label)

        kpe_box = Rectangle(width=1.2, height=0.9, color=C_RED, fill_opacity=0.25)
        kpe_val = MathTex(str(qkv_data[0][2]), font_size=30, color=C_RED)
        kpe_val.move_to(kpe_box.get_center())
        kpe_label = MathTex(r"k_{pe}", font_size=24, color=C_RED)
        kpe_label.next_to(kpe_box, DOWN, buff=0.15)
        kpe_grp = VGroup(kpe_box, kpe_val, kpe_label)

        cq_grp.move_to(LEFT * 2.5 + DOWN * 0.5)
        ckv_grp.next_to(cq_grp, RIGHT, buff=1.0)
        kpe_grp.next_to(ckv_grp, RIGHT, buff=1.0)

        self.play(
            FadeIn(cq_grp), FadeIn(ckv_grp), FadeIn(kpe_grp),
            FadeOut(qkv_mat), FadeOut(qkv_name),
            run_time=0.8
        )
        self.wait(0.6)

        # ═══════════════════════════════════════════════════════
        # Step 3: Q path - c_q → q_nope
        # ═══════════════════════════════════════════════════════
        step3_label = MathTex(
            r"\textcircled{3}\ c_q \xrightarrow{\text{Norm, } W_B^q} q_{\text{nope}}",
            font_size=28, color=C_BLUE)
        step3_label.move_to(step2_label.get_center())

        self.play(FadeOut(step2_label), FadeIn(step3_label), run_time=0.4)

        qnope_data = [[1, 2, 0]]
        qnope_mat = Matrix(qnope_data, h_buff=0.5, v_buff=0.4).scale(0.85)
        qnope_mat.next_to(cq_grp, DOWN, buff=1.0)
        qnope_name = MathTex(r"q_{\text{nope}}", font_size=26, color=C_BLUE)
        qnope_name.next_to(qnope_mat, DOWN, buff=0.12)

        arrow1 = Arrow(cq_grp.get_bottom(), qnope_mat.get_top(), color=C_BLUE, buff=0.1)

        self.play(Create(arrow1), run_time=0.5)
        self.play(FadeIn(qnope_mat), FadeIn(qnope_name), run_time=0.7)
        self.wait(0.6)

        # Clean canvas - keep only q_nope, move c_kv to corner
        ckv_grp_small = ckv_grp.copy().scale(0.6)
        ckv_grp_small.to_corner(UR, buff=0.4)

        self.play(
            FadeOut(cq_grp), FadeOut(kpe_grp), FadeOut(arrow1),
            ReplacementTransform(ckv_grp, ckv_grp_small),
            qnope_mat.animate.move_to(LEFT * 2.5 + UP * 0.3),
            qnope_name.animate.next_to(qnope_mat.target, UP, buff=0.12),
            run_time=0.7
        )

        # ═══════════════════════════════════════════════════════
        # Step 4: Weight absorption
        # ═══════════════════════════════════════════════════════
        step4_label = MathTex(
            r"\textcircled{4}\ q_{\text{nope}} \times W_{kc} \rightarrow q_{\text{abs}}",
            font_size=28, color=C_ORANGE)
        step4_label.move_to(step3_label.get_center())

        self.play(FadeOut(step3_label), FadeIn(step4_label), run_time=0.4)

        Wkc_data = [[1, 0],
                    [0, 1],
                    [2, 1]]
        qabs_data = (np.array(qnope_data) @ np.array(Wkc_data)).tolist()

        Wkc_mat = Matrix(Wkc_data, h_buff=0.5, v_buff=0.4).scale(0.85)
        Wkc_mat.next_to(qnope_mat, RIGHT, buff=0.8)
        Wkc_name = MathTex(r"W_{kc}", font_size=26, color=C_YELLOW)
        Wkc_name.next_to(Wkc_mat, UP, buff=0.12)

        qabs_mat = Matrix([["\\cdot", "\\cdot"]], h_buff=0.5, v_buff=0.4).scale(0.85)
        qabs_mat.next_to(Wkc_mat, RIGHT, buff=0.8)
        qabs_name = MathTex(r"q_{\text{abs}}", font_size=26, color=C_ORANGE)
        qabs_name.next_to(qabs_mat, UP, buff=0.12)

        times4 = MathTex(r"\times", font_size=26)
        times4.move_to((qnope_mat.get_right() + Wkc_mat.get_left()) / 2)
        eq4 = MathTex(r"=", font_size=26)
        eq4.move_to((Wkc_mat.get_right() + qabs_mat.get_left()) / 2)

        self.play(
            FadeIn(Wkc_mat), FadeIn(Wkc_name),
            FadeIn(qabs_mat), FadeIn(qabs_name),
            Write(times4), Write(eq4),
            run_time=1.0
        )

        # Detailed computation
        r_rect2 = SurroundingRectangle(qnope_mat.get_rows()[0], color=C_BLUE, buff=0.06)
        c_rect2 = SurroundingRectangle(Wkc_mat.get_columns()[0], color=C_YELLOW, buff=0.06)
        self.play(Create(r_rect2), Create(c_rect2), run_time=0.35)

        eq_str2 = f"1\\cdot1 + 2\\cdot0 + 0\\cdot2 = {qabs_data[0][0]}"
        eq_mob2 = MathTex(eq_str2, font_size=22, color=C_TXT)
        eq_mob2.to_edge(DOWN, buff=1.0)
        self.play(Write(eq_mob2), run_time=0.7)

        # Fill result
        for j, val in enumerate(qabs_data[0]):
            cell = qabs_mat.get_entries()[j]
            val_mob = MathTex(str(val), font_size=24, color=C_ORANGE)
            val_mob.move_to(cell.get_center())
            if j == 0:
                self.play(FadeOut(cell), FadeIn(val_mob), run_time=0.4)
            else:
                cell.become(val_mob)

        self.play(FadeOut(r_rect2), FadeOut(c_rect2), FadeOut(eq_mob2), run_time=0.3)
        self.wait(0.6)

        # Move q_abs to corner
        qabs_small = VGroup(qabs_mat, qabs_name).copy().scale(0.7)
        qabs_small.to_corner(UL, buff=0.4)

        self.play(
            FadeOut(qnope_mat), FadeOut(qnope_name),
            FadeOut(Wkc_mat), FadeOut(Wkc_name),
            FadeOut(times4), FadeOut(eq4),
            ReplacementTransform(VGroup(qabs_mat, qabs_name), qabs_small),
            run_time=0.7
        )

        # ═══════════════════════════════════════════════════════
        # Step 5: Attention (symbolic)
        # ═══════════════════════════════════════════════════════
        step5_label = MathTex(
            r"\textcircled{5}\ \text{Attention}",
            font_size=28, color=C_GREEN)
        step5_label.move_to(step4_label.get_center())

        attn_formula = MathTex(
            r"\text{softmax}\left(\frac{q_{\text{abs}} \cdot c_{kv}^\top}{\sqrt{d}}\right) \cdot c_{kv}",
            font_size=24, color=C_TXT)
        attn_formula.move_to(ORIGIN)

        note = MathTex(
            r"\text{Uses } c_{kv} \text{ directly — no K up-projection!}",
            font_size=22, color=C_YELLOW)
        note.next_to(attn_formula, DOWN, buff=0.5)

        self.play(FadeOut(step4_label), FadeIn(step5_label), run_time=0.4)
        self.play(Write(attn_formula), run_time=0.9)
        self.play(FadeIn(note), run_time=0.6)
        self.wait(1.2)

        self.play(FadeOut(attn_formula), FadeOut(note), run_time=0.5)

        # ═══════════════════════════════════════════════════════
        # Step 6: V absorption
        # ═══════════════════════════════════════════════════════
        step6_label = MathTex(
            r"\textcircled{6}\ \text{attn} \times W_{vc} \rightarrow \text{output}",
            font_size=28, color=C_PURPLE)
        step6_label.move_to(step5_label.get_center())

        self.play(FadeOut(step5_label), FadeIn(step6_label), run_time=0.4)

        attn_data = [[1, 2]]
        Wvc_data = [[1, 0],
                    [0, 1]]
        out_data = (np.array(attn_data) @ np.array(Wvc_data)).tolist()

        attn_mat = Matrix(attn_data, h_buff=0.5, v_buff=0.4).scale(0.85)
        attn_mat.move_to(LEFT * 2.5 + UP * 0.3)
        attn_mat_name = MathTex(r"\text{attn}", font_size=26, color=C_PURPLE)
        attn_mat_name.next_to(attn_mat, UP, buff=0.12)

        Wvc_mat = Matrix(Wvc_data, h_buff=0.5, v_buff=0.4).scale(0.85)
        Wvc_mat.next_to(attn_mat, RIGHT, buff=0.8)
        Wvc_name = MathTex(r"W_{vc}", font_size=26, color=C_ORANGE)
        Wvc_name.next_to(Wvc_mat, UP, buff=0.12)

        out_mat = Matrix([["\\cdot", "\\cdot"]], h_buff=0.5, v_buff=0.4).scale(0.85)
        out_mat.next_to(Wvc_mat, RIGHT, buff=0.8)
        out_name = MathTex(r"\text{output}", font_size=26, color=C_GREEN)
        out_name.next_to(out_mat, UP, buff=0.12)

        times6 = MathTex(r"\times", font_size=26)
        times6.move_to((attn_mat.get_right() + Wvc_mat.get_left()) / 2)
        eq6 = MathTex(r"=", font_size=26)
        eq6.move_to((Wvc_mat.get_right() + out_mat.get_left()) / 2)

        self.play(
            FadeIn(attn_mat), FadeIn(attn_mat_name),
            FadeIn(Wvc_mat), FadeIn(Wvc_name),
            FadeIn(out_mat), FadeIn(out_name),
            Write(times6), Write(eq6),
            run_time=1.0
        )

        # Detailed computation
        r_rect3 = SurroundingRectangle(attn_mat.get_rows()[0], color=C_PURPLE, buff=0.06)
        c_rect3 = SurroundingRectangle(Wvc_mat.get_columns()[0], color=C_ORANGE, buff=0.06)
        self.play(Create(r_rect3), Create(c_rect3), run_time=0.35)

        eq_str3 = f"1\\cdot1 + 2\\cdot0 = {out_data[0][0]}"
        eq_mob3 = MathTex(eq_str3, font_size=22, color=C_TXT)
        eq_mob3.to_edge(DOWN, buff=1.0)
        self.play(Write(eq_mob3), run_time=0.7)

        # Fill result
        for j, val in enumerate(out_data[0]):
            cell = out_mat.get_entries()[j]
            val_mob = MathTex(str(val), font_size=24, color=C_GREEN)
            val_mob.move_to(cell.get_center())
            if j == 0:
                self.play(FadeOut(cell), FadeIn(val_mob), run_time=0.4)
            else:
                cell.become(val_mob)

        self.play(FadeOut(r_rect3), FadeOut(c_rect3), FadeOut(eq_mob3), run_time=0.3)
        self.wait(0.8)

        # ═══════════════════════════════════════════════════════
        # Final summary
        # ═══════════════════════════════════════════════════════
        summary = MathTex(
            r"\text{MLA: 576 dims/token vs MHA 40960 (71x compression)}",
            font_size=24, color=C_YELLOW)
        summary.to_edge(DOWN, buff=0.5)

        self.play(FadeOut(step6_label), Write(summary), run_time=0.9)
        self.wait(3)
