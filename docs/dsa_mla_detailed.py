"""
MLA Complete Detailed Flow

Full MLA forward pass from input x to final o_proj output.
Shows ALL intermediate matrices: down-proj, split, Q/K up-proj,
weight absorption, attention, V absorption, output projection.
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


class MLADetailedFlow(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BG

        title = MathTex(
            r"\text{MLA: Complete Detailed Flow}",
            font_size=36, color=C_TXT)
        title.to_edge(UP, buff=0.15)
        self.play(Write(title), run_time=0.8)
        self.wait(0.4)

        self.step1_down_proj()
        self.step2_split()
        self.step3_q_up_proj()
        self.step4_weight_absorption()
        self.step5_rope()
        self.step6_attention()
        self.step7_v_absorption()
        self.step8_output_proj()

        summary = MathTex(
            r"\text{Complete: } x \to \text{qkv} \to q,k,v \to \text{attn} \to o",
            font_size=20, color=C_YELLOW)
        summary.to_edge(DOWN, buff=0.3)
        self.play(Write(summary), run_time=0.8)
        self.wait(3)

    def step1_down_proj(self):
        label = MathTex(
            r"\textcircled{1}\ \mathbf{x} \times W_A \to \text{qkv\_latent}",
            font_size=24, color=C_BLUE)
        label.to_edge(UP, buff=0.6)
        self.play(FadeIn(label), run_time=0.3)

        x_data = [[1, 2, 0, 1]]
        WA_data = [[1, 0, 2], [0, 1, 1], [2, 1, 0], [1, 1, 1]]
        qkv_data = (np.array(x_data) @ np.array(WA_data)).tolist()

        x_mat = Matrix(x_data, h_buff=0.5, v_buff=0.4).scale(0.9)
        WA_mat = Matrix(WA_data, h_buff=0.5, v_buff=0.4).scale(0.9)
        qkv_mat = Matrix([["?", "?", "?"]], h_buff=0.5, v_buff=0.4).scale(0.9)

        x_mat.move_to(LEFT * 3 + UP * 0.5)
        WA_mat.next_to(x_mat, RIGHT, buff=0.6)
        qkv_mat.next_to(WA_mat, RIGHT, buff=0.6)

        x_name = MathTex(r"x", font_size=22, color=C_BLUE)
        x_name.next_to(x_mat, UP, buff=0.1)
        WA_name = MathTex(r"W_A", font_size=22, color=C_YELLOW)
        WA_name.next_to(WA_mat, UP, buff=0.1)
        qkv_name = MathTex(r"\text{qkv}", font_size=22, color=C_GREEN)
        qkv_name.next_to(qkv_mat, UP, buff=0.1)

        times = MathTex(r"\times", font_size=20)
        times.move_to((x_mat.get_right() + WA_mat.get_left()) / 2)
        eq = MathTex(r"=", font_size=20)
        eq.move_to((WA_mat.get_right() + qkv_mat.get_left()) / 2)

        self.play(
            FadeIn(x_mat), FadeIn(x_name),
            FadeIn(WA_mat), FadeIn(WA_name),
            FadeIn(qkv_mat), FadeIn(qkv_name),
            Write(times), Write(eq),
            run_time=0.8
        )

        # Compute
        for j, val in enumerate(qkv_data[0]):
            cell = qkv_mat.get_entries()[j]
            val_mob = MathTex(str(val), font_size=16, color=C_GREEN)
            val_mob.move_to(cell.get_center())
            cell.become(val_mob)

        self.wait(0.5)

        # Save qkv for next step, clean original
        self.qkv_result = VGroup(qkv_mat, qkv_name).copy()
        self.qkv_result.move_to(UP * 1.2)
        self.play(
            FadeOut(x_mat), FadeOut(x_name),
            FadeOut(WA_mat), FadeOut(WA_name),
            FadeOut(qkv_mat), FadeOut(qkv_name),
            FadeOut(times), FadeOut(eq), FadeOut(label),
            FadeIn(self.qkv_result),
            run_time=0.6
        )

    def step2_split(self):
        label = MathTex(
            r"\textcircled{2}\ \text{split: } [c_q \mid c_{kv} \mid k_{pe}]",
            font_size=24, color=C_PURPLE)
        label.to_edge(UP, buff=0.6)
        self.play(FadeIn(label), run_time=0.3)

        qkv_data = [[2, 4, 2]]
        cq_box = Rectangle(width=1.0, height=0.7, color=C_BLUE, fill_opacity=0.2)
        cq_val = MathTex(str(qkv_data[0][0]), font_size=16, color=C_BLUE)
        cq_val.move_to(cq_box)
        cq_lbl = MathTex(r"c_q", font_size=18, color=C_BLUE)
        cq_lbl.next_to(cq_box, DOWN, buff=0.1)
        self.cq_grp = VGroup(cq_box, cq_val, cq_lbl)

        ckv_box = Rectangle(width=1.0, height=0.7, color=C_PURPLE, fill_opacity=0.2)
        ckv_val = MathTex(str(qkv_data[0][1]), font_size=16, color=C_PURPLE)
        ckv_val.move_to(ckv_box)
        ckv_lbl = MathTex(r"c_{kv}", font_size=18, color=C_PURPLE)
        ckv_lbl.next_to(ckv_box, DOWN, buff=0.1)
        self.ckv_grp = VGroup(ckv_box, ckv_val, ckv_lbl)

        kpe_box = Rectangle(width=1.0, height=0.7, color=C_RED, fill_opacity=0.2)
        kpe_val = MathTex(str(qkv_data[0][2]), font_size=16, color=C_RED)
        kpe_val.move_to(kpe_box)
        kpe_lbl = MathTex(r"k_{pe}", font_size=18, color=C_RED)
        kpe_lbl.next_to(kpe_box, DOWN, buff=0.1)
        self.kpe_grp = VGroup(kpe_box, kpe_val, kpe_lbl)

        self.cq_grp.move_to(LEFT * 2.2 + DOWN * 0.3)
        self.ckv_grp.next_to(self.cq_grp, RIGHT, buff=0.8)
        self.kpe_grp.next_to(self.ckv_grp, RIGHT, buff=0.8)

        self.play(
            FadeIn(self.cq_grp), FadeIn(self.ckv_grp), FadeIn(self.kpe_grp),
            FadeOut(self.qkv_result),
            run_time=0.7
        )
        self.wait(0.5)
        self.play(FadeOut(label), run_time=0.3)

    def step3_q_up_proj(self):
        label = MathTex(
            r"\textcircled{3}\ c_q \xrightarrow{W_B^q} q_{\text{nope}}",
            font_size=24, color=C_BLUE)
        label.to_edge(UP, buff=0.6)
        self.play(FadeIn(label), run_time=0.3)

        qnope_data = [[1, 2, 0]]
        qnope_mat = Matrix(qnope_data, h_buff=0.45, v_buff=0.35).scale(0.75)
        qnope_mat.next_to(self.cq_grp, DOWN, buff=0.8)
        qnope_name = MathTex(r"q_{\text{nope}}", font_size=22, color=C_BLUE)
        qnope_name.next_to(qnope_mat, DOWN, buff=0.1)
        self.qnope_grp = VGroup(qnope_mat, qnope_name)

        arrow = Arrow(self.cq_grp.get_bottom(), qnope_mat.get_top(), color=C_BLUE, buff=0.1)

        self.play(Create(arrow), run_time=0.4)
        self.play(FadeIn(self.qnope_grp), run_time=0.6)
        self.wait(0.5)
        # Clean ALL from step 2
        self.play(
            FadeOut(arrow), FadeOut(self.cq_grp),
            FadeOut(self.ckv_grp), FadeOut(self.kpe_grp),
            FadeOut(label),
            run_time=0.4
        )

    def step4_weight_absorption(self):
        label = MathTex(
            r"\textcircled{4}\ q_{\text{nope}} \times W_{kc} \to q_{\text{abs}}",
            font_size=24, color=C_ORANGE)
        label.to_edge(UP, buff=0.6)
        self.play(FadeIn(label), run_time=0.3)

        self.play(self.qnope_grp.animate.move_to(LEFT * 2.5 + UP * 0.5), run_time=0.5)

        Wkc_data = [[1, 0], [0, 1], [2, 1]]
        qabs_data = [[1, 2]]

        Wkc_mat = Matrix(Wkc_data, h_buff=0.45, v_buff=0.35).scale(0.75)
        Wkc_mat.next_to(self.qnope_grp, RIGHT, buff=0.6)
        Wkc_name = MathTex(r"W_{kc}", font_size=22, color=C_YELLOW)
        Wkc_name.next_to(Wkc_mat, UP, buff=0.1)

        qabs_mat = Matrix([["?", "?"]], h_buff=0.45, v_buff=0.35).scale(0.75)
        qabs_mat.next_to(Wkc_mat, RIGHT, buff=0.6)
        qabs_name = MathTex(r"q_{\text{abs}}", font_size=22, color=C_ORANGE)
        qabs_name.next_to(qabs_mat, UP, buff=0.1)
        self.qabs_grp = VGroup(qabs_mat, qabs_name)

        times = MathTex(r"\times", font_size=20)
        times.move_to((self.qnope_grp.get_right() + Wkc_mat.get_left()) / 2)
        eq = MathTex(r"=", font_size=20)
        eq.move_to((Wkc_mat.get_right() + qabs_mat.get_left()) / 2)

        self.play(
            FadeIn(Wkc_mat), FadeIn(Wkc_name),
            FadeIn(qabs_mat), FadeIn(qabs_name),
            Write(times), Write(eq),
            run_time=0.8
        )

        for j, val in enumerate(qabs_data[0]):
            cell = qabs_mat.get_entries()[j]
            val_mob = MathTex(str(val), font_size=16, color=C_ORANGE)
            val_mob.move_to(cell.get_center())
            cell.become(val_mob)

        self.wait(0.5)

        # Re-create ckv, kpe for attention step (fresh objects)
        ckv_box_new = Rectangle(width=0.6, height=0.5, color=C_PURPLE, fill_opacity=0.2)
        ckv_val_new = MathTex("4", font_size=16, color=C_PURPLE)
        ckv_val_new.move_to(ckv_box_new)
        ckv_lbl_new = MathTex(r"c_{kv}", font_size=14, color=C_PURPLE)
        ckv_lbl_new.next_to(ckv_box_new, DOWN, buff=0.08)
        ckv_grp_new = VGroup(ckv_box_new, ckv_val_new, ckv_lbl_new)
        ckv_grp_new.to_corner(UR, buff=0.3)

        kpe_box_new = Rectangle(width=0.6, height=0.5, color=C_RED, fill_opacity=0.2)
        kpe_val_new = MathTex("2", font_size=16, color=C_RED)
        kpe_val_new.move_to(kpe_box_new)
        kpe_lbl_new = MathTex(r"k_{pe}", font_size=14, color=C_RED)
        kpe_lbl_new.next_to(kpe_box_new, DOWN, buff=0.08)
        kpe_grp_new = VGroup(kpe_box_new, kpe_val_new, kpe_lbl_new)
        kpe_grp_new.to_corner(DR, buff=0.3)

        self.ckv_display = ckv_grp_new
        self.kpe_display = kpe_grp_new

        self.play(FadeIn(self.ckv_display), FadeIn(self.kpe_display), run_time=0.4)

        self.play(
            FadeOut(self.qnope_grp), FadeOut(Wkc_mat), FadeOut(Wkc_name),
            FadeOut(times), FadeOut(eq), FadeOut(label),
            self.qabs_grp.animate.move_to(UP * 0.8),
            run_time=0.6
        )

    def step5_rope(self):
        label = MathTex(
            r"\textcircled{5}\ \text{RoPE}(q_{pe}, k_{pe})",
            font_size=24, color=C_RED)
        label.to_edge(UP, buff=0.6)
        self.play(FadeIn(label), run_time=0.3)

        note = MathTex(
            r"\text{Apply rotary position embedding}",
            font_size=20, color=C_TXT)
        note.move_to(ORIGIN)
        self.play(Write(note), run_time=0.6)
        self.wait(0.8)
        self.play(FadeOut(note), FadeOut(label), run_time=0.4)

    def step6_attention(self):
        label = MathTex(
            r"\textcircled{6}\ \text{Attention: } q_{\text{abs}} \cdot c_{kv}^\top",
            font_size=24, color=C_GREEN)
        label.to_edge(UP, buff=0.6)
        self.play(FadeIn(label), run_time=0.3)

        formula = MathTex(
            r"\text{attn} = \text{softmax}\left(\frac{q \cdot k^\top}{\sqrt{d}}\right) \cdot c_{kv}",
            font_size=20, color=C_TXT)
        formula.move_to(DOWN * 0.5)

        note = MathTex(
            r"\text{Uses } c_{kv} \text{ directly (no K up-proj)}",
            font_size=18, color=C_YELLOW)
        note.next_to(formula, DOWN, buff=0.4)

        self.play(Write(formula), run_time=0.8)
        self.play(FadeIn(note), run_time=0.5)
        self.wait(1.0)

        attn_result = MathTex(r"[1, 2]", font_size=22, color=C_GREEN)
        attn_result.next_to(formula, DOWN, buff=0.8)
        attn_lbl = MathTex(r"\text{attn\_out}", font_size=20, color=C_GREEN)
        attn_lbl.next_to(attn_result, DOWN, buff=0.1)
        self.attn_grp = VGroup(attn_result, attn_lbl)

        self.play(FadeIn(self.attn_grp), run_time=0.5)
        self.wait(0.5)
        self.play(
            FadeOut(formula), FadeOut(note), FadeOut(self.qabs_grp),
            FadeOut(self.ckv_display), FadeOut(self.kpe_display), FadeOut(label),
            self.attn_grp.animate.move_to(UP * 0.8),
            run_time=0.6
        )

    def step7_v_absorption(self):
        label = MathTex(
            r"\textcircled{7}\ \text{attn} \times W_{vc} \to v_{\text{out}}",
            font_size=24, color=C_PURPLE)
        label.to_edge(UP, buff=0.6)
        self.play(FadeIn(label), run_time=0.3)

        self.play(self.attn_grp.animate.move_to(LEFT * 2.5 + UP * 0.5), run_time=0.5)

        Wvc_data = [[1, 0], [0, 1]]
        vout_data = [[1, 2]]

        Wvc_mat = Matrix(Wvc_data, h_buff=0.45, v_buff=0.35).scale(0.75)
        Wvc_mat.next_to(self.attn_grp, RIGHT, buff=0.6)
        Wvc_name = MathTex(r"W_{vc}", font_size=22, color=C_ORANGE)
        Wvc_name.next_to(Wvc_mat, UP, buff=0.1)

        vout_mat = Matrix([["?", "?"]], h_buff=0.45, v_buff=0.35).scale(0.75)
        vout_mat.next_to(Wvc_mat, RIGHT, buff=0.6)
        vout_name = MathTex(r"v_{\text{out}}", font_size=22, color=C_PURPLE)
        vout_name.next_to(vout_mat, UP, buff=0.1)
        self.vout_grp = VGroup(vout_mat, vout_name)

        times = MathTex(r"\times", font_size=20)
        times.move_to((self.attn_grp.get_right() + Wvc_mat.get_left()) / 2)
        eq = MathTex(r"=", font_size=20)
        eq.move_to((Wvc_mat.get_right() + vout_mat.get_left()) / 2)

        self.play(
            FadeIn(Wvc_mat), FadeIn(Wvc_name),
            FadeIn(vout_mat), FadeIn(vout_name),
            Write(times), Write(eq),
            run_time=0.8
        )

        for j, val in enumerate(vout_data[0]):
            cell = vout_mat.get_entries()[j]
            val_mob = MathTex(str(val), font_size=16, color=C_PURPLE)
            val_mob.move_to(cell.get_center())
            cell.become(val_mob)

        self.wait(0.5)
        self.play(
            FadeOut(self.attn_grp), FadeOut(Wvc_mat), FadeOut(Wvc_name),
            FadeOut(times), FadeOut(eq), FadeOut(label),
            self.vout_grp.animate.move_to(UP * 0.8),
            run_time=0.6
        )

    def step8_output_proj(self):
        label = MathTex(
            r"\textcircled{8}\ v_{\text{out}} \times W_o \to \text{output}",
            font_size=24, color=C_GREEN)
        label.to_edge(UP, buff=0.6)
        self.play(FadeIn(label), run_time=0.3)

        self.play(self.vout_grp.animate.move_to(LEFT * 2.5 + UP * 0.5), run_time=0.5)

        Wo_data = [[2, 1], [0, 1]]
        out_data = [[2, 3]]

        Wo_mat = Matrix(Wo_data, h_buff=0.45, v_buff=0.35).scale(0.75)
        Wo_mat.next_to(self.vout_grp, RIGHT, buff=0.6)
        Wo_name = MathTex(r"W_o", font_size=22, color=C_YELLOW)
        Wo_name.next_to(Wo_mat, UP, buff=0.1)

        out_mat = Matrix([["?", "?"]], h_buff=0.45, v_buff=0.35).scale(0.75)
        out_mat.next_to(Wo_mat, RIGHT, buff=0.6)
        out_name = MathTex(r"\text{output}", font_size=22, color=C_GREEN)
        out_name.next_to(out_mat, UP, buff=0.1)

        times = MathTex(r"\times", font_size=20)
        times.move_to((self.vout_grp.get_right() + Wo_mat.get_left()) / 2)
        eq = MathTex(r"=", font_size=20)
        eq.move_to((Wo_mat.get_right() + out_mat.get_left()) / 2)

        self.play(
            FadeIn(Wo_mat), FadeIn(Wo_name),
            FadeIn(out_mat), FadeIn(out_name),
            Write(times), Write(eq),
            run_time=0.8
        )

        for j, val in enumerate(out_data[0]):
            cell = out_mat.get_entries()[j]
            val_mob = MathTex(str(val), font_size=20, color=C_GREEN)
            val_mob.move_to(cell.get_center())
            cell.become(val_mob)

        self.wait(0.8)
        self.play(FadeOut(label), run_time=0.3)
