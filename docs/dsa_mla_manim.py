"""
MLA (Multi-Head Latent Attention) Matrix Computation Visualization — Manim CE

Shows the actual matrix operations in DeepSeek V3's MLA, step by step,
with real dimension labels and the weight absorption derivation.

Context: Round-Robin CP path (DSA prefill).

Scenes:
  MLAProjection  — Down-projection + Q up-projection + dimension flow
  MLAAbsorption  — Weight absorption derivation (the key MLA trick)
  MLAFullFlow    — Complete forward pass with KV cache savings

Usage:
  manim -qm --media_dir docs/media docs/dsa_mla_manim.py MLAProjection
  manim -qm --media_dir docs/media docs/dsa_mla_manim.py MLAAbsorption
  manim -qm --media_dir docs/media docs/dsa_mla_manim.py MLAFullFlow
"""

from manim import *
import numpy as np
import html as _html

_ManimText = Text


def Text(text, font_size=14, color=WHITE, weight=NORMAL, **kwargs):
    escaped = _html.escape(str(text))
    is_bold = weight not in (NORMAL, None, "normal", "NORMAL")
    w = "bold" if is_bold else "normal"
    markup = (
        f'<span font_family="Times New Roman" '
        f'letter_spacing="-400" weight="{w}">{escaped}</span>'
    )
    return MarkupText(markup, font_size=font_size, color=color, **kwargs)


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
    return MathTex(
        tex_str, font_size=font_size, color=color, tex_template=_TEX_TEMPLATE
    )


# ─── Colors (low-saturation palette) ────────────────────────
BG = "#1a1a2e"
COL_INPUT = "#7A95AE"       # muted steel blue — activations
COL_WEIGHT = "#B89A88"      # muted terracotta — weight matrices
COL_LATENT = "#9A88B8"      # muted lavender — compressed latent
COL_OUTPUT = "#7BAA8E"      # muted sage — output
COL_ROPE = "#A87A7C"        # muted rose — RoPE components
COL_HIGHLIGHT = "#FFD700"   # gold — highlights
COL_KV = "#6B9E8A"          # muted teal — KV cache
COL_DIM = "#aaaacc"
COL_CODE = "#88aacc"
COL_ABSORB = "#C4A050"      # muted gold — absorbed result

# ─── DeepSeek V3 actual dimensions ──────────────────────────
D_MODEL = 7168
N_HEADS = 128
D_NOPE = 128
D_ROPE = 64
D_V = 128
D_CQ = 1536
D_CKV = 512


# ─── Helpers ────────────────────────────────────────────────

def mat_block(name_tex, shape_tex, color, w=1.4, h=0.7, name_fs=20, shape_fs=14):
    rect = RoundedRectangle(
        width=w, height=h, corner_radius=0.06,
        fill_color=color, fill_opacity=0.85,
        stroke_color=WHITE, stroke_width=1.0,
    )
    name = latex_label(name_tex, font_size=name_fs, color=WHITE)
    shape = latex_label(shape_tex, font_size=shape_fs, color="#ddddee")
    grp = VGroup(name, shape).arrange(DOWN, buff=0.04)
    grp.move_to(rect.get_center())
    return VGroup(rect, grp)


def op_circle(symbol, color=COL_HIGHLIGHT, r=0.18, fs=18):
    circ = Circle(radius=r, fill_color=color, fill_opacity=0.9,
                  stroke_color=WHITE, stroke_width=0.8)
    sym = latex_label(symbol, font_size=fs, color=WHITE)
    sym.move_to(circ.get_center())
    return VGroup(circ, sym)


def thin_arrow(start, end, color=WHITE):
    return Arrow(start, end, color=color, stroke_width=1.5,
                 tip_length=0.12, buff=0.05)


def make_brace_label(mob, direction, text, color=COL_DIM, fs=12):
    br = Brace(mob, direction, color=color, buff=0.06)
    lbl = latex_label(text, font_size=fs, color=color)
    lbl.next_to(br, direction, buff=0.04)
    return VGroup(br, lbl)


# ═════════════════════════════════════════════════════════════
# Scene 1: MLA Projections — Down + Up + Split
# ═════════════════════════════════════════════════════════════

class MLAProjection(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BG

        # ── Title ──
        title = Text("MLA: Low-Rank Projection Flow",
                      font_size=28, color=WHITE, weight=BOLD)
        source = Text("deepseek_v2.py — DeepseekV2AttentionMLA.forward_absorb_prepare()",
                       font_size=11, color=COL_CODE)
        hdr = VGroup(title, source).arrange(DOWN, buff=0.06)

        # ════════════════════════════════════════
        # Step 1: Input
        # ════════════════════════════════════════
        step1_label = latex_label(r"\textbf{Step 1:}\ \text{Input}", color=COL_HIGHLIGHT)
        x_block = mat_block(r"\mathbf{x}", r"[n, 7168]", COL_INPUT, w=1.6, h=0.7)
        step1 = VGroup(step1_label, x_block).arrange(DOWN, buff=0.12)

        # ════════════════════════════════════════
        # Step 2: Fused A-projection (down)
        # ════════════════════════════════════════
        step2_label = latex_label(
            r"\textbf{Step 2:}\ W_{dqkv}\ \text{(fused down-proj)}",
            color=COL_HIGHLIGHT)

        x2 = mat_block(r"\mathbf{x}", r"[n, 7168]", COL_INPUT, w=1.2, h=0.55)
        mul1 = op_circle(r"\times")
        w_a = mat_block(r"W_{\!A}", r"[7168, 2112]", COL_WEIGHT, w=1.4, h=0.55)
        eq1 = op_circle(r"=", color="#555577")
        qkv = mat_block(r"\text{qkv}", r"[n, 2112]", COL_LATENT, w=1.2, h=0.55)

        proj_row = VGroup(x2, mul1, w_a, eq1, qkv).arrange(RIGHT, buff=0.15)

        # split result
        split_label = latex_label(r"\text{split}(-1)", font_size=18, color=COL_DIM)
        c_q = mat_block(r"c_q", r"[n, 1536]", COL_LATENT, w=1.1, h=0.5)
        c_kv = mat_block(r"c_{kv}", r"[n, 512]", COL_KV, w=0.9, h=0.5)
        k_pe = mat_block(r"k_{pe}", r"[n, 64]", COL_ROPE, w=0.8, h=0.5)

        split_row = VGroup(c_q, c_kv, k_pe).arrange(RIGHT, buff=0.2)
        step2 = VGroup(step2_label, proj_row, split_label, split_row).arrange(DOWN, buff=0.12)

        dim_note = latex_label(
            r"2112 = \underbrace{1536}_{d_{cq}} + \underbrace{512}_{d_{ckv}} + \underbrace{64}_{d_{rope}}",
            font_size=18, color=COL_DIM)

        # ════════════════════════════════════════
        # Step 3: Q path (up-projection)
        # ════════════════════════════════════════
        step3_label = latex_label(
            r"\textbf{Step 3:}\ \text{Q up-projection}", color=COL_HIGHLIGHT)

        cq2 = mat_block(r"c_q", r"[n, 1536]", COL_LATENT, w=1.0, h=0.5)
        norm_box = mat_block(r"\text{Norm}", r"", "#666688", w=0.7, h=0.5, shape_fs=10)
        mul2 = op_circle(r"\times")
        w_b = mat_block(r"W_{B}^{q}", r"[1536, h{\cdot}192]", COL_WEIGHT, w=1.5, h=0.5)
        eq2 = op_circle(r"=", color="#555577")
        q_full = mat_block(r"\mathbf{q}", r"[n, h, 192]", COL_INPUT, w=1.2, h=0.5)

        q_row = VGroup(cq2, norm_box, mul2, w_b, eq2, q_full).arrange(RIGHT, buff=0.12)

        split2_label = latex_label(r"\text{split}(d_{nope}, d_{rope})",
                                   font_size=18, color=COL_DIM)
        q_nope = mat_block(r"q_{\text{nope}}", r"[n, h, 128]", COL_INPUT, w=1.1, h=0.5)
        q_pe = mat_block(r"q_{pe}", r"[n, h, 64]", COL_ROPE, w=0.9, h=0.5)
        q_split = VGroup(q_nope, q_pe).arrange(RIGHT, buff=0.25)

        step3 = VGroup(step3_label, q_row, split2_label, q_split).arrange(DOWN, buff=0.12)

        # ════════════════════════════════════════
        # Step 4: KV norm + RoPE
        # ════════════════════════════════════════
        step4_label = latex_label(
            r"\textbf{Step 4:}\ \text{RMSNorm}(c_{kv}),\ \text{RoPE}(q_{pe}, k_{pe})",
            color=COL_HIGHLIGHT)
        ckv2 = mat_block(r"\hat{c}_{kv}", r"[n, 512]", COL_KV, w=1.0, h=0.5)
        norm2 = mat_block(r"\text{Norm}", r"", "#666688", w=0.6, h=0.5, shape_fs=10)
        arrow_norm = thin_arrow(LEFT * 0.3, RIGHT * 0.3, color=COL_DIM)
        ckv3 = mat_block(r"c_{kv}", r"[n, 512]", COL_KV, w=1.0, h=0.5)
        kv_norm_row = VGroup(ckv2, arrow_norm, norm2, arrow_norm.copy(), ckv3).arrange(RIGHT, buff=0.1)

        rope_note = latex_label(
            r"q_{pe}, k_{pe} \leftarrow \text{RoPE}(\text{pos},\ q_{pe},\ k_{pe})",
            font_size=18, color=COL_ROPE)

        step4 = VGroup(step4_label, rope_note).arrange(DOWN, buff=0.12)

        # ── Full layout ──
        all_content = VGroup(hdr, step1, step2, dim_note, step3, step4)
        all_content.arrange(DOWN, buff=0.25)

        # ── Animate ──
        self.play(Write(title), FadeIn(source), run_time=0.8)
        self.play(self.camera.auto_zoom(all_content, margin=0.4), run_time=0.3)

        # Step 1
        self.play(FadeIn(step1_label), run_time=0.3)
        self.play(FadeIn(x_block), run_time=0.5)
        self.wait(0.5)

        # Step 2
        self.play(FadeIn(step2_label), run_time=0.3)
        self.play(
            LaggedStart(FadeIn(x2), FadeIn(mul1), FadeIn(w_a),
                        FadeIn(eq1), FadeIn(qkv), lag_ratio=0.1),
            run_time=1.2)
        self.wait(0.3)
        self.play(FadeIn(split_label), run_time=0.2)
        self.play(
            LaggedStart(FadeIn(c_q), FadeIn(c_kv), FadeIn(k_pe), lag_ratio=0.15),
            run_time=0.8)
        self.play(FadeIn(dim_note), run_time=0.5)
        self.wait(0.5)

        # Step 3
        self.play(FadeIn(step3_label), run_time=0.3)
        self.play(
            LaggedStart(FadeIn(cq2), FadeIn(norm_box), FadeIn(mul2),
                        FadeIn(w_b), FadeIn(eq2), FadeIn(q_full), lag_ratio=0.08),
            run_time=1.2)
        self.wait(0.3)
        self.play(FadeIn(split2_label), run_time=0.2)
        self.play(FadeIn(q_nope), FadeIn(q_pe), run_time=0.6)
        self.wait(0.5)

        # Step 4
        self.play(FadeIn(step4_label), run_time=0.3)
        self.play(FadeIn(rope_note), run_time=0.5)
        self.wait(2)


# ═════════════════════════════════════════════════════════════
# Scene 2: Weight Absorption Derivation
# ═════════════════════════════════════════════════════════════

class MLAAbsorption(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("MLA: Weight Absorption Trick",
                      font_size=28, color=WHITE, weight=BOLD)
        source = Text("forward_mla.py — forward_absorb_prepare(): q_nope @ W_kc",
                       font_size=11, color=COL_CODE)
        hdr = VGroup(title, source).arrange(DOWN, buff=0.06)

        # ════════════════════════════════════════
        # Part A: Standard MHA (per head h)
        # ════════════════════════════════════════
        partA_label = latex_label(
            r"\textbf{Standard MHA}\ \text{(per head } h \text{)}",
            font_size=22, color="#aaddff")

        eq_k = latex_label(
            r"k_h = c_{kv} \cdot W_{uk,h}^\top",
            font_size=22, color=WHITE)
        eq_k_dim = latex_label(
            r"[n, 512] \times [512, 128] \to [n, 128]",
            font_size=16, color=COL_DIM)
        k_line = VGroup(eq_k, eq_k_dim).arrange(DOWN, buff=0.04)

        eq_s = latex_label(
            r"S_h = q_{\text{nope},h} \cdot k_h^\top",
            font_size=22, color=WHITE)
        eq_s_dim = latex_label(
            r"[n, 128] \times [128, n] \to [n, n]",
            font_size=16, color=COL_DIM)
        s_line = VGroup(eq_s, eq_s_dim).arrange(DOWN, buff=0.04)

        partA = VGroup(partA_label, k_line, s_line).arrange(DOWN, buff=0.15)

        # ════════════════════════════════════════
        # Part B: Substitution
        # ════════════════════════════════════════
        partB_label = latex_label(
            r"\textbf{Substitute}\ k_h \text{:}",
            font_size=22, color="#aaddff")

        eq_sub1 = latex_label(
            r"S_h = q_{\text{nope},h} \cdot (c_{kv} \cdot W_{uk,h}^\top)^\top",
            font_size=22, color=WHITE)
        eq_sub2 = latex_label(
            r"= q_{\text{nope},h} \cdot W_{uk,h} \cdot c_{kv}^\top",
            font_size=22, color=WHITE)

        partB = VGroup(partB_label, eq_sub1, eq_sub2).arrange(DOWN, buff=0.1)

        # ════════════════════════════════════════
        # Part C: Rearrange (absorption!)
        # ════════════════════════════════════════
        partC_label = latex_label(
            r"\textbf{Absorb}\ W_{uk,h}\ \text{into}\ q \text{:}",
            font_size=22, color=COL_HIGHLIGHT)

        eq_abs1 = latex_label(
            r"S_h = \underbrace{(q_{\text{nope},h} \cdot W_{uk,h})}_{q_{\text{absorbed},h}} \cdot c_{kv}^\top",
            font_size=24, color=WHITE)

        eq_abs2 = latex_label(
            r"q_{\text{absorbed},h} = q_{\text{nope},h} \cdot W_{kc,h}",
            font_size=22, color=COL_ABSORB)
        eq_abs2_dim = latex_label(
            r"[n, 128] \times [128, 512] \to [n, 512]",
            font_size=16, color=COL_DIM)
        abs_line = VGroup(eq_abs2, eq_abs2_dim).arrange(DOWN, buff=0.04)

        partC = VGroup(partC_label, eq_abs1, abs_line).arrange(DOWN, buff=0.12)

        # ════════════════════════════════════════
        # Part D: Result — no K up-projection at inference!
        # ════════════════════════════════════════
        partD_label = latex_label(
            r"\textbf{Result:}",
            font_size=22, color=COL_HIGHLIGHT)

        result1 = latex_label(
            r"S_h = q_{\text{absorbed},h} \cdot c_{kv}^\top",
            font_size=22, color="#55ff55")
        result1_note = latex_label(
            r"\text{Attention score uses } c_{kv} \text{ directly — no up-projection!}",
            font_size=16, color="#aaddff")

        partD = VGroup(partD_label, result1, result1_note).arrange(DOWN, buff=0.08)

        # ════════════════════════════════════════
        # Part E: Same trick for V
        # ════════════════════════════════════════
        partE_label = latex_label(
            r"\textbf{Same for V:}",
            font_size=22, color="#aaddff")

        eq_v1 = latex_label(
            r"o_h = P_h \cdot v_h = P_h \cdot c_{kv} \cdot W_{uv,h}^\top",
            font_size=20, color=WHITE)
        eq_v2 = latex_label(
            r"\Rightarrow\ o_h^{(\text{latent})} = P_h \cdot c_{kv}",
            font_size=20, color=WHITE)
        eq_v3 = latex_label(
            r"\text{then } o_h = o_h^{(\text{latent})} \cdot W_{vc,h}",
            font_size=20, color=COL_ABSORB)
        eq_v3_dim = latex_label(
            r"[n, 512] \times [512, 128] \to [n, 128]",
            font_size=16, color=COL_DIM)
        v_line = VGroup(eq_v3, eq_v3_dim).arrange(DOWN, buff=0.04)
        eq_v_note = latex_label(
            r"\text{Post-multiply } W_{vc} \text{ after attention (BMM)}",
            font_size=16, color="#aaddff")

        partE = VGroup(partE_label, eq_v1, eq_v2, v_line, eq_v_note).arrange(DOWN, buff=0.08)

        # ── Code reference ──
        code_ref = latex_label(
            r"\texttt{w\_kc, w\_vc} \text{ are pre-computed from } \texttt{kv\_b\_proj}",
            font_size=16, color=COL_CODE)

        # ── Layout ──
        all_content = VGroup(hdr, partA, partB, partC, partD, partE, code_ref)
        all_content.arrange(DOWN, buff=0.22)

        # ── Animate ──
        self.play(Write(title), FadeIn(source), run_time=0.8)
        self.play(self.camera.auto_zoom(all_content, margin=0.4), run_time=0.3)

        # Part A: Standard MHA
        self.play(FadeIn(partA_label), run_time=0.3)
        self.play(FadeIn(k_line), run_time=0.6)
        self.play(FadeIn(s_line), run_time=0.6)
        self.wait(1)

        # Part B: Substitute
        self.play(FadeIn(partB_label), run_time=0.3)
        self.play(FadeIn(eq_sub1), run_time=0.8)
        self.play(FadeIn(eq_sub2), run_time=0.8)
        self.wait(0.5)

        # Part C: Absorption!
        self.play(FadeIn(partC_label), run_time=0.3)
        self.play(FadeIn(eq_abs1), run_time=1.0)
        self.wait(0.5)
        self.play(FadeIn(abs_line), run_time=0.8)
        self.wait(1)

        # Part D: Result
        self.play(FadeIn(partD_label), run_time=0.3)
        self.play(FadeIn(result1), run_time=0.6)
        self.play(FadeIn(result1_note), run_time=0.5)
        self.wait(1)

        # Part E: V absorption
        self.play(FadeIn(partE_label), run_time=0.3)
        self.play(FadeIn(eq_v1), run_time=0.6)
        self.play(FadeIn(eq_v2), run_time=0.6)
        self.play(FadeIn(v_line), run_time=0.6)
        self.play(FadeIn(eq_v_note), run_time=0.5)
        self.wait(0.5)

        self.play(FadeIn(code_ref), run_time=0.4)
        self.wait(2)


# ═════════════════════════════════════════════════════════════
# Scene 3: Full MLA Forward Pass with KV Cache
# ═════════════════════════════════════════════════════════════

class MLAFullFlow(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("MLA: Complete Forward Pass (Round-Robin CP Path)",
                      font_size=26, color=WHITE, weight=BOLD)
        source = Text("forward_mla.py — forward_absorb_prepare() + forward_absorb_core()",
                       font_size=11, color=COL_CODE)
        hdr = VGroup(title, source).arrange(DOWN, buff=0.06)

        # ════════════════════════════════════════
        # Row 1: Fused down-projection
        # ════════════════════════════════════════
        r1_label = latex_label(
            r"\textcircled{1}\ \text{Fused down-proj:}\ "
            r"\mathbf{x} \cdot W_A \to [c_q \mid c_{kv} \mid k_{pe}]",
            font_size=20, color=COL_HIGHLIGHT)

        x1 = mat_block(r"\mathbf{x}", r"[n, 7168]", COL_INPUT, w=1.0, h=0.45)
        arr1 = thin_arrow(LEFT * 0.2, RIGHT * 0.2)
        w1 = mat_block(r"W_{\!A}", r"[7168, 2112]", COL_WEIGHT, w=1.2, h=0.45)
        arr2 = thin_arrow(LEFT * 0.2, RIGHT * 0.2)
        cq1 = mat_block(r"c_q", r"[n,1536]", COL_LATENT, w=0.85, h=0.45, name_fs=16, shape_fs=12)
        ckv1 = mat_block(r"c_{kv}", r"[n,512]", COL_KV, w=0.75, h=0.45, name_fs=16, shape_fs=12)
        kpe1 = mat_block(r"k_{pe}", r"[n,64]", COL_ROPE, w=0.65, h=0.45, name_fs=16, shape_fs=12)
        row1_blocks = VGroup(x1, arr1, w1, arr2, cq1, ckv1, kpe1).arrange(RIGHT, buff=0.08)
        row1 = VGroup(r1_label, row1_blocks).arrange(DOWN, buff=0.08)

        # ════════════════════════════════════════
        # Row 2: Q up-projection + split
        # ════════════════════════════════════════
        r2_label = latex_label(
            r"\textcircled{2}\ c_q \xrightarrow{\text{Norm}} W_B^q \to "
            r"[q_{\text{nope}} \mid q_{pe}]",
            font_size=20, color=COL_HIGHLIGHT)

        cq2 = mat_block(r"c_q", r"[n,1536]", COL_LATENT, w=0.85, h=0.45, name_fs=16, shape_fs=12)
        arr3 = thin_arrow(LEFT * 0.2, RIGHT * 0.2)
        norm1 = mat_block(r"\text{Norm}", r"", "#666688", w=0.5, h=0.45, name_fs=14, shape_fs=10)
        arr4 = thin_arrow(LEFT * 0.2, RIGHT * 0.2)
        wb = mat_block(r"W_B^q", r"[1536, 24576]", COL_WEIGHT, w=1.1, h=0.45, name_fs=16, shape_fs=11)
        arr5 = thin_arrow(LEFT * 0.2, RIGHT * 0.2)
        qnope = mat_block(r"q_{\text{nope}}", r"[n,h,128]", COL_INPUT, w=0.9, h=0.45, name_fs=14, shape_fs=11)
        qpe = mat_block(r"q_{pe}", r"[n,h,64]", COL_ROPE, w=0.75, h=0.45, name_fs=16, shape_fs=11)
        row2_blocks = VGroup(cq2, arr3, norm1, arr4, wb, arr5, qnope, qpe).arrange(RIGHT, buff=0.06)
        row2 = VGroup(r2_label, row2_blocks).arrange(DOWN, buff=0.08)

        # ════════════════════════════════════════
        # Row 3: Weight absorption (BMM #1)
        # ════════════════════════════════════════
        r3_label = latex_label(
            r"\textcircled{3}\ \text{Absorb } W_K \text{:}\ "
            r"q_{\text{abs}} = q_{\text{nope}} \cdot W_{kc}\ \ \textbf{(BMM)}",
            font_size=20, color=COL_HIGHLIGHT)

        qn2 = mat_block(r"q_{\text{nope}}", r"[h,n,128]", COL_INPUT, w=0.9, h=0.45, name_fs=14, shape_fs=11)
        mul3 = op_circle(r"\times", r=0.14, fs=14)
        wkc = mat_block(r"W_{kc}", r"[h,128,512]", COL_WEIGHT, w=1.0, h=0.45, name_fs=16, shape_fs=11)
        eq3 = op_circle(r"=", color="#555577", r=0.14, fs=14)
        qabs = mat_block(r"q_{\text{abs}}", r"[h,n,512]", COL_ABSORB, w=0.9, h=0.45, name_fs=14, shape_fs=11)
        row3_blocks = VGroup(qn2, mul3, wkc, eq3, qabs).arrange(RIGHT, buff=0.1)
        row3 = VGroup(r3_label, row3_blocks).arrange(DOWN, buff=0.08)

        # ════════════════════════════════════════
        # Row 4: RoPE + Assemble Q, K
        # ════════════════════════════════════════
        r4_label = latex_label(
            r"\textcircled{4}\ \text{RoPE} + \text{assemble:}",
            font_size=20, color=COL_HIGHLIGHT)

        q_combined = mat_block(
            r"\mathbf{q}", r"[n,h,576]", COL_INPUT, w=0.9, h=0.45, name_fs=16, shape_fs=11)
        eq_q = latex_label(r"= [q_{\text{abs}} \mid q_{pe}]", font_size=16, color=WHITE)

        k_combined = mat_block(
            r"\mathbf{k}", r"[n,\mathbf{1},576]", COL_KV, w=0.9, h=0.45, name_fs=16, shape_fs=11)
        eq_k2 = latex_label(r"= [c_{kv} \mid k_{pe}]", font_size=16, color=WHITE)

        q_line = VGroup(q_combined, eq_q).arrange(RIGHT, buff=0.1)
        k_line2 = VGroup(k_combined, eq_k2).arrange(RIGHT, buff=0.1)

        kv_note = latex_label(
            r"\text{K shared across all } h=128 \text{ heads (MQA-style)!}",
            font_size=16, color="#55ff55")

        row4 = VGroup(r4_label, q_line, k_line2, kv_note).arrange(DOWN, buff=0.06)

        # ════════════════════════════════════════
        # Row 5: KV Cache (the punchline!)
        # ════════════════════════════════════════
        r5_label = latex_label(
            r"\textcircled{5}\ \textbf{KV Cache:}", font_size=20, color=COL_HIGHLIGHT)

        kv_cache = mat_block(
            r"\text{cache}", r"[n, 576]", COL_KV, w=1.0, h=0.55)
        kv_eq = latex_label(r"= [c_{kv} \mid k_{pe}]", font_size=18, color=WHITE)
        kv_row = VGroup(kv_cache, kv_eq).arrange(RIGHT, buff=0.12)

        mha_compare = latex_label(
            r"\text{Standard MHA: } 128 \times (128{+}64{+}128) = 40960\ \text{per token}",
            font_size=16, color="#ff8888")
        mla_compare = latex_label(
            r"\text{MLA: } 512 + 64 = 576\ \text{per token}",
            font_size=16, color="#55ff55")
        ratio = latex_label(
            r"\textbf{Compression: } 71\times \text{ smaller!}",
            font_size=20, color=COL_HIGHLIGHT)

        compare_box_content = VGroup(mha_compare, mla_compare, ratio).arrange(DOWN, buff=0.06)
        compare_box = SurroundingRectangle(
            compare_box_content, color="#4466aa", fill_color="#222244",
            fill_opacity=0.85, corner_radius=0.08, buff=0.12)
        compare_g = VGroup(compare_box, compare_box_content)

        row5 = VGroup(r5_label, kv_row, compare_g).arrange(DOWN, buff=0.1)

        # ════════════════════════════════════════
        # Row 6: Attention + V absorption (BMM #2) + output
        # ════════════════════════════════════════
        r6_label = latex_label(
            r"\textcircled{6}\ \text{Attention} + W_{vc}\ \textbf{(BMM)} + W_o",
            font_size=20, color=COL_HIGHLIGHT)

        attn_eq = latex_label(
            r"\text{attn} = \text{softmax}\!\left(\frac{\mathbf{q} \cdot \mathbf{k}^\top}{\sqrt{d}}\right) "
            r"\cdot c_{kv}",
            font_size=20, color=WHITE)
        attn_dim = latex_label(
            r"[n, h, 576] \times [576, n] \to [n, n] \to \times [n, 512] \to [n, h, 512]",
            font_size=14, color=COL_DIM)
        attn_line = VGroup(attn_eq, attn_dim).arrange(DOWN, buff=0.04)

        bmm2_eq = latex_label(
            r"\text{output}_h = \text{attn}_h \cdot W_{vc,h}",
            font_size=20, color=COL_ABSORB)
        bmm2_dim = latex_label(
            r"[h, n, 512] \times [h, 512, 128] \to [h, n, 128]",
            font_size=14, color=COL_DIM)
        bmm2_line = VGroup(bmm2_eq, bmm2_dim).arrange(DOWN, buff=0.04)

        out_eq = latex_label(
            r"\text{output} = W_o \cdot \text{flatten}(\text{output}_h)",
            font_size=20, color=COL_OUTPUT)
        out_dim = latex_label(
            r"[h{\cdot}128, 7168]^\top \times [n, h{\cdot}128] \to [n, 7168]",
            font_size=14, color=COL_DIM)
        out_line = VGroup(out_eq, out_dim).arrange(DOWN, buff=0.04)

        row6 = VGroup(r6_label, attn_line, bmm2_line, out_line).arrange(DOWN, buff=0.1)

        # ── Full layout ──
        all_content = VGroup(hdr, row1, row2, row3, row4, row5, row6)
        all_content.arrange(DOWN, buff=0.2)

        # ── Animate ──
        self.play(Write(title), FadeIn(source), run_time=0.8)
        self.play(self.camera.auto_zoom(all_content, margin=0.35), run_time=0.3)

        # Row 1: Down-projection
        self.play(FadeIn(r1_label), run_time=0.3)
        self.play(
            LaggedStart(*[FadeIn(m) for m in row1_blocks], lag_ratio=0.06),
            run_time=1.5)
        self.wait(0.5)

        # Row 2: Q path
        self.play(FadeIn(r2_label), run_time=0.3)
        self.play(
            LaggedStart(*[FadeIn(m) for m in row2_blocks], lag_ratio=0.06),
            run_time=1.5)
        self.wait(0.5)

        # Row 3: Weight absorption BMM
        self.play(FadeIn(r3_label), run_time=0.3)
        self.play(
            LaggedStart(*[FadeIn(m) for m in row3_blocks], lag_ratio=0.08),
            run_time=1.2)
        self.wait(0.8)

        # Row 4: RoPE + assemble
        self.play(FadeIn(r4_label), run_time=0.3)
        self.play(FadeIn(q_line), FadeIn(k_line2), run_time=0.8)
        self.play(FadeIn(kv_note), run_time=0.5)
        self.wait(0.5)

        # Row 5: KV Cache comparison
        self.play(FadeIn(r5_label), run_time=0.3)
        self.play(FadeIn(kv_row), run_time=0.5)
        self.play(FadeIn(mha_compare), run_time=0.5)
        self.play(FadeIn(mla_compare), run_time=0.5)
        self.play(FadeIn(compare_box), FadeIn(ratio), run_time=0.8)
        self.wait(1.5)

        # Row 6: Attention + output
        self.play(FadeIn(r6_label), run_time=0.3)
        self.play(FadeIn(attn_line), run_time=0.8)
        self.wait(0.3)
        self.play(FadeIn(bmm2_line), run_time=0.8)
        self.wait(0.3)
        self.play(FadeIn(out_line), run_time=0.8)
        self.wait(3)
