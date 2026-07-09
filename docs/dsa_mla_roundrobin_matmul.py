"""
MLA Round-Robin Matrix Operations Visualization — 3Blue1Brown Style
Manim Community Edition

Shows step-by-step matrix computations in MLA with Round-Robin CP split:
1. Fused down-projection: x @ W_A → [c_q | c_kv | k_pe]
2. Q up-projection: c_q (Norm) @ W_B^q → [q_nope | q_pe]
3. Weight absorption (BMM): q_nope @ W_kc → q_absorbed
4. Attention BMM: attn @ W_vc → output

Based on:
- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
  forward_absorb_prepare() L137-400, forward_absorb_core() L402-623
- python/sglang/srt/layers/attention/dsa/utils.py
  dsa_cp_round_robin_split_data() L98-127

Dimensions (DeepSeek V3):
  d_model=7168, n_heads=128, d_cq=1536, d_ckv=512, d_rope=64, d_nope=128, d_v=128

Usage:
  manim -qm --media_dir=/tmp/mla_media docs/dsa_mla_roundrobin_matmul.py FusedDownProjection
  manim -qm --media_dir=/tmp/mla_media docs/dsa_mla_roundrobin_matmul.py QUpProjection
  manim -qm --media_dir=/tmp/mla_media docs/dsa_mla_roundrobin_matmul.py WeightAbsorptionBMM
  manim -qm --media_dir=/tmp/mla_media docs/dsa_mla_roundrobin_matmul.py AttentionBMM
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


# ─── 3Blue1Brown Color Palette ──────────────────────────
BG = "#1a1a2e"
C_INPUT = "#7A95AE"       # steel blue — input activations
C_WEIGHT = "#B89A88"      # terracotta — weight matrices
C_LATENT = "#9A88B8"      # lavender — compressed latent
C_OUTPUT = "#7BAA8E"      # sage — output
C_ROPE = "#A87A7C"        # rose — RoPE components
C_HIGHLIGHT = "#FFD700"   # gold — highlights
C_KV = "#6B9E8A"          # teal — KV cache
C_DIM = "#aaaacc"
C_CODE = "#88aacc"
C_ABSORB = "#C4A050"      # gold — absorbed q

# Round-Robin CP split colors (4 ranks)
RANK_COLORS = ["#7A95AE", "#B89A88", "#7BAA8E", "#A87A7C"]

# ─── DeepSeek V3 Dimensions (scaled for viz) ────────────
D_MODEL = 7168
N_HEADS = 128
D_NOPE = 128
D_ROPE = 64
D_V = 128
D_CQ = 1536
D_CKV = 512
CP_SIZE = 4


# ─── Helpers ─────────────────────────────────────────────


def mat_block(name_tex, shape_tex, color, w=1.4, h=0.7, name_fs=20, shape_fs=14):
    """Matrix block with name + shape annotation."""
    rect = RoundedRectangle(
        width=w,
        height=h,
        corner_radius=0.06,
        fill_color=color,
        fill_opacity=0.85,
        stroke_color=WHITE,
        stroke_width=1.0,
    )
    name = latex_label(name_tex, font_size=name_fs, color=WHITE)
    shape = latex_label(shape_tex, font_size=shape_fs, color="#ddddee")
    grp = VGroup(name, shape).arrange(DOWN, buff=0.04)
    grp.move_to(rect.get_center())
    return VGroup(rect, grp)


def op_circle(symbol, color=C_HIGHLIGHT, r=0.18, fs=18):
    """Operation symbol in a circle."""
    circ = Circle(
        radius=r, fill_color=color, fill_opacity=0.9, stroke_color=WHITE, stroke_width=0.8
    )
    sym = latex_label(symbol, font_size=fs, color=WHITE)
    sym.move_to(circ.get_center())
    return VGroup(circ, sym)


def thin_arrow(start, end, color=WHITE):
    return Arrow(start, end, color=color, stroke_width=1.5, tip_length=0.12, buff=0.05)


def make_matrix_cells(rows, cols, data, color_fn=None, w=0.35, h=0.3, gap=0.02, fs=11):
    """Make a matrix of labeled cells."""
    matrix = VGroup()
    for r in range(rows):
        row_group = VGroup()
        for c in range(cols):
            label = data[r][c]
            color = color_fn(r, c, label) if color_fn else C_INPUT
            rect = RoundedRectangle(
                width=w, height=h, corner_radius=0.04,
                fill_color=color, fill_opacity=1,
                stroke_color=WHITE, stroke_width=0.8,
            )
            txt = Text(str(label), font_size=fs, color=WHITE, weight=BOLD)
            txt.move_to(rect.get_center())
            cell = VGroup(rect, txt)
            row_group.add(cell)
        row_group.arrange(RIGHT, buff=gap)
        matrix.add(row_group)
    matrix.arrange(DOWN, buff=gap)
    return matrix


def dim_text(text, font_size=10):
    return Text(text, font_size=font_size, color=C_DIM)


# ═════════════════════════════════════════════════════════════
# Scene 1: Fused Down-Projection — x @ W_A → [c_q | c_kv | k_pe]
# Shows matrix multiplication and split into 3 parts
# ═════════════════════════════════════════════════════════════


class FusedDownProjection(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BG

        # ── Title ──
        title = Text(
            "MLA Step 1: Fused Down-Projection",
            font_size=28,
            color=WHITE,
            weight=BOLD,
        )
        source = Text(
            "forward_mla.py:137 — forward_absorb_prepare()",
            font_size=11,
            color=C_CODE,
        )
        hdr = VGroup(title, source).arrange(DOWN, buff=0.06)

        # ── Input x [n, 7168] ──
        step1_label = latex_label(
            r"\textbf{Input:}", font_size=22, color=C_HIGHLIGHT
        )
        x_mat = mat_block(r"\mathbf{x}", r"[n, 7168]", C_INPUT, w=1.6, h=0.7)
        step1 = VGroup(step1_label, x_mat).arrange(DOWN, buff=0.12)

        # ── Matmul: x @ W_A ──
        step2_label = latex_label(
            r"\textbf{Down-proj:}\ x \cdot W_A", font_size=22, color=C_HIGHLIGHT
        )
        x2 = mat_block(r"\mathbf{x}", r"[n, 7168]", C_INPUT, w=1.2, h=0.55)
        mul1 = op_circle(r"\times")
        w_a = mat_block(r"W_{\!A}", r"[7168, 2112]", C_WEIGHT, w=1.4, h=0.55)
        eq1 = op_circle(r"=", color="#555577")
        qkv = mat_block(r"\text{qkv}", r"[n, 2112]", C_LATENT, w=1.2, h=0.55)

        proj_row = VGroup(x2, mul1, w_a, eq1, qkv).arrange(RIGHT, buff=0.15)

        # ── Split into 3 parts ──
        split_label = latex_label(
            r"\text{torch.split}(-1,\ [1536, 512, 64])", font_size=18, color=C_DIM
        )
        c_q = mat_block(r"c_q", r"[n, 1536]", C_LATENT, w=1.1, h=0.5)
        c_kv = mat_block(r"c_{kv}", r"[n, 512]", C_KV, w=0.9, h=0.5)
        k_pe = mat_block(r"k_{pe}", r"[n, 64]", C_ROPE, w=0.8, h=0.5)

        split_row = VGroup(c_q, c_kv, k_pe).arrange(RIGHT, buff=0.2)

        step2 = VGroup(step2_label, proj_row, split_label, split_row).arrange(
            DOWN, buff=0.12
        )

        dim_note = latex_label(
            r"2112 = \underbrace{1536}_{d_{cq}} + \underbrace{512}_{d_{ckv}} + \underbrace{64}_{d_{rope}}",
            font_size=18,
            color=C_DIM,
        )

        # ── Layout ──
        all_content = VGroup(hdr, step1, step2, dim_note)
        all_content.arrange(DOWN, buff=0.25)

        # ── Animate ──
        self.play(Write(title), FadeIn(source), run_time=0.8)
        self.play(self.camera.auto_zoom(all_content, margin=0.4), run_time=0.3)

        # Step 1: Input
        self.play(FadeIn(step1_label), run_time=0.3)
        self.play(FadeIn(x_mat), run_time=0.5)
        self.wait(0.5)

        # Step 2: Matmul
        self.play(FadeIn(step2_label), run_time=0.3)
        self.play(
            LaggedStart(
                FadeIn(x2),
                FadeIn(mul1),
                FadeIn(w_a),
                FadeIn(eq1),
                FadeIn(qkv),
                lag_ratio=0.1,
            ),
            run_time=1.2,
        )
        self.wait(0.3)

        # Split
        self.play(FadeIn(split_label), run_time=0.2)
        self.play(
            LaggedStart(FadeIn(c_q), FadeIn(c_kv), FadeIn(k_pe), lag_ratio=0.15),
            run_time=0.8,
        )
        self.play(FadeIn(dim_note), run_time=0.5)
        self.wait(2)


# ═════════════════════════════════════════════════════════════
# Scene 2: Q Up-Projection — c_q → Norm → @ W_B^q → [q_nope | q_pe]
# Shows normalization + matmul + split
# ═════════════════════════════════════════════════════════════


class QUpProjection(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BG

        title = Text(
            "MLA Step 2: Q Up-Projection", font_size=28, color=WHITE, weight=BOLD
        )
        source = Text(
            "forward_mla.py:223-262 — q_a_layernorm + q_b_proj",
            font_size=11,
            color=C_CODE,
        )
        hdr = VGroup(title, source).arrange(DOWN, buff=0.06)

        # ── Input c_q ──
        step1_label = latex_label(
            r"\textbf{From Step 1:}", font_size=22, color=C_HIGHLIGHT
        )
        cq_in = mat_block(r"c_q", r"[n, 1536]", C_LATENT, w=1.2, h=0.6)
        step1 = VGroup(step1_label, cq_in).arrange(DOWN, buff=0.12)

        # ── Norm + Matmul ──
        step2_label = latex_label(
            r"\textbf{Norm + Up-proj:}", font_size=22, color=C_HIGHLIGHT
        )
        cq2 = mat_block(r"c_q", r"[n, 1536]", C_LATENT, w=1.0, h=0.5)
        arr1 = thin_arrow(LEFT * 0.2, RIGHT * 0.2, color=C_DIM)
        norm_box = mat_block(
            r"\text{Norm}", r"", "#666688", w=0.7, h=0.5, shape_fs=10
        )
        arr2 = thin_arrow(LEFT * 0.2, RIGHT * 0.2, color=C_DIM)
        mul2 = op_circle(r"\times", r=0.16, fs=16)
        w_b = mat_block(
            r"W_{B}^{q}", r"[1536, h{\cdot}192]", C_WEIGHT, w=1.5, h=0.5
        )
        eq2 = op_circle(r"=", color="#555577", r=0.16, fs=16)
        q_full = mat_block(r"\mathbf{q}", r"[n, h, 192]", C_INPUT, w=1.2, h=0.5)

        q_row = VGroup(cq2, arr1, norm_box, arr2, mul2, w_b, eq2, q_full).arrange(
            RIGHT, buff=0.1
        )

        # ── Split q ──
        split_label = latex_label(
            r"\text{split}(-1,\ [d_{nope}, d_{rope}])",
            font_size=18,
            color=C_DIM,
        )
        q_nope = mat_block(
            r"q_{\text{nope}}", r"[n, h, 128]", C_INPUT, w=1.1, h=0.5
        )
        q_pe = mat_block(r"q_{pe}", r"[n, h, 64]", C_ROPE, w=0.9, h=0.5)
        q_split = VGroup(q_nope, q_pe).arrange(RIGHT, buff=0.25)

        step2 = VGroup(step2_label, q_row, split_label, q_split).arrange(
            DOWN, buff=0.12
        )

        dim_note = latex_label(
            r"192 = 128_{\text{nope}} + 64_{\text{rope}}",
            font_size=18,
            color=C_DIM,
        )

        # ── Layout ──
        all_content = VGroup(hdr, step1, step2, dim_note)
        all_content.arrange(DOWN, buff=0.25)

        # ── Animate ──
        self.play(Write(title), FadeIn(source), run_time=0.8)
        self.play(self.camera.auto_zoom(all_content, margin=0.4), run_time=0.3)

        self.play(FadeIn(step1_label), run_time=0.3)
        self.play(FadeIn(cq_in), run_time=0.5)
        self.wait(0.5)

        self.play(FadeIn(step2_label), run_time=0.3)
        self.play(
            LaggedStart(
                *[FadeIn(m) for m in [cq2, arr1, norm_box, arr2, mul2, w_b, eq2, q_full]],
                lag_ratio=0.08,
            ),
            run_time=1.5,
        )
        self.wait(0.3)

        self.play(FadeIn(split_label), run_time=0.2)
        self.play(FadeIn(q_nope), FadeIn(q_pe), run_time=0.6)
        self.play(FadeIn(dim_note), run_time=0.5)
        self.wait(2)


# ═════════════════════════════════════════════════════════════
# Scene 3: Weight Absorption BMM — q_nope @ W_kc → q_absorbed
# The key MLA trick: absorb K up-projection weight into Q
# Shows per-head batch matrix multiplication
# ═════════════════════════════════════════════════════════════


class WeightAbsorptionBMM(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BG

        title = Text(
            "MLA Step 3: Weight Absorption (BMM)",
            font_size=28,
            color=WHITE,
            weight=BOLD,
        )
        source = Text(
            "forward_mla.py:287-368 — q_nope @ w_kc (per-head BMM)",
            font_size=11,
            color=C_CODE,
        )
        hdr = VGroup(title, source).arrange(DOWN, buff=0.06)

        # ── Derivation reminder ──
        deriv_label = latex_label(
            r"\textbf{Why:}\ S_h = q_{\text{nope},h} \cdot k_h^\top = q_{\text{nope},h} \cdot (c_{kv} \cdot W_{uk,h}^\top)^\top",
            font_size=18,
            color=C_DIM,
        )
        deriv_eq = latex_label(
            r"= (q_{\text{nope},h} \cdot W_{uk,h}) \cdot c_{kv}^\top = q_{\text{abs},h} \cdot c_{kv}^\top",
            font_size=18,
            color=C_ABSORB,
        )
        deriv = VGroup(deriv_label, deriv_eq).arrange(DOWN, buff=0.06)

        # ── BMM operation ──
        bmm_label = latex_label(
            r"\textbf{Batch matmul:}\ \text{(per head)}", font_size=22, color=C_HIGHLIGHT
        )

        qn = mat_block(
            r"q_{\text{nope}}", r"[h, n, 128]", C_INPUT, w=1.0, h=0.5
        )
        mul3 = op_circle(r"\times", r=0.16, fs=16)
        wkc = mat_block(
            r"W_{kc}", r"[h, 128, 512]", C_WEIGHT, w=1.0, h=0.5
        )
        eq3 = op_circle(r"=", color="#555577", r=0.16, fs=16)
        qabs = mat_block(
            r"q_{\text{abs}}", r"[h, n, 512]", C_ABSORB, w=0.9, h=0.5
        )

        bmm_row = VGroup(qn, mul3, wkc, eq3, qabs).arrange(RIGHT, buff=0.12)
        step1 = VGroup(bmm_label, bmm_row).arrange(DOWN, buff=0.12)

        # ── Code note ──
        code_note = latex_label(
            r"\texttt{torch.bmm}(q\_nope.transpose(0, 1),\ w\_kc)",
            font_size=16,
            color=C_CODE,
        )

        # ── Result ──
        result_label = latex_label(
            r"\textbf{Result:}\ \text{q absorbs K up-projection, saves KV cache space}",
            font_size=18,
            color="#55ff55",
        )

        # ── Layout ──
        all_content = VGroup(hdr, deriv, step1, code_note, result_label)
        all_content.arrange(DOWN, buff=0.25)

        # ── Animate ──
        self.play(Write(title), FadeIn(source), run_time=0.8)
        self.play(self.camera.auto_zoom(all_content, margin=0.4), run_time=0.3)

        # Derivation
        self.play(FadeIn(deriv_label), run_time=0.5)
        self.play(FadeIn(deriv_eq), run_time=0.8)
        self.wait(1)

        # BMM
        self.play(FadeIn(bmm_label), run_time=0.3)
        self.play(
            LaggedStart(*[FadeIn(m) for m in [qn, mul3, wkc, eq3, qabs]], lag_ratio=0.1),
            run_time=1.2,
        )
        self.wait(0.5)

        self.play(FadeIn(code_note), run_time=0.5)
        self.wait(0.5)

        self.play(FadeIn(result_label), run_time=0.8)
        self.wait(2)


# ═════════════════════════════════════════════════════════════
# Scene 4: Attention BMM — attn @ W_vc → output
# Shows the second absorption: V up-projection weight applied after attention
# ═════════════════════════════════════════════════════════════


class AttentionBMM(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BG

        title = Text(
            "MLA Step 4: Attention + V Absorption (BMM)",
            font_size=28,
            color=WHITE,
            weight=BOLD,
        )
        source = Text(
            "forward_mla.py:549-623 — attn @ w_vc (per-head BMM)",
            font_size=11,
            color=C_CODE,
        )
        hdr = VGroup(title, source).arrange(DOWN, buff=0.06)

        # ── Attention operation (simplified) ──
        attn_label = latex_label(
            r"\textbf{Attention:}\ \text{(uses absorbed q, latent kv)}",
            font_size=22,
            color=C_HIGHLIGHT,
        )
        attn_eq = latex_label(
            r"\text{attn}_h = \text{softmax}\left(\frac{q_{\text{abs},h} \cdot c_{kv}^\top}{\sqrt{d}}\right) \cdot c_{kv}",
            font_size=20,
            color=WHITE,
        )
        attn_dim = latex_label(
            r"[n, h, 512]", font_size=16, color=C_DIM
        )
        attn_line = VGroup(attn_eq, attn_dim).arrange(DOWN, buff=0.04)
        step1 = VGroup(attn_label, attn_line).arrange(DOWN, buff=0.12)

        # ── V absorption BMM ──
        bmm_label = latex_label(
            r"\textbf{V absorption:}\ \text{(per head)}", font_size=22, color=C_HIGHLIGHT
        )

        attn_mat = mat_block(
            r"\text{attn}_h", r"[h, n, 512]", C_LATENT, w=1.0, h=0.5
        )
        mul4 = op_circle(r"\times", r=0.16, fs=16)
        wvc = mat_block(
            r"W_{vc}", r"[h, 512, 128]", C_WEIGHT, w=1.0, h=0.5
        )
        eq4 = op_circle(r"=", color="#555577", r=0.16, fs=16)
        out_h = mat_block(
            r"o_h", r"[h, n, 128]", C_OUTPUT, w=0.9, h=0.5
        )

        bmm_row = VGroup(attn_mat, mul4, wvc, eq4, out_h).arrange(RIGHT, buff=0.12)
        step2 = VGroup(bmm_label, bmm_row).arrange(DOWN, buff=0.12)

        # ── Final output ──
        final_label = latex_label(
            r"\textbf{Final:}\ \text{flatten} \to W_o",
            font_size=22,
            color=C_HIGHLIGHT,
        )
        final_eq = latex_label(
            r"\text{output} = W_o \cdot \text{flatten}(o_h)\ \to [n, 7168]",
            font_size=20,
            color=C_OUTPUT,
        )
        step3 = VGroup(final_label, final_eq).arrange(DOWN, buff=0.08)

        # ── Code note ──
        code_note = latex_label(
            r"\texttt{torch.bmm}(attn.transpose(0, 1),\ w\_vc)",
            font_size=16,
            color=C_CODE,
        )

        # ── Layout ──
        all_content = VGroup(hdr, step1, step2, code_note, step3)
        all_content.arrange(DOWN, buff=0.25)

        # ── Animate ──
        self.play(Write(title), FadeIn(source), run_time=0.8)
        self.play(self.camera.auto_zoom(all_content, margin=0.4), run_time=0.3)

        # Attention
        self.play(FadeIn(attn_label), run_time=0.3)
        self.play(FadeIn(attn_line), run_time=0.8)
        self.wait(1)

        # V absorption BMM
        self.play(FadeIn(bmm_label), run_time=0.3)
        self.play(
            LaggedStart(
                *[FadeIn(m) for m in [attn_mat, mul4, wvc, eq4, out_h]], lag_ratio=0.1
            ),
            run_time=1.2,
        )
        self.wait(0.5)

        self.play(FadeIn(code_note), run_time=0.5)
        self.wait(0.5)

        # Final
        self.play(FadeIn(step3), run_time=0.8)
        self.wait(2)
