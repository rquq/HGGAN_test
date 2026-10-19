# HGGAN Main Style Reference Sampling & Long String Generation Analysis

**Checkpoint**: [`/home/quq/machineLearning/HTG/chkpoints/30_8_2026/best_fid_3.0260_main.pth`](file:///home/quq/machineLearning/HTG/chkpoints/30_8_2026/best_fid_3.0260_main.pth) (Epoch 56, Best FID: 3.0260, KID: 0.0633)  
**Dataset**: [`/home/quq/machineLearning/HTG/iam/iam_test_64.hdf5`](file:///home/quq/machineLearning/HTG/iam/iam_test_64.hdf5) (25,432 test samples across 161 writers)  
**Architecture**: Native 64-pixel Global-Local Adversarial Model with Multi-Token Style Encoder ($8 \times 32$ tokens) and Style-Content Character Fusion Generator

---

## 1. Executive Summary & Objective

This study systematically evaluates how the **HGGAN Main (64px)** generator and multi-token style encoder behave when conditioned on short, single-character, numeric, and punctuation reference images:
1. **Single Punctuation Marks** (`.`, `,`, `?`, `!`, `"`, `-`)
2. **Single Digits** (`1`, `2`, `3`, `7`, `8`, `9`)
3. **Multi-Digit Numbers** (`37`, `11`, `17`, `1604`)
4. **Capitalized Letters / Words** (`I`, `A`, `London`)
5. **Full Word Controls** (from matching authors) & **Unconditioned Baselines**

These styles were used to synthesize **long target strings** (13 to 50 characters) including words (`"extraordinary"`, `"differentiation"`), phrases (`"Machine Learning"`), and extended sentences (`"the quick brown fox jumps over the lazy dog"`, `"Handwriting text generation with HGGAN in 2026."`, `"Meeting at 10:30, room 402, with Dr. Watson."`).

---

## 2. Visual Showcases (Native x64 Resolution)

### Showcase 1: Full-Alphabet Long Sentence (43 Characters)
> Target: `"the quick brown fox jumps over the lazy dog"`

![Showcase 1: HGGAN Main Pangram Generation across reference styles](./media/special_digits/hggan_fine_grained_showcase_the_quick_brown_fox_jumps.png)

---

### Showcase 2: Complex 47-Character Alphanumeric & Punctuation Sentence
> Target: `"Handwriting text generation with HGGAN in 2026."`

![Showcase 2: HGGAN Main 47-Char Sentence Generation](./media/special_digits/hggan_fine_grained_showcase_Handwriting_text_generati.png)

---

### Showcase 3: Single Long Word Generation (13 Characters)
> Target: `"extraordinary"`

![Showcase 3: HGGAN Main Extraordinary Generation](./media/special_digits/hggan_fine_grained_showcase_extraordinary.png)

---

### Showcase 4: Multi-Author Style Transfer Comparison
> Comparing style consistency across 4 different writers under Punctuation vs Uppercase vs Full Word vs Unconditioned baselines.

![Showcase 4: HGGAN Main Multi-Author Comparison](./media/special_digits/hggan_master_comparison_the_quick_brown_fox_jumps.png)

---

## 3. Quantitative Style Token Statistics (HGGAN Main)

In HGGAN Main, the `StyleEncoder` outputs an $8 \times 32$ tensor $\mathbf{Z} \in \mathbb{R}^{8 \times 32}$:
* **Token 0 ($\mathbf{z}_0 \in \mathbb{R}^{32}$)**: Global Style Context (modulates generator CCBN).
* **Tokens 1..7 ($\mathbf{z}_{1:7} \in \mathbb{R}^{7 \times 32}$)**: Local Stroke Attentive Tokens (modulate character-level allographs via `style_content_mix`).

| Reference Category | Sample Count ($N$) | Mean Global Norm $||\mathbf{z}_0||_2$ | Std Dev | Total Norm $||\mathbf{Z}||_F$ | Characteristic Generation Quality |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full Long Words** (`>= 6` chars) | 50 | **$6.46$** | $\pm 1.43$ | **$6.53$** | **Gold Standard**: Exceptional fidelity, smooth ligatures, realistic ink pressure. |
| **Short Words** (`<= 5` chars) | 50 | **$7.68$** | $\pm 1.61$ | **$7.74$** | **High Fidelity**: Crisp letter shapes and consistent baseline. |
| **Capitalized Words** (`London`, `Mr`) | 50 | **$7.69$** | $\pm 2.33$ | **$7.75$** | **High Fidelity**: Natural ascender height and kerning. |
| **Single Digits** (`1`, `2`, `3`, `7`, `8`, `9`) | 16 | **$8.42$** | $\pm 1.54$ | **$8.48$** | **High Stylistic Diversity**: Accurate transfer of pen slant, weight, and roundness. |
| **Multi-digit Numbers** (`37`, `11`, `17`) | 35 | **$8.64$** | $\pm 2.24$ | **$8.70$** | **Clean & Stable**: Broad spatial support (64–160px) yields natural handwriting. |
| **Single Capital Letters** (`I`, `A`) | 50 | **$10.07$** | $\pm 2.24$ | **$10.11$** | **Strong Downstroke Guidance**: Solid letter structures. |
| **Punctuation Marks** (`.`, `,`, `?`, `!`, `"`) | 50 | **$9.73$** | $\pm 2.46$ | **$9.79$** | **Bimodal**: Directional marks (`,`, `?`, `!`) excel; isolated dots (`.`) show vertical elongation. |

---

## 4. Key Architectural & Empirical Insights

### 1. Multi-Token Disentanglement ($8 \times 32$ vs SpiS-GAN's $1 \times 96$)
* **SpiS-GAN**: Uses a single pooled vector $\mu \in \mathbb{R}^{96}$ concatenated with 32-dim Gaussian noise. Punctuation inputs collapse into a 1-column average with severe norm inflation ($||\mu|| = 30.71$).
* **HGGAN Main**: Uses an 8-token representation where local tokens $\mathbf{z}_{1:7}$ are extracted via cross-attention with fixed orthogonal slot anchors:
  $$\mathbf{z}_{\text{local}} = \text{Linear}(\text{LN}(\mathbf{Q}_{\text{anchors}} + \sigma(\mathbf{W}_{\text{gate}}) \cdot \text{CrossAttn}(\mathbf{Q}, \mathbf{K}, \mathbf{V})))$$
* **Impact**: Total style norm remains tightly bounded between $6.53$ and $10.11$ across all categories without exploding on short crops.

### 2. Behavior on Punctuation Marks
* **Directional Punctuation (`,`, `?`, `!`, `"`)**:
  * Unlike SpiS-GAN which produced margin smudges, HGGAN Main produces **exceptionally clean, high-resolution handwriting**:
    * Reference `'?'` (writer 1): Generates graceful, light-grey pencil-style cursive script.
    * Reference `'!'` (writer 19): Generates dynamic, forward-slanted handwriting.
    * Reference `'"'` (writer 17): Generates bold, dark marker-pen handwriting.
* **Isolated Point Marks (`.`, `-`)**:
  * When conditioned on an isolated 32px dot crop, the spatial cross-attention keys lack horizontal stroke context, inducing tall, narrow, vertical stroke artifacts in character stems.

### 3. High-Fidelity Style Transfer from Single Digits
* Single digit conditioning demonstrates precise control over penmanship characteristics:
  * **Digit `'1'` (w13, sharp forward slant)**: Transferred a high-slant, sharp cursive flow across the entire 43-character sentence.
  * **Digit `'2'` (w29, upright round loop)**: Produced upright, rounded, medium-weight letters.
  * **Digit `'3'` (w34, relaxed open shape)**: Induced relaxed, open-kerning lowercase cursive.
  * **Digit `'7'` (w32, heavy pressure)**: Produced rich, dark, high-contrast ink strokes with natural ligature connections (`f-o-x`, `j-u-m-p-s`).
  * **Digit `'9'` (w137, delicate hairline)**: Transferred ultra-thin ballpoint pen writing with delicate ascenders.

### 4. Native 64-Pixel Resolution Advantages over 32-Pixel Baselines
* **Texture & Anti-Aliasing**: HGGAN Main includes a `StyleFrequencyRefinement` module that generates realistic high-frequency edge variations and pen-pressure gradients.
* **Long-Sequence Legibility**: On 47-character sentences (`"Handwriting text generation with HGGAN in 2026."`), all character ascenders, descenders, and ligatures remain distinct without blurring or repetition.

---

## 5. Direct Comparison: SpiS-GAN vs HGGAN Main

| Metric / Dimension | SpiS-GAN (x32) | HGGAN Main (x64) |
| :--- | :--- | :--- |
| **Output Resolution** | $32 \times W$ | **$64 \times W$ (2x vertical detail)** |
| **Best IAM FID** | $4.4595$ | **$3.0260$ (32.1% lower FID)** |
| **Style Representation** | 96-dim scalar pooled vector $+ 32$-dim noise | **$8 \times 32$-dim Multi-Token (Global + 7 Local Attentive Slots)** |
| **Punctuation Vector Norm** | $30.71 \pm 18.24$ (out-of-distribution inflation) | **$9.73 \pm 2.46$ (stable & bounded)** |
| **Punctuation Visual Artifacts** | Gray horizontal background smudges on `.` and `-` | **No background smudges; clean white paper** |
| **Single Digit Style Transfer** | Captures general slant & weight | **Captures fine stroke pressure, ligatures, and glyph curvature** |
| **Long Sentence Consistency** | Good character structure, 32px pixelation | **Pristine, publication-quality 64px cursive rendering** |

---

## 6. Artifact Locations

* **Pangram Showcase Image**: [`./media/special_digits/hggan_fine_grained_showcase_the_quick_brown_fox_jumps.png`](./media/special_digits/hggan_fine_grained_showcase_the_quick_brown_fox_jumps.png)
* **47-Char Sentence Showcase**: [`./media/special_digits/hggan_fine_grained_showcase_Handwriting_text_generati.png`](./media/special_digits/hggan_fine_grained_showcase_Handwriting_text_generati.png)
* **Single Word Showcase**: [`./media/special_digits/hggan_fine_grained_showcase_extraordinary.png`](./media/special_digits/hggan_fine_grained_showcase_extraordinary.png)
* **Cross-Author Pangram Comparison**: [`./media/special_digits/hggan_master_comparison_the_quick_brown_fox_jumps.png`](./media/special_digits/hggan_master_comparison_the_quick_brown_fox_jumps.png)
