# SpiS-GAN Style Reference Sampling & Long String Generation Analysis

**Checkpoint**: [`/home/quq/machineLearning/HTG/chkpoints/SpiS-GAN/bestIAM.pth`](file:///home/quq/machineLearning/HTG/chkpoints/SpiS-GAN/bestIAM.pth) (Epoch 98, FID: 4.4595, KID: 0.0758)  
**Dataset**: [`/home/quq/machineLearning/HTG/iam/iam_test_32.hdf5`](file:///home/quq/machineLearning/HTG/iam/iam_test_32.hdf5) (25,766 test samples across 161 writers)  
**Architecture**: Native 32-pixel BigGAN StarSpiral Generator with SharedBackbone and StyleEncoder

---

## 1. Executive Summary & Objective

This study systematically investigates how **SpiS-GAN** behaves when style reference images are constrained to short, single-character, and non-alphabetic inputs:
1. **Single Punctuation Marks** (`.`, `,`, `?`, `!`, `"`, `-`)
2. **Single Digits** (`1`, `2`, `3`, `7`, `8`, `9`)
3. **Multi-Digit Numbers** (`1815`, `1911`, `1922`, `1914`)
4. **Capitalized Letters / Words** (`I`, `A`, `London`)
5. **Full Word Controls** (from matching authors) & **Unconditioned Baselines**

These extracted styles were conditioned to synthesize **long target strings** (13 to 50 characters) to evaluate how well the StarSpiral Generator preserves stroke weight, slant, baseline stability, and legibility over long sequence lengths.

---

## 2. Visual Showcases

### Showcase 1: Full-Alphabet Long Sentence (43 Characters)
> Target: `"the quick brown fox jumps over the lazy dog"`

![Showcase 1: The Quick Brown Fox](./media/special_digits/fine_grained_showcase_the_quick_brown_fox_jumps.png)

---

### Showcase 2: Complex 50-Character Alphanumeric & Punctuation Sentence
> Target: `"Handwriting text generation with SpiS-GAN in 2026."`

![Showcase 2: Complex 50-Char Sentence](./media/special_digits/fine_grained_showcase_Handwriting_text_generati.png)

---

### Showcase 3: Single Long Word Generation (13 Characters)
> Target: `"extraordinary"`

![Showcase 3: Single Word Extraordinary](./media/special_digits/fine_grained_showcase_extraordinary.png)

---

### Showcase 4: Multi-Author Style Consistency Study
> Comparing style transfer across 4 different writers under Punctuation vs Uppercase vs Full Word vs Unconditioned baselines.

![Showcase 4: Multi-Author Pangram Comparison](./media/special_digits/master_comparison_the_quick_brown_fox_jumps.png)

---

## 3. Quantitative Style Vector Statistics

We analyzed the style latent representations produced by the `StyleEncoder` ($\boldsymbol{\mu} \in \mathbb{R}^{96}$) across all reference categories in the IAM test set:

| Reference Category | Sample Count ($N$) | Mean $L_2$ Norm $||\boldsymbol{\mu}||_2$ | Std Dev | Characteristic Generation Quality |
| :--- | :---: | :---: | :---: | :--- |
| **Full Long Words** (`>= 6` chars) | 50 | **$10.91$** | $\pm 2.16$ | **Gold Standard**: Crisp, stable, uniform stroke weight, zero margin artifacts. |
| **Capitalized Words** (`London`, `Mr`) | 50 | **$11.55$** | $\pm 2.74$ | **High Fidelity**: Smooth letter spacing and crisp baselines. |
| **Short Words** (`<= 5` chars) | 50 | **$13.51$** | $\pm 3.35$ | **High Fidelity**: Slight variance in stroke thickness. |
| **Multi-digit Numbers** (`1815`, `1911`) | 35 | **$15.40$** | $\pm 5.45$ | **High Quality**: Broad spatial support (40–80px) yields natural cursive flow. |
| **Single Capital Letters** (`I`, `A`) | 50 | **$15.59$** | $\pm 7.03$ | **Good**: Strong vertical downstroke guidance; occasional contrast elevation. |
| **Single Digits** (`1`, `2`, `3`, `7`, `8`, `9`) | 25 | **$17.51$** | $\pm 5.45$ | **Distinct Style Transfer**: Each digit imparts a distinct stroke curvature / slant. |
| **Punctuation Marks** (`.`, `,`, `?`, `!`, `"`) | 50 | **$30.71$** | $\pm 18.24$ | **Volatile / Out-of-Distribution**: $\approx 3\times$ higher norm; can cause background smudging. |

---

## 4. Key Findings & Mechanistic Insights

### 1. The Punctuation Pooling Anomaly ($L=1$ Problem)
* **Architecture**: In SpiS-GAN's `StyleEncoder`, input images are processed by the CNN backbone and downsampled horizontally by a factor of 16. The style vector is obtained via 1D temporal average pooling:
  $$\text{style} = \frac{1}{L} \sum_{l=1}^L \mathbf{F}_l, \quad \text{where } L = \left\lfloor \frac{W}{16} \right\rfloor$$
* **Mechanics**: For a 16px wide crop containing a single dot (`.`) or hyphen (`-`), $L = 1$. The feature map consists of only a single column.
* **Consequence**: Without multi-character averaging across diverse ascenders and descenders, the background padding values ($-1.0$) dominate the convolution activations. The linear projection layers receive an out-of-distribution feature vector with an abnormally high magnitude ($||\boldsymbol{\mu}|| \approx 30.71 \pm 18.24$).
* **Visual Effect**: When generating long strings, this high-norm style vector causes horizontal gray margin artifacts and blurred ink blooming across the top and bottom borders.

### 2. Stroke-Rich Punctuation Succeeds (`,` and `?`)
* When punctuation marks contain directional stroke trajectories (such as the sharp tail of a comma `,` or the loop and dot of `?`), the encoder successfully captures pen pressure and slant.
* Conditioning on a comma (`,`) produces bold, high-contrast, forward-slanted cursive across all 50 generated characters with zero background collapse.

### 3. Single Digits Act as Potent Style Priors (`1`, `2`, `3`, `7`, `8`, `9`)
* Each single digit encapsulates distinct handwriting traits that propagate coherently across long sentences:
  * **Digit `'1'` (forward slant)**: Injected a clean, italicized cursive slant across all target words.
  * **Digit `'2'` (open loops)**: Produced open, airy, rounded lowercase letter shapes.
  * **Digit `'3'` (upright printed)**: Induced upright, block-like lowercase font structures.
  * **Digit `'7'` (heavy top bar)**: Transferred dark, high-pressure cursive script.
  * **Digit `'9'` (delicate loop & descender)**: Transferred thin, light pen-pressure handwriting.

### 4. Multi-Digit Numbers Match Full-Word Stability (`1815`, `1911`, `1922`, `1914`)
* Multi-digit numbers span 40–80 pixels ($L = 2 \dots 5$ feature columns).
* This provides sufficient spatial coverage for the shared CNN backbone to produce smooth, representative style vectors ($||\boldsymbol{\mu}|| \approx 15.4$) that match full words in quality and consistency.

### 5. StarSpiral BigGAN Long-Range Stability
* Across 43-character pangrams and 50-character sentences, the StarSpiral spatial attention modules preserve stroke width, slant angle, and letter height uniformly from the first character to the last character without fading, repetition collapse, or drift.
* **Latent Composition Requirement**: SpiS-GAN requires a 128-dimensional latent vector $\mathbf{z} = [\mathbf{z}_{\text{noise}}, \boldsymbol{\mu}_{\text{style}}]$, where the first 32 dimensions must be non-zero standard normal Gaussian noise $\mathcal{N}(0, I_{32})$ and the remaining 96 dimensions contain the style vector.

---

## 5. Artifact Locations

* **Pangram Showcase Image**: [`./media/special_digits/fine_grained_showcase_the_quick_brown_fox_jumps.png`](./media/special_digits/fine_grained_showcase_the_quick_brown_fox_jumps.png)
* **50-Char Sentence Showcase**: [`./media/special_digits/fine_grained_showcase_Handwriting_text_generati.png`](./media/special_digits/fine_grained_showcase_Handwriting_text_generati.png)
* **Single Word Showcase**: [`./media/special_digits/fine_grained_showcase_extraordinary.png`](./media/special_digits/fine_grained_showcase_extraordinary.png)
* **Cross-Author Pangram Comparison**: [`./media/special_digits/master_comparison_the_quick_brown_fox_jumps.png`](./media/special_digits/master_comparison_the_quick_brown_fox_jumps.png)
