
# CDN-SAA: CDN-SAA: Correlation Decoupling Network with Scribble-guided Agent Attention for Infrared and Visible Image Fusion
 Lei Maa,b, Liang Hu,b, Chengliang Jiang*,b, 

## 📌 Abstract 

> Infrared and visible image fusion (IVIF) aims to generate a more comprehensive scene representation by combining the complementary characteristics of infrared and visible, which effectively addresses the limitations of single-sensor perception in complex environments. However, existing image fusion methods with semantic guidance still face three challenges: 1) they usually rely on task-specific supervision or automatically generated semantic priors and lack a mechanism for users to explicitly specify regions of interest, 2) cross-modal representations may not clearly distinguish modalityshared features from modality-specific detail features, which may hinder the subsequent interaction of complementary information, and 3) semantic priors may be gradually diluted during long-range cross-modal interaction and multi-scale feature propagation. To address these issues, we propose a correlation decoupling network with scribble-guided agent attention for infrared and visible image fusion. Specifically, we construct semantic masks using scribble prompts and exploit the powerful zero-shot segmentation capability of SAM to guide cross-modal feature interaction in fusion, where such masks intuitively embody human visual preferences. Subsequently, we propose a shallow feature
enhancement module with a novel correlation-decoupled constraint loss to improve shallow feature representations and alleviate modal discrepancies by maximizing the correlation of low-frequency features and minimizing the correlation of high-frequency features. In addition, we introduce a novel masked agent cross-attention module to balance global semantic perception and local detail retention by realizing effective interaction between agent features and semantic mask-guided key region features, which can suppress the interference from misalignment artifacts and redundant background information. Finally, we design a multi-scale feature calibration module to achieve mutual calibration of cross-scale fused features under semantic mask guidance, so as to produce fusion images enriched with abundant semantic information. Extensive experiments on several public benchmark datasets demonstrate our model significantly improves the perceptual contrast and information richness of fused images.

## 🖼️ Network Architecture

<p align="center">
  <img src="./fig/method.png" alt="Comparison of fusion frameworks" width="90%">
</p>

*Figure 1: Comparison between conventional task-oriented IVIF methods and the proposed CDN-SAA.*

<p align="center">
  <img src="./fig/model.png" alt="CDN-SAA architecture" width="90%">
</p>

*Figure 2: Overview of the CDN-SAA architecture for interactive infrared and visible image fusion.*

---

## 🚀 How to Run

### 1. Prerequisites

- Python 3.8+
- PyTorch 1.12+
- A compatible NVIDIA GPU and CUDA environment

### 2. Installation

```bash
git clone https://github.com/jiangchengliang59-code/CDN-SAA.git
cd CDN-SAA

conda create -n cdnsaa python=3.8 -y
conda activate cdnsaa
```

### 3. Data Preparation

Organize the datasets as follows:

```text
datasets/
├── M3FD/
│   ├── ir/
│   └── vi/
├── RoadScene/
│   ├── ir/
│   └── vi/
└── TNO/
    ├── ir/
    └── vi/
```

Before training or testing, configure the dataset paths, hyperparameters,
and model settings in `options/fusion.yml`.

<!-- Add the actual training and testing commands here. -->

---

## 📊 Qualitative and Quantitative Results

### 1. Visual Comparison

<p align="center">
  <img src="./fig/M3FD.png" alt="Visual comparison on M3FD" width="100%">
</p>

*Figure 3: Qualitative fusion results for scene "01937" from M3FD.*

<p align="center">
  <img src="./fig/round.png" alt="Visual comparison on RoadScene" width="100%">
</p>

*Figure 4: Qualitative fusion results for scene "FLIR_00977" from RoadScene.*

<p align="center">
  <img src="./fig/TNO.png" alt="Visual comparison on TNO" width="100%">
</p>

*Figure 5: Qualitative fusion results for scene "03" from TNO.*

### 2. Quantitative Comparison

Quantitative results on M3FD, RoadScene, and TNO are reported below.
**Bold** indicates the best result, and *italics* indicate the second-best result.

The compared methods are grouped as follows:

- **I:** Pixel-level or frequency-based fusion
- **II:** Task-oriented fusion
- **III:** Implicit semantic-aware fusion

#### M3FD

| Category | Method | EN ↑ | SD ↑ | SF ↑ | AG ↑ | Q<sub>abf</sub> ↑ | VIF ↑ |
| :---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| I | LRR | 6.437 | 27.183 | 10.683 | 3.591 | 0.498 | 0.634 |
| I | CDDFuse | 6.899 | 37.240 | 14.785 | 4.862 | 0.615 | 0.847 |
| I | SHIP | 6.842 | 35.820 | 16.015 | 5.249 | 0.638 | 0.857 |
| I | SIBA | 6.915 | 37.214 | 14.350 | 4.803 | **0.668** | *0.897* |
| I | ISFL | 6.842 | 33.163 | 15.468 | 5.236 | 0.525 | 0.644 |
| I | EMMA | 6.924 | 38.264 | 15.233 | 5.328 | 0.592 | 0.820 |
| II | BDLFusion | 6.684 | 29.327 | 8.410 | 2.993 | 0.449 | 0.681 |
| II | CAF | 6.923 | 35.259 | 13.567 | 4.357 | 0.512 | 0.663 |
| II | PSFusion | *7.331* | *49.557* | *20.724* | *6.901* | 0.573 | 0.845 |
| II | SeAFusion | 6.846 | 35.433 | 13.954 | 4.770 | 0.599 | 0.776 |
| II | DCEvo | 6.838 | 35.602 | 14.113 | 4.614 | *0.639* | 0.869 |
| III | MaeFuse | 7.021 | 36.101 | 9.528 | 3.748 | 0.386 | 0.591 |
| III | Text-IF | 6.516 | 28.070 | 11.323 | 3.748 | 0.608 | 0.785 |
| III | SAGE | 6.850 | 36.466 | 12.729 | 4.249 | 0.583 | 0.768 |
| III | EVAFusion | 6.842 | 35.581 | 14.108 | 4.619 | 0.633 | 0.861 |
| III | **CDN-SAA (Ours)** | **7.382** | **49.683** | **25.072** | **8.822** | 0.449 | **0.920** |

#### RoadScene

| Category | Method | EN ↑ | SD ↑ | SF ↑ | AG ↑ | Q<sub>abf</sub> ↑ | VIF ↑ |
| :---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| I | LRR | 7.139 | 41.827 | 11.651 | 4.260 | 0.334 | 0.567 |
| I | CDDFuse | 7.124 | 46.012 | 13.151 | 4.461 | 0.542 | 0.771 |
| I | SHIP | 7.213 | 42.840 | *17.028* | 6.160 | 0.522 | 0.731 |
| I | SIBA | 7.371 | 48.203 | 14.572 | 5.521 | **0.631** | **0.886** |
| I | ISFL | 7.083 | 38.838 | 14.914 | 5.528 | 0.468 | 0.601 |
| I | EMMA | 7.331 | 49.324 | 15.219 | 5.739 | 0.444 | 0.720 |
| II | BDLFusion | 7.050 | 35.567 | 8.243 | 3.374 | 0.385 | 0.655 |
| II | CAF | 7.231 | 41.539 | 13.781 | 5.178 | 0.517 | 0.701 |
| II | PSFusion | **7.498** | **50.374** | 16.113 | *6.355* | 0.539 | 0.678 |
| II | SeAFusion | 7.324 | 47.439 | 15.356 | 5.930 | 0.502 | 0.697 |
| II | DCEvo | 7.216 | 43.775 | 14.201 | 5.173 | 0.569 | 0.813 |
| III | MaeFuse | 7.301 | 44.068 | 11.752 | 4.897 | 0.478 | 0.685 |
| III | Text-IF | 7.144 | 28.070 | 13.120 | 5.250 | *0.626* | *0.821* |
| III | SAGE | 7.101 | 47.372 | 10.435 | 3.871 | 0.392 | 0.655 |
| III | EVAFusion | 7.234 | 44.752 | 15.211 | 6.182 | 0.493 | 0.814 |
| III | **CDN-SAA (Ours)** | *7.424* | *50.256* | **19.636** | **7.378** | 0.523 | 0.693 |

#### TNO

| Category | Method | EN ↑ | SD ↑ | SF ↑ | AG ↑ | Q<sub>abf</sub> ↑ | VIF ↑ |
| :---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| I | LRR | 6.898 | 38.571 | 9.437 | 3.626 | 0.366 | 0.592 |
| I | CDDFuse | 7.116 | 46.002 | 13.149 | 4.895 | 0.539 | 0.818 |
| I | SHIP | 6.966 | 41.420 | 13.899 | 5.030 | 0.569 | 0.828 |
| I | SIBA | 6.950 | 42.802 | 12.103 | 4.495 | **0.601** | *0.886* |
| I | ISFL | 6.754 | 31.827 | 13.773 | 5.209 | 0.424 | 0.564 |
| I | EMMA | 7.158 | 46.772 | 11.743 | 4.759 | 0.486 | 0.750 |
| II | BDLFusion | 6.956 | 39.503 | 8.454 | 3.452 | 0.452 | 0.678 |
| II | CAF | 6.815 | 37.053 | 13.766 | 4.517 | 0.446 | 0.621 |
| II | PSFusion | **7.305** | *50.369* | *15.451* | *6.070* | 0.531 | 0.762 |
| II | SeAFusion | 7.103 | 44.199 | 12.412 | 4.939 | 0.506 | 0.733 |
| II | DCEvo | 6.951 | 40.778 | 11.821 | 4.356 | *0.581* | **0.889** |
| III | MaeFuse | 6.993 | 37.364 | 8.795 | 3.919 | 0.389 | 0.607 |
| III | Text-IF | 6.917 | 37.240 | 10.686 | 5.030 | 0.570 | 0.762 |
| III | SAGE | 6.882 | 41.392 | 10.504 | 3.870 | 0.464 | 0.686 |
| III | EVAFusion | 6.972 | 41.878 | 12.923 | 4.476 | 0.573 | 0.884 |
| III | **CDN-SAA (Ours)** | *7.263* | **50.694** | **18.109** | **7.114** | 0.472 | 0.764 |
