<p align="center">
  <img src="assets/cqu.png" height="150" alt="Chongqing University" />
</p>

<h1 align="center">DoFu-Net: A Tri-Branch Fusion Network for LiDAR-Camera Place Recognition</h1>

<p align="center">
  CHONGQING UNIVERSITY · Key Laboratory of Optoelectronic Technology and Systems of Ministry of Education
</p>

<p align="center">
  <b>LiDAR-Camera Place Recognition · Panoramic Vision · Geometric Multi-View Interaction · Robust Sensor Fusion</b>
</p>

<p align="center">
  <a href="#-overview">Overview</a> ·
  <a href="#-method">Method</a> ·
  <a href="#-visualizations--demo">Demo</a> ·
  <a href="#-experimental-results">Results</a> ·
  <a href="#-installation--usage">Usage</a> ·
  <a href="#-citation">Citation</a>
</p>

<!-- TODO: Replace this file with the exported overview figure from the manuscript. -->
<p align="center">
  <img src="assets/图片1.svg" width="95%" alt="Overview of DoFu-Net" />
</p>

**DoFu-Net** is a tri-branch LiDAR-camera place recognition framework that learns discriminative and robust **256-dimensional multimodal global descriptors** from LiDAR point clouds and panoramic multi-camera images. It integrates a geometry-guided LiDAR backbone, a panoramic visual continuity module, and lightweight cross-modal feature interactions.

> **Paper status:** Manuscript under revision for *ISPRS Journal of Photogrammetry and Remote Sensing*.
>
> **Repository status:** This README is a project-page draft. Source-code commands, checkpoints, and download links will be added after the repository is organized.

---

## 🔍 Overview

DoFu-Net addresses three challenges in multimodal place recognition:

1. **LiDAR geometric organization:** learning complementary geometric structures across spherical views rather than treating each view independently.
2. **Panoramic visual continuity:** modeling local and cross-view feature relationships across concatenated surround-camera features.
3. **Robust multimodal fusion:** improving descriptor stability when one sensor modality becomes degraded.

The network contains three branches:

- **LiDAR Branch:** spherical **Pitch-Yaw (PY)**, **Radius-Yaw (RY)**, and **Radius-Pitch (RP)** representations, with geometry-guided axial interactions and progressive feature updates.
- **Visual Branch:** a shared-weight **ConvNeXtV2** encoder followed by the proposed **Local Continuity Graph (LCG)** module.
- **Fusion Branch:** low-rank multiplicative cross-modal interactions, contextual refinement, residual write-back, and a lightweight concatenation-based descriptor head.

---

## 🧠 Method

### 1. Geometry-Guided LiDAR Backbone

Building on the spherical three-view representation developed in our previous [Dome](https://github.com/ljkljk123/Dome-master) work, DoFu-Net introduces **learnable, progressive multi-view feature interaction** rather than reusing Dome's handcrafted descriptor-matching pipeline.

- **Pitch interaction:** associates the **PY and RP views** through their shared Pitch coordinate.
- **Yaw interaction:** associates the **PY and RY views** through their shared Yaw coordinate.
- **Axial Fusion Layer (AFL):** combines complementary view information through two geometry-guided attention paths.
- **Write Back Gate (WBG):** controls updates to the primary view, while FiLM-based feedback supports progressive auxiliary-view enhancement.

<!-- TODO: Export the AFL/WBG architecture or create its animation. -->
<p align="center">
  <img src="assets/图片3.svg" width="90%" alt="Axial Fusion Layer and Write Back Gate" />
</p>

### 2. Panoramic Local Continuity Graph (LCG)

The proposed **LCG** module models adaptive local structural relationships in the panoramic feature map instead of relying exclusively on fixed neighborhood mixing. Its main operations include:

- Local differences between each feature node and its candidate neighbors.
- A learned spatially adaptive threshold and candidate-edge scoring.
- Discrete edge selection using an **STE-based hard binary gate**.
- Valid-neighbor-normalized message aggregation and feature updating.
- Horizontal circular neighborhood sampling for panoramic boundary continuity.

<!-- TODO: Replace with the LCG figure or graph-animation thumbnail. -->
<p align="center">
  <img src="assets/图片5.svg" width="80%" alt="Local Continuity Graph module" />
</p>

### 3. Lightweight LiDAR-Camera Fusion

The Fusion Branch performs **low-rank multiplicative feature interaction**, contextual refinement, and **residual write-back** into the unimodal branches. Separate LiDAR and visual global descriptors are then combined using a lightweight **concatenation + linear projection** head to produce a normalized 256-dimensional fused descriptor.

**Geometric alignment note:** Resizing LiDAR and panoramic visual features to a common feature-grid size does **not** imply point-to-pixel geometric alignment. DoFu-Net learns implicit feature associations between approximately directionally organized representations; its sensitivity to extrinsic perturbations is evaluated experimentally.

<!-- TODO: Export fusion-branch figure from the manuscript. -->
<p align="center">
  <img src="assets/图片7.svg" width="90%" alt="LiDAR-camera fusion branch" />
</p>

---

## 🎬 Visualizations & Demo

### Place Recognition and SLAM Demonstration

<!-- TODO: Create a concise, muted preview GIF; link it to a complete video. -->
<p align="center">
  <img src="assets/slam_demo.gif" width="88%" alt="DoFu-Net place recognition demo" />
</p>

**Full SLAM demonstration:** [Video link to be added]

### How the Modules Work

<!-- TODO: Optional animated illustrations with the same style and color palette. -->

- **AFL animation:** shared Pitch/Yaw coordinates → dual-path interactions → WBG update.
- **LCG animation:** local candidate neighbors → adaptive edge selection → message aggregation.
- **Robustness animation:** correct retrieval examples under LiDAR occlusion and visual degradation.

---

## 📊 Experimental Results

Experiments are conducted on **nuScenes**, **Argoverse 2 Sensor (AV2S)**, **NCLT**, and three self-collected sequences. The public-dataset results below use the paper's standard **9 m matching threshold**.

### Place Recognition Accuracy

| Dataset | Test sequence | DoFu-Net Recall@1 (%) | LCPR Recall@1 (%) |
|:--|:--|--:|--:|
| nuScenes | BS | **98.40** | 93.45 |
| nuScenes | SON | **99.52** | 98.46 |
| nuScenes | SHV | **99.23** | 86.12 |
| AV2S | DTW | **100.00** | 99.89 |
| AV2S | MIA | **97.24** | 94.14 |
| AV2S | PIT | **99.58** | 96.44 |
| NCLT | 2012-05-26 | **98.57** | 94.23 |
| NCLT | 2013-01-10 | **98.86** | 96.11 |
| NCLT | 2013-02-23 | **99.61** | 95.26 |

**Average Recall@1: 99.00%** across the nine public test sequences. Compared with LCPR, DoFu-Net achieves a **4.32% relative improvement** in average Recall@1 under the reported settings.

<!-- TODO: Insert a compact figure showing PR curves or retrieval matches. -->
<p align="center">
  <img src="assets/benchmark.png" width="95%" alt="Public-dataset benchmark results" />
</p>

### Robustness and Efficiency

- **LiDAR occlusion:** At **90° sector occlusion**, the LiDAR branch achieves a **36.26% relative gain** in mean Recall@1 over RangePlace across the evaluated AV2S sequences.
- **Single-modality degradation:** Fused-descriptor mean Recall@1 reaches **94.53%** with smoke-degraded visual inputs and **94.74%** with occluded LiDAR inputs.
- **Extrinsic perturbation:** Evaluated under both joint Database/Query perturbations and Query-only perturbations, covering Roll, Pitch, Yaw, and translation.
- **Cross-platform tests:** Evaluated on three real-world sequences collected with a six-camera, 16-beam-LiDAR platform.
- **Model efficiency:** **10.75 M** parameters, approximately **0.26 GB** measured GPU memory, and **12.76 ms** mean end-to-end query time on the three AV2S sequences using the evaluation setup reported in the paper.

<!-- TODO: Insert carefully selected robustness comparisons. -->
<p align="center">
  <img src="assets/robustness.png" width="95%" alt="Robustness experiments" />
</p>

---

## 🗂️ Datasets

| Dataset | Sensors used | Role |
|:--|:--|:--|
| nuScenes | Top LiDAR + six surround-view cameras | Training / validation / testing; cross-region testing |
| AV2S | LiDAR + seven surround-view cameras | Training / validation / testing |
| NCLT | Velodyne LiDAR + six cameras | Cross-date training / validation / testing |
| Self-collected | 16-beam LiDAR + six cameras | Cross-platform evaluation only |

Datasets must be obtained independently from their respective providers. Please follow the original providers' licenses and usage terms.

<!-- TODO: Add download instructions and the repository's data-preprocessing layout. -->

---

## 📦 Installation & Usage

> **Pending repository integration.** The commands below should be populated from the actual released code, rather than guessed from the paper.

### Environment

- OS: **[to be confirmed]**
- Python: **[to be confirmed]**
- PyTorch / CUDA: **[to be confirmed]**
- Required packages: **[to be generated from environment specification]**
- Training hardware reported in the paper: **NVIDIA RTX 4060 GPU**

### Repository Setup

```bash
# TODO: Add verified repository clone and environment setup commands.
```

### Dataset Preparation

```bash
# TODO: Add verified dataset preprocessing / configuration commands.
```

### Training

```bash
# TODO: Add verified training entry point and configuration example.
```

### Evaluation

```bash
# TODO: Add verified evaluation entry point and output description.
```

### Pretrained Weights

| Dataset | Checkpoint | Notes |
|:--|:--|:--|
| nuScenes | To be added | Include model and evaluation configuration |
| AV2S | To be added | Include model and evaluation configuration |
| NCLT | To be added | Include model and evaluation configuration |

---

## 📄 Citation

If you find DoFu-Net useful, please cite the paper:

**DoFu-Net: A Tri-Branch Fusion Network for LiDAR-Camera Place Recognition**

*Manuscript under revision for ISPRS Journal of Photogrammetry and Remote Sensing.*

The official BibTeX entry will be added when the final bibliographic information is available.

### Related Work

The spherical three-view representation builds upon our earlier work:

- [Dome: Fast and Robust LiDAR Place Recognition via Spherical Three-View Feature Fusion](https://github.com/ljkljk123/Dome-master), *Measurement*, 2026.

---

## 📌 License & Contact

- **License:** [To be selected and added as a `LICENSE` file.]
- **Academic inquiries / commercial use:** [Add the appropriate contact and usage conditions.]

<!-- TODO before publishing: replace all pending links and image placeholders; verify setup commands; ensure claimed released assets actually exist. -->
