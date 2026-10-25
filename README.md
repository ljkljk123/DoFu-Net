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
> **Repository status:** The nuScenes source implementation and usage instructions are available in this release. Trained checkpoints and the other dataset adapters will be added separately.

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

**Full SLAM demonstration:** [https://www.bilibili.com/video/BV1fbHg6UEpQ/?share_source=copy_web&vd_source=646c84d0e91d0a42c3b1130dc776744e]

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

### Cross-Platform Generalization on Self-Collected Sequences

The following animations visualize the **top-1 place recognition results on Seq02** of our self-collected dataset. The first row shows five LiDAR-camera methods, and the second row shows four single-modality baselines. **Green:** correct matches; **red:** incorrect matches; **blue:** missed loop detections. Click an animation to view it at full size.

<!-- GIF directory: assets/gifs/<Algorithm>/seq02_matches3d.gif -->
<table align="center">
  <tr>
    <td align="center">
      <a href="assets/gifs/DoFu/seq02_matches3d.gif"><img src="assets/DoFu_seq02_matches3d.gif" width="150" alt="DoFu-Net Seq02 place recognition matches" /></a>
      <br/><strong>DoFu-Net</strong>
    </td>
    <td align="center">
      <a href="assets/gifs/LCPR/seq02_matches3d.gif"><img src="assets/LCPR_seq02_matches3d.gif" width="150" alt="LCPR Seq02 place recognition matches" /></a>
      <br/><strong>LCPR</strong>
    </td>
    <td align="center">
      <a href="assets/gifs/AdaFusion/seq02_matches3d.gif"><img src="assets/AdaFusion_seq02_matches3d.gif" width="150" alt="AdaFusion Seq02 place recognition matches" /></a>
      <br/><strong>AdaFusion</strong>
    </td>
    <td align="center">
      <a href="assets/gifs/MinkLoc++/seq02_matches3d.gif"><img src="assets/MinkLoc_seq02_matches3d.gif" width="150" alt="MinkLoc++ Seq02 place recognition matches" /></a>
      <br/><strong>MinkLoc++</strong>
    </td>
    <td align="center">
      <a href="assets/gifs/UMF/seq02_matches3d.gif"><img src="assets/UMF_seq02_matches3d.gif" width="150" alt="UMF Seq02 place recognition matches" /></a>
      <br/><strong>UMF</strong>
    </td>
  </tr>
</table>

<table align="center">
  <tr>
    <td align="center">
      <a href="assets/gifs/BoQ/seq02_matches3d.gif"><img src="assets/BoQ_seq02_matches3d.gif" width="170" alt="BoQ Seq02 place recognition matches" /></a>
      <br/><strong>BoQ</strong>
    </td>
    <td align="center">
      <a href="assets/gifs/SelaVPR/seq02_matches3d.gif"><img src="assets/SelaVPR_seq02_matches3d.gif" width="170" alt="SelaVPR Seq02 place recognition matches" /></a>
      <br/><strong>SelaVPR</strong>
    </td>
    <td align="center">
      <a href="assets/gifs/OverlapMamba/seq02_matches3d.gif"><img src="assets/OverlapMamba_seq02_matches3d.gif" width="170" alt="OverlapMamba Seq02 place recognition matches" /></a>
      <br/><strong>OverlapMamba</strong>
    </td>
    <td align="center">
      <a href="assets/gifs/RangePlace/seq02_matches3d.gif"><img src="assets/RangePlace_seq02_matches3d.gif" width="170" alt="RangePlace Seq02 place recognition matches" /></a>
      <br/><strong>RangePlace</strong>
    </td>
  </tr>
</table>

---

## 🗂️ Datasets

| Dataset | Sensors used | Role |
|:--|:--|:--|
| nuScenes | Top LiDAR + six surround-view cameras | Training / validation / testing; cross-region testing |
| AV2S | LiDAR + seven surround-view cameras | Training / validation / testing |
| NCLT | Velodyne LiDAR + six cameras | Cross-date training / validation / testing |
| Self-collected | 16-beam LiDAR + six cameras | Cross-platform evaluation only |

Datasets must be obtained independently from their respective providers. Please follow the original providers' licenses and usage terms.

For nuScenes metadata generation, database/query splits, and LiDAR three-view preprocessing, see [Dataset Preparation (nuScenes)](#dataset-preparation-nuscenes) below.

---

## 📦 Installation & Usage

This repository contains the **nuScenes implementation** of DoFu-Net. The code covers LiDAR spherical three-view preprocessing, multi-camera loading, training, feature extraction, and place-recognition evaluation on BS, SON, and SHV. It does **not** yet include the AV2S, NCLT, or self-collected dataset adapters. No dataset or pretrained DoFu-Net checkpoint is bundled with the code.

### Environment

The reported environment was exported from the authors' Conda installation; the OS release was not recorded in the export.

- **OS:** Linux (distribution/version not available in the environment export)
- **Python:** 3.10.13
- **PyTorch / TorchVision:** 2.1.1+cu118 / 0.16.1+cu118
- **CUDA:** PyTorch CUDA 11.8 build
- **FAISS:** faiss-gpu 1.8.0
- **Key packages:** timm 0.9.2, NumPy 1.26.2, SciPy 1.11.4, scikit-learn 1.3.2, h5py 3.7.0, nuscenes-devkit 1.1.10, OpenCV 4.8.1.78, safetensors 0.4.1
- **Training GPU reported in the paper:** NVIDIA RTX 4060

The following commands are a portable *installation starting point* based on the exported versions; installation on a new machine has not yet been verified end to end. A compatible NVIDIA GPU driver is required for CUDA acceleration.

```bash
conda create -n dofunet python=3.10.13 -y
conda activate dofunet

# PyTorch CUDA 11.8 wheels (the author's original PyTorch build).
python -m pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
  --index-url https://download.pytorch.org/whl/cu118

# FAISS GPU availability can depend on your Linux architecture and channels.
conda install -c pytorch -c conda-forge faiss-gpu=1.8.0

# Remaining Python packages from the exported environment.
python -m pip install -r requirements.txt
```

### Repository Setup

Clone this repository after publishing it to GitHub, then run all following commands from its root directory:

```bash
# Replace <YOUR_GITHUB_REPOSITORY_URL> with the published DoFu-Net repository URL.
git clone <YOUR_GITHUB_REPOSITORY_URL> DoFu-Net-nuScenes
cd DoFu-Net-nuScenes

# Verify the installed packages and project imports.
python -m scripts.verify_imports
```

The code is organized as follows:

```text
DoFu-Net-nuScenes/
├── config/config.yaml              # Paths and training hyperparameters
├── dataset/NuScenesDataset.py      # LiDAR and six-camera data loaders
├── modules/DoFuNet.py              # DoFu-Net architecture
├── modules/netvlad.py              # NetVLAD descriptor head
├── tools/gen_info.py               # nuScenes metadata extraction
├── tools/gen_index.py              # Database/query index generation
├── tools/gen_range.py              # PY/RY/RP three-view generation
├── tools/runner.py                 # Training and evaluation utilities
├── tools/utils.py                  # Shared configuration/path helpers
├── train.py                        # Training entry point
├── test.py                         # Test/evaluation entry point
└── requirements.txt                # Core Python package versions
```

### Dataset Preparation (nuScenes)

Download nuScenes **v1.0-trainval** and **v1.0-test**, including their sensor samples and metadata, according to the dataset license. Place both versions under the nuScenes root directory expected by the devkit.

Choose your local dataset root using the environment variable `NUSCENES_ROOT`, or edit `data.data_root_dir` in `config/config.yaml`.

```bash
export NUSCENES_ROOT=/path/to/nuscenes

# 1. Generate metadata files (BS, SON, SHV, SQ).
python -m tools.gen_info

# 2. Generate database/query split files.
python -m tools.gen_index

# 3. Generate three-view LiDAR .npz files.
python -m tools.gen_range
```

After preprocessing, the relevant files should be arranged as follows:

```text
/path/to/nuscenes/
├── samples/
│   ├── CAM_FRONT/...              # Original nuScenes images
│   ├── LIDAR_TOP/...              # Original nuScenes point clouds
│   └── RANGE_DOFU_3VIEW/
│       └── <LiDAR sample name>.npz
├── nuscenes_infos-bs.pkl
├── nuscenes_infos-son.pkl
├── nuscenes_infos-shv.pkl
├── nuscenes_infos-sq.pkl
├── bs_db.npy
├── bs_train_query.npy
├── bs_val_query.npy
├── bs_test_query.npy
├── son_db.npy
├── son_query.npy
├── shv_db.npy
├── shv_query.npy
└── ...
```

Each generated LiDAR `.npz` contains `py`, `ry`, `rp`, `pitch_min`, and `pitch_max`. The three view sizes generated by this code are **PY: 32×768**, **RY: 64×768**, and **RP: 64×32**. The loader pads these to a common `3×64×768` tensor without changing the underlying three-view data.

**If you already generated these files using the older experimental code, you do not need to recompute them.** Set `data.lidar_view_dir` in `config/config.yaml` to your *existing* folder relative to the nuScenes root. It may use a different name; no directory rename is necessary. This setting is shared by training, evaluation, and preprocessing.

### Training

The original nuScenes configuration is retained: six surrounding cameras, image resize **192×512**, four triplet negatives, margin **0.5**, initial learning rate **1e-5**, StepLR step **6** / gamma **0.5**, and **120 epochs**. The feature cache is refreshed every **3 epochs** and also initialized before the first training epoch when missing.

The visual branch is initialized using a local ConvNeXtV2-Atto `.safetensors` checkpoint. Place that checkpoint in the path specified by `runner.convnextv2_ckpt_path` in `config/config.yaml`, or point the field to its actual absolute path. The example path is a placeholder; the pretrained file is **not** included.

```bash
export NUSCENES_ROOT=/path/to/nuscenes
# Edit runner.convnextv2_ckpt_path in config/config.yaml first.
python train.py --config config/config.yaml
```

Training checkpoints are saved under `runs/checkpoints/` as `DoFuNet_epoch_<N>.pth.tar`, e.g. `DoFuNet_epoch_120.pth.tar`; validation and cache files are stored in their respective `runs/` subdirectories. To resume, set `runner.resume_path` in the YAML to your checkpoint. If you switch to a different experiment/checkpoint, remove any stale cached features before training.

**Checkpoint compatibility:** the original neural layers and their `state_dict` parameter names are preserved; only the *outer class name* and file names have changed. Existing checkpoints containing a `net` state dictionary can still be selected by path. The original checkpoint filename itself does not need to be renamed.

### Evaluation

Provide a **trained fused DoFu-Net** checkpoint explicitly. Do not substitute a visual-only ablation checkpoint or the ConvNeXtV2 backbone initialization weights. By default, the script evaluates BS, SON, and SHV; SQ is optional.

```bash
export NUSCENES_ROOT=/path/to/nuscenes
python test.py \
  --config config/config.yaml \
  --checkpoint runs/checkpoints/DoFuNet_epoch_120.pth.tar \
  --splits BS SON SHV

# Optional: test just one region, or include SQ if its query files exist.
python test.py --checkpoint /path/to/your/trained_dofunet.pth.tar --splits SHV
```

The evaluator extracts **256-D** global descriptors, retrieves candidates using FAISS L2 search, and reports Recall@1/5/10/20, precision-recall curves, ROC curves, and maximum F1. Outputs are written under `runs/results/<SPLIT>/` with a combined `runs/results/nuscenes_summary.csv`. The matching radius used by the supplied code is **9 m**.

### Pretrained Weights

| Dataset | Checkpoint | Notes |
|:--|:--|:--|
| nuScenes | To be released | The fused DoFu-Net trained checkpoint is not included in this source package |
| AV2S | To be added | Dataset-specific implementation and checkpoint not included here |
| NCLT | To be added | Dataset-specific implementation and checkpoint not included here |

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
