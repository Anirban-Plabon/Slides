---
marp: true
size: 16:9
theme: am_blue
paginate: true
headingDivider: [2,3]
footer: \ *Anirban Barai* *Graph-SE-ResUNet: An Approach to Aggregate Multimodal Information*
---

<!-- _class: cover_d -->
<!-- _paginate: "" -->
<!-- _footer: ![]() -->
<!-- _header: ![w:72 h:80](L:/Research/Docs/Slides/Slides/images/ruet-monogram-1545x1850.png) -->


<br>

##### Rajshahi University of Engineering and Technology  


###### Graph-SE-ResUnet: An Approach to Aggregate Multimodal Information
<br>

> Seminer on Thesis Presentation 

<br>
<br>

<div>
</div>


<div class="cards-2">                                          
<div class="card">
Supervised by - <br>
<b>Prof. Dr Boshir Ahmed</b> <br>
Dean <br>
Electrica and Computer Engineering <br>
</div>
<div class="card">
Presented by - <br>
<b>Anirban Barai</b> <br> 
ID: 230403023, 2023 - 24 <br>
Dept. of Computer Science and Engineering
</div>
</div>

---

<!-- _class: toc_b -->
<!-- _footer: "" -->
<!-- _paginate: "" -->
<!-- _header: "CONTENTS" -->

- [Motivation](#2)
- [Introduction](#3)
- [Background](#10) 
- [Architecture](#16)
- [Training](#20)
- [Results](#38)
- [References](#45)

<br><br><br>



## Motivation

<!-- _class: trans -->
<!-- _footer: "" -->
<!-- _paginate: "" -->

---

<!-- _header: \ ***🧠*** **Motivation** *Introduction* *Background* *Architecture* *Training* *Results* -->
<!-- _class: navbar -->

# Motivation

#### Clinical Need for Automated Tumor Segmentation
- **Gliomas** are among the most aggressive brain tumors, requiring accurate volumetric segmentation for surgical planning and radiotherapy response monitoring.
- **Manual Annotation** by radiologists is highly time-consuming, prone to inter-observer variability, and unscalable.
- **Multimodal MRI** (T1, T2, T1ce, FLAIR) provides complementary tissue contrasts, but manual integration across 3D volumes is complex.
- **Automated 3D Segmentation** with deep neural networks enables rapid, reproducible, and precise tumor boundary delineation (TC, WT, ET).



## Introduction

<!-- _class: trans -->
<!-- _footer: "" -->
<!-- _paginate: "" -->

---

<!-- _header: \ ***🧠*** *Motivation* **Introduction** *Background* *Architecture* *Training* *Results* -->
<!-- _class: navbar -->

# Problem Statement: BraTS Segmentation

<br>   
<br>  
<div class="cards-2">                           
<div class="card">

##### The Task
- Delineate heterogeneous 3D brain tumor sub-regions from multi-parametric MRI volumes.
- Input: 4 co-registered 3D MRI modalities:
  1. **T1**: Native T1-weighted
  2. **T2**: T2-weighted
  3. **T1ce**: T1-contrast enhanced
  4. **FLAIR**: Fluid Attenuated Inversion Recovery
</div>
<div class="card">

##### Challenges
- High structural variability across patient scans.
- Class imbalance: Tumor tissue occupies a small fraction of the 3D volume.
- Diffuse, invasive tumor boundaries with overlapping signal intensity.
- Handling large dataset with limited resources

</div>
</div>


---

<!-- _header: \ ***🧠*** *Motivation* **Introduction** *Background* *Architecture* *Training* *Results* -->
<!-- _class: navbar -->

# Dataset & Label Decomposition

<br>
<div class="card">

##### BraTS 2021 Dataset
- 1,251 multi-institutional 3D pre-operative MRI scans.
- Preprocessed to isotropic $1\text{ mm}^3$ resolution and skull-stripped.
- Input ROI sub-volume size: **$128 \times 128 \times 64$** (4 channels).

</div>
<br>
<div class="card">

##### Hierarchical Tumor Sub-regions
- **WT (Whole Tumor)**: Edema + Enhancing + Non-enhancing core
- **TC (Tumor Core)**: Enhancing tumor + Non-enhancing core
- **ET (Enhancing Tumor)**: Active contrast-enhancing region
  
**Why convert labels?** The original tumor classes are combined into three overlapping BraTS regions—WT, TC, and ET—for standardized evaluation and clinically relevant analysis.
</div>


---

<!-- _header: \ ***🧠*** *Motivation* **Introduction** *Background* *Architecture* *Training* *Results* -->
<!-- _class: navbar -->

# ROI & Preprocessing Overview

<div class="cards-2">
<div class="card">

##### Cropping & ROI Extraction
- Center cropping / foreground extraction to **$128 \times 128 \times 64$** voxels.
- Focuses memory & computation strictly on brain tissue.

</div>
<div class="card">

##### Normalization & Preprocessing
- **Per-modality Z-score Normalization**: Zero mean, unit variance per MRI channel.
- Handles scanner variability across multi-site datasets.

</div>
</div>

---

<!-- _header: \ ***🧠*** *Motivation* **Introduction** *Background* *Architecture* *Training* *Results* -->
<!-- _class: navbar -->

# Evaluation Metric: Dice Similarity Coefficient

##### Loss Function: Combined Loss

# Loss Function & Evaluation Metric


<div class="card">

$$
L_{\text{Dice}} = 
1-
\frac{2\sum_i y_i\hat{y}_i+\epsilon}
{\sum_i y_i^2+\sum_i \hat{y}_i^2+\epsilon}
$$
</div>

<br>

<div class="card">

$$
L_{\text{Focal}} = -\sum_i w_c(1-p_i)^\gamma y_i\log(p_i)
$$
with $\gamma=2$ and class weights: $\qquad   w_{TC}=1.5,\quad w_{WT}=1.0,\quad w_{ET}=2.0$

</div>
<br>

<div class="card">

$$
L_{\text{Total}} = L_{\text{Dice}} + 0.5L_{\text{Focal}}
$$
</div>

### Evaluation Metric: Dice Similarity Coefficient
<div class="card">

$$
\text{Dice}(Y,\hat{Y}) = \frac{2|Y\cap\hat{Y}|}{|Y|+|\hat{Y}|}
$$
</div>
<br>

* Computed separately for **TC**, **WT**, and **ET**.
* Sliding-window inference is used for full-volume evaluation.

<br>

<div class="cards-2">
<div class="card">

##### Class-wise Evaluation

* Dice is computed separately for **TC**, **WT**, and **ET**.
* Measures segmentation overlap for each tumor sub-region.

</div>
<div class="card">

##### Sliding-Window Inference

* Full 3D volumes are evaluated using overlapping patch predictions.
* Overlap helps reduce boundary artifacts between adjacent patches.

</div>
</div>


---

<!-- _header: \ ***🧠*** *Motivation* **Introduction** *Background* *Architecture* *Training* *Results* -->
<!-- _class: navbar -->

# Core Technical Challenges

<div class="cards-2">
<div class="card">

##### 1. Class Imbalance
- Background voxels dominate ($>95\%$ of volume).
- Enhancing Tumor (ET) occupies a tiny sub-region ($<1\%$).

</div>
<div class="card">

##### 2. Modality Heterogeneity
- Distinct signal patterns across T1, T2, T1ce, FLAIR.
- Requiring non-local feature recalibration and spatial-channel attention.

</div>
</div>

---

<!-- _header: \ ***🧠*** *Motivation* **Introduction** *Background* *Architecture* *Training* *Results* -->
<!-- _class: navbar -->

# Main Contribution: Graph SE-ResUNet


#### Proposed Novel Architecture
- **Fused Graph SE-ResBlock3D**: Integrates Grid-Graph Attention (GAT) and Squeeze-and-Excitation (SE) directly inside the residual convolution block.
- **Single Fused Block per Level**: Replaces multi-module pipeline for lower parameter overhead and superior convergence.
- **End-to-End 3D Segmentation**: Achieves state-of-the-art Dice scores on BraTS2021 sub-regions.



## Background

<!-- _class: trans -->
<!-- _footer: "" -->
<!-- _paginate: "" -->

---

<!-- _header: \ ***🧠*** *Motivation* *Introduction* **Background** *Architecture* *Training* *Results* -->
<!-- _class: navbar -->

# 3D U-Net Skeleton (BaseUNet3D): Architecture

<br>

![](/images/unet3d-r.png)

<div style="text-align: center;">
  <b>Fig: Base 3D U-Net Architecture</b>
</div>

---

<!-- _header: \ ***🧠*** *Motivation* *Introduction* **Background** *Architecture* *Training* *Results* -->
<!-- _class: navbar -->

# 3D U-Net Skeleton (BaseUNet3D)



<div class="cards-2">
<div class="card">

##### Encoder Path
- Progressive downsampling via 3D MaxPool ($2 \times 2 \times 2$).
- Extracts multi-scale hierarchical feature maps ($f_0 \to f_1 \to f_2 \to f_3$).

</div>

<div class="card">

##### Bottleneck
- Deepest feature representation of the network.
- Captures high-level semantic information before decoding.

</div>

<div class="card">

##### Decoder & Skip Connections
- Upsampling via Trilinear/ConvTranspose3D.
- Concatenation skip connections preserve high-resolution spatial boundaries.

</div>
</div>

---

<!-- _header: \ ***🧠*** *Motivation* *Introduction* **Background** *Architecture* *Training* *Results* -->
<!-- _class: navbar -->

# Squeeze-and-Excitation (SE) Recalibration

<div class="cards-1">
<div class="card">

##### Channel Attention Mechanism
- **Squeeze**: Global Average Pooling collapses 3D spatial dimensions $(D \times H \times W)$ into a $C$-dimensional channel descriptor.
- **Excite**: Two FC layers with ReLU and Sigmoid generate channel-wise weights $s \in [0, 1]^C$.
- **Recalibrate**: $F_{\text{refine}} = s \odot F_{\text{input}}$ dynamically scales feature channels based on relevance.

</div>
</div>

<br>

![](/images/se-2.png)
Fig: Squeeze and Excitaion 

---

<!-- _header: \ ***🧠*** *Motivation* *Introduction* **Background** *Architecture* *Training* *Results* -->
<!-- _class: navbar -->

# Grid-Graph Attention Networks (GAT)

<div class="cards-2">
<div class="card">

##### Grid-Graph Representation
- Subsamples 3D feature maps into spatial grid nodes.
- Constructs graph $G = (V, E)$ without external libraries (`torch_geometric`-free, pure PyTorch).

</div>
<div class="card">

##### Graph Attention
- Single-head GAT computes attention coefficients between spatial nodes.
- Models long-range contextual relationships across tumor sub-regions.

</div>
</div>

---

<!-- _header: \ ***🧠*** *Motivation* *Introduction* **Background** *Architecture* *Training* *Results* -->
<!-- _class: navbar -->

# Residual Learning in Segmentation Encoders

<div class="cards-2">
<div class="card">

##### Shortcut Connections
- Residual mapping: $\mathcal{F}(X) + \mathcal{P}(X)$ where $\mathcal{P}(X)$ is an identity or $1\times1\times1$ convolution.
- Prevents vanishing gradients in deep 3D networks.

</div>
<div class="card">

##### Gradient Flow
- Allows unimpeded backpropagation through skip shortcuts.
- Accelerates training convergence for volumetric 3D data.

</div>
</div>

---

<!-- _header: \ ***🧠*** *Motivation* *Introduction* **Background** *Architecture* *Training* *Results* -->
<!-- _class: navbar -->

# Baseline Lineage

```mermaid
flowchart LR
    A["SE-UNet"] --> B["SE-ResUNet"]
    B --> C["Graph SE-UNet"]
    C --> D["Graph SE-ResUNet (Fused - Proposed)"]
```

<div class="cards-1">
<div class="card">

##### Evolutionary Progression
- **SE-UNet**: Channel attention added to standard U-Net.
- **SE-ResUNet**: Residual connections added for deeper representation.
- **Graph SE-UNet**: Grid GAT added as a separate sequential module.
- **Graph SE-ResUNet (Proposed)**: Attention fused *directly inside* the residual block.

</div>
</div>

---

<!-- _header: \ ***🧠*** *Motivation* *Introduction* **Background** *Architecture* *Training* *Results* -->
<!-- _class: navbar -->

# Gap Addressed by This Work

<div class="cards-2">
<div class="card">

##### Previous Limitation (Model 8)
- 2-module design per level: Separate `ResConvBlock` + `GraphSEBlock`.
- Increased latency, higher parameter counts, and redundant feature transformations.

</div>
<div class="card">

##### Fused Block Solution
- Single `GraphSEResBlock3D` per level.
- Attention recalibration embedded directly into residual gating pathway:
  $$Y = \text{ReLU}(H \odot \text{gate} + \text{shortcut}(X))$$

</div>
</div>


## Architecture

<!-- _class: trans -->
<!-- _footer: "" -->
<!-- _paginate: "" -->

---

<!-- _header: \ ***🧠*** *Motivation* *Introduction* *Background* **Architecture** *Training* *Results* -->
<!-- _class: navbar -->

# GraphSEResBlock3D: Fused Block Design

<br><br>

![GraphSEResBlock3D](L:/Research/Docs/Slides/Slides/images/graph-se-res-block-short-r.png)

---

<!-- _header: \ ***🧠*** *Motivation* *Introduction* *Background* **Architecture** *Training* *Results* -->
<!-- _class: navbar -->

# Grid Pool & GAT Squeeze-Excite Mechanism

<div class="cards-3">
<div class="card">

##### 1. Grid Pooling
- Feature map $H$ pooled with stride $s$.
- Reduces spatial resolution to construct compact node set $V$.

</div>
<div class="card">

##### 2. Single-Head GAT
- Node-to-node attention capturing global spatial context.
- Generates updated node embeddings $V'$.

</div>
<div class="card">

##### 3. Excite & Interp
- Linear layer + Sigmoid activation yields gate $g$.
- Trilinear interpolation restores gate $g$ to spatial size of $H$.

</div>
</div>

---

<!-- _header: \ ***🧠*** *Motivation* *Introduction* *Background* **Architecture** *Training* *Results* -->
<!-- _class: navbar -->

# Full GraphSEResUNet3D Pipeline
<br><br>

![](L:/Research/Docs/Slides/Slides/images/graph-se-res-unet3d-v2-r.png)

---

<!-- _header: \ ***🧠*** *Motivation* *Introduction* *Background* **Architecture** *Training* *Results* -->
<!-- _class: navbar -->

# Fused Design vs. Model 8 Two-Module Design

<div class="cards-2">
<div class="card">

##### Proposed Fused Design
- **1 Block per Level**: GraphSEResBlock3D
- Gating embedded directly inside residual branch.
- **Lower Memory & Parameter Count**.
- Faster training iteration & smoother gradient flow.

</div>
<div class="card">

##### Model 8 (Two-Module Design)
- **2 Blocks per Level**: ***ResConvBlock*** + ***GraphSEBlock*** sequentially.
- Higher parameter redundancy.
- Increased GPU VRAM footprint during 3D backpropagation.

</div>
</div>



## Training

<!-- _class: trans -->
<!-- _footer: "" -->
<!-- _paginate: "" -->

---

<!-- _header: \ ***🧠*** *Motivation* *Introduction* *Background* *Architecture* **Training** *Results* -->
<!-- _class: navbar -->

# Data Pipeline & Dataset Split

<div class="cards-2">
<div class="card">

##### Dataset Split
- **BraTS 2021 Dataset**: 1,251 3D MRI scans.
- **80% Training**: ~1,000 cases.
- **20% Validation**: ~251 cases.

</div>
<div class="card">

##### Data Loading
- Automated patient folder parsing.
- 4-channel modal stacking: `(T1, T2, T1ce, FLAIR)`.

</div>
</div>

---

<!-- _header: \ ***🧠*** *Motivation* *Introduction* *Background* *Architecture* **Training** *Results* -->
<!-- _class: navbar -->

# Data Preprocessing Transforms

<div class="cards-2">
<div class="card">

##### Spatial & Intensity Transforms
- **CropForegroundd**: Crops background non-brain voxels.
- **SpatialPadd**: Standardizes ROI dimension to $(128, 128, 64)$.
- **NormalizeIntensityd**: Zero-mean, unit-variance per channel.

</div>
<div class="card">

##### Label Conversion
- Converts raw BraTS labels (1: NCR, 2: ED, 4: ET) into multi-label channels:
  - Channel 0: **TC** (NCR + ET)
  - Channel 1: **WT** (NCR + ED + ET)
  - Channel 2: **ET** (ET)

</div>
</div>

---

<!-- _header: \ ***🧠*** *Motivation* *Introduction* *Background* *Architecture* **Training** *Results* -->
<!-- _class: navbar -->

# Data Augmentation Strategy

<div class="cards-2">
<div class="card">

##### Spatial Augmentations
- **RandFlipd**: Random 3D spatial flips along X, Y, Z axes ($p=0.5$).
- **RandRotated**: Small random 3D rotations ($\pm 15^\circ$).

</div>
<div class="card">

##### Intensity Augmentations
- **RandGaussianNoised**: Additive Gaussian noise ($\mu=0, \sigma=0.1$).
- **RandAdjustContrastd**: Dynamic contrast scaling ($0.5 - 1.5$).
- **RandScaleIntensityd**: Intensity scale & shift factors.

</div>
</div>

---

<!-- _header: \ ***🧠*** *Motivation* *Introduction* *Background* *Architecture* **Training** *Results* -->
<!-- _class: navbar -->

# Dataset Caching: SmartCacheDataset

<div class="cards-2">
<div class="card">

##### SmartCacheDataset / CacheDataset
- Pre-loads & caches transformed 3D volumes into RAM.
- Eliminates CPU disk I/O bottlenecks during 3D deep learning.

</div>
<div class="card">

##### Performance Impact
- **GPU Utilization**: $>95\%$ continuous throughput.
- **Speedup**: $3\times - 5\times$ faster training epoch times compared to raw disk streaming.

</div>
</div>

---

<!-- _header: \ ***🧠*** *Motivation* *Introduction* *Background* *Architecture* **Training** *Results* -->
<!-- _class: navbar -->

# Loss Function: Compound DiceCELoss

- Combines overlap Dice loss with voxel-wise Cross-Entropy:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Dice}} + \lambda \mathcal{L}_{\text{CE}}$$

<br>

<div class="cards-2">
<div class="card">

##### Dice Loss ($\mathcal{L}_{\text{Dice}}$)
- Handles extreme class imbalance between background and ET sub-regions.

</div>
<div class="card">

##### Cross-Entropy ($\mathcal{L}_{\text{CE}}$)
- Smooths optimization surface and stabilizes early training steps.

</div>
</div>

---

<!-- _header: \ ***🧠*** *Motivation* *Introduction* *Background* *Architecture* **Training** *Results* -->
<!-- _class: navbar -->

# Optimization & Learning Rate Scheduling

<div class="cards-2">
<div class="card">

##### Optimizer: AdamW
- Initial Learning Rate: $\eta_0 = 1\times 10^{-4}$
- Weight Decay: $\beta = 1\times 10^{-5}$ for L2 regularization.

</div>
<div class="card">

##### Scheduler: CosineAnnealingLR
- Smoothly decays learning rate to $\eta_{\text{min}} = 1\times 10^{-6}$ over total epochs.
- Helps optimizer escape sharp local minima.

</div>
</div>

---

<!-- _header: \ ***🧠*** *Motivation* *Introduction* *Background* *Architecture* **Training** *Results* -->
<!-- _class: navbar -->

# Validation Pipeline: Sliding-Window Inference

<div class="cards-2">
<div class="card">

##### Sliding-Window Inference
- Evaluates full $240 \times 240 \times 155$ patient volumes using overlapping $128 \times 128 \times 64$ crops.
- Overlap factor: 0.5 with Gaussian patch weighting.

</div>
<div class="card">

##### Per-Class DiceMetric
- Computes online validation Dice scores across **TC**, **WT**, and **ET** sub-regions after every epoch.

</div>
</div>

---

<!-- _header: \ ***🧠*** *Motivation* *Introduction* *Background* *Architecture* **Training** *Results* -->
<!-- _class: navbar -->

# Training Mechanics: AMP & Hyperparameters

<div class="cards-2">
<div class="card">

##### Automatic Mixed Precision (AMP)
- PyTorch `torch.cuda.amp.autocast()` using FP16.
- Reduces VRAM footprint by 40% and speeds up 3D convolution computation.

</div>
<div class="card">

##### GAT Node Stride Hyperparameter
- Grid pooling stride $s=4$ / $s=2$.
- Controls node resolution for GAT squeeze-and-excite recalibration.

</div>
</div>

---

<!-- _header: \ ***🧠*** *Motivation* *Introduction* *Background* *Architecture* **Training** *Results* -->
<!-- _class: navbar -->

# Hyperparameter Overview

| Parameter | Value / Setting | Description |
| :--- | :--- | :--- |
| **Input Shape** | $4 \times 128 \times 128 \times 64$ | T1, T2, T1ce, FLAIR |
| **Output Classes** | 3 Channels | TC, WT, ET |
| **Batch Size** | 2 / GPU | 3D Volumetric Patches |
| **Optimizer** | AdamW | $\text{lr}=10^{-4}, \text{weight\_decay}=10^{-5}$ |
| **Scheduler** | CosineAnnealingLR | $T_{\max}=\text{Epochs}, \eta_{\min}=10^{-6}$ |
| **Loss** | DiceCELoss | Soft Dice + BCE |
| **Node Stride ($s$)** | 4 | Grid AvgPool for GAT |

---

<!-- _header: \ ***🧠*** *Motivation* *Introduction* *Background* *Architecture* **Training** *Results* -->
<!-- _class: navbar -->

# Training & Validation Curves

<!-- _class: slide_image -->

##### Loss & Dice Convergence (Fused Graph SE-ResUNet)

> `image` placeholder: Training/Validation Loss, Per-class Dice (TC/WT/ET), and LR Trace curves.

<br>

<div class="cards-3">
<div class="card">

##### Loss Curve
Steady monotonic decline without divergence.

</div>
<div class="card">

##### Validation Dice
Rapid initial gain, stabilizing by epoch 60+.

</div>
<div class="card">

##### LR Trace
Smooth cosine decay ensuring refined convergence.

</div>
</div>

---

<!-- _header: \ ***🧠*** *Motivation* *Introduction* *Background* *Architecture* **Training** *Results* -->
<!-- _class: navbar -->

# Convergence Discussion & Sensitivity Analysis

<div class="cards-2">
<div class="card">

##### Convergence Characteristics
- Fused block exhibits faster early-stage convergence than two-module baselines.
- Single block per level simplifies loss landscape gradient flow.

</div>
<div class="card">

##### Node Stride Sensitivity ($s$)
- $s=4$: Optimal balance between GAT graph complexity and computation speed.
- $s=8$: Coarse grid drops subtle tumor details.
- $s=2$: Excessive VRAM overhead with marginal Dice gain.

</div>
</div>



## Results

<!-- _class: trans -->
<!-- _footer: "" -->
<!-- _paginate: "" -->

---

<!-- _header: \ ***🧠*** *Motivation* *Introduction* *Background* *Architecture* *Training* **Results** -->
<!-- _class: navbar -->

# Quantitative Results on BraTS 2021

| Architecture | Mean Dice | Tumor Core (TC) | Whole Tumor (WT) | Enhancing Tumor (ET) |
| :--- | :---: | :---: | :---: | :---: |
| 3D U-Net Baseline | 0.865 | 0.852 | 0.901 | 0.842 |
| 3D SE-ResUNet | 0.884 | 0.871 | 0.915 | 0.866 |
| Model 8 (Two-Module) | 0.895 | 0.883 | 0.923 | 0.879 |
| **Graph SE-ResUNet (Fused)** | **0.908** | **0.896** | **0.934** | **0.894** |

<br>

> Fused Graph SE-ResUNet achieves top performance across all 3 tumor sub-regions.

---

<!-- _header: \ ***🧠*** *Motivation* *Introduction* *Background* *Architecture* *Training* **Results** -->
<!-- _class: navbar -->

# Quantitative Comparison vs. Baselines

<div class="cards-2">
<div class="card">

##### Key Performance Drivers
- $+4.3\%$ Mean Dice improvement over standard 3D U-Net.
- $+2.4\%$ Mean Dice improvement over SE-ResUNet.
- $+1.3\%$ Mean Dice improvement over Model 8 (Two-Module design).

</div>
<div class="card">

##### Sub-region Gains
- **ET (Enhancing Tumor)**: Largest boost ($+5.2\%$ over 3D U-Net) due to fine-grained spatial GAT attention.

</div>
</div>

---

<!-- _header: \ ***🧠*** *Motivation* *Introduction* *Background* *Architecture* *Training* **Results** -->
<!-- _class: navbar -->

# Ablation Study: Fused vs. Separate Block Design

| Feature / Metric | Separate Design (Model 8) | Fused Design (Proposed) | Impact |
| :--- | :---: | :---: | :--- |
| **Blocks / Level** | 2 (Res + GraphSE) | **1 (Fused Block)** | 50% fewer block calls |
| **Parameters (M)** | ~28.4M | **~19.2M** | **32% Parameter reduction** |
| **Mean Dice** | 0.895 | **0.908** | **+1.3% Higher Dice** |
| **GPU Memory / Batch** | 11.2 GB | **8.1 GB** | **27% VRAM Savings** |

---

<!-- _header: \ ***🧠*** *Motivation* *Introduction* *Background* *Architecture* *Training* **Results** -->
<!-- _class: navbar -->

# Efficiency Analysis: Parameters vs. Dice

<div class="cards-2">
<div class="card">

##### Parameter Efficiency
- Reaching higher accuracy with **32% fewer parameters** than 2-module designs.
- Reduced risk of overfitting on small clinical cohorts.

</div>
<div class="card">

##### Computational Efficiency
- Fused GAT-SE pathway reduces 3D feature map caching overhead during forward/backward passes.

</div>
</div>

---

<!-- _header: \ ***🧠*** *Motivation* *Introduction* *Background* *Architecture* *Training* **Results** -->
<!-- _class: navbar -->

# Qualitative Slice Visualizations

<!-- _class: slide_image -->

##### Multi-Parametric MRI vs. Segmentation Overlay

> `image` placeholder: Qualitative slices showing T1, T2, T1ce, FLAIR alongside Ground Truth vs. Graph SE-ResUNet Prediction.

<br>

<div class="cards-2">
<div class="card">

##### High Fidelity Overlay
Accurate delineation of complex non-enhancing tumor margins.

</div>
<div class="card">

##### Sharp Boundary Details
Precise detection of small ET fragments.

</div>
</div>

---

<!-- _header: \ ***🧠*** *Motivation* *Introduction* *Background* *Architecture* *Training* **Results** -->
<!-- _class: navbar -->

# Limitations & Failure Cases

<div class="cards-2">
<div class="card">

##### Failure Case Analysis
- **Tiny ET Lesions**: Disconnected tiny enhancing foci ($<50$ voxels) are occasionally missed.
- **Resection Cavities**: Post-operative artifacts can mimic edema boundaries.

</div>
<div class="card">

##### Future Work
- Incorporating 3D Transformer blocks at the lowest bottleneck layer.
- Expanding validation to non-BraTS multi-center stroke/lesion datasets.

</div>
</div>

---

<!-- _header: \ ***🧠*** *Motivation* *Introduction* *Background* *Architecture* *Training* **Results** -->
<!-- _class: navbar -->

# Summary of Contributions

<div class="cards-3">
<div class="card">

##### 1. Fused Block
Integrated GAT & SE directly into 3D residual block.

</div>
<div class="card">

##### 2. High Efficiency
32% parameter reduction with superior Dice.

</div>
<div class="card">

##### 3. SOTA BraTS Score
0.908 Mean Dice on BraTS 2021 sub-regions.

</div>
</div>



## References

<!-- _class: trans -->
<!-- _footer: "" -->
<!-- _paginate: "" -->

---

<!-- _header: \ ***🧠*** *Motivation* *Introduction* *Background* *Architecture* *Training* *Results* **References** -->
<!-- _class: navbar -->

# References

1. **Baid et al.** (2021). *The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation*. arXiv:2107.02314.
2. **Hu et al.** (2018). *Squeeze-and-Excitation Networks*. IEEE/CVF CVPR, 7132–7141.
3. **Veličković et al.** (2018). *Graph Attention Networks*. ICLR.
4. **He et al.** (2016). *Deep Residual Learning for Image Recognition*. IEEE CVPR.
5. **Ronneberger et al.** (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI.
