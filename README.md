# XYScanNet: A State Space Model for Single Image Deblurring

<p align="center">
  <img src="https://img.shields.io/badge/CVPR--NTIRE%202025-Accepted-brightgreen.svg">
  <img src="https://img.shields.io/badge/arXiv-2412.10338-blue">
</p>

[![NTIRE](https://img.shields.io/badge/CVPR--NTIRE%202025-Accepted-brightgreen.svg)](https://arxiv.org/abs/2412.10338)
[![arXiv](https://img.shields.io/badge/arXiv-2412.10338-b31b1b.svg)](https://arxiv.org/abs/2412.10338)

Official implementation of our **CVPR Workshop NTIRE 2025** accepted paper:

> **XYScanNet: A State Space Model for Single Image Deblurring**  
> *Hanzhou Liu, Chengkai Liu, Jiacong Xu, Peng Jiang, Mi Lu*  
> [arXiv 2412.10338](https://arxiv.org/abs/2412.10338)

---

## 📝 Abstract

Deep state-space models (SSMs), like recent Mamba archi-
tectures, are emerging as a promising alternative to CNN
and Transformer networks. Existing Mamba-based restora-
tion methods process visual data by leveraging a flatten-
and-scan strategy that converts image patches into a 1D se-
quence before scanning. However, this scanning paradigm
ignores local pixel dependencies and introduces spatial
misalignment by positioning distant pixels incorrectly ad-
jacent, which reduces local noise-awareness and degrades
image sharpness in low-level vision tasks. To overcome
these issues, we propose a novel slice-and-scan strategy
that alternates scanning along intra- and inter-slices. We
further design a new Vision State Space Module (VSSM)
for image deblurring, and tackle the inefficiency challenges
of the current Mamba-based vision module. Building upon
this, we develop XYScanNet, an SSM architecture integrated
with a lightweight feature fusion module for enhanced im-
age deblurring. XYScanNet, maintains competitive dis-
tortion metrics and significantly improves perceptual per-
formance. Experimental results show that XYScanNet en-
hances KID by 17% compared to the nearest competitor.

---

## 🚀 Highlights

- ✅ **Accepted at CVPR 2025 - NTIRE Workshop**
- 📈 Competitive results on GoPro, HIDE, and RealBlur datasets.

---

## 📁 Code & Dataset

Coming soon! Stay tuned.

---

## ✅ TODO

- [ ] 🔓 **Release Pretrained Models**
- [ ] 🔄 **Extend to Other Restoration Tasks**
  - [ ] Image Denoising
  - [ ] Super-Resolution
  - [ ] Compression Artifact Removal
- [ ] 📊 **Complexity Analysis**
  - [ ] FLOPs and Runtime Benchmarks
  - [ ] Model Scaling Trend

---

## 📖 Citation

If you find our work useful, please consider citing us:

```bibtex
@article{liu2024xyscannet,
  title={XYScanNet: An Interpretable State Space Model for Perceptual Image Deblurring},
  author={Liu, Hanzhou and Liu, Chengkai and Xu, Jiacong and Jiang, Peng and Lu, Mi},
  journal={arXiv preprint arXiv:2412.10338},
  year={2024}
}
