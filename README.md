# XYScanNet: A State Space Model for Single Image Deblurring

[![NTIRE](https://img.shields.io/badge/CVPR--NTIRE%202025-Accepted-brightgreen.svg)](https://arxiv.org/abs/2412.10338)
[![arXiv](https://img.shields.io/badge/arXiv-2412.10338-b31b1b.svg)](https://arxiv.org/abs/2412.10338)

Official implementation of our **CVPR Workshop NTIRE 2025** accepted paper:

> **XYScanNet: A State Space Model for Single Image Deblurring**  
> *Hanzhou Liu, Chengkai Liu, Jiacong Xu, Peng Jiang, Mi Lu*  
> [arXiv 2412.10338](https://arxiv.org/abs/2412.10338)

---

## 📝 Abstract

Deep state-space models (SSMs), like recent Mamba architectures, are emerging as a promising alternative to CNN and Transformer networks. Existing Mamba-based restoration methods process visual data by leveraging a flatten-and-scan strategy that converts image patches into a 1D sequence before scanning. However, this scanning paradigm ignores local pixel dependencies and introduces spatial misalignment by positioning distant pixels incorrectly adjacent, which reduces local noise-awareness and degrades image sharpness in low-level vision tasks.

To overcome these issues, we propose a novel slice-and-scan strategy that alternates scanning along intra- and inter-slices. We further design a new Vision State Space Module (VSSM) for image deblurring and tackle the inefficiency challenges of the current Mamba-based vision module. Building upon this, we develop XYScanNet, an SSM architecture integrated with a lightweight feature fusion module for enhanced image deblurring. XYScanNet maintains competitive distortion metrics and significantly improves perceptual performance. Experimental results show that XYScanNet enhances KID by 17% compared to the nearest competitor.

---

## 🚀 Highlights

- **Accepted at CVPR 2025 - NTIRE Workshop**
- Novel slice-and-scan scanning for spatially aware state-space modeling
- Lightweight Vision State Space Module (VSSM) designed for efficiency
- Competitive results on GoPro, HIDE, RealBlur_J/R, and RWBI datasets

---

## 📦 Pretrained Models and Visual Results

The following table lists our released pretrained models and qualitative visual examples. All models are available from Google Drive.

| Dataset         | Trained On       | Model Weights                                                                 | Sample Results Preview |
|-----------------|------------------|-------------------------------------------------------------------------------|-------------------------|
| **GoPro**       | GoPro            | [Download](https://drive.google.com/drive/folders/10nu5WiA05Dv4q12A0XiISGMmXd6gyiYE?usp=drive_link) | ![GoPro Sample](images/gopro_sample.png) |
| **HIDE**        | GoPro            | [Download](https://drive.google.com/drive/folders/10nu5WiA05Dv4q12A0XiISGMmXd6gyiYE?usp=drive_link) | ![HIDE Sample](images/hide_sample.png)  |
| **RWBI**        | GoPro            | [Download](https://drive.google.com/drive/folders/10nu5WiA05Dv4q12A0XiISGMmXd6gyiYE?usp=drive_link) | ![RWBI Sample](images/rwbi_sample.png)  |
| **RealBlur_J**  | RealBlur_J       | [Download](https://drive.google.com/drive/folders/1LEBIHQpqZAzudPwkHVQiVkrTwI4jtJOq?usp=drive_link) | ![RealBlurJ Sample](images/realblurj_sample.png) |
| **RealBlur_R**  | RealBlur_R       | [Download](https://drive.google.com/drive/folders/10TXboH85HMfp_9TjBLm3Xw-jN1Q-Z9Bi?usp=drive_link) | ![RealBlurR Sample](images/realblurr_sample.png) |

> 📌 *Please make sure the `images/` directory contains the corresponding sample images for proper rendering on GitHub.*

---

## ✅ TODO

- [x] **Release Pretrained Models**
- [x] **Release Test Images**
- [ ] **Extend to Other Restoration Tasks**
  - [ ] Image Denoising
  - [ ] Super-Resolution
  - [ ] Compression Artifact Removal

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
