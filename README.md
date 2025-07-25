# XYScanNet: A State Space Model for Single Image Deblurring

[![NTIRE](https://img.shields.io/badge/CVPR--NTIRE%202025-Accepted-brightgreen.svg)](https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/papers/Liu_XYScanNet_A_State_Space_Model_for_Single_Image_Deblurring_CVPRW_2025_paper.pdf)
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
## Environment Setup
```
conda create -n XYScanNet python=3.8 -y
conda activate XYScanNet
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

---

## Training on GoPro
Download "[GoPro](https://drive.google.com/drive/folders/1BdV2l7A5MRXLWszGonMxR88eV27geb_n?usp=sharing)" dataset into './datasets'
for example: './datasets/GoPro'. Note: we say the blur images is A and the sharp images is B, e.g., ./GOPRO/test/sharp <-> .GOPRO/test/testB. </br>

Download "[VGG19 Pretrained Weights](https://drive.google.com/file/d/1r2_clZ02-ai6xM7EOHW9APqY9IxkPYsS/view?usp=drive_link)" into './models',
which is used to calculate ContrastLoss.  </br>

**We train our XYScanNetPlus in two stages:** </br>
* We pre-train XYScanNetPlus for 4000 epochs with patch size 252x252 </br> 
* Run the following command 
```
python train_XYScanNet_stage1.py --job_name xyscannetp_gopro
```

* After 4000 epochs, we keep training XYScanNetPlus for 4000 epochs with patch size 320x320 </br>
* Run the following command 
```
python train_XYScanNet_stage2.py --job_name xyscannetp_gopro
```

📄 **Training Log:** [GoPro](https://drive.google.com/drive/folders/1hxyuTzYumivNFhsTBemWLM7shFQ-tdlK?usp=drive_link) (can be viewed with linux ```cat``` commands)


## Training on RealBlur
I forgot the training details on RealBlur datasets. As much as I can remember, I used the gopro-trained model (final .pth in stage 1) and fine-tuned it based on the provided config file.

📄 **Training Log:** [RealBlur_J](https://drive.google.com/drive/folders/1FdxXs95c-kTjNp9jlm-fQ6ayPpwH5kZv?usp=drive_link)
📄 **Training Log:** [RealBlur_R](https://drive.google.com/drive/folders/1ci-D0slKuv3cVX_HOK0_aUj7Gck4fSsf?usp=drive_link)

---

## Testing
**For testing on GoPro dataset** </br>
* Create folders with a structure results/xyscannetp_gopro/models, where models is the folder that saves the network weights.
* Download "[GoPro](https://drive.google.com/file/d/1Fp0MuEwFlzT_NKAFjr3SpuQl3Sm0cFYA/view?usp=sharing)" full dataset or test set into './datasets' (For example: './datasets/GoPro/test') </br>
* Change the dataset path in predict_GoPro_test_results.py line 34.
* Run the following command
```
python predict_GoPro_test_results.py --job_name xyscannetp_gopro
```
**For testing on HIDE dataset** </br>
* Create folders with a structure results/xyscannetp_hide/models, where models is the folder that saves the network weights.
* Download "[HIDE](https://drive.google.com/drive/folders/1BdV2l7A5MRXLWszGonMxR88eV27geb_n?usp=sharing)" into './datasets' </br>
* Run the following command
```
python predict_HIDE_results.py --job_name xyscannetp_gopro
```
**For testing on RealBlur test sets** </br>
* Create folders with a structure results/xyscannetp_realj(or realr)/models, where models is the folder that saves the network weights.
* Download "[RealBlur_J](https://drive.google.com/drive/folders/1BdV2l7A5MRXLWszGonMxR88eV27geb_n?usp=sharing)" and "[RealBlur_R](https://drive.google.com/drive/folders/1BdV2l7A5MRXLWszGonMxR88eV27geb_n?usp=sharing)" into './datasets' </br>
* Run the following command
```
python predict_RealBlur_J_test_results.py --job_name xyscannetp_realj
```
```
python predict_RealBlur_R_test_results.py --job_name xyscannetp_realr
```

---

## 📦 Pretrained Models and Visual Results

The following table lists our released pretrained models and qualitative visual examples. All models are available from Google Drive.

**PS: We have trained multiple versions of XYScanNetPlus. We are still double-checking whether the released models and images match with the paper results.**

| Dataset         | Trained On       | Model Weights                                                                 | Sample Results Preview |
|-----------------|------------------|-------------------------------------------------------------------------------|-------------------------|
| **GoPro**       | GoPro            | [Download](https://drive.google.com/drive/folders/10nu5WiA05Dv4q12A0XiISGMmXd6gyiYE?usp=drive_link) | [GoPro Sample](https://drive.google.com/file/d/1lQlRzjhcG_8L4Ikr2COS_qjVObhaKOZV/view?usp=drive_link) |
| **HIDE**        | GoPro            | [Download](https://drive.google.com/drive/folders/10nu5WiA05Dv4q12A0XiISGMmXd6gyiYE?usp=drive_link) | [HIDE Sample](https://drive.google.com/file/d/1_3MOt1gGR9aQ8bp3yceyDwhEoOD1LmH6/view?usp=drive_link)  |
| **RWBI**        | GoPro            | [Download](https://drive.google.com/drive/folders/10nu5WiA05Dv4q12A0XiISGMmXd6gyiYE?usp=drive_link) | [RWBI Sample](https://drive.google.com/file/d/1P_GXhmIdGvs6dB3h_7SVuDXpA7tCDE3G/view?usp=drive_link)  |
| **RealBlur_J**  | RealBlur_J       | [Download](https://drive.google.com/drive/folders/1LEBIHQpqZAzudPwkHVQiVkrTwI4jtJOq?usp=drive_link) | [RealBlurJ Sample](https://drive.google.com/file/d/1B3J7IEj7wPTc0dhtzDkinPmxJ5mEl1gt/view?usp=drive_link) |
| **RealBlur_R**  | RealBlur_R       | [Download](https://drive.google.com/drive/folders/10TXboH85HMfp_9TjBLm3Xw-jN1Q-Z9Bi?usp=drive_link) | [RealBlurR Sample](https://drive.google.com/file/d/12K0o9g7TX0m2urrPBHB31603i7t-ZfbH/view?usp=drive_link) |

---

## ✅ TODO

- [x] **Release Pretrained Models**
- [x] **Release Test Images**
- [x] **Release Training Logs**
- [x] **Release Environment Setup**

---

## 📖 Citation

If you find our work useful, please consider citing us:

```bibtex
@InProceedings{Liu_2025_CVPR,
    author    = {Liu, Hanzhou and Liu, Chengkai and Xu, Jiacong and Jiang, Peng and Lu, Mi},
    title     = {XYScanNet: A State Space Model for Single Image Deblurring},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR) Workshops},
    month     = {June},
    year      = {2025},
    pages     = {779-789}
}
