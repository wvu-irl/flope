<h1 align="center">🌻 FloPE: Flower Pose Estimation for Precision Pollination</h1>

<p align="center">
  <a href="https://wvu-irl.github.io/flope-irl/" target="_blank">
    <img src="https://img.shields.io/badge/Project_Page-%23007ACC?style=for-the-badge&logo=github" alt="Project Page">
  </a>
  <a href="https://arxiv.org/pdf/2503.11692" target="_blank">
    <img src="https://img.shields.io/badge/Paper-PDF-informational?style=for-the-badge&logo=adobeacrobatreader" alt="Paper">
  </a>
  <a href="https://arxiv.org/abs/2503.11692" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-2503.11692-b31b1b?style=for-the-badge" alt="arXiv">
  </a>
  <a href="https://www.youtube.com/watch?v=7FnDFMThjGs" target="_blank">
    <img src="https://img.shields.io/badge/Video-Youtube-red?style=for-the-badge&logo=youtube" alt="Video">
  </a>
  <a href="https://wvu-irl.github.io/flope-irl/static/images/flope_poster.png" target="_blank">
    <img src="https://img.shields.io/badge/Poster-Image-blue?style=for-the-badge&logo=picture" alt="Poster">
  </a>
  <a href="https://github.com/wvu-irl/flope/releases/tag/release1" target="_blank">
    <img src="https://img.shields.io/badge/Data-Dataset-orange?style=for-the-badge&logo=databricks" alt="Data">
  </a>
</p>


![Teaser](media/github_teaser.png)

# 🛠️ Installation Guide

## 📦 Prerequisites

Make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) installed on your system.

You can check by running:

```bash
conda --version
```

## 🚀 Environment Setup
```bash
git clone https://github.com/wvu-irl/flope.git
cd flope
conda env create -f environment.yml
echo "export PYTHONPATH=$(pwd):\$PYTHONPATH" >> ~/.bashrc # use zshrc if you are using zsh terminal
source ~/.bashrc
conda activate sunflower
```

# 🔧 FloPE Pose Annotator Tool

<p align="left">
    <img src="media/flope_annotator.png" alt="FloPE Pose Annotator Tool" width="500">
</p>

```bash
cd pose_annotator
python annotator.py
```

# BibTex

```
@article{shrestha2025flope,
  title={FloPE: Flower Pose Estimation for Precision Pollination},
  author={Shrestha, Rashik and Rijal, Madhav and Smith, Trevor and Gu, Yu},
  journal={arXiv preprint arXiv:2503.11692},
  year={2025}
}
```
