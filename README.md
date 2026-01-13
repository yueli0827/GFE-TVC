# GFE-TVC：A General Unsupervised Telerobotic Trajectory Segmentation Method

## Contents
- [Installation](#installation)
- [Datasets](#datasets)
- [Getting Started](#getting-started)
  - [Feature Extraction](#Feature-extraction)
  - [Clustering](#Clustering)
  - [Postpromoting](#Postpromoting)
- [Acknowledgement](#acknowledgement)

## Installation

First clone this repo, and then run the following commands in the given order to install the dependency for GFE-TVC.

```bash
conda create -n GFE_TVC python=3.9.19
conda activate GFE_TVC
cd GFE_TVC
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```
## Datasets

To demonstrate the generalizability of the proposed method, we validated GFE-TVC on three publicly available datasets.

REASSEMBLE Dataset link:
JIGSAWS Dataset link:
NursBot Dataset link:(It has been incorporated into this repo)

## Getting Started

In the following, we provide example scripts for Feature Extraction, Clustering and Postpromoting.

### Feature Extraction

The following is a example of unsupervised skill discovery on `libero_object` dataset.

```bash
cd lotus/skill_learning
```

### Clustering

The following is a example of unsupervised skill discovery on `libero_object` dataset.

```bash
cd lotus/skill_learning
```


### Postpromoting

The following is a example of unsupervised skill discovery on `libero_object` dataset.

```bash
cd lotus/skill_learning
```


