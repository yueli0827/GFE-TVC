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

[REASSEMBLE Dataset](https://tuwien-asl.github.io/REASSEMBLE_page/)
[JIGSAWS Dataset](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/)
NursBot Dataset has been incorporated into this repo.

## Getting Started

In the following, we provide example scripts for Feature Extraction, Clustering and Postpromoting.

### Feature Extraction

The following is a example of feature extraction on `REASSEMBLE` dataset.

```bash
cd feature extraction/
```

### Clustering

The following is a example of spectral clustering on `JIGSAWS` dataset.

```bash
cd clustering/
```


### Postpromoting

The following is a example of postpromoting on `NursBot` dataset.

```bash
cd postpromoting/
```
## Acknowledgement

The clustering component of GFE-TVC is based on [Spectral Clustering](https://github.com/xjnine/GBSC), GMM, K-means, [Graph Clustering](https://github.com/gayanku/scgc), [Hierarchical Clustering](https://github.com/wazenmai/hc-smoe), and [DeepDPM](https://github.com/BGU-CS-VIL/DeepDPM). We also conducted performance validation of GFE-TVC using three datasets(REASSEMBLE, JIGSAWS, and NursBot). Many thanks to their wonderful work!
