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
conda create -n GFE_TVC python=3.9
conda activate GFE_TVC
cd GFE_TVC
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
## Datasets

To demonstrate the generalizability of the proposed method, we validated GFE-TVC on three publicly available datasets.

[REASSEMBLE Dataset](https://tuwien-asl.github.io/REASSEMBLE_page/) 

[JIGSAWS Dataset](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/)


NursBot Dataset has been incorporated into this repo.



## Getting Started

In the following, we provide example scripts for Feature Extraction, Clustering and Postpromoting.

### Feature Extraction

The following is a example of feature extraction.

```bash
cd feature_extraction/
python feature_extraction.py --video_path PATH_TO_YOUR_VIDEO --num_frames 1000 --batch_size 8 --feat_key "mid"
```

### Clustering

The following is a example of spectral clustering.

```bash
cd clustering/
python spectral_clustering.py --input_path YOUR_FEATURE_DIR --out_dir YOUR_OUTPUT_DIR --k_mode deepdpm --k_min 2 --k_max 20
```


### Postpromoting

The following is a example of postpromoting.

```bash
cd postpromoting/
python postpromoting.py -d YOUR_DATA_SET_NAME -clustering-model YOUR_CLUSTERING_MODEL_NAME -at 0.3 -ae 0.6 --n-epochs 200 --batch-size 32 --tvc-flag true
```
## Acknowledgement

The clustering component of GFE-TVC is based on [Spectral Clustering](https://github.com/xjnine/GBSC), GMM, K-means, [Graph Clustering](https://github.com/gayanku/scgc), [Hierarchical Clustering](https://github.com/wazenmai/hc-smoe), and [DeepDPM](https://github.com/BGU-CS-VIL/DeepDPM). We also conducted performance validation of GFE-TVC using three datasets(REASSEMBLE, JIGSAWS, and NursBot). Many thanks to their wonderful work!
