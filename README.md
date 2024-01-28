# Medical Image Segmentation Project

## Overview

This repository contains the code and documentation for a research project focused on comparing different deep learning methods for the segmentation of medical images, with a particular emphasis on gastro-intestinal tract cancer treatment using radiation therapy.

## Objective

The primary objective of this study is to evaluate and compare the effectiveness of various deep learning approaches, including Unet, Unet++, and Deeplabv3, for automating the segmentation of stomach and intestines in integrated magnetic resonance imaging (IMRI) scans. The goal is to streamline the manual segmentation process, making medical image analysis more efficient and accessible.

## Dataset

The dataset used in this project consists of IMRI scans from patients diagnosed with gastro-intestinal tract cancer. These scans, acquired through MR-Linac systems, enable the visualization of daily tumor and organ positions. The dynamic nature of organ movement underscores the need for automated segmentation to optimize radiation delivery.

link : https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation

![output-onlinegiftools-ezgif com-optimize](https://github.com/Se2007/Medical-Image-Segmentation/assets/112750879/5b261c39-4642-41e0-9b45-f14e5d31e37f)

## Methodology

- **Deep Learning Models:** Unet, Unet++, Deeplabv3
- **Encoder Variations:** Different encoders used in Unet and Unet++
- **Evaluation Metrics:** Dice metric, recall, accuracy, IOU score, F1 score

## How to Use

1. Installation :

   ```bash
   git clone https://github.com/Se2007/Medical-Image-Segmentation
   cd Medical-Image-Segmentation
   python -m pip install -r requirements.txt

2. Dataset and Trained Models:
  To request a pre-processed dataset and trained models, simply send an email with specific details, such as your intended use and license type.

