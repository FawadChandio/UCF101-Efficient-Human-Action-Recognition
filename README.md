# UCF101 Efficient Human Action Recognition

Compute-Budgeted Temporal Evidence Selection for Efficient CNN-Based Human Action Recognition
An experimental research project investigating efficient temporal frame selection for RGB-based Human Action Recognition (HAR), using MobileNetV3-Large and the UCF101 dataset.

## Project Overview
This project investigates whether selecting informative video frames can improve computational efficiency while maintaining competitive action-recognition performance.
The proposed approach compares:
- Uniform temporal frame sampling
- Temporal-difference-based frame selection

A fixed candidate pool of 32 frames is considered, with frame budgets of 4, 8, 12, and 16.

## Methodology
- Dataset: UCF101
- Official split: Split 1
- Training videos: 7,933
- Validation videos: 1,604
- Testing videos: 3,783
- Action classes: 101
- Backbone: MobileNetV3-Large
- Input: RGB video frames
- Temporal aggregation: Feature-level temporal average pooling

The validation set is derived from the official training partition using source-group separation to reduce data leakage. The official test set is reserved for final evaluation.

## Experimental Comparisons
1. Uniform sampling with different frame budgets.
2. Temporal-difference sampling with the same frame budgets.
3. Accuracy-versus-computation analysis.

## Repository Structure
```text
notebooks/
    HAR_Research_Development.ipynb
```

## Current Status
Research and implementation in progress.

Dataset partitioning and source-group leakage checks have been completed. Further model training, evaluation, computational profiling, and comparative analysis are pending.

No final performance claims are made at this stage.

## Tools and Technologies
- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Kaggle GPU

## Dataset
UCF101 is a publicly available human action recognition dataset. Users must obtain the dataset from its official source and follow its terms of use.

## Research Disclaimer
This repository documents an ongoing experimental research project. Results and conclusions will be updated after controlled experiments and validation.

## Author
SoftMAT Developers

GitHub: [FawadChandio](https://github.com/FawadChandio)

