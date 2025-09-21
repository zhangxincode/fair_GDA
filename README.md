# FairGDA: Source-Free Graph Domain Adaptation for Social Fairness

This repository contains the official implementation of **FairGDA**, a novel framework designed to achieve **group fairness** and **domain adaptation** in graph neural networks (GNNs) under **source-free settings**.

## 🔍 Overview

**FairGDA** explores *fair feature learning* for **Graph Domain Adaptation (GDA)** *without using source data*, aiming to balance fairness and adaptation across domains with distribution shifts.
Most existing fairness research in GNNs assumes access to labeled source data, which is often unrealistic in real-world privacy-sensitive scenarios (e.g., social networks). FairGDA addresses this gap by learning fair features from unlabeled target data, while ensuring that sensitive attributes are disentangled from other node features.

## ⚙️ Framework

FairGDA consists of **three synergistic components**:

### 1. **Disentanglement of Fairness Information**

- Separates node representations into *fairness-related* and *bias-related* subspaces via a dual-branch learning structure (FairNet & BiasNet).

### 2. **Feature Reverse Alignment**

- Enhances independence between the two subspaces through **similarity minimization** and **contrastive learning**, reducing the influence of bias-related information on fairness.

### 3. **Fairness-Aware Data Augmentation**

- Introduces iterative augmentation to increase data diversity and expose the model to varying *bias intensities*, improving robustness and fairness under domain shifts.

## 🔧 Installation
Install the required dependencies:
```angular2html
pip install -r requirements.txt
```
## 🚀 Run the code
To run FairGDA on a given dataset, you need to train the model on the source domain. Of course, we also provide pre-trained parameters in the `model_para` folder.

```angular2html
python train_source.py --dataset <dataset_name> --inid <domain>
```
Replace `<dataset_name>` with the name of the dataset you want to use (e.g., `bail`, `credit`).
Replace `<domain>` with the name of the domain you want to use (e.g., `_B0`, `_B2`).
run `test.py` to test the model on the source domain.


Then, you can train the model on other target domains, for example:
```angular2html
python train_target.py --dataset <dataset_name> --inid <domain>
```

