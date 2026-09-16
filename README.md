# FairGDA: Source-Free Graph Domain Adaptation for Social Fairness

This repository contains the official implementation of **FairGDA**, a framework designed to achieve **group fairness** and **graph domain adaptation** under **source-free settings**.

## 🔍 Overview

**FairGDA** explores fair feature learning for **Graph Domain Adaptation (GDA)** without accessing source data during target adaptation. It aims to improve both predictive performance and group fairness under graph distribution shifts.

Most existing fair GNN methods assume that source training data remain available during deployment or adaptation, which may be impractical in privacy-sensitive applications such as social networks. FairGDA addresses this issue by adapting a pretrained source model to unlabeled target graphs while reducing the influence of sensitive attribute information.

## ⚙️ Framework

FairGDA consists of three main components:

### 1. Disentanglement of Fairness Information

Separates node representations into fairness-related and bias-related subspaces through a dual-branch learning structure.

### 2. Feature Reverse Alignment

Enhances the independence between the two representation subspaces through similarity minimization and contrastive learning, reducing the influence of bias-related information.

### 3. Fairness-Aware Data Augmentation

Introduces iterative graph augmentation with different bias intensities to improve robustness and fairness under domain shifts.

## 🔧 Environment and Installation

The implementation is based on Python and PyTorch. A CUDA-enabled GPU is recommended for efficient training, although the code can also be executed on CPU for small datasets.

We recommend creating an isolated Python environment before installing the dependencies:

```bash
conda create -n fairgda python=3.9
conda activate fairgda
```

Install all required packages using:

```bash
pip install -r requirements.txt
```

The detailed package dependencies are provided in `requirements.txt`.

After installation, make sure that the datasets and pretrained model parameters are placed in the corresponding directories specified in the repository.

## 🚀 Execution Workflow

The complete experimental workflow consists of three main steps:

### Step 1: Train the Source Model

First, train a model on the selected source domain:

```bash
python train_source.py --dataset <dataset_name> --inid <domain>
```

For example:

```bash
python train_source.py --dataset bail --inid _B0
```

Here,

* `<dataset_name>` specifies the dataset, such as `bail` or `credit`.
* `<domain>` specifies the source domain, such as `_B0` or `_B2`.

We also provide pretrained source model parameters in the `model_para` folder. Therefore, this step can be skipped when directly using the provided pretrained models.

### Step 2: Evaluate the Source Model

The pretrained or newly trained source model can be evaluated by running:

```bash
python test.py --dataset <dataset_name> --inid <domain>
```

This step verifies the performance and fairness of the source model before target-domain adaptation.

### Step 3: Perform Source-Free Target Adaptation

After obtaining the source model, adapt it to a target domain without accessing the original source data:

```bash
python train_target.py --dataset <dataset_name> --inid <domain>
```

For example:

```bash
python train_target.py --dataset bail --inid _B2
```

During this stage, only the pretrained source model and unlabeled target-domain graph are used, following the source-free graph domain adaptation setting.

## 📌 Recommended Workflow

A typical experimental pipeline is:

```text
Prepare dataset
      ↓
Train source model
      ↓
Evaluate source model
      ↓
Perform source-free adaptation on target domain and evaluate target-domain accuracy and fairness
```

If the provided pretrained parameters are used, the source training stage can be omitted and the experiment can start directly from target-domain adaptation.
