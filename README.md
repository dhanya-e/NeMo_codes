# NeMo Scripts for Inference and Fine-Tuning

This repository contains a collection of scripts for running inference, fine-tuning and other experiments using NVIDIA NeMo ASR models.

---

# Requirements

Before running the scripts, ensure the following are installed:

- Python 3.10+
- PyTorch
- CUDA-compatible GPU (recommended)
- pip or conda

---

# Install NVIDIA NeMo

Follow installation instructions from the official repository:

https://github.com/NVIDIA/NeMo

# Running Inference

To run inference using the provided bash script:
```
bash run_inference.sh
```
The script executes the inference pipeline using a pretrained NeMo model.

Modify the parameters inside the script to configure:

model path
input data
output location

# Running Fine-Tuning

To fine-tuning a model:
```
bash run_finetuning.sh
```

This script launches the fine-tuning workflow using the fine-tuning training configuration defined in the script. [NeMo ASR Finetuning Config](https://github.com/NVIDIA-NeMo/NeMo/blob/main/examples/asr/conf/asr_finetune/speech_to_text_finetune.yaml)

Modify:

dataset path
training parameters
learning rate
number of epochs
checkpoint paths