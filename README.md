# Itraoperative-US
GenerativeAI model for synthesizing Intraoperative Ultrasound (iUS)

## Install
This repository relys on the [PyTorch](https://pytorch.org/get-started/locally/) and [Diffuser](https://huggingface.co/docs/diffusers/installation). Before install the requirements please read the official documentation and install torch and diffuser. here a simple example:

create a virtual env with conda
```bash
conda create --name ius python==3.10
conda deactivate
conda activate ius
```

install torch
```bash
pip install torch torchvision torchaudio
```

install diffuser and transformers
```bash
pip install diffusers["torch"] transformers
```

downlosd the repository using ssh connection
```bash
git clone git@github.com:AngeloLasala/Itraoperative-US.git
```

install the requirements
```bash
pip install -e .
```
