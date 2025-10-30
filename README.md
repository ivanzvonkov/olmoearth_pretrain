<div align="center">
  <img src="assets/OlmoEarth-logo.png" alt="OlmoEarth Logo" style="width: 600px; margin-left:'auto' margin-right:'auto' display:'block'"/>
  <br>
  <br>
</div>
<p align="center">
  <a href="huggingface.co/collections/allenai/olmoearth">
    <img alt="Model Checkpoints" src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-Models-yellow">
  </a>
</p>

OlmoEarth is a a flexible, multi-modal, spatio-temporal family of foundation models for Earth Observations.

## Installation

We recommend Python 3.12, and recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/).
To install dependencies with uv, run:

```bash
git clone git@github.com:allenai/olmoearth_pretrain.git
cd olmoearth_pretrain
uv sync --locked --all-groups --python 3.12
# only necessary for development
uv tool install pre-commit --with pre-commit-uv --force-reinstall
```

uv installs everything into a venv, so to keep using python commands you can activate uv's venv: `source .venv/bin/activate`. Otherwise, swap to `uv run python`.

OlmoEarth is built using [OLMo-core](https://github.com/allenai/OLMo-core.git). OLMo-core's published [Docker images](https://github.com/orgs/allenai/packages?repo_name=OLMo-core) contain all core and optional dependencies.

## Model Summary

The OlmoEarth models are trained on three satellite modalities (Sentinel 2, Sentinel 1 and Landsat) and six derived maps (OpenStreetMap, WorldCover).
| Model Size | Weights | Encoder Params | Decoder Params |
| --- | --- | --- | --- |
| Nano | [link](https://huggingface.co/allenai/OlmoEarth-v1-Nano) | 1.4M | 800K |
| Tiny | [link](https://huggingface.co/allenai/OlmoEarth-v1-Tiny) | 6.2M | 1.9M |
| Base | [link](https://huggingface.co/allenai/OlmoEarth-v1-Base) | 89M | 30M |

## Data Summary

Our pretraining dataset contains around 300,000 samples from around the world of 2.56km×2.56km regions, although many samples contain only a subset of the timesteps and modalities.

The distribution of the samples is available below:

<img src="assets/datamap.png" alt="Training sample distribution" style="width: 500px; margin-left:'auto' margin-right:'auto' display:'block'"/>

The dataset can be downloaded [here](https://huggingface.co/datasets/allenai/olmoearth_pretrain_dataset).

## Training scripts

Detailed instructions on how to pretrain your own OlmoEarth model are available in [Pretraining.md](docs/Pretraining.md).

# Evaluations

Detailed instructions on how to replicate our evaluations is available in #TODO.
