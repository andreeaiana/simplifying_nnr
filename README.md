<div align="center">

# Simplifying Content-Based News Recommendation

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

This is the code accompanying the paper [Simplifying Content-Based Neural News Recommendation: On User Modeling and Training Objectives](https://arxiv.org/abs/2304.03112) in which we propose a unified framework allowing for a systematic and fair comparison of news recommenders across three crucial design dimensions: (i) candidate-awareness in user modeling, (ii) click behavior fusion, and (iii) training objectives. 

![](./framework.png)

## Project Structure

The directory structure of the project looks like this:

```
├── configs                   <- Hydra configuration files
│   ├── callbacks                <- Callbacks configs
│   ├── datamodule               <- Datamodule configs
│   ├── debug                    <- Debugging configs
│   ├── experiment               <- Experiment configs
│   ├── extras                   <- Extra utilities configs
│   ├── hparams_search           <- Hyperparameter search configs
│   ├── hydra                    <- Hydra configs
│   ├── local                    <- Local configs
│   ├── logger                   <- Logger configs
│   ├── model                    <- Model configs
│   ├── paths                    <- Project paths configs
│   ├── trainer                  <- Trainer configs
│   │
│   ├── eval.yaml             <- Main config for evaluation
│   └── train.yaml            <- Main config for training
│
├── data                   <- Project data
│
├── logs                   <- Logs generated by hydra and lightning loggers
│
├── scripts                <- Shell scripts
│
├── src                    <- Source code
│   ├── datamodules              <- Lightning datamodules
│   ├── models                   <- Lightning models
│   ├── utils                    <- Utility scripts
│   │
│   ├── eval.py                  <- Run evaluation
│   └── train.py                 <- Run training
│
├── .gitignore                <- List of files ignored by git
├── requirements.txt          <- File for installing python dependencies
├── setup.py                  <- File for installing project as a package
└── README.md
```

## Data
We use the [MIND](https://msnews.github.io/) dataset in all experiments. <br>
The datasets are automatially downloaded, cached, and pre-processed when running the [train.py](src/train.py) pipeline. <br>
Alternatively, the datasets can be manually downloaded into the [data](data/) directory using the URLs from the [MIND data config](configs/datamodule/mind.yaml).


## How to run

Install dependencies

```bash
# clone project
git clone git@github.com:andreeaiana/simplifying_nnr.git
cd simplifying_nnr

# [OPTIONAL] create conda environment
conda create -n simplifying_nnr_env python=3.9
conda activate simplifying_nnr_env

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
# train on CPU
python src/train.py experiment=experiment_name.yaml trainer=cpu

# train on GPU
python src/train.py experiment=experiment_name.yaml trainer=gpu
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 datamodule.batch_size=64
```

## Citation

```bibtex
@article{iana2023simplifying,
    title={Simplifying Content-Based Neural News Recommendation: On User Modeling and Training Objectives},
    author={Andreea Iana and Goran Glavaš and Heiko Paulheim},
    booktitle={Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
    pages={2384--2388},
    year={2023}
}
```
