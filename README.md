# Generative Playground

The original idea of this code was to test if gradient ascent on the logits function of a classifier is a good generative model (spoiler, in the simple implementation here it isn't). 
The code also contains a simple score matching implementation from scratches (the original idea was to use it as a baseline) and can be useful to someone.

This repository contains code for comparing two generation options on the scikit-learn Moons dataset:
1) score matching, implemented from scratches following [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364).
2) gradient ascent on the logits of a simple binary classifier, where points in the Moons dataset are labeled 1 and outside 0.



## Installation

1. Clone this repository:
```sh
git clone git@github.com:IreneTallini/generative-playground.git
```

2. Navigate to the project folder:
```sh
cd generative-playground
```

3. Create and activate the conda environment:
```sh
conda env create -f env.yaml
conda activate gp
```

## Usage
running 
`python main.py`
will train the score model, the classifier, and sample with the two methods.

## Project Structure

- `datasets.py`: Contains data provider classes for generating synthetic datasets.
- `modules.py`: Contains model definitions for the denoising model and classifier.
- `main.py`: Contains functions for training models and generating samples.
- `utils.py`: Contains plotting utils
- `environment.yaml`: Conda environment configuration file.
- `setup.py`: Package setup file.