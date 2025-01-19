# Generative Playground

This repository contains code for comparing two generation options on the Moons dataset, 
1) score matching, implemented from scratches following https://arxiv.org/abs/2206.00364 (Elucidating the Design Space of Diffusion-Based Generative Models).
2) gradient ascent on the logits of a simple binary classifier, where points in the Moons dataset are labeled 1 and outside 0.
Note that this is just a proof of concept, parameters need to be tuned for optimal performance of the two methods.

## Installation

1. Clone this repository:
```sh
git clone git@github.com:IreneTallini/generative-playground.git
```

2. Navigate to the project folder:
```sh
cd generative_playground
```

3. Create and activate the conda environment:
```sh
conda env create -f env.yaml
conda activate generative-playground
```

4. Install the package:
```sh
pip install -e .
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