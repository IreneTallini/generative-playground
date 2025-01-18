# Generative Playground

This repository contains code for comparing two generation options on the Moons dataset, 
1) score matching, implemented from scratches
2) gradient ascent on the logits of a simple binary classifier, where points in the Moons dataset are labeled 1 and outside 0.

## Installation

1. Clone this repository:
```sh
git clone <repository-url>
```

2. Navigate to the project folder:
```sh
cd generative_playground
```

3. Create and activate the conda environment:
```sh
conda env create -f environment.yaml
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

- `data_providers.py`: Contains data provider classes for generating synthetic datasets.
- `modules.py`: Contains model definitions for the denoising model and classifier.
- `main.py`: Contains functions for training models and generating samples.
- `environment.yaml`: Conda environment configuration file.
- `setup.py`: Package setup file.

## License

This project is licensed under the MIT License.