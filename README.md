# Auto Glossing Stem Translation

## Overview

Welcome to the Auto Glossing Stem Translation repository! This repository contains the code accompanying our system description paper. Our system is based on the framework presented in [Tü-CL at SIGMORPHON 2023: Straight-Through Gradient Estimation for Hard Attention](https://aclanthology.org/2023.sigmorphon-1.17/).

## Setup

To get started, follow these steps:

1. **Create a virtual environment**: We recommend using Anaconda for managing environments. You can create a new environment named `glossing` with Python 3.9 and pip by running:
    ```
    conda create -n glossing python=3.9 pip
    ```

2. **Activate the environment**: Activate the newly created environment with:
    ```
    conda activate glossing
    ```

3. **Install dependencies**: Install the required dependencies listed in [requirements.txt](requirements.txt) using pip:
    ```
    pip install -r requirements.txt
    ```

Ensure that you have a folder named `data` in the repository. You can place your own data into this folder. The data format should align with the format obtained from [the shared task's main repository](https://github.com/sigmorphon/2023glossingST).

## Training a Model

To train a single model and obtain predictions for the corresponding test set, execute the following command:
python main.py --language LANGUAGE --model MODEL --track TRACK

Replace `LANGUAGE` with the desired language from the shared task dataset. The `MODEL` parameter should be set to `morph` for the joint segmentation and glossing model. The `TRACK` parameter can be either `1` for the closed track or `2` for the open track. For additional hyperparameters, refer to the help section:


## Hyperparameter Tuning

To find the best hyperparameters, you can utilize the `hyperparameter_tuning.py` script:
 ```
python hyperparameter_tuning.py
--language LANGUAGE
--model MODEL
--track TRACK
--trials TRIALS
 ```
The `TRIALS` parameter specifies the number of evaluated hyperparameter combinations. We used 50 trials to obtain the hyperparameters provided in [best_hyperparameters.json](best_hyperparameters.json), which is included in this repository.

To retrain all models with the best hyperparameters, run:

To obtain predictions for the test data from the trained models, use:
python predict_from_model.py


## Citation

If you find this code useful, please consider citing our paper. The citation will be updated soon.
