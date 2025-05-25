# Soil Classification CNN Model

This repository contains code to train and run inference on a Convolutional Neural Network (CNN) model for soil type classification based on images.

---

## Contents

- `train_soil_classification.ipynb` — Notebook to train the CNN model on the soil image dataset.
- `inference_soil_classification.ipynb` — Notebook to load the trained model and run predictions on test images.
- `submission.csv` — Sample output file with predicted soil types for test images.

---

## Dataset

The dataset used consists of soil images categorized into 4 classes:

- Alluvial soil
- Black soil
- Clay soil
- Red soil

The dataset CSV files (`train_labels.csv` and `test_ids.csv`) and image folders are expected to be in the `/kaggle/input/soil-classification/soil_classification-2025/` directory when running on Kaggle.

---

## Environment Setup

To run this project locally, install the required Python packages listed in `requirements.txt`:

pip install -r requirements.txt

## How to Run
## Training
Open train_soil_classification.ipynb.

Ensure the dataset CSVs and images are accessible and paths updated if running locally.

Run all cells to preprocess data, define the CNN model, and train it.

The model will be saved as soil_classification_model.h5.

## Inference
Open inference_soil_classification.ipynb.

Load the saved model file (soil_classification_model.h5).

Preprocess the test images similarly to the training pipeline.

Run predictions and generate submission.csv with predicted soil types.

Submit submission.csv for evaluation or use for further analysis.

## Notes
The training notebook uses ImageDataGenerator for efficient image loading and augmentation.

Model checkpoints and early stopping callbacks can be added to improve training efficiency.

Adjust batch size, epochs, and learning rates in the notebook as needed for your environment.

The inference notebook assumes test images are located in the /kaggle/input/soil-classification/soil_classification-2025/test/ directory.
