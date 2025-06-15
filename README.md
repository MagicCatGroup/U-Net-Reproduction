# U-Net Model Implementation for Biomedical Image Segmentation

This repository provides a full implementation and reproduction of the U-Net architecture, specifically applied to biomedical image segmentation tasks using the ISBI2012 dataset.

## Project Overview

U-Net is a convolutional neural network architecture widely used for image segmentation tasks, particularly effective in biomedical domains. This project demonstrates an end-to-end pipeline, including data preparation, model training, evaluation, and prediction.

## Dataset

The implementation utilizes the **ISBI2012 dataset**, specifically curated for biomedical image segmentation challenges. The dataset comprises microscopy images and their corresponding segmentation masks.

Dataset details:

* Source: [ISBI Challenge: Segmentation of Neuronal Structures in EM Stacks](https://imagej.net/events/isbi-2012-segmentation-challenge)
* Format: Grayscale images with associated masks

## Project Structure

```plaintext
├── main.ipynb          # Jupyter notebook containing end-to-end training and evaluation workflow
├── U_Net.py            # Python script defining the U-Net model architecture
├── data/               # Directory for storing the ISBI2012 dataset
├── models/            # Directory for saving trained models and segmentation outputs
└── README.md           # Project documentation
```

## Requirements

Install the required libraries via pip:

```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

1. Download the ISBI2012 dataset and place it under the `data/` directory.
2. Ensure your data directory structure matches the expectations in `main.ipynb`.

### Training and Evaluation

Open and run the provided Jupyter notebook (`main.ipynb`) sequentially to perform:

* Data loading and preprocessing
* Model training
* Performance evaluation
* Result visualization

### Model Definition

The U-Net model architecture is implemented in `U_Net.py`. Adjustments to the architecture (e.g., depth, number of channels) can be made directly in this file.

## Results

Trained model checkpoints and segmentation results are saved under the `models/` directory. Visualizations of model performance are included in the notebook.

## License

This project is open-sourced under the MIT License. Feel free to adapt and use it for your projects.

---

## References

* Ronneberger, O., Fischer, P., & Brox, T. (2015). [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). MICCAI.
