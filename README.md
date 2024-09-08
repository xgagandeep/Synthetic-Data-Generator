# Synthetic Data Generator


## Table of Contents
- [Introduction](#introduction)

- [Motivation](#motivation)

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the GAN](#training-the-gan)
  - [Generating Synthetic Images](#generating-synthetic-images)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

The **Synthetic Data Generator** is a project designed to create synthetic images of handwritten digits using a Conditional Generative Adversarial Network (cGAN) trained on the MNIST dataset. This tool aims to augment existing datasets, enabling the training of more robust neural networks by providing additional diverse data samples.
![Output1](https://github.com/user-attachments/assets/60afb48e-4a4a-48f1-bc75-4e627e1fa952)

## Motivation

Data scarcity is a common challenge in training deep learning models. High-quality synthetic data can help overcome this limitation by providing additional training examples, enhancing model performance, and reducing overfitting. This project focuses on generating synthetic handwritten digits to support various machine learning applications.

## Features

- **Conditional GAN Architecture**: Generates images conditioned on specific digit labels, allowing control over the generated output.
- **Training on MNIST**: Utilizes the well-known MNIST dataset for training, ensuring compatibility with standard benchmarks.
- **Google Colab Integration**: Easily train the model using Google Colab with seamless Google Drive integration for saving checkpoints and generated images.
- **Streamlit Web App**: Provides an interactive web interface to generate and download synthetic images on demand.
- **Model Persistence**: Save and load trained models for continued training or deployment.

## Architecture

The project consists of two main components:

1. **Training Script (Colab Notebook)**:
   - **Data Preparation**: Loads and preprocesses the MNIST dataset.
   - **Model Building**: Constructs the Generator and Discriminator models using TensorFlow and Keras.
   - **Training Loop**: Trains the GAN, saving model checkpoints and generated images at specified intervals.
   - **Model Saving**: Saves the trained models in both checkpoint and `.h5` formats.

2. **Deployment Script (Streamlit App)**:
   - **Model Loading**: Loads the pre-trained Generator model.
   - **Image Generation**: Generates synthetic images based on user-specified labels.
   - **User Interface**: Allows users to generate single or multiple digits and download the results as a ZIP file.

## Installation

### Prerequisites

- **Python 3.7+**
- **Google Colab Account** (for training)
- **Google Drive** (for storing models and images)
- **Streamlit** (for the web application)

### Clone the Repository

```bash
git clone https://github.com/yourusername/synthetic-data-generator.git
cd synthetic-data-generator
```

### Set Up Virtual Environment (Optional but Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

*If `requirements.txt` is not provided, install the necessary packages manually:*

```bash
pip install tensorflow matplotlib numpy streamlit pillow
```

## Usage

### Training the GAN

1. **Open the Colab Notebook**

   Navigate to the [Colab Notebook](https://colab.research.google.com/drive/1lU4ELnWSmaJVIalx5F1Myxju2xlGvK5K) provided in the repository.

2. **Mount Google Drive**

   The notebook includes code to mount your Google Drive, ensuring that model checkpoints and generated images are saved persistently.

3. **Run the Notebook**

   Execute all cells in the notebook. Training parameters such as epochs, batch size, and latent dimensions are predefined but can be adjusted as needed.

4. **Monitor Training**

   The notebook displays generated images at regular intervals and prints loss metrics for both the Generator and Discriminator.

5. **Save Models**

   After training, the Generator and Discriminator models are saved in both checkpoint and `.h5` formats in your Google Drive under the `Project/Model/` directory.

### Generating Synthetic Images

1. **Navigate to the Streamlit App**

   The deployment script is located in the repository and can be run locally.

2. **Run the Streamlit App**

   ```bash
   streamlit run app.py
   ```

3. **Interact with the Interface**

   - **Select Generation Option**: Choose whether to generate a specific digit or all digits.
   - **Specify Parameters**: Input the digit label and the number of images to generate.
   - **Generate Images**: Click the "Generate" button to create synthetic images.
   - **Download Images**: After generation, download the images as a ZIP file for use in your projects.

## Project Structure

```
synthetic-data-generator/
├── project.ipynb        # Training script
├── model.py                      # Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── Weights/                    # Final model generator and discriminator files
│   ├── Gepoch_*.ckpt
│   └-─ Depoch_*.ckpt
├── Generator/
│   └-─ Final_model.h5           # final model generator to be used by model.py
├── Images/
│   └── *.png                    # Generated images
└── ...                          # Additional files and directories
```

## Dependencies

- **TensorFlow**: Deep learning framework for building and training models.
- **Keras**: High-level neural networks API, integrated with TensorFlow.
- **Matplotlib**: Plotting library for visualizing generated images.
- **NumPy**: Fundamental package for numerical computations.
- **Streamlit**: Framework for building interactive web applications.
- **Pillow**: Python Imaging Library for image processing.

*Install all dependencies using:*

```bash
pip install -r requirements.txt
```


## Acknowledgements

- **MNIST Dataset**: Thanks to Yann LeCun for the MNIST dataset, a staple in the machine learning community.
- **TensorFlow and Keras**: For providing robust tools to build and train deep learning models.
- **Streamlit**: For making it easy to create interactive web applications for machine learning projects.
- **OpenAI**: For inspiring advancements in synthetic data generation.

---

*Feel free to contribute to this project by submitting issues or pull requests. Your feedback and contributions are welcome!*
