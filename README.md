# Flower Type Detection Without Using Libraries

This project demonstrates a machine learning approach to detect flower types from images without utilizing any external libraries. The goal is to build the entire process from scratch, including data preprocessing, model building, training, and evaluation.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Building](#model-building)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [How to Run](#how-to-run)
- [Contributing](#contributing)
- [Contact](#contact)

## Introduction

The **Flower Type Detection Without Using Libraries** project is a machine learning project aimed at detecting different types of flowers from images using a custom-built machine learning model. The project refrains from using any machine learning libraries such as TensorFlow, PyTorch, or Scikit-learn, and instead focuses on implementing the algorithms and techniques from scratch.

## Project Structure

The project files and folders are organized as follows:

```plaintext
Flower_Type_Detection_Without_Using_Library/
│
├── data/
│   ├── train/
│   ├── test/
│   └── labels.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── results/
│   ├── confusion_matrix.png
│   ├── accuracy_plot.png
│   └── model_weights.pkl
│
├── README.md
├── requirements.txt
└── LICENSE
```

- **data/**: Contains the training and test datasets.
- **src/**: Contains the source code for data preprocessing, model building, training, and evaluation.
- **results/**: Contains the results of the model such as plots and saved weights.
- **README.md**: This file.
- **requirements.txt**: List of dependencies required to run the project.
- **LICENSE**: License for the project.

## Dataset

The dataset consists of images of various flower types. The images are categorized into different folders representing different classes of flowers. The dataset is split into training and testing sets:

- **Training set**: Used to train the model.
- **Testing set**: Used to evaluate the model's performance.

### Dataset Details

- **Number of Classes**: 5 (e.g., Daisy, Dandelion, Rose, Sunflower, Tulip)
- **Total Images**: 4,000
- **Image Size**: 128x128 pixels

## Data Preprocessing

Data preprocessing involves several steps to prepare the dataset for training the model. The following operations are performed:

1. **Image Resizing**: All images are resized to a consistent size of 128x128 pixels.
2. **Normalization**: Pixel values are normalized to a range of 0 to 1.
3. **Label Encoding**: Flower labels are encoded into numerical values.

The preprocessing script can be found in `src/data_preprocessing.py`.

## Model Building

The model architecture is implemented from scratch using basic Python. The model consists of the following layers:

1. **Input Layer**: Accepts the preprocessed image data.
2. **Convolutional Layers**: Extracts features from the images.
3. **Fully Connected Layers**: Learns the complex patterns to classify the flowers.
4. **Output Layer**: Outputs the probabilities for each class.

The model implementation can be found in `src/model.py`.

## Training

The model is trained using a custom-built training loop. The training process involves:

- **Forward Pass**: Passing the input data through the network.
- **Loss Calculation**: Calculating the difference between predicted and actual labels.
- **Backward Pass**: Adjusting the model weights to minimize the loss.
- **Optimization**: Using gradient descent to optimize the model parameters.

The training script can be found in `src/train.py`.

### Hyperparameters

- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Epochs**: 50

## Evaluation

The model is evaluated on the test dataset using various metrics such as accuracy, precision, recall, and F1-score. A confusion matrix is also generated to visualize the model's performance across different classes.

The evaluation script can be found in `src/evaluate.py`.

## Results

The model achieved the following results on the test dataset:

- **Accuracy**: 92%
- **Precision**: 90%
- **Recall**: 91%
- **F1-Score**: 90.5%

## How to Run

To run the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Flower_Type_Detection_Without_Using_Library.git
   cd Flower_Type_Detection_Without_Using_Library
   ```

2. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Preprocess the data**:
   ```bash
   python src/data_preprocessing.py
   ```

4. **Train the model**:
   ```bash
   python src/train.py
   ```

5. **Evaluate the model**:
   ```bash
   python src/evaluate.py
   ```

## Contributing

Contributions are welcome! If you want to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes.
4. Push your branch to your forked repository.
5. Open a pull request.

Please ensure your code adheres to the project's coding standards and includes appropriate tests.

## Contact

For any inquiries or feedback, feel free to contact me at:

- **Email**: [gorkemturkut@hotmail.com](mailto:gorkemturkut@hotmail.com)
- **GitHub**: [https://github.com/gorkemturkut57](https://github.com/gorkemturkut57)
