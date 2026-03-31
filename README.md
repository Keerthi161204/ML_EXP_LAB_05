# ML_EXP_LAB_05

📘 Perceptron vs Multilayer Perceptron (MLP)

 🔍 Overview

This project implements and compares two machine learning models:

* Perceptron Learning Algorithm (PLA) – implemented from scratch
* Multilayer Perceptron (MLP) – implemented using Scikit-learn

The goal is to evaluate their performance on a handwritten digits dataset and understand the impact of model complexity and hyperparameter tuning.

🎯 Objectives

* Implement PLA and MLP models
* Apply preprocessing and training
* Perform hyperparameter tuning for MLP
* Compare performance using evaluation metrics

 📂 Dataset

* Dataset used: `load_digits()` from Scikit-learn
* Type: Grayscale images of handwritten digits
* Preprocessing:

  * Feature scaling using `StandardScaler`
  * Train-test split (80% training, 20% testing)



⚙️ Implementation Details

 🔹 Perceptron Learning Algorithm (PLA)

* Implemented from scratch
* Uses **One-vs-Rest** for multi-class classification
* Linear classifier
* Weight update rule:

  ```
  w = w + η * y * x
  ```

🔹 Multilayer Perceptron (MLP)

* Hidden layers: `(128, 64)`
* Activation: `ReLU`
* Optimizer: `Adam`
* Learning rate: `0.001`
* Batch size: `64`
* Epochs: `200`


 📊 Results

| Model | Accuracy | Precision | Recall | F1-score |
| ----- | -------- | --------- | ------ | -------- |
| PLA   | 79.44%   | 0.79      | 0.79   | 0.79     |
| MLP   | 96.94%   | 0.97      | 0.97   | 0.97     |

These results are consistent with the lab report findings 


 📈 Key Observations

* PLA performs poorly because it can only learn **linear decision boundaries**
* MLP captures **nonlinear patterns**, leading to significantly higher accuracy
* Hyperparameters like:

  * learning rate
  * optimizer
  * number of layers
    have a strong impact on performance
* Increasing layers improves accuracy but may cause **overfitting**


🧠 Conclusion

MLP clearly outperforms PLA in handwritten digit classification due to its ability to model complex, nonlinear relationships. Proper hyperparameter tuning further enhances its performance and stability.


## 🚀 How to Run

bash
pip install numpy scikit-learn
python your_script_name.py

