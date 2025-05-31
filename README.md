# breast-cancer-detection
Breast Cancer Classification with Logistic Regression
In this project, we explore how to use Logistic Regression for binary classification on the Breast Cancer Wisconsin (Diagnostic) Dataset. Our goal is to build a model that can effectively distinguish between malignant (cancerous) and benign (non-cancerous) tumors using Python and Scikit-learn.

📌 Objective
The primary aim of this project is to create a binary classifier that can accurately identify whether a tumor is malignant or benign. We will also evaluate the model's performance using various classification metrics to ensure its reliability.

📊 Dataset
Name: Breast Cancer Wisconsin (Diagnostic) Data Set
Source: You can find the dataset on Kaggle - uciml/breast-cancer-wisconsin-data.
Features: The dataset contains 30 numeric features derived from digitized images of fine needle aspirates (FNA) of breast masses.
Target Variable:
M = Malignant (1)
B = Benign (0)
🛠️ Technologies Used
To accomplish our objectives, we will utilize the following technologies:

Python 🐍 for programming
Pandas and NumPy for data manipulation and analysis
Matplotlib and Seaborn for data visualization
Scikit-learn for building and evaluating our machine learning model
🚀 How to Run the Project
To get started with the project, follow these steps:

Download the dataset from Kaggle.

Place the data.csv file in your project directory.

Data Loading & Cleaning

We will load the dataset and remove unnecessary columns (like id and Unnamed: 32).
The target variable will be encoded, converting M to 1 and B to 0.
Train/Test Split & Standardization

We will split the data into training (70%) and testing (30%) sets.
The StandardScaler will be applied to standardize the input features.
Model Training

We will implement Logistic Regression using the sklearn.linear_model library.
Evaluation

The model's performance will be assessed using a confusion matrix, classification report (including precision, recall, and F1-score), and ROC-AUC score and curve.
Threshold Tuning

We will experiment with different thresholds (e.g., 0.3) to find a balance between precision and recall.
Sigmoid Function Visualization

Finally, we will visualize how logistic regression uses the sigmoid function to map outputs to probabilities.
