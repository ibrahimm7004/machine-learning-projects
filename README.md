### 🛡️ Fraud Detection Using Machine Learning & Linear Algebra

#### 📌 Overview

This project implements a **credit card fraud detection system** using **Support Vector Machines (SVM)** and **Principal Component Analysis (PCA)**. The model analyzes financial transactions to classify them as **fraudulent or authentic**, helping mitigate risks in digital financial systems.

#### 🛠️ Technologies Used

- **Machine Learning:** SVM, PCA
- **Libraries:** Scikit-learn, NumPy, Pandas, Matplotlib
- **Dataset:** Financial transaction records with fraud labels

#### 🔑 Key Features

- **Dimensionality Reduction:** PCA improves model efficiency.
- **Fraud Classification:** SVM handles high-dimensional transaction data.
- **Data Preprocessing:** Balanced dataset using oversampling for better fraud detection.

#### 📊 Hierarchical Fraud Classification

Fraudsters are categorized based on their corporate and community level roles.

![Fraudster Hierarchy](assets/tree.png)

#### 🔍 Principal Component Analysis (PCA)

To reduce dimensionality, PCA was applied, and the scree plot below shows the eigenvalues of each principal component.

![PCA Scree Plot](assets/scree-plot-pca.png)

#### ⚡ ML Pipeline

Our pipeline standardizes the data, applies PCA for feature selection, and then uses an **SVM classifier** to predict fraudulent transactions.

![SVM Pipeline](assets/svm-pipeline.png)

#### 🚀 Results

- **Model Accuracy:** **72.63% (test), 72.93% (train)**
- **Fraudulent Transactions Identified:** Mainly found in **Transfer & Cash-Out transactions**.
- **Dimensionality Reduction Success:** PCA helped optimize performance while retaining fraud detection accuracy.

🔗 **[Full Report & Code](fraud-detection/)**
