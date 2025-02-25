## 🚀 Mini Machine Learning Projects Repository

This repository contains a collection of **mini Machine Learning projects**, covering various domains such as **fraud detection, financial sentiment analysis, and more**. Each project is self-contained, demonstrating a specific ML/AI concept with clear implementations and results.

---

### 🛡️ Fraud Detection Using Machine Learning

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

![Fraudster Hierarchy](fraud-detection/assets/tree.png)

#### 🔍 Principal Component Analysis (PCA)

To reduce dimensionality, PCA was applied, and the scree plot below shows the eigenvalues of each principal component.

![PCA Scree Plot](fraud-detection/assets/scree-plot-pca.png)

#### ⚡ ML Pipeline

Our pipeline standardizes the data, applies PCA for feature selection, and then uses an **SVM classifier** to predict fraudulent transactions.

![SVM Pipeline](fraud-detection/assets/svm-pipeline.png)

#### 🚀 Results

- **Model Accuracy:** **72.63% (test), 72.93% (train)**
- **Fraudulent Transactions Identified:** Mainly found in **Transfer & Cash-Out transactions**.
- **Dimensionality Reduction Success:** PCA helped optimize performance while retaining fraud detection accuracy.

🔗 **[Full Report & Code](fraud-detection/)**

---

### 💰 Financial News Sentiment Analysis Application

#### 📌 Overview

This project implements a **news sentiment analysis application** using the **DistilRoBERTa model fine-tuned for financial news sentiment analysis**, accessible via the **Hugging Face API**. The model classifies financial texts, such as market reports and news articles, into different sentiment categories to help users analyze the market sentiment.

#### 🛠️ Technologies Used

- **Machine Learning:** DistilRoBERTa (fine-tuned for financial sentiment analysis)
- **Libraries:** Hugging Face Transformers, Flask, PostgreSQL
- **Deployment:** Flask API, hosted on Heroku

#### 🔑 Key Features

- **Real-Time Sentiment Analysis:** Uses Hugging Face API for instant results.
- **Financial-Specific Model:** Trained on financial news to improve accuracy in economic contexts.
- **Web Application Interface:** Built using Flask, allowing users to input text and receive real-time analysis.

#### 🚀 How It Works

1. **User inputs financial text** (e.g., a market report or company earnings statement).
2. **The text is sent to the Hugging Face API**, which classifies sentiment as **positive, negative, or neutral**.
3. **The results are displayed** in a user-friendly interface.

#### 📈 Model Used

The **pretrained model** used for this task:
🔗 **[DistilRoBERTa fine-tuned for financial sentiment analysis](https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis)**

🔗 **[Full Code & Implementation](news-sentiment-analysis/)**

---
