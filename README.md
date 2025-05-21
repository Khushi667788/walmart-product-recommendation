# walmart-product-recommendation
Categorical encoding using LabelEncoder  Creation of User-Item matrix  KNN to find nearest neighbors for each product  Recommendations based on most similar users/products
# 🛒 Walmart Product Recommendation System

A machine learning-based personalized recommendation engine that suggests products to Walmart users based on their historical purchase behavior.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-Learn](https://img.shields.io/badge/ML-KNN-green)
![Flask](https://img.shields.io/badge/API-Flask-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📦 Features

- Content-based filtering using KNN
- Personalized product recommendations
- Flask API or Streamlit interface
- Clean data preprocessing & encoding
- Model saved with Joblib for fast inference

---

## 📁 Dataset

Used: [Walmart Sales Dataset (Kaggle)](https://www.kaggle.com/datasets/sachin27/walmart)

**Columns:**
- `User_ID`, `Product_ID`, `Gender`, `Age`, `Occupation`, `City_Category`, `Stay_In_Current_City_Years`, `Marital_Status`, `Product_Category`, `Purchase`

Place it in the `/data` folder as `walmart_data.csv`.

---

## 🚀 Installation

```bash
git clone https://github.com/yourusername/walmart-product-recommendation.git
cd walmart-product-recommendation
pip install -r requirements.txt
