import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("../data/walmart_data.csv")

user_encoder = LabelEncoder()
product_encoder = LabelEncoder()

df['User_ID'] = user_encoder.fit_transform(df['User_ID'])
df['Product_ID'] = product_encoder.fit_transform(df['Product_ID'])

user_item_matrix = df.pivot_table(index='User_ID', columns='Product_ID', values='Purchase', fill_value=0)

model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(user_item_matrix)

joblib.dump(model, "../models/knn_model.pkl")
joblib.dump(user_item_matrix, "../models/user_item_matrix.pkl")
