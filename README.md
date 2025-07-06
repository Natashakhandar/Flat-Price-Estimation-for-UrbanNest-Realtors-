# Flat-Price-Estimation-for-UrbanNest-Realtors-

# AIM:-
To build a machine learning model that predicts the price of flats using features such as area, bedrooms, amenities, and distance to metro, helping UrbanNest Realtors make data  driven pricing decisions.

# Tools & Libraries Used:-
•	Python
•	pandas, numpy
•	matplotlib, seaborn
•	scikit learn
•	joblib (model persistence)
•	Streamlit (UI)

# Step  by  Step Implementation:-

Jupyter Notebook

# 1.Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib    


# 2.Load the Dataset 

df = pd.read_csv('Flat Price Estimation for UrbanNest Realtors (2).csv')
df.head()   
![image](https://github.com/user-attachments/assets/0405a22b-b951-4e94-b519-ce83c7c91eb0)


# 3.Understand the Dataset

print(df.info())
print(df.describe())
print(df.columns.tolist())

![image](https://github.com/user-attachments/assets/3717c0ec-d316-4b3d-bb02-7202223eeb52)

![image](https://github.com/user-attachments/assets/4f46673a-23bd-469f-8349-d6994e90de1e)




# 4.Exploratory Data Analysis

//Correlation matrix

plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

![image](https://github.com/user-attachments/assets/11c86dd4-a3e7-40a0-aa17-f1d557067c0f)
![image](https://github.com/user-attachments/assets/1baf1628-3894-46d0-8cd0-3bb16413de4c)

# 5.Scatter plot: Area vs Price
sns.scatterplot(x='area_sqft', y='flat_price', data=df)
plt.title("Area vs Flat Price")
plt.show()  
  
![image](https://github.com/user-attachments/assets/6c9151a5-d10a-4150-a8df-e6a3acce397b)



# 6.Define Features & Target

X = df.drop('flat_price', axis=1)
y = df['flat_price']


# 7. Train  Test Split
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   
 
# 8.Feature Scaling
    
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
    

# 9.Train Models

// Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

//Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

// Gradient Boosting
gb = GradientBoostingRegressor()
gb.fit(X_train, y_train)

![image](https://github.com/user-attachments/assets/f41573e6-8811-4e80-b3e4-d9d4bf2022b2)


    
# 10.Evaluate Models

def evaluate_model(model, X, y_true, model_name):
    y_pred = model.predict(X)
    print(f"\n📊 {model_name}")
    print("R² Score:", r2_score(y_true, y_pred))
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))

// Evaluate all
evaluate_model(lr, X_test_scaled, y_test, "Linear Regression")
evaluate_model(rf, X_test, y_test, "Random Forest")
evaluate_model(gb, X_test, y_test, "Gradient Boosting")


   
   ![image](https://github.com/user-attachments/assets/1da45662-9af5-4eb4-a290-8854d5b85d48)



# 11.Save the Best Model 
   
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
pipeline.fit(X, y)


![image](https://github.com/user-attachments/assets/78113aa5-d8a5-408a-87f4-2914680036eb)


// Save the pipeline
joblib.dump(pipeline, 'flat_price_model.pkl')    


![image](https://github.com/user-attachments/assets/0ab48a1e-cd8a-4342-a5a8-2ff0c23fc05f)


# App.py 

import streamlit as st
import joblib
import numpy as np

st.title(" UrbanNest Flat Price Estimator")

model = joblib.load('flat_price_model.pkl')


// Input Fields
area = st.slider("Area (in sq.ft)", 300, 5000, step=50)
bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5])
distance = st.slider("Distance to Metro (km)", 0.1, 10.0, step=0.1)
age = st.slider("Age of Flat (years)", 0, 50)
amenity = st.slider("Amenities Score", 0, 10)

if st.button("Estimate Price"):
    input_data = np.array([[area, bedrooms, distance, age, amenity]])
    prediction = model.predict(input_data)
    st.success(f"Estimated Price: ₹{prediction[0]:,.2f}")



# Files inside Folder:-

![image](https://github.com/user-attachments/assets/9121c89d-b5db-405f-8d5d-463802f4f89e)




# Run Streamlit App Command:-

python -m streamlit run app.py


  # OUTPUT:-

   ![image](https://github.com/user-attachments/assets/4255829d-dc8f-414e-af36-2c80b57d5d06)

    
  ![image](https://github.com/user-attachments/assets/325814d6-2962-4fe5-a4ac-c59f9e15100b)























