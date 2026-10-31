from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

model = LinearRegression()

# Training data
Data={
    "Salary_DataSet" : [20000,25000,30000,35000,40000,45000,50000,55000,60000,65000,70000,100000],
    "Experience" : [2,3,4,5,6,7,8,9,10,11,15,20]
}

df = pd.DataFrame(Data)

# print(df.describe)


# Spliting the data 
x=df[['Experience']]
y=df['Salary_DataSet']
X_train , X_test , y_train , y_test = train_test_split(x , y , random_state=42 , test_size=.2)

model.fit(X_train , y_train)

i = int(input("Enter Your Experince : "))
pred_X = pd.DataFrame({"Experience": [i]})
pred_Sal=model.predict(pred_X[['Experience']])
print(f"Based on your Experience your Salary might be {pred_Sal[0]:,.2f} ")

