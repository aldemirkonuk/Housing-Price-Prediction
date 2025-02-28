import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df= pd.read_csv('Housing_Prediction.csv',skiprows=1,sep=';')


# print(df.head())  # print first 5 rows of the dataframe
# print(df.tail())        # View the last 5 rows
# print(df.info())         # Summary of the DataFrame (column types, non-null values)
# print(df.describe())     # Summary statistics for numerical columns
# print(df.columns)        # List of column names
# print(df.shape)          # Get number of rows and columns
# Print all column names
# print(df.columns)
# print(df.head())

df['mainroad'] = df['mainroad'].map({'yes': 1, 'no': 0})
df['guestroom'] = df['guestroom'].map({'yes': 1, 'no': 0})
df['basement'] = df['basement'].map({'yes': 1, 'no': 0})
df['hotwaterheating'] = df['hotwaterheating'].map({'yes': 1, 'no': 0})
df['airconditioning'] = df['airconditioning'].map({'yes': 1, 'no': 0})
df['prefarea'] = df['prefarea'].map({'yes': 1, 'no': 0})

df['furnishingstatus'] = df['furnishingstatus'].map({'furnished': 1, 'semi-furnished': 2, 'unfurnished': 3})
# df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
X = df[features]
Y = df['price']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Print dataset sizes
print(f'Training data: {X_train.shape}')
print(f'Testing data: {X_test.shape}\n')
# print(Y_train.shape)
# print(Y_test.shape)

# Initialize the model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, Y_train)  # .fit(X_train, y_train) → This tells the model to learn the relationship between the features (X_train) and the house prices (y_train).

Y_pred = model.predict(X_test)

mse = mean_squared_error(Y_test, Y_pred) #lower it is, more accurate the model is
rmse= mse**0.5 
r2 = r2_score(Y_test, Y_pred) #closer to 1 the better., variance of the model


# import matplotlib.pyplot as plt
# import seaborn as sns
# # Scatter plot of Actual vs Predicted Prices
# plt.figure(figsize=(8, 5))
# sns.scatterplot(x=Y_test, y=Y_pred, alpha=0.7)
# plt.xlabel('Actual Prices')
# plt.ylabel('Predicted Prices')
# plt.title('Actual vs Predicted House Prices')
# plt.show()
