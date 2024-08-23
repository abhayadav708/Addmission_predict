import pandas as pd

data = pd.read_csv('Admission_Predict.csv')
print(data.head())
print(data.isnull().sum())

# Drop unnecessary columns if any
data.drop(columns=['Serial No.'], inplace=True)

# Split the data into features (X) and target (y)
X = data.drop(columns=['Chance of Admit '])
y = data['Chance of Admit ']

# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import matplotlib.pyplot as plt
import seaborn as sns

# Plot correlations
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()

# Visualize the distribution of CGPA vs. Chance of Admit
plt.scatter(data['CGPA'], data['Chance of Admit'])
plt.xlabel('CGPA')
plt.ylabel('Chance of Admit')
plt.title('CGPA vs. Chance of Admit')
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f'Root Mean Squared Error: {rmse}')
print(f'R^2 Score: {r2}')

new_student = [[330, 115, 4, 4.5, 4.5, 9.0, 1]]

# Predict the chance of admission
predicted_chance = model.predict(new_student)
print(f'Predicted Chance of Admission: {predicted_chance[0]:.2f}')

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    new_student = [[data['GRE'], data['TOEFL'], data['University Rating'], data['SOP'], data['LOR'], data['CGPA'], data['Research']]]
    prediction = model.predict(new_student)
    return jsonify({'Chance of Admit': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)