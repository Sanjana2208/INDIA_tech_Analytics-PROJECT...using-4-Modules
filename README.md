📊 Module 1: Technological Growth Analysis Across Indian States


python
import pandas as pd
•	✅ Why: Loads and manages tabular data using DataFrames.
•	📌 Used for: Creating and manipulating the dataset of Indian states.


python
from sklearn.preprocessing import StandardScaler
•	✅ Why: Normalizes features to have mean = 0 and standard deviation = 1.
•	📌 Used for: Ensuring fair distance calculations in KNN.


python
from sklearn.neighbors import KNeighborsClassifier
•	✅ Why: Imports the KNN algorithm.
•	📌 Used for: Classifying states into High or Low tech growth.


python
from sklearn.model_selection import train_test_split
•	✅ Why: Splits data into training and testing sets.
•	📌 Used for: Evaluating model performance on unseen data.

python
from sklearn.metrics import accuracy_score
•	✅ Why: Measures how many predictions were correct.
•	📌 Used for: Evaluating model accuracy.

python
data = {
'State': [...],
'Startups': [...],
'TechParks': [...],
'R&D_Centers': [...],
'Label': [...]
}
•	✅ Why: Simulates real-world tech data for 15 Indian states.
•	📌 Used for: Providing input features and target labels.

python
df = pd.DataFrame(data)
•	✅ Why: Converts dictionary to a structured DataFrame.
•	📌 Used for: Easier data handling and analysis.

python
X = df[['Startups', 'TechParks', 'R&D_Centers']]
y = df['Label']
•	✅ Why: Separates features (X) and target (y).
•	📌 Used for: Training the model with inputs and expected outputs.

python
X_scaled = StandardScaler().fit_transform(X)
•	✅ Why: Scales features to equalize their influence.
•	📌 Used for: Improving KNN performance.

python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)
•	✅ Why: Splits data into 70% training and 30% testing.
•	📌 Used for: Training and validating the model.

python
model = KNeighborsClassifier(n_neighbors=3)
•	✅ Why: Initializes KNN with 3 neighbors.
•	📌 Used for: Classifying based on majority vote of nearest neighbors.

python
model.fit(X_train, y_train)
•	✅ Why: Trains the model using training data.
•	📌 Used for: Learning patterns from known labels.

python
y_pred = model.predict(X_test)
•	✅ Why: Predicts labels for test data.
•	📌 Used for: Evaluating model performance.

python
print("Tech Growth Accuracy:", accuracy_score(y_test, y_pred))
•	✅ Why: Prints accuracy score.
•	📌 Used for: Showing how well the model performed.


