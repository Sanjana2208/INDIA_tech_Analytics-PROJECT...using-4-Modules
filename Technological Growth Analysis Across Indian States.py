import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Simulated data for 15 states
data = {
    'State': ['Karnataka', 'Maharashtra', 'Tamil Nadu', 'Telangana', 'Kerala',
              'Gujarat', 'Delhi', 'Punjab', 'Rajasthan', 'Uttar Pradesh',
              'West Bengal', 'Bihar', 'Odisha', 'Assam', 'Jharkhand'],
    'Startups': [1200, 1100, 950, 1000, 800, 700, 850, 400, 300, 250, 500, 150, 200, 180, 160],
    'TechParks': [15, 14, 12, 13, 10, 9, 11, 5, 4, 3, 6, 2, 3, 2, 2],
    'R&D_Centers': [50, 48, 45, 47, 40, 35, 38, 20, 18, 15, 25, 10, 12, 11, 10],
    'Label': [1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]  # 1 = High Growth, 0 = Low Growth
}

df = pd.DataFrame(data)

# Features and labels
X = df[['Startups', 'TechParks', 'R&D_Centers']]
y = df['Label']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# KNN Model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
