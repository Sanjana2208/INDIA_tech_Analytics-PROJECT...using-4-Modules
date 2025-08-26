import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


#📊 Module 1: Technological Growth Analysis Across Indian States
def tech_growth_analysis():
    print("\n📊 Technological Growth Analysis Across Indian States")
    data = {
        'State': ['Karnataka', 'Maharashtra', 'Tamil Nadu', 'Telangana', 'Kerala',
                  'Gujarat', 'Delhi', 'Punjab', 'Rajasthan', 'Uttar Pradesh',
                  'West Bengal', 'Bihar', 'Odisha', 'Assam', 'Jharkhand'],
        'Startups': [1200, 1100, 950, 1000, 800, 700, 850, 400, 300, 250, 500, 150, 200, 180, 160],
        'TechParks': [15, 14, 12, 13, 10, 9, 11, 5, 4, 3, 6, 2, 3, 2, 2],
        'R&D_Centers': [50, 48, 45, 47, 40, 35, 38, 20, 18, 15, 25, 10, 12, 11, 10],
        'Label': [1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]
    }
    df = pd.DataFrame(data)
    X = df[['Startups', 'TechParks', 'R&D_Centers']]
    y = df['Label']
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

#...................................................................................................................

#🚦 Module 2: Smart Traffic Management Using ML
def smart_traffic_management():
    print("\n🚦 Smart Traffic Management Using ML")
    np.random.seed(0)
    vehicle_count = np.random.randint(50, 500, 100)
    avg_speed = np.random.randint(20, 80, 100)
    density = ['Low' if v < 150 else 'Medium' if v < 300 else 'High' for v in vehicle_count]
    df = pd.DataFrame({
        'VehicleCount': vehicle_count,
        'AvgSpeed': avg_speed,
        'Density': density
    })
    df['DensityLabel'] = df['Density'].map({'Low': 0, 'Medium': 1, 'High': 2})
    X = df[['VehicleCount', 'AvgSpeed']]
    y = df['DensityLabel']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
    plt.title("Traffic Density Confusion Matrix")
    plt.show()

#...............................................................................................................................

#🗑️ Module 3: Garbage Level Monitoring System
def garbage_monitoring():
    print("\n🗑️ Garbage Level Monitoring System")
    np.random.seed(1)
    distance = np.random.uniform(5, 50, 100)
    status = ['Full' if d < 15 else 'Half' if d < 30 else 'Empty' for d in distance]
    df = pd.DataFrame({
        'Distance': distance,
        'Status': status
    })
    df['StatusLabel'] = df['Status'].map({'Empty': 0, 'Half': 1, 'Full': 2})
    X = df[['Distance']]
    y = df['StatusLabel']
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

#.............................................................................................................


#❤️ Module 4: Personal Health Monitoring via Smartphone Sensors
def health_monitoring():
    print("\n❤️ Personal Health Monitoring Using Smartphone Sensors")
    np.random.seed(2)
    heart_rate = np.random.randint(60, 150, 100)
    motion_level = np.random.uniform(0.1, 2.0, 100)
    status = ['Normal' if hr < 90 else 'Warning' if hr < 120 else 'Critical' for hr in heart_rate]
    df = pd.DataFrame({
        'HeartRate': heart_rate,
        'MotionLevel': motion_level,
        'Status': status
    })
    df['StatusLabel'] = df['Status'].map({'Normal': 0, 'Warning': 1, 'Critical': 2})
    X = df[['HeartRate', 'MotionLevel']]
    y = df['StatusLabel']
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    model = KNeighborsClassifier(n_neighbors=4)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

#........................................................................................................................................

def main_menu():
    while True:
        print("\n📌 Select a Module to Run:")
        print("1. Technological Growth Analysis")
        print("2. Smart Traffic Management")
        print("3. Garbage Level Monitoring")
        print("4. Personal Health Monitoring")
        print("5. Exit")
        choice = input("Enter your choice (1-5): ")
        if choice == '1':
            tech_growth_analysis()
        elif choice == '2':
            smart_traffic_management()
        elif choice == '3':
            garbage_monitoring()
        elif choice == '4':
            health_monitoring()
        elif choice == '5':
            print("Exiting... 🚪")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main_menu()

#............................................................................................................................