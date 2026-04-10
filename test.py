print("Test script running")
import pandas as pd
print("pandas imported")
df = pd.read_csv(r'c:\Users\deepak thayat\Desktop\Project\Data\processed_data.csv')
print("Data loaded, shape:", df.shape)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
print("sklearn imported")

features = ['RPM', 'Feed_mm_per_sec', 'DOC_mm']
targets = ['Tool_Life_HSS_min', 'Max_Force_X', 'Max_Force_Y', 'Max_Force_Z']
X = df[features]
y = df[targets]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split")

model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=10, random_state=42))  # small n for test
model.fit(X_train, y_train)
print("Model fitted")
print("Test completed")