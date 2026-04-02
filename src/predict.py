import joblib
import numpy as np

model=joblib.load("models/wine_model.pkl")

sample = np.array([[11.2,0.28,0.56,1.9,0.075,17.0,60.0,0.998,3.16,0.58,9.8,3]])
predictions=model.predict(sample)
print("Predicted output: ",predictions)