from data_loader import load_data
from preprocess import preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

model=RandomForestClassifier()

df=load_data("data\\raw\\WineQT.csv")
X_train,X_test,y_train, y_test=preprocess_data(df)

model.fit(X_train,y_train)

predictions=model.predict(X_test)

accuracy=accuracy_score(y_test,predictions)
print(f"Accuracy:  {accuracy}")

joblib.dump(model,"models/wine_model.pkl")