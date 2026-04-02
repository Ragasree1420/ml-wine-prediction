import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    df['quality_binary']=(df['quality']>=7).astype(int)
    X=df.drop(['quality','quality_binary'],axis=1)
    y=df['quality_binary']
    X_train,X_test, y_train, y_test=train_test_split(X,y,random_state=42,test_size=0.2)
    return X_train,X_test,y_train, y_test