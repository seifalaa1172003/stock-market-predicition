import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    # Fill missing dates with the last available date
    df['Date'] = pd.to_datetime(df['Date']).ffill()
    
    # Fill missing numeric values with the mean of the respective column
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Encode the price movements into numerical labels
    label_encoder = LabelEncoder()
    df['Price Movement '] = label_encoder.fit_transform(df['Price Movement '])
    
    # Split the dataframe into features (X) and labels (y)
    X = df[['AMZN', 'DPZ', 'BTC', 'NFLX']]
    y = df['Price Movement ']

    # Standardize the features
    feature_names = X.columns  
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

    # Extract the dates (not used in modeling)
    dates = df['Date']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=21)
    
    return X_train, X_test, y_train, y_test, label_encoder, scaler


# Loading the data
df = pd.read_csv('stock.csv')

# Preprocessing data
X_train, X_test, y_train, y_test, label_encoder, scaler = preprocess_data(df)

# Initializing and training Logistic Regression model
lr_model = LogisticRegression(solver='liblinear', C=10, random_state=0)
lr_model.fit(X_train, y_train)

# Initializing and training Random Forest model
rf_model = RandomForestClassifier(n_estimators=5, max_depth=2,
                                             random_state=0)
rf_model.fit(X_train, y_train)

# Initializing and training Decision Tree model
dt_model = DecisionTreeClassifier(criterion='gini', random_state=21, max_depth=5, max_leaf_nodes=10, ccp_alpha=0)
dt_model.fit(X_train, y_train)

# Initializing and training SVM model
svm_model = SVC(random_state=0)
svm_model.fit(X_train, y_train)

# Return models and data preprocessing components
def get_models():
    return lr_model, rf_model, dt_model, svm_model

def get_preprocessing_components():
    return X_train, X_test, y_train, y_test, label_encoder, scaler
