import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import dice_ml
from dice_ml import Dice

#read data
df = pd.read_csv("loan_sanction_train.csv", sep = ";")
df = df.drop(columns=["Loan_ID"])

#Select numeric and categorical columns
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = df.select_dtypes(include=[object]).columns.tolist()

#Fill in missing values
for col in numeric_columns:
    df[col].fillna(df[col].mean(), inplace=True)
for col in categorical_columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

#Split data into feature and target variables
X= df.drop(columns=["Loan_Status"])
y= df["Loan_Status"]

#Convert target variable to binary
y = y.map({"N": 0, "Y": 1})

#Convert categorical to numerical using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train randomforest classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

#Predict on test set and print accuracy score
y_pred_rf = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(rf_accuracy)

#Print feature importances and feature names
print(rf.feature_names_in_)
print(rf.feature_importances_)

#check for rejected instances
rejected_indexes = np.where(y_pred_rf == "N")[0]
print("Rejected indexes:", rejected_indexes)

#Concatenate X and y for DICE
df_copy = pd.concat([X,y], axis=1)

#Create DICE object
d = dice_ml.Data(
    dataframe=df_copy,
    continuous_features=numeric_columns,
    outcome_name= "Loan_Status",
)
#Create model object
m = dice_ml.Model(model=rf, backend="sklearn")
exp = Dice(d, m)

#Choose index for rejected instances
i = np.where(rf.predict(X_test) == 0)
query_instance = X_test.iloc[i]

#Print query instance
dice_exp = exp.generate_counterfactuals(query_instance,
                                        total_CFs=3,
                                        desired_class="opposite")

pd.set_option('display.max_columns', None)
#Visualize the counterfactuals as a dataframe
dice_exp.visualize_as_dataframe()