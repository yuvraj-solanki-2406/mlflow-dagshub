from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import dagshub
import mlflow
import mlflow.sklearn
import mlflow.xgboost

# initilaize DagsHub
dagshub.init(repo_owner='yuvraj-solanki-2406', repo_name='my-first-repo', mlflow=True)


X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest
params = {"max_depth": 4, "random_state": 42}
model = RandomForestRegressor(**params)
model.fit(X_train, y_train)

# SVR
params2 = {"gamma": "scale", "C": 3}
model2 = SVR(**params2)
model2.fit(X_train, y_train)

# Xgboost
params3 = {"booster": "gbtree"}
model3 = XGBRegressor(**params3)
model3.fit(X_train, y_train)

# Random forest prediction
y_pred = model.predict(X_test)
# SVR predictions
y_pred2 = model2.predict(X_test)
# XGBoostRegressor predictions
y_pred3 = model3.predict(X_test)

print("Error in Random Forest", mean_squared_error(y_pred, y_test))
print("Error in SVR", mean_squared_error(y_pred2, y_test))
print("Error in Xgboost", mean_squared_error(y_pred3, y_test))


# configure ml flow
with mlflow.start_run() as run:
    # Log parameters and metrics using the MLflow APIs
    mlflow.log_params(params)
    mlflow.log_params(params2)
    mlflow.log_params(params3)

    # Random forest error metrics
    mlflow.log_metrics({"mse": mean_squared_error(y_test, y_pred)})
    mlflow.log_metrics({"rmse": mean_squared_error(y_test, y_pred)})
    
    # SVR error metrics
    mlflow.log_metrics(
        {
            "mse": mean_squared_error(y_test, y_pred2),
            "rmse": mean_squared_error(y_test, y_pred2)
        }
    )
    
    # XGBoostRegressor error metrics
    mlflow.log_metrics({"mse": mean_squared_error(y_test, y_pred3)})
    mlflow.log_metrics({"rmse": mean_squared_error(y_test, y_pred3)})


    # Log the sklearn model and register as version 1
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="sklearn-model",
        input_example=X_train,
        registered_model_name="sk-learn-random-forest-reg-model",
    )
    
    mlflow.sklearn.log_model(
        sk_model=model2,
        artifact_path="sklearn-model-2",
        input_example=X_train,
        registered_model_name="sk-learn-svr-reg-model",
    )

    mlflow.xgboost.log_model(
        xgb_model=model3,
        artifact_path="xgboost-model",
        input_example=X_train,
        registered_model_name="xgboost-reg-model",
    )
