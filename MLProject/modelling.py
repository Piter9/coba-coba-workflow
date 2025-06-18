import pandas as pd
import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Set tracking URI agar menyimpan lokal ke folder mlruns
mlflow.set_tracking_uri("file:./mlruns")

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True)
args = parser.parse_args()

# Load data
data = pd.read_csv(args.data_path)

# Fitur dan target
X = data.drop("stroke", axis=1)
y = data["stroke"]

# Bagi data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Inisialisasi model dan grid search
model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy')
# Mulai tracking

with mlflow.start_run():
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Logging manual parameter terbaik
    best_params = grid_search.best_params_
    mlflow.log_params(best_params)

    # Prediksi dan hitung metrik
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    # Logging manual metrik
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]))
    mlflow.log_param("model", "RandomForestClassifier")
  

    # Simpan model ke mlruns/0/<run_id>/artifacts/model
    mlflow.sklearn.log_model(
    sk_model=best_model,
    artifact_path="model",
    input_example=X_test.iloc[:5],  # Contoh input
    signature=mlflow.models.infer_signature(X_test, best_model.predict(X_train)) 
)
