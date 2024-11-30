import optuna
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def load_data():
    """Carga el dataset California Housing y división en datos de entrenamiento y prueba."""
    data = fetch_california_housing()
    X, y = data.data, data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    """Función objetivo para optimizar los hiperparámetros de XGBoost."""
    param = {
        "verbosity": 0,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, step=0.01),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
    }

    # Entrenar el modelo
    model = xgb.XGBRegressor(**param, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Métrica a minimizar
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse

if __name__ == "__main__":
    # Cargar datos
    X_train, X_test, y_train, y_test = load_data()

    # Crear estudio de Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    # Mejores hiperparámetros
    print("Mejores hiperparámetros:", study.best_params)

    # Entrenar modelo final
    best_params = study.best_params
    best_model = xgb.XGBRegressor(**best_params, random_state=42)
    best_model.fit(X_train, y_train)

    # Evaluar modelo final
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2 = r2_score(y_test, y_test_pred)

    print(f"RMSE Entrenamiento: {train_rmse}")
    print(f"RMSE Prueba: {test_rmse}")
    print(f"R² Prueba: {r2}")
