
# **California Housing Optimization with Optuna and XGBoost**

Este proyecto implementa un modelo de machine learning utilizando **XGBoost** para predecir el precio medio de viviendas en California, optimizando los hiperparámetros con **Optuna** y registrando los resultados en **MLflow**. Este documento incluye la descripción del proyecto, los pasos para ejecutarlo, y las conclusiones derivadas del análisis.

---

## **Estructura del Proyecto**

```
california_housing_optuna/
├── california_housing_optuna/
│   ├── __init__.py        # Archivo de inicialización
│   ├── main.py            # Código principal
├── tests/
│   ├── __init__.py        # Archivo de inicialización para pruebas
├── pyproject.toml         # Configuración del entorno de Poetry
└── README.md              # Documentación general
```

---

## **Requisitos**

- **Python 3.8 o superior**
- **Poetry** para la gestión del entorno virtual

### **Instalación de Dependencias**
1. Instala **Poetry** si aún no lo tienes:
   ```bash
   pip install poetry
   ```

2. Instala las dependencias del proyecto:
   ```bash
   poetry install
   ```

---

## **Ejecución del Proyecto**

1. Activa el entorno virtual de Poetry:
   ```bash
   poetry shell
   ```

2. Ejecuta el archivo principal:
   ```bash
   poetry run python california_housing_optuna/main.py
   ```

---

## **Librerías Usadas**

- **Optuna**: Optimización de hiperparámetros.
- **XGBoost**: Modelo basado en boosting para predicción.
- **Scikit-learn**: División de datos y cálculo de métricas.
- **MLflow**: Registro de experimentos y modelos.
- **Matplotlib** y **Seaborn**: Visualización de datos.

---

## **Flujo del Proyecto**

1. **Carga de Datos**:
   - El dataset `California Housing` es cargado desde `sklearn`.
   - Los datos se dividen en conjuntos de entrenamiento (80%) y prueba (20%).

2. **Optimización con Optuna**:
   - Se optimizan los hiperparámetros del modelo **XGBoost** en 50 combinaciones mediante búsqueda aleatoria.

3. **Entrenamiento y Evaluación**:
   - Se entrena el modelo con los mejores hiperparámetros encontrados.
   - Se evalúan las métricas clave: **RMSE** y **R²**.

4. **Registro con MLflow**:
   - Se registran los parámetros, métricas y el modelo final en MLflow para trazabilidad.

---

## **Resultados Finales**

### **Mejores Hiperparámetros Encontrados**
```plaintext
{'max_depth': 9, 'learning_rate': 0.13, 'n_estimators': 272, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 6.66, 'reg_lambda': 7.90}
```

### **Métricas del Modelo Final**
| Métrica           | Entrenamiento | Prueba     |
|-------------------|---------------|------------|
| **RMSE**          | `0.2741`      | `0.4390`   |
| **R²**            | `97.15%`      | `85.29%`   |

---

## **Interpretación**

1. **Regularización Efectiva**:
   - Los valores altos de `reg_alpha` y `reg_lambda` (6.66 y 7.90 respectivamente) ayudaron a controlar el sobreajuste.

2. **Capacidad Predictiva**:
   - El **R² de prueba** indica que el modelo explica el 85.29% de la varianza en los datos no vistos, lo que es un excelente desempeño.

3. **Optimización con Optuna**:
   - Optuna permitió encontrar una configuración óptima de hiperparámetros en solo 50 pruebas.

4. **Evaluación General**:
   - El modelo generaliza bien, con un error promedio bajo en la predicción de precios.

---

## **Gráficos de Resultados**

El proyecto incluye gráficos de:
- **Distribución de datos**: Para validar la calidad y normalización del dataset.
- **Residuos del modelo**: Para evaluar errores en las predicciones.
- **Línea ideal vs Predicciones**: Para visualizar la cercanía entre predicciones y valores reales.

---

## **Conclusiones**

1. **Uso de Optuna**:
   - La optimización automática con Optuna simplifica la selección de hiperparámetros, mejorando el desempeño del modelo en menos tiempo.

2. **Registro con MLflow**:
   - Permite reproducibilidad y facilita la comparación de experimentos.

3. **Rendimiento del Modelo**:
   - El RMSE de prueba de `0.4390` y un R² de `85.29%` muestran que el modelo es preciso y generaliza bien en datos no vistos.

4. **Mejoras Futuras**:
   - Probar técnicas de ensamblado (stacking o blending).
   - Incorporar validación cruzada para mejorar la robustez del modelo.
   - Experimentar con características adicionales derivadas de datos externos.

---

## **Exploración con MLflow**

1. Inicia el servidor de MLflow:
   ```bash
   mlflow ui
   ```

2. Accede al dashboard:
   - URL: [http://127.0.0.1:5000](http://127.0.0.1:5000)

3. Explora los experimentos registrados para comparar métricas y parámetros.

---

## **Siguientes Pasos**
1. **Validación Adicional**:
   - Implementar validación cruzada para garantizar que el modelo generalice mejor.
   
2. **Comparación de Modelos**:
   - Implementar otros algoritmos como Gradient Boosting o Random Forest.

3. **Despliegue en Producción**:
   - Usar el modelo registrado en MLflow para implementar un sistema de predicción en tiempo real.

Este proyecto es un ejemplo sólido de cómo usar herramientas modernas como **Optuna** y **MLflow** para resolver problemas complejos de machine learning. 🚀
