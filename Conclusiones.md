# **Conclusiones del Proyecto**

Este documento resume los hallazgos y aprendizajes clave obtenidos del desarrollo y ejecución del proyecto **Predicción de Precios de Viviendas en California con XGBoost y Optuna**.

---

## **Resultados del Modelo**
El modelo optimizado con Optuna y registrado en MLflow obtuvo los siguientes resultados:

| **Métrica**        | **Entrenamiento** | **Prueba**      |
|--------------------|------------------|----------------|
| **RMSE**           | 0.2741           | 0.4390         |
| **R²**             | -                | 0.8529         |

| **Mejores Hiperparámetros** |
|-----------------------------|
| `max_depth=9`              |
| `learning_rate=0.13`       |
| `n_estimators=272`         |
| `subsample=0.8`            |
| `colsample_bytree=0.8`     |
| `reg_alpha=6.66`           |
| `reg_lambda=7.90`          |

---

## **Interpretación de Resultados**
1. **Capacidad Predictiva**:
   - El modelo explica el **85.29% de la varianza** en los precios de vivienda del conjunto de prueba, lo que demuestra un desempeño sólido.
   - El RMSE en prueba de `0.4390` indica un error promedio moderado en las predicciones.

2. **Equilibrio de Desempeño**:
   - La diferencia entre el RMSE en entrenamiento (`0.2741`) y en prueba (`0.4390`) refleja un modelo bien ajustado, con un leve sobreajuste controlado gracias a la regularización (`reg_alpha`, `reg_lambda`).

3. **Impacto de la Optimización**:
   - El uso de **Optuna** permitió encontrar combinaciones óptimas de hiperparámetros de manera eficiente, maximizando el desempeño del modelo.

---

## **Consideraciones**
1. **Regularización**:
   - Los parámetros `reg_alpha` y `reg_lambda` desempeñaron un papel crucial para evitar sobreajuste en un modelo con profundidad máxima de 9.

2. **Dataset y Preprocesamiento**:
   - El dataset California Housing está bien estructurado, pero la normalización y manejo de outliers contribuyeron a mejorar el rendimiento del modelo.

3. **Trazabilidad**:
   - Con **MLflow**, el proyecto aseguró reproducibilidad al registrar todos los hiperparámetros, métricas y el modelo final, facilitando el análisis y la comparación.

---

## **Lecciones Aprendidas**
1. **Optimización Inteligente**:
   - El uso de **Optuna** para la optimización de hiperparámetros reduce significativamente los tiempos de experimentación y mejora los resultados.

2. **Importancia de la Regularización**:
   - Modelos complejos como XGBoost requieren parámetros de regularización adecuados para evitar sobreajuste, especialmente en datasets de alta dimensionalidad.

3. **Registro de Modelos**:
   - Herramientas como MLflow son indispensables para asegurar la trazabilidad y evaluar la efectividad de los modelos en entornos profesionales.

---

**Conclusión General**:
El modelo XGBoost optimizado con Optuna demostró ser una herramienta poderosa para resolver problemas de predicción de precios de viviendas, con un balance ideal entre sesgo y varianza.
