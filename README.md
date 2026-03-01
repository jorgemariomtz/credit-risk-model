# Credit Risk Modeling – Probability of Default Prediction

Credit Risk Modeling – German Credit Dataset

1. Business Problem

Desarrollar un modelo predictivo para estimar la probabilidad de incumplimiento (default) y soportar decisiones de aprobación de crédito, alineando el desempeño estadístico con la función de pérdida del negocio.

El foco principal es minimizar pérdidas financieras derivadas de falsos negativos (clientes riesgosos aprobados).

2. Dataset

*   1000 observaciones.
*	Tasa de default ≈ 30%.
*	Variables numéricas y categóricas.
*	Colinealidad parcial entre duración y monto del crédito ($~0.62$).
*	Dataset moderadamente desbalanceado.


4. Exploratory Data Analysis (EDA)
   
Principales hallazgos:

* Préstamos de mayor duración presentan tasas de default superiores al $50%$.
* El historial crediticio muestra alto poder discriminatorio ($57%$ vs $17%$).
* El propósito del crédito contribuye de forma moderada al riesgo.
* El riesgo es multivariado; la relación marginal no explica completamente el comportamiento del target.


4. Model Comparison

Se evaluaron tres modelos:   

* Logistic Regression
*	Random Forest
*	XGBoost

AUC Comparison

| Model                | AUC          |
|----------------------|-------------|
| Logistic Regression  | ~0.80       |
| Random Forest        | ~0.79       |
| XGBoost              | Similar rango |


Logistic mostró mejor estabilidad y desempeño consistente.


5. Threshold Optimization

Se implementó función de costo:

Cost = 5×FN + FP

Comparación de políticas

| Model              | Threshold | FN | FP  | Recall (Default) | Cost |
|--------------------|----------|----|-----|------------------|------|
| Logistic (0.5)     | 0.50     | 43 | 23  | 0.52             | 238  |
| Logistic (F1)      | 0.304    | 21 | 53  | 0.77             | 158  |
| Logistic (Cost)    | 0.228    | 14 | 74  | 0.84             | 144  |
| XGBoost (Cost)     | 0.105    | 9  | 100 | 0.90             | 145  |


El threshold optimizado por costo minimiza pérdida esperada


6. Interpretability

La regresión logística permite interpretar drivers de riesgo mediante Odds Ratios.

Principales variables:

*	Property_A124 ($OR ≈ 2.27$)
*	Purpose_A46 ($OR ≈ 1.97$)
*	Duration_Months ($OR ≈ 1.66$)
*	Installment_Rate ($OR ≈ 1.33$)

Estos resultados son consistentes con teoría financiera de riesgo.

7. Final Recommendation

Se recomienda utilizar Logistic Regression con threshold optimizado por costo debido a:

*	AUC sólido (~0.80)
*	Reducción significativa de falsos negativos
*	Costo esperado mínimo
*	Alta interpretabilidad
*	Adecuación a contextos regulatorios

8. Business Implications

Reducir falsos negativos disminuye pérdidas financieras esperadas, aunque incrementa el rechazo de clientes solventes.

La selección final del threshold debe alinearse con el apetito de riesgo y estrategia comercial.

9. Reproducibility:

```bash
pip install -r requirements.txt
jupyter notebook
```

11. Future Improvements
    
*	Cross-validation con optimización de threshold.
*	Calibración de probabilidades.
*	Tuning avanzado de XGBoost.
*	Análisis de estabilidad temporal.
*	Backtesting en múltiples muestras.

