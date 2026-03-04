# Credit Risk Modeling: Probability of Default Prediction  
**German Credit Dataset (UCI Machine Learning Repository)**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter Notebook](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=jupyter)](https://jupyter.org/)

**Proyecto de portafolio** – Transición de docencia (6 años) a analista de datos. Aplicación de estadística aplicada y business intelligence para resolver un problema real de banca: predecir incumplimiento crediticio minimizando pérdidas financieras.

## 1. Business Problem
Desarrollar un modelo de **Probability of Default (PD)** para apoyar decisiones de aprobación de crédito.  
**Objetivo principal**: Minimizar **pérdidas por falsos negativos** (aprobar clientes riesgosos), alineando el modelo con la función de pérdida del negocio (cost matrix: 5×FN + 1×FP).

**Contexto**: Dataset clásico de riesgo crediticio (1000 observaciones, ~30% default rate). Muy relevante en finanzas, fintech y regulación (e.g., APRA en Australia).

## 2. Dataset
- Fuente: [UCI Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)  
- 1000 instancias, 20 atributos (7 numéricos, 13 categóricos).  
- Target: 1 = Good credit (70%), 2 = Bad credit (30%) → imbalance moderado.  
- Sin valores faltantes.  
- Colinealidad notable: duración y monto del crédito (corr ≈ 0.62).

**Ejemplos de variables clave** (códigos originales mapeados):
- `checking_status`: A11 (< 0 DM), A12 (0–200 DM), A13 (≥200 DM), A14 (sin cuenta).  
- `credit_history`: A30 (sin créditos), A34 (buena historia).  
- `purpose`: A40 (auto nuevo), A43 (TV/radio), A46 (educación), etc.  
- `duration_months`, `credit_amount`, `age`, etc. 
  
## 3. Exploratory Data Analysis (EDA) – Hallazgos clave
- Préstamos de **mayor duración** (>36 meses) → default >50%.  
- **Historial crediticio** tiene alto poder predictivo (e.g., A34 bueno vs A33/A32 peor).  
- Propósitos como **educación (A46)** y **mantenimiento** elevan riesgo moderadamente.  
- Relaciones multivariadas dominan → la relación marginal no explica completamente el comportamiento del target.

   ## ROC Curve Comparison

![Histogramas](figures/hist.png)

<img width="980" height="645" alt="image" src="https://github.com/user-attachments/assets/07753e0e-80f5-4517-b6b8-e18edff76fea" />

<img width="980" height="645" alt="image" src="https://github.com/user-attachments/assets/0858dbc6-1f53-43c9-adca-71c944cd3254" />

## 4. Model Comparison
Modelos evaluados (train-test split + cross-validation estratificada recomendada):

| Modelo              | AUC   | Recall (Default) @0.5 | Notas                              |
|---------------------|-------|-----------------------|------------------------------------|
| Logistic Regression | ~0.80 | ~0.52                 | Mejor estabilidad e interpretabilidad |
| Random Forest       | ~0.79 | –                     | Buen desempeño, pero menos interpretable |
| XGBoost             | ~0.80 | –                     | Similar rango, más sensible a hiperparámetros |

**Logistic Regression** destaca por consistencia y explicabilidad (ideal para banca regulada).

Logistic mostró mejor estabilidad y desempeño consistente.


## 5. Threshold Optimization & Cost-Sensitive Learning
Función de costo: **Cost = 5 × FN + 1 × FP** (basado en UCI cost matrix).

| Modelo              | Threshold | FN   | FP   | Recall (Default) | Cost Total |
|---------------------|-----------|------|------|------------------|------------|
| Logistic (default 0.5) | 0.50    | 43   | 23   | 0.52             | 238        |
| Logistic (F1-opt)   | 0.30      | 42   | 15   | 0.77             | 158        |
| Logistic (Cost-opt) | 0.22      | 14   | 74   | 0.84             | 144        |
| XGBoost (Cost-opt)  | 0.10      | 5    | 910  | 0.90             | 145        |

**Hallazgo**: Threshold ~0.22 minimiza costo esperado → reduce FN drásticamente (mejor alineación


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

