# Credit Risk Modeling – Probability of Default Prediction

Credit Risk Modeling – German Credit Dataset

1. Business Problem

Desarrollar un modelo predictivo para estimar la probabilidad de incumplimiento (default) y soportar decisiones de aprobación de crédito, alineando el desempeño estadístico con la función de pérdida del negocio.

El foco principal es minimizar pérdidas financieras derivadas de falsos negativos (clientes riesgosos aprobados).

2. Dataset

	*   1000 observaciones.
	*	Tasa de default ≈ 30%.
	*	Variables numéricas y categóricas.
	*	Colinealidad parcial entre duración y monto del crédito (~0.62).
	*	Dataset moderadamente desbalanceado.


4. Exploratory Data Analysis (EDA)

Principales hallazgos:   
	*	Préstamos de mayor duración presentan tasas de default superiores al 50%.
	*	El historial crediticio muestra alto poder discriminatorio (57% vs 17%).
	*	El propósito del crédito contribuye de forma moderada al riesgo.
	*	El riesgo es multivariado; la relación marginal no explica completamente el comportamiento del target.


4. Model Comparison

Se evaluaron tres modelos:   
	*	Logistic Regression
	*	Random Forest
	*	XGBoost

AUC Comparison
