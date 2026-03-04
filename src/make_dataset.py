#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# src/data/make_dataset.py

import pandas as pd
import os

def load_german_credit_data(
    raw_path="Data/german.data",
    processed_path=None
):
    """
    Carga el dataset German Credit (UCI).
    
    Parameters:
    -----------
    raw_path : str
        Ruta al archivo raw (german.data)
    processed_path : str, optional
        Si se pasa, guarda el DataFrame limpio en CSV y lo retorna
    
    Returns:
    --------
    pd.DataFrame
        Dataset con columnas correctamente nombradas y Target mapeado a 0/1
    """
    column_names = [
        "Status_Checking_Account", "Duration_Months", "Credit_History", "Purpose",
        "Credit_Amount", "Savings_Account", "Employment_Since", "Installment_Rate",
        "Personal_Status_Sex", "Other_Debtors", "Residence_Since", "Property",
        "Age", "Other_Installment_Plans", "Housing", "Existing_Credits",
        "Job", "Number_of_Dependents", "Telephone", "Foreign_Worker", "Target"
    ]
    
    # Verificar que el archivo existe
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"No se encuentra el archivo: {raw_path}")
    
    df = pd.read_csv(raw_path, sep=r"\s+", header=None)
    df.columns = column_names
    
    # Mapear Target: 1 → 0 (good), 2 → 1 (bad) → estándar en credit scoring
    df["Target"] = df["Target"].map({1: 0, 2: 1})
    
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Opcional: guardar procesado
    if processed_path:
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)
        print(f"Guardado en: {processed_path}")
    
    return df

