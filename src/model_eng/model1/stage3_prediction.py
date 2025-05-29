import joblib
import numpy as np
import pandas as pd
import os


def get_latest_model_path(model_dir="models", filename="model_rf.pkl"):
    folders = sorted(os.listdir(model_dir), reverse=True)
    for folder in folders:
        full_path = os.path.join(model_dir, folder, filename)
        if os.path.isfile(full_path):
            return full_path
    raise FileNotFoundError("No trained model found.")

model_path = get_latest_model_path()
print(model_path)
model = joblib.load(model_path)

# Datos de prueba como diccionario
sample_data = {
    "po_/_so_#": 0.00,
    "asn/dn_#": 0.00,
    "country": 38.00,
    "fulfill_via": 0.00,
    "vendor_inco_term": 5.00,
    "sub_classification": 5.00,
    "unit_of_measure_(per_pack)": 240.00,
    "line_item_quantity": 1000.00,
    "pack_price": 6.2,
    "unit_price": 0.03,
    "first_line_designation": 1.00,
    "freight_cost_(usd)": 4521.5,
    "shipment_mode": 0.00,
    "line_item_insurance_(usd)": 47.04,
    "days_to_process": -930
}

# Crear DataFrame de prueba
sample_df = pd.DataFrame([sample_data])

output = model.predict(sample_df)

print(output)
