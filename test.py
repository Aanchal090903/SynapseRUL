import pickle
xgb_model = pickle.load(open("xgb_shap.pkl", "rb"))

print("📋 Feature Names in SHAP Model:")
print(xgb_model.get_booster().feature_names)
