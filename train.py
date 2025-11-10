

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pickle, warnings, os
warnings.filterwarnings("ignore")

print("\n🚚 Predictive Maintenance Model Training Started...\n")

# ----------------------------------------------------------
# STEP 1: Load dataset
# ----------------------------------------------------------
df = pd.read_csv("engine_data2.csv")
print(f"✅ Dataset loaded successfully! Total Records: {len(df)}")

target_col = "Engine Condition"
feature_cols = [c for c in df.columns if c != target_col]

X, y = df[feature_cols], df[target_col]

# ----------------------------------------------------------
# STEP 2: Simulate real-world noise for robustness
# ----------------------------------------------------------
noise_fraction = 0.15   # mild sensor drift
flip_fraction = 0.03    # mislabeled data fraction

num_noisy = int(len(X) * noise_fraction)
noise_idx = np.random.choice(X.index, num_noisy, replace=False)
X.loc[noise_idx] += np.random.normal(0, 0.25, X.loc[noise_idx].shape)

flip_idx = np.random.choice(y.index, int(len(y) * flip_fraction), replace=False)
y.loc[flip_idx] = 1 - y.loc[flip_idx]

# ----------------------------------------------------------
# STEP 3: Scale features
# ----------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------------------------
# STEP 4: Split dataset
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42
)

# ----------------------------------------------------------
# STEP 5: Define Models
# ----------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=70, max_depth=4, min_samples_split=12, min_samples_leaf=8,
    max_features=0.5, random_state=42, class_weight='balanced_subsample'
)

xgb = XGBClassifier(
    n_estimators=70, learning_rate=0.15, max_depth=3,
    subsample=0.6, colsample_bytree=0.5, reg_lambda=4, reg_alpha=3,
    random_state=42, eval_metric='logloss', use_label_encoder=False
)

# ----------------------------------------------------------
# STEP 6: Train Base Models Individually
# ----------------------------------------------------------
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

print("✅ Base models (RF, XGB) trained successfully!")

# ----------------------------------------------------------
# STEP 7: Ensemble (Soft Voting)
# ----------------------------------------------------------
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb)],
    voting='soft', weights=[1, 1]
)
ensemble.fit(X_train, y_train)
print("✅ Ensemble model training completed successfully!")

# ----------------------------------------------------------
# STEP 8: Evaluate Performance
# ----------------------------------------------------------
y_pred = ensemble.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Test Accuracy: {acc*100:.2f}%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cv = cross_val_score(ensemble, X_scaled, y, cv=5)
print(f"\n🔁 Cross-Validation Accuracy: {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")

# ----------------------------------------------------------
# STEP 9: Feature Importance Analysis
# ----------------------------------------------------------
rf_importance = rf.feature_importances_
xgb_importance = xgb.feature_importances_
avg_importance = (rf_importance + xgb_importance) / 2

importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": avg_importance
}).sort_values(by="Importance", ascending=False)

os.makedirs("artifacts", exist_ok=True)
importance_df.to_csv("artifacts/feature_importance.csv", index=False)

# --- Plot Feature Importance ---
plt.figure(figsize=(8, 5))
plt.barh(importance_df["Feature"], importance_df["Importance"], color="teal")
plt.title("Average Feature Importance (RF + XGB)")
plt.xlabel("Relative Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("artifacts/feature_importance.png")
plt.close()
print("📊 Feature importance chart saved to artifacts/feature_importance.png")

# ----------------------------------------------------------
# STEP 10: Save Model Artifacts
# ----------------------------------------------------------
with open("artifacts/model.pkl", "wb") as f:
    pickle.dump(ensemble, f)
with open("artifacts/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n✅ Model & Scaler saved successfully in 'artifacts/' directory.")
print("🚀 Training Completed — Model Ready for LIME Explainability!\n")
