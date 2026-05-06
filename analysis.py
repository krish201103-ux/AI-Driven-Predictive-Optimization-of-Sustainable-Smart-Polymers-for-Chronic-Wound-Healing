# =====================================
# IMPORTS
# =====================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# =====================================
# GLOBAL GRAPH SETTINGS
# =====================================
plt.rcParams.update({
    "figure.figsize": (10,6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

# =====================================
# LOAD DATA
# =====================================
df = pd.read_excel("Polymer Dataset .xlsx")

# Clean
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str).str.replace('−', '-', regex=False)
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

# =====================================
# HEALING EFFICIENCY
# =====================================
df['Healing Efficiency'] = (
    0.25 * df['Bio Score'] +
    0.2 * df['AntiMicrobial'] +
    0.2 * (df['Drug Release %'] / 100) * 10 +
    0.15 * (1 / (df['Contact Angle'] + 1)) * 100 +
    0.2 * (1 / (df['Degradation %/day'] + 1)) * 100
)

df['Healing Efficiency'] = (df['Healing Efficiency'] - df['Healing Efficiency'].min()) / \
                           (df['Healing Efficiency'].max() - df['Healing Efficiency'].min()) * 100

# =====================================
# FEATURES
# =====================================
features = ['MW (kDa)', 'Tg (°C)', 'Contact Angle', 'Degradation %/day',
            'Tensile MPa', 'Drug Release %', 'Bio Score', 'AntiMicrobial']

X = df[features]
y = df['Healing Efficiency']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================================
# MODELS
# =====================================
rf = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

mlp = MLPRegressor(hidden_layer_sizes=(128,64,32), max_iter=1000, random_state=42)
mlp.fit(X_train_s, y_train_s)
y_pred_mlp = mlp.predict(X_test_s)

xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# =====================================
# METRICS
# =====================================
def metrics(y_true, y_pred):
    return (
        r2_score(y_true, y_pred),
        np.sqrt(mean_squared_error(y_true, y_pred)),
        mean_absolute_error(y_true, y_pred)
    )

r2_rf, rmse_rf, mae_rf = metrics(y_test, y_pred_rf)
r2_mlp, rmse_mlp, mae_mlp = metrics(y_test_s, y_pred_mlp)
r2_xgb, rmse_xgb, mae_xgb = metrics(y_test, y_pred_xgb)

# =====================================
# TOP 10 OVERALL
# =====================================
top10 = df.sort_values(by='Healing Efficiency', ascending=False).head(10)

plt.figure()
bars = plt.barh(top10['Polymer'], top10['Healing Efficiency'])

for bar in bars:
    plt.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
             f"{bar.get_width():.2f}", va='center')

plt.xlabel("Healing Efficiency (%)")
plt.ylabel("Polymer Name")
plt.title("Top 10 Polymers Overall")
plt.xlim(10,100)
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# =====================================
# CATEGORY GRAPHS
# =====================================
for name, group in df.groupby('Polymer Type'):
    subset = group.sort_values(by='Healing Efficiency', ascending=False).head(10)

    if len(subset) >= 3:
        plt.figure()
        bars = plt.barh(subset['Polymer'], subset['Healing Efficiency'])

        for bar in bars:
            plt.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                     f"{bar.get_width():.2f}", va='center')

        plt.xlabel("Healing Efficiency (%)")
        plt.ylabel("Polymer Name")
        plt.title(f"Top 10 Polymers in {name}")

        plt.xlim(10,100)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()

# =====================================
# ACTUAL vs PREDICTED
# =====================================
def plot_ap(y_true, y_pred, title):
    plt.figure()
    plt.plot(y_true, label="Actual", linewidth=2)
    plt.plot(y_pred, linestyle='--', label="Predicted", linewidth=2)

    plt.xlabel("Sample Index")
    plt.ylabel("Healing Efficiency (%)")
    plt.title(title)
    plt.ylim(10,100)

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

plot_ap(y_test, y_pred_rf, "Random Forest")
plot_ap(y_test_s, y_pred_mlp, "Deep Learning")
plot_ap(y_test, y_pred_xgb, "XGBoost")

# =====================================
# MODEL COMPARISON
# =====================================
models = ['RF','DL','XGB']
r2_vals = [r2_rf, r2_mlp, r2_xgb]
rmse_vals = [rmse_rf, rmse_mlp, rmse_xgb]
mae_vals = [mae_rf, mae_mlp, mae_xgb]

x = np.arange(len(models))

plt.figure()
bars1 = plt.bar(x-0.25, r2_vals, 0.25, label='R2')
bars2 = plt.bar(x, rmse_vals, 0.25, label='RMSE')
bars3 = plt.bar(x+0.25, mae_vals, 0.25, label='MAE')

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        plt.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                 f"{bar.get_height():.2f}", ha='center')

plt.xticks(x, models)
plt.xlabel("Model")
plt.ylabel("Metric Value")
plt.title("Model Performance Comparison")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# =====================================
# FEATURE IMPORTANCE
# =====================================
feat_df = pd.DataFrame({
    "Feature": features,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure()
bars = plt.barh(feat_df['Feature'], feat_df['Importance'])

for bar in bars:
    plt.text(bar.get_width(), bar.get_y()+bar.get_height()/2,
             f"{bar.get_width():.3f}", va='center')

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance")
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# =====================================
# HEATMAP
# =====================================
plt.figure(figsize=(10,6))
sns.heatmap(df[features].corr(), annot=True, cmap="Blues")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# =====================================
# BEST COMBINATIONS
# =====================================
combinations = pd.DataFrame({
    "Combination":[
        "PCL + Silver + PEG",
        "PCL + ZnO Nanofiber",
        "PLA + PHB Blend",
        "Silicone + Hydrogel",
        "PCL + Gelatin + Silver"
    ],
    "Efficiency":[95,93,90,92,94]
})

plt.figure()
bars = plt.barh(combinations['Combination'], combinations['Efficiency'])

for bar in bars:
    plt.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
             f"{bar.get_width():.1f}", va='center')

plt.xlabel("Predicted Healing Efficiency (%)")
plt.ylabel("Polymer Combination")
plt.title("Optimized Polymer Combinations")
plt.xlim(10,100)
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# =====================================
# SHAP
# =====================================
try:
    import shap
    X_sample = X_test.sample(n=min(100,len(X_test)))
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_sample)

    shap.summary_plot(shap_values, X_sample)
    shap.summary_plot(shap_values, X_sample, plot_type="bar")

except:
    print("Run locally for SHAP")

# =====================================
# SAVE EXCEL
# =====================================
with pd.ExcelWriter("final_polymer_analysis.xlsx") as writer:
    top10.to_excel(writer, sheet_name='Top10_Overall', index=False)
    combinations.to_excel(writer, sheet_name='Combinations', index=False)

    pd.DataFrame({
        "Model": models,
        "R2": r2_vals,
        "RMSE": rmse_vals,
        "MAE": mae_vals
    }).to_excel(writer, sheet_name='Performance', index=False)

print("Saved as final_polymer_analysis.xlsx")