import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# === Load dataset ===
df = pd.read_csv("Mountain Height Anchor.csv")  # uncomment for generating results for other anchor tests
#df = pd.read_csv("Birth Rate Anchor.csv")
#df = pd.read_csv("Willingness to Pay Anchor.csv")

# === Basic cleaning ===
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# === Drop missing / invalid values ===
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["anchor", "estimate", "true_value", "anchorhigh"])

# === 1️⃣ Correlation ===
corr = df["anchor"].corr(df["estimate"])
print(f"\n🔹 Anchor–Estimate correlation: {corr:.3f}")

# === 2️⃣ Regression ===
X = sm.add_constant(df["anchor"])
y = df["estimate"]
model = sm.OLS(y, X).fit()

slope = model.params["anchor"]
interpretation = (
    "Strong anchoring" if abs(slope) > 0.3 else
    "Moderate anchoring" if abs(slope) > 0.1 else
    "Weak anchoring"
)
print(f"Behavioral interpretation: slope = {slope:.3f} → {interpretation}")

# === 3️⃣ Directional Bias ===
df["bias"] = df["estimate"] - df["true_value"]

# Split by anchor type
bias_summary = (
    df.groupby("anchorhigh")
    .agg(mean_estimate=("estimate", "mean"),
         mean_bias=("bias", "mean"),
         std_bias=("bias", "std"),
         count=("bias", "count"))
    .rename(index={0: "Low Anchor", 1: "High Anchor"})
)

print("\n🔹 Directional bias summary by anchor:")
print(bias_summary.round(2))

# === 4️⃣ Z-standardization ===
df["z_anchor"] = (df["anchor"] - df["anchor"].mean()) / df["anchor"].std()
df["z_estimate"] = (df["estimate"] - df["estimate"].mean()) / df["estimate"].std()

# === 5️⃣ Combined Plot (All Anchors + True Value) ===
sns.set(style="whitegrid", font_scale=1.1)
plt.figure(figsize=(9, 6))

# Plot estimates
sns.scatterplot(
    data=df, x=df.index, y="estimate",
    hue="anchorhigh", palette={0: "steelblue", 1: "darkorange"},
    alpha=0.7, s=60
)

# Plot anchor lines
low_anchor_val = df.loc[df["anchorhigh"] == 0, "anchor"].iloc[0]
high_anchor_val = df.loc[df["anchorhigh"] == 1, "anchor"].iloc[0]
true_value_val = df["true_value"].mean()

plt.axhline(y=low_anchor_val, color="blue", linestyle="--", linewidth=1.5, label=f"Low Anchor = {low_anchor_val:.0f}")
plt.axhline(y=high_anchor_val, color="orange", linestyle="--", linewidth=1.5, label=f"High Anchor = {high_anchor_val:.0f}")
plt.axhline(y=true_value_val, color="green", linestyle="--", linewidth=1.5, label=f"True Value ≈ {true_value_val:.0f}")

plt.title("Estimates by Anchor Type with True Value")
plt.xlabel("Participant Index")
plt.ylabel("Estimate")
plt.legend()
plt.tight_layout()
plt.show()

# === 7️⃣ Boxplot of Estimates ===
plt.figure(figsize=(6, 5))
df["anchorhigh"] = df["anchorhigh"].astype(float)  # ensure consistent dtype
sns.boxplot(
    data=df,
    x="anchorhigh",
    y="estimate",
    hue="anchorhigh",
    palette={0.0: "steelblue", 1.0: "darkorange"},
    legend=False
)
plt.xticks([0, 1], ["Low Anchor", "High Anchor"])
plt.title("Estimate Spread by Anchor Type")
plt.xlabel("Anchor Type")
plt.ylabel("Estimate")
plt.tight_layout()
plt.show()


print("\n✅ All plots and summaries generated successfully.")
