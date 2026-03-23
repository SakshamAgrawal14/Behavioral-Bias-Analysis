import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# === Load and clean ===
df = pd.read_csv("wason_dataset.csv", sep=";", engine="python", on_bad_lines="skip")
df.columns = df.columns.str.strip()

# Drop blanks in key columns
df = df.dropna(subset=["wason_condition", "response_correct", "rt"])

# Force binary numeric for response_correct
df["response_correct"] = (
    df["response_correct"]
    .astype(str)
    .str.strip()
    .str.lower()
    .replace({"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0})
)
df["response_correct"] = pd.to_numeric(df["response_correct"], errors="coerce")

# Convert rt and bonus
df["rt"] = pd.to_numeric(df["rt"], errors="coerce")
df["replication_experiment_with_bonus"] = pd.to_numeric(
    df["replication_experiment_with_bonus"], errors="coerce"
)

# Filter only Arbitrary and Realistic
df = df[df["wason_condition"].isin(["Arbitrary", "Realistic"])]
print(f"✅ Dataset filtered to Arbitrary & Realistic only. Rows remaining: {len(df)}")

# --- Plot 1: Accuracy by condition ---
acc_by_cond = (
    df.groupby("wason_condition")["response_correct"]
    .mean()
    .reset_index()
    .rename(columns={"response_correct": "mean_accuracy"})
)
print("\n📊 Accuracy Data Used for Plot:")
print(acc_by_cond)

sns.barplot(data=acc_by_cond, x="wason_condition", y="mean_accuracy", hue="wason_condition", legend=False)
plt.title("Mean Accuracy by Wason Condition")
plt.show()

# --- Plot 2: Accuracy by RT Split ---
df["rt_split"] = (df["rt"] > df["rt"].median()).map({True: "Slow", False: "Fast"})
acc_by_rt_split = (
    df.groupby("rt_split")["response_correct"]
    .mean()
    .reset_index()
    .rename(columns={"response_correct": "mean_accuracy"})
)
print("\n📊 Accuracy by RT Split:")
print(acc_by_rt_split)

sns.barplot(data=acc_by_rt_split, x="rt_split", y="mean_accuracy", hue="rt_split", legend=False, palette="Set2")
plt.title("Accuracy by Reaction Time Split (Fast vs Slow)")
plt.show()

# --- Plot 3: Two-way comparison (Condition × RT Split) ---
two_way = (
    df.groupby(["wason_condition", "rt_split"])["response_correct"]
    .mean()
    .reset_index()
    .rename(columns={"response_correct": "mean_accuracy"})
)
print("\n📊 Two-Way Data (Condition × RT Split):")
print(two_way)

sns.barplot(data=two_way, x="wason_condition", y="mean_accuracy", hue="rt_split", palette="muted")
plt.title("Accuracy by Condition × Reaction Time Split")
plt.show()

# --- Plot 4: Error pattern analysis ---
for col in [
    "pc_choice_includes_antecedent_true",
    "pc_choice_includes_consequent_true",
    "pc_choice_includes_consequent_false",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df["pattern_type"] = "Other"
df.loc[
    (df["pc_choice_includes_antecedent_true"] == 1)
    & (df["pc_choice_includes_consequent_true"] == 1),
    "pattern_type",
] = "Confirmation"
df.loc[
    (df["pc_choice_includes_antecedent_true"] == 1)
    & (df["pc_choice_includes_consequent_false"] == 1),
    "pattern_type",
] = "Falsification"

pattern_data = df.groupby(["wason_condition", "pattern_type"]).size().reset_index(name="count")
print("\n📊 Error Pattern Data Used for Plot:")
print(pattern_data)

sns.barplot(data=pattern_data, x="wason_condition", y="count", hue="pattern_type", palette="pastel")
plt.title("Error Pattern Distribution by Condition")
plt.show()

# --- Logistic Regression ---
print("\n⚙️ Running logistic regression for Realistic vs Arbitrary...\n")

# Drop rows with missing response_correct
df_reg = df.dropna(subset=["response_correct", "rt", "replication_experiment_with_bonus"])

model = smf.logit(
    formula="response_correct ~ rt + C(wason_condition) + replication_experiment_with_bonus",
    data=df_reg
).fit(disp=False)

summary = model.summary2().tables[1][["Coef.", "P>|z|"]]
summary["Significant?"] = summary["P>|z|"] < 0.05
print("\n📈 Key Regression Results:")
print(summary.to_string(index=False))
