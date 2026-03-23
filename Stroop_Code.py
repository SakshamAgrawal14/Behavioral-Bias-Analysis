import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# 1️⃣ Load Data
# ==============================
df = pd.read_csv("stroopdata.csv")

print("\n🔹 Basic Info:")
print(df.info())

print("\n🔹 First 5 Rows:")
print(df.head())

# ==============================
# 2️⃣ Mean Reaction Time & Accuracy per Condition
# ==============================
mean_rt = df.groupby("condition")["reaction_time_ms"].mean().reset_index()
mean_acc = df.groupby("condition")["accuracy"].mean().reset_index()

print("\n📊 Mean Reaction Time per Condition:")
print(mean_rt)

print("\n📈 Mean Accuracy per Condition:")
print(mean_acc)

# ==============================
# 3️⃣ Subject-Level Averages (for t-test)
# ==============================
subject_means = (
    df.groupby(["subject", "condition"])
      .agg(mean_rt=("reaction_time_ms", "mean"),
           mean_acc=("accuracy", "mean"))
      .reset_index()
)

print("\n👤 Subject-Level Means (first 10 shown):")
print(subject_means.head(10))

# Pivot for paired comparisons
pivot_rt = subject_means.pivot(index="subject", columns="condition", values="mean_rt")
pivot_acc = subject_means.pivot(index="subject", columns="condition", values="mean_acc")

# ==============================
# 4️⃣ Paired t-tests
# ==============================
t_rt, p_rt = stats.ttest_rel(pivot_rt["congruent"], pivot_rt["incongruent"])
t_acc, p_acc = stats.ttest_rel(pivot_acc["congruent"], pivot_acc["incongruent"])

print("\n🧠 Paired t-test Results:")
print(f"Reaction Time: t = {t_rt:.3f}, p = {p_rt:.5f}")
print(f"Accuracy:      t = {t_acc:.3f}, p = {p_acc:.5f}")

# ==============================
# 5️⃣ Speed–Accuracy Tradeoff
# ==============================
# Correlation between reaction time and accuracy (subject-level)
tradeoff = subject_means.groupby("subject").agg(
    mean_rt=("mean_rt", "mean"),
    mean_acc=("mean_acc", "mean")
).reset_index()

corr = tradeoff["mean_rt"].corr(tradeoff["mean_acc"])
print("\n⚖️ Speed–Accuracy Tradeoff (subject-level):")
print(tradeoff.head())
print(f"\nCorrelation between mean RT and mean accuracy = {corr:.3f}")

# Plot tradeoff
plt.figure(figsize=(6, 4))
sns.scatterplot(data=tradeoff, x="mean_rt", y="mean_acc", s=70)
plt.title("Speed–Accuracy Tradeoff")
plt.xlabel("Mean Reaction Time (ms)")
plt.ylabel("Mean Accuracy")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.barplot(data=mean_acc, x="condition", y="accuracy", palette="Set2")
plt.title("Mean Accuracy by Condition")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()


# ==============================
# 6️⃣ Logistic Regression: accuracy ~ reaction_time_ms + condition
# ==============================
model = smf.logit("accuracy ~ reaction_time_ms + C(condition)", data=df).fit(disp=False)
print("\n📉 Logistic Regression Summary:")
print(model.summary())

# ==============================
# 7️⃣ Visualization Data (for your own plots)
# ==============================
print("\n🗂️ Data for Plotting — Mean RT per Condition:")
print(mean_rt)

print("\n🗂️ Data for Plotting — Accuracy per Condition:")
print(mean_acc)

# Optional example plot
plt.figure(figsize=(6, 4))
sns.barplot(data=mean_rt, x="condition", y="reaction_time_ms")
plt.title("Mean Reaction Time by Condition")
plt.ylabel("Reaction Time (ms)")
plt.tight_layout()
plt.show()
