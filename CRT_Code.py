# 🧠 COGNITIVE REFLECTION TEST (CRT) DATA ANALYSIS
# =================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.cluster import KMeans

# -------------------------------
#  Load Data
# -------------------------------
df = pd.read_csv("CRT_Dataset.csv")
df = df[["crt1", "crt2", "crt3"]].astype(int)

# -------------------------------
# 1️⃣ DESCRIPTIVE ANALYSIS
# -------------------------------

# % correct per question
percent_correct = (df.mean() * 100).round(2)
print("\n📊 Percentage Correct per Question:")
print(percent_correct)

# Total CRT score
df["total_score"] = df.sum(axis=1)

# Distribution of total CRT scores
score_distribution = (
    df["total_score"].value_counts(normalize=True).sort_index() * 100
).round(2)
print("\n📈 Distribution of Total CRT Scores (as %):")
print(score_distribution)

# --- Visuals ---
sns.barplot(x=percent_correct.index, y=percent_correct.values, palette="coolwarm")
plt.title("Percentage Correct per CRT Question")
plt.ylabel("% Correct")
plt.xlabel("Question")
plt.show()

sns.histplot(df["total_score"], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], kde=False)
plt.title("Distribution of Total CRT Scores (0–3)")
plt.xlabel("Total Correct Answers")
plt.ylabel("Count")
plt.show()


# -------------------------------
# 2️⃣ ERROR PATTERN ANALYSIS
# -------------------------------

# Create response pattern (e.g., 101)
df["pattern"] = df[["crt1", "crt2", "crt3"]].astype(str).agg("".join, axis=1)

pattern_counts = df["pattern"].value_counts().sort_index()
print("\n🧩 Response Pattern Counts:")
print(pattern_counts)

# --- Visuals ---

# Pie chart for response types
df["response_type"] = df["total_score"].map({
    0: "All Intuitive",
    1: "Mostly Intuitive",
    2: "Mostly Reflective",
    3: "All Correct"
})
print("\n🥧 Response Type Distribution:")
print(df["response_type"].value_counts())

df["response_type"].value_counts().plot.pie(
    autopct="%1.1f%%", colors=sns.color_palette("pastel")
)
plt.title("Overall Response Type Distribution")
plt.ylabel("")
plt.show()


# -------------------------------
# 3️⃣ WITHIN-SUBJECT COMPARATIVE ANALYSIS
# -------------------------------

# Conditional probabilities
cond_probs = {
    "P(Q2=1 | Q1=1)": df.loc[df["crt1"] == 1, "crt2"].mean(),
    "P(Q2=1 | Q1=0)": df.loc[df["crt1"] == 0, "crt2"].mean(),
    "P(Q3=1 | Q2=1)": df.loc[df["crt2"] == 1, "crt3"].mean(),
    "P(Q3=1 | Q2=0)": df.loc[df["crt2"] == 0, "crt3"].mean(),
    "P(Q3=1 | Q1=1)": df.loc[df["crt1"] == 1, "crt3"].mean(),
    "P(Q3=1 | Q1=0)": df.loc[df["crt1"] == 0, "crt3"].mean(),
}
print("\n📈 Conditional Probabilities:")
for k, v in cond_probs.items():
    print(f"{k}: {v:.2f}")

# Transition matrices
transitions = {
    "Q1→Q2": pd.crosstab(df["crt1"], df["crt2"], normalize="index").round(2),
    "Q2→Q3": pd.crosstab(df["crt2"], df["crt3"], normalize="index").round(2),
    "Q1→Q3": pd.crosstab(df["crt1"], df["crt3"], normalize="index").round(2),
}

for name, mat in transitions.items():
    print(f"\n🔄 Transition Matrix {name}:")
    print(mat)
    sns.heatmap(mat, annot=True, cmap="YlGnBu")
    plt.title(f"Transition Matrix {name} (Proportion)")
    plt.xlabel(f"{name.split('→')[1]} Correct (1=Yes, 0=No)")
    plt.ylabel(f"{name.split('→')[0]} Correct (1=Yes, 0=No)")
    plt.show()


# -------------------------------
# 4️⃣ PREDICTIVE MODELING
# -------------------------------

# Q2_correct ~ Q1_correct
model_q2 = smf.logit("crt2 ~ crt1", data=df).fit()
print("\n📊 Logistic Regression: Q2_correct ~ Q1_correct")
print(model_q2.summary())

# Q3_correct ~ Q1_correct + Q2_correct
model_q3 = smf.logit("crt3 ~ crt1 + crt2", data=df).fit()
print("\n📊 Logistic Regression: Q3_correct ~ Q1_correct + Q2_correct")
print(model_q3.summary())

print("\n✅ All analyses complete.")
