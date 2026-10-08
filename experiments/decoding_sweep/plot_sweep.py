import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# =========================================================
# LOAD CSV
# =========================================================

import os
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_results.csv"))

print(df.head())
print(df.describe())

# =========================================================
# CONFIG
# =========================================================

score_col = "overall_acc"

metric_cols = [
    "acc_Enter",
    "acc_Ancie",
    "acc_Scien",
    "acc_Maths",
    "acc_Philo",
    "acc_News"
]

param_cols = [
    "temperature",
    "top_p",
    "num_beams"
]

# =========================================================
# BEST CONFIG
# =========================================================

best = df.sort_values(score_col, ascending=False).head(10)

print("\n=== TOP CONFIGS ===")
print(best[
    param_cols +
    [score_col, "mean_score", "std_score"]
])

# =========================================================
# CORRELATION MATRIX
# =========================================================

plt.figure(figsize=(12, 10))

corr = df.corr(numeric_only=True)

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# =========================================================
# TEMPERATURE VS ACCURACY
# =========================================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="temperature",
    y=score_col,
    hue="top_p",
    size="num_beams",
    palette="viridis"
)

plt.title("Temperature vs Accuracy")
plt.tight_layout()
plt.show()

# =========================================================
# TOP_P VS ACCURACY
# =========================================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="top_p",
    y=score_col,
    hue="temperature",
    size="num_beams",
    palette="plasma"
)

plt.title("Top_p vs Accuracy")
plt.tight_layout()
plt.show()

# =========================================================
# NUM_BEAMS EFFECT
# =========================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="num_beams",
    y=score_col
)

plt.title("Effect of num_beams")
plt.tight_layout()
plt.show()

# =========================================================
# HEATMAP TEMPERATURE x TOP_P
# =========================================================

pivot = df.pivot_table(
    values=score_col,
    index="temperature",
    columns="top_p",
    aggfunc="mean"
)

plt.figure(figsize=(12, 8))

sns.heatmap(
    pivot,
    annot=True,
    cmap="viridis"
)

plt.title("Heatmap Temperature vs Top_p")
plt.tight_layout()
plt.show()

# =========================================================
# 3D SURFACE
# =========================================================

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

x = df["temperature"]
y = df["top_p"]
z = df[score_col]

ax.scatter(x, y, z, c=z, cmap="viridis", s=60)

ax.set_xlabel("Temperature")
ax.set_ylabel("Top_p")
ax.set_zlabel(score_col)

plt.title("3D Parameter Space")
plt.show()

# =========================================================
# SCORE DISTRIBUTION
# =========================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df[score_col],
    bins=20,
    kde=True
)

plt.title("Score Distribution")
plt.tight_layout()
plt.show()

# =========================================================
# STABILITY ANALYSIS
# =========================================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="std_score",
    y=score_col,
    hue="temperature",
    size="num_beams"
)

plt.title("Stability vs Performance")
plt.tight_layout()
plt.show()

# =========================================================
# PARAMETER IMPORTANCE
# =========================================================

for param in param_cols:

    plt.figure(figsize=(10, 6))

    sns.regplot(
        data=df,
        x=param,
        y=score_col,
        scatter_kws={"alpha":0.7}
    )

    plt.title(f"{param} influence on {score_col}")
    plt.tight_layout()
    plt.show()

# =========================================================
# RADAR CHART BEST CONFIG
# =========================================================

best_row = df.sort_values(score_col, ascending=False).iloc[0]

values = [best_row[c] for c in metric_cols]
values += values[:1]

angles = np.linspace(0, 2*np.pi, len(metric_cols), endpoint=False).tolist()
angles += angles[:1]

fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

ax.plot(angles, values)
ax.fill(angles, values, alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metric_cols)

plt.title("Best Config Radar Chart")
plt.show()

# =========================================================
# PAIRPLOT
# =========================================================

sns.pairplot(
    df[
        param_cols +
        [score_col]
    ],
    diag_kind="kde"
)

plt.show()

# =========================================================
# DOMAIN ANALYSIS
# =========================================================

domain_means = df[metric_cols].mean()

plt.figure(figsize=(10, 6))

domain_means.sort_values().plot(kind="bar")

plt.title("Average Accuracy by Domain")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.show()

# =========================================================
# TOP CONFIGS VISUALIZATION
# =========================================================

top10 = df.sort_values(score_col, ascending=False).head(10)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=top10,
    x=top10.index,
    y=score_col
)

plt.title("Top 10 Configurations")
plt.xlabel("Config")
plt.ylabel(score_col)

plt.tight_layout()
plt.show()

# =========================================================
# OUTLIERS
# =========================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    y=score_col
)

plt.title("Outlier Detection")
plt.tight_layout()
plt.show()

# =========================================================
# PRINT INSIGHTS
# =========================================================

print("\n=== CORRELATIONS WITH ACCURACY ===")

corrs = corr[score_col].sort_values(ascending=False)

print(corrs)

print("\n=== MOST STABLE CONFIGS ===")

stable = df.sort_values("std_score").head(10)

print(stable[
    param_cols +
    [score_col, "std_score"]
])

print("\n=== BEST MEAN SCORE ===")

best_mean = df.sort_values("mean_score", ascending=False).head(10)

print(best_mean[
    param_cols +
    ["mean_score", score_col]
])