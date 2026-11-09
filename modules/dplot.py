import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statannotations.Annotator import Annotator


rng = np.random.default_rng(42)

def plot_violin(df, ax, x="Model",color=None, hue=None,palette=None,alpha=0.8):
    sns.violinplot(data=df,x=x,y="Accuracy",hue=hue,ax=ax,color=color,width=0.5,palette=palette)
    for pc in ax.collections:
        pc.set_alpha(alpha)
    ax.set_ylabel("Accuracy")
def plot_points(xx, arr, ax, size=6):
    for i in range(arr.shape[0]):
        jitter = rng.normal(0, 0.03, size=size)
        ax.plot(xx + jitter, arr[i], color="#7A7A7A", alpha=0.4, linewidth=0.8, marker="o", markersize=3)
def plot_scatter(xx, arr, ax, size=6):
    for i in range(arr.shape[0]):
        jitter = rng.normal(0, 0.03, size=size)
        ax.scatter(xx + jitter, arr[i], color="#7A7A7A", alpha=0.4, linewidth=0.8, marker="o", markersize=3)
        

def add_chance_test(ax, df, chance, height):
    models = df["Model"].unique()

    for i, model in enumerate(models):
        values = df.loc[df["Model"] == model, "Accuracy"].values
        # one-sided one-sample t-test: accuracy > chance
        _, p = stats.ttest_1samp(values, chance, alternative="greater")
        if p < 0.0001:
            text = "****"
        elif p < 0.001:
            text = "***"
        elif p < 0.01:
            text = "**"
        elif p < 0.05:
            text = "*"
        else:
            text = "ns"

        ax.text(i, height, text, ha="center", va="bottom")