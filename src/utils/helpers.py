# src/utils/helpers.py

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches

def plot_roc_curve(model, X_test, y_test, model_name, save_path):
    roc_display = RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f'ROC Curve for {model_name} (AUC = {roc_display.roc_auc:.2f})')
    plt.show()
    plt.savefig(save_path)
    plt.close()

def plot_classification_report(report, model_name, save_path):
    metrics_df = pd.DataFrame(report).transpose().loc[['0', '1'], ['precision', 'recall', 'f1-score']]
    metrics_df.index = ['NoFailure', 'PreFailure']
    metrics_df = metrics_df * 100

    colors = ['#C62828', '#FF9800', '#4CAF50']
    hatches = ['/', '\\', 'x']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = metrics_df.plot(kind='bar', ax=ax, color=colors, rot=0)
    for i, bar_container in enumerate(ax.containers):
        for bar in bar_container:
            bar.set_hatch(hatches[i % len(hatches)])

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.2f}%', (p.get_x() + p.get_width() / 2, height), ha='center', va='bottom', fontsize=10)

    handles = [mpatches.Patch(facecolor=color, hatch=hatch, label=label) 
            for color, hatch, label in zip(colors, hatches, metrics_df.columns)]

    # ax.legend(
    #     handles=handles,
    #     title='Metrics',
    #     loc='center',
    #     ncol=3,
    #     frameon=False,
    #     fontsize=1,         # 🔹 Taille du texte de légende
    #     title_fontsize=13,   # 🔹 Taille du titre "Metrics"
    #     handlelength=4,      # 🔹 Taille du rectangle légende
    #     handleheight=3,    # 🔹 Hauteur du rectangle légende
    #     borderpad=1.2        # 🔹 Espacement intérieur
    # )
    ax.legend(handles=handles, title='Metrics', bbox_to_anchor=(0.5, 0.5),  loc='center',borderaxespad=0., handleheight=2, handlelength=3)
    plt.ylabel('Score (%)')
    # plt.ylim(99.5, 100) #for accuracy >99.5%
    plt.ylim(0, 100)
    # plt.title(f'Classification Report for {model_name}')
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path)
    plt.close()