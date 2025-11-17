# shap_analysis.py
# --------------------------------------------------------
# SHAP explainability utilities
# --------------------------------------------------------

import shap
import matplotlib.pyplot as plt
import pandas as pd


def shap_summary(model, X_sample, out_path="plots/shap_summary.png"):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def shap_beeswarm(model, X_sample, out_path="plots/shap_beeswarm.png"):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
