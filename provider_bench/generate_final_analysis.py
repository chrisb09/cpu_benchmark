import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Final C23MM Benchmark Analysis\n",
                "This notebook aggregates and analyzes the 17 distinct provider/configuration benchmarks, each containing 50 inferences (5 models * 10 consecutive runs).\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import glob\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "sns.set_theme(style=\"whitegrid\", context=\"talk\")\n",
                "\n",
                "# Load all CSVs\n",
                "csv_files = glob.glob('results_c23mm/*.csv')\n",
                "df_list = []\n",
                "for f in csv_files:\n",
                "    df_list.append(pd.read_csv(f))\n",
                "df = pd.concat(df_list, ignore_index=True)\n",
                "\n",
                "# Clean up\n",
                "df['time_s'] = pd.to_numeric(df['time_s'], errors='coerce')\n",
                "df['max_rss_mb'] = pd.to_numeric(df['max_rss_mb'], errors='coerce')\n",
                "\n",
                "# Feature engineering\n",
                "def determine_bind(row):\n",
                "    provider = str(row['provider'])\n",
                "    if 'SINGLE' in provider:\n",
                "        return 'SINGLE'\n",
                "    elif 'NOBIND' in provider:\n",
                "        return 'NOBIND'\n",
                "    elif 'BIND' in provider:\n",
                "        return 'BIND'\n",
                "    else:\n",
                "        return 'OTHER'\n",
                "\n",
                "df['bind_type'] = df.apply(determine_bind, axis=1)\n",
                "\n",
                "# Success rates\n",
                "df['is_success'] = df['status'] == 'SUCCESS'\n",
                "success_summary = df.groupby(['model', 'provider']).agg(\n",
                "    total_runs=('status', 'count'),\n",
                "    successes=('is_success', 'sum')\n",
                ").reset_index()\n",
                "success_summary['fail_rate'] = 100 * (1 - success_summary['successes'] / success_summary['total_runs'])\n",
                "print(\"Failure rates across configurations:\")\n",
                "display(success_summary.sort_values(['fail_rate', 'model'], ascending=[False, True]))\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Median Inference Time (Bar Plots)\n",
                "The bar plots depict the median execution time for each configuration, grouped by model."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df_success = df[df['status'] == 'SUCCESS'].copy()\n",
                "\n",
                "models = sorted(df_success['model'].unique())\n",
                "palette = {\"BIND\": \"#4C72B0\", \"NOBIND\": \"#C44E52\", \"OTHER\": \"#55A868\", \"SINGLE\": \"#FF9F00\"}\n",
                "\n",
                "# Sort providers logically so they are consistent across plots\n",
                "def get_provider_order(df_s):\n",
                "    return sorted(df_s['provider'].unique())\n",
                "\n",
                "for model in models:\n",
                "    data_m = df_success[df_success['model'] == model]\n",
                "    order = get_provider_order(data_m)\n",
                "    \n",
                "    plt.figure(figsize=(16, 8))\n",
                "    ax = sns.barplot(data=data_m, x='provider', y='time_s', hue='bind_type', \n",
                "                     palette=palette, estimator=np.median, order=order, dodge=False, errorbar=None)\n",
                "    \n",
                "    # Add value labels on top of the bars\n",
                "    for container in ax.containers:\n",
                "        ax.bar_label(container, fmt='%.1f', padding=3, fontsize=10)\n",
                "    \n",
                "    plt.title(f\"Median Inference Time for Model: {model}\", fontsize=18)\n",
                "    plt.ylabel(\"Median Time (s)\", fontsize=14)\n",
                "    plt.xlabel(\"Provider / Configuration\", fontsize=14)\n",
                "    plt.xticks(rotation=45, ha='right', fontsize=12)\n",
                "    plt.legend(title=\"Binding Type\")\n",
                "    plt.tight_layout()\n",
                "    plt.show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Variance and Confidence Intervals (Box Plots)\n",
                "To accurately measure the stability of the 10 consecutive executions per configuration, these Box Plots map out the complete variance.\n",
                "Whiskers are set to `whis=(2.5, 97.5)` to explicitly outline the **95% Confidence Interval** boundaries. Outliers (the 5% extremes) are rendered as individual diamonds.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "for model in models:\n",
                "    data_m = df_success[df_success['model'] == model]\n",
                "    order = get_provider_order(data_m)\n",
                "    \n",
                "    plt.figure(figsize=(16, 8))\n",
                "    # whis=(2.5, 97.5) visualizes the 95% empirical confidence interval limits directly on the whiskers\n",
                "    sns.boxplot(data=data_m, x='provider', y='time_s', hue='bind_type', \n",
                "                palette=palette, order=order, dodge=False, whis=(2.5, 97.5), fliersize=6, flierprops=dict(marker='d'))\n",
                "    \n",
                "    plt.title(f\"Execution Time 95% CI (10 Runs) for Model: {model}\", fontsize=18)\n",
                "    plt.ylabel(\"Time (s)\", fontsize=14)\n",
                "    plt.xlabel(\"Provider / Configuration\", fontsize=14)\n",
                "    plt.xticks(rotation=45, ha='right', fontsize=12)\n",
                "    plt.legend(title=\"Binding Type\")\n",
                "    plt.tight_layout()\n",
                "    plt.show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Maximum RSS Memory (Bar Plots)\n",
                "The bar plots depict the median maximum resident set size (Max RSS) in MB across all runs."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "for model in models:\n",
                "    data_m = df_success[df_success['model'] == model]\n",
                "    order = get_provider_order(data_m)\n",
                "    \n",
                "    plt.figure(figsize=(16, 8))\n",
                "    ax = sns.barplot(data=data_m, x='provider', y='max_rss_mb', hue='bind_type', \n",
                "                     palette=palette, estimator=np.median, order=order, dodge=False, errorbar=None)\n",
                "    \n",
                "    # Add value labels on top of the bars\n",
                "    for container in ax.containers:\n",
                "        ax.bar_label(container, fmt='%.1f', padding=3, fontsize=10)\n",
                "    \n",
                "    plt.title(f\"Median Max RSS Memory for Model: {model}\", fontsize=18)\n",
                "    plt.ylabel(\"Memory (MB)\", fontsize=14)\n",
                "    plt.xlabel(\"Provider / Configuration\", fontsize=14)\n",
                "    plt.xticks(rotation=45, ha='right', fontsize=12)\n",
                "    plt.legend(title=\"Binding Type\")\n",
                "    plt.tight_layout()\n",
                "    plt.show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Maximum RSS Memory Variance (Box Plots)\n",
                "Box plots showing the 95% confidence variance of peak memory usage across consecutive inferences."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "for model in models:\n",
                "    data_m = df_success[df_success['model'] == model]\n",
                "    order = get_provider_order(data_m)\n",
                "    \n",
                "    plt.figure(figsize=(16, 8))\n",
                "    sns.boxplot(data=data_m, x='provider', y='max_rss_mb', hue='bind_type', \n",
                "                palette=palette, order=order, dodge=False, whis=(2.5, 97.5), fliersize=6, flierprops=dict(marker='d'))\n",
                "    \n",
                "    plt.title(f\"Memory Max RSS 95% CI (10 Runs) for Model: {model}\", fontsize=18)\n",
                "    plt.ylabel(\"Memory (MB)\", fontsize=14)\n",
                "    plt.xlabel(\"Provider / Configuration\", fontsize=14)\n",
                "    plt.xticks(rotation=45, ha='right', fontsize=12)\n",
                "    plt.legend(title=\"Binding Type\")\n",
                "    plt.tight_layout()\n",
                "    plt.show()\n"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.9.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("final_c23mm_analysis.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated successfully: final_c23mm_analysis.ipynb")
