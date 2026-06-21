import json
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# CPU Benchmark Results Analysis\n",
                "Aggregating data from different model runs, ignoring `inter` to simulate multiple runs per `(ranks, intra)` configuration."
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
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "# Load all results.csv from the benchmarks directory\n",
                "base_dir = '/hpcwork/ro092286/smartsim/cpu_benchmark/benchmarks'\n",
                "csv_files = glob.glob(os.path.join(base_dir, '*_mini_app_*', 'results.csv')) + \\\n",
                "            glob.glob(os.path.join(base_dir, '*_mmcp_*', 'results.csv'))\n",
                "\n",
                "dfs = []\n",
                "for f in csv_files:\n",
                "    model_folder = os.path.basename(os.path.dirname(f))\n",
                "    if 'mini_app' in model_folder:\n",
                "        model_name = model_folder.split('_mini_app')[0]\n",
                "    elif 'mmcp' in model_folder:\n",
                "        model_name = model_folder.split('_mmcp')[0]\n",
                "    else:\n",
                "        model_name = model_folder.split('_')[0]\n",
                "        \n",
                "    try:\n",
                "        df = pd.read_csv(f)\n",
                "        df['model'] = model_name\n",
                "        dfs.append(df)\n",
                "    except Exception as e:\n",
                "        print(f\"Error reading {f}: {e}\")\n",
                "\n",
                "df_all = pd.concat(dfs, ignore_index=True)\n",
                "\n",
                "# Filter successful runs\n",
                "df_succ = df_all[df_all['status'] == 'SUCCESS'].copy()\n",
                "\n",
                "# Group by model, ranks, intra to compute means across inter (which we treat as identical runs)\n",
                "df_agg = df_succ.groupby(['model', 'ranks', 'intra']).agg({\n",
                "    'time_s': ['mean', 'std', 'count'],\n",
                "    'max_rss_mb': 'mean'\n",
                "}).reset_index()\n",
                "\n",
                "# Flatten multi-level columns\n",
                "df_agg.columns = ['model', 'ranks', 'intra', 'time_mean', 'time_std', 'measurements', 'memory_mean']\n",
                "df_agg.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Execution Time Heatmaps\n",
                "Heatmaps are often much clearer than 3D plots for visualizing two independent variables (ranks and intra) against a dependent variable (time). The color intensity provides an immediate intuition for performance hotspots."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "models = df_agg['model'].unique()\n",
                "\n",
                "for model in models:\n",
                "    plt.figure(figsize=(8, 6))\n",
                "    df_model = df_agg[df_agg['model'] == model]\n",
                "    pivot_table = df_model.pivot(index='intra', columns='ranks', values='time_mean')\n",
                "    \n",
                "    # Sort indices just in case so they display logically\n",
                "    pivot_table = pivot_table.sort_index(ascending=False) \n",
                "    \n",
                "    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='viridis_r', cbar_kws={'label': 'Mean Time (s)'})\n",
                "    plt.title(f\"Execution Time Heatmap: {model} (Mean of N={int(df_model['measurements'].max())} runs)\")\n",
                "    plt.xlabel('Ranks')\n",
                "    plt.ylabel('Intra-op Threads')\n",
                "    plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Memory Usage\n",
                "Memory usage primarily scales linearly with the number of ranks. Here is a simple line plot showing this relationship across all models."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(10, 6))\n",
                "sns.lineplot(data=df_agg, x='ranks', y='memory_mean', hue='model', marker='o', linewidth=2, markersize=8)\n",
                "plt.title('Memory Usage vs MPI Ranks')\n",
                "plt.xlabel('Number of MPI Ranks')\n",
                "plt.ylabel('Mean Memory (MB)')\n",
                "plt.grid(True, ls='--', alpha=0.7)\n",
                "plt.show()"
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
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.9.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}
with open('/hpcwork/ro092286/smartsim/cpu_benchmark/cpu_results_analysis.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)
