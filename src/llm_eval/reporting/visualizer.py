import matplotlib
matplotlib.use('Agg') # Set backend for headless environments (Docker)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Dict, List
import os
import math
import numpy as np

class Visualizer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_histograms(self, results: pd.DataFrame):
        """Generates histograms for each numeric column in results."""
        numeric_cols = results.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            plt.figure(figsize=(10, 6))
            sns.histplot(results[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.xlabel("Score")
            plt.ylabel("Frequency")
            filename = f"hist_{col.replace('/', '_').replace(' ', '_')}.png"
            plt.savefig(os.path.join(self.output_dir, filename))
            plt.close()

    def generate_radar_chart(self, summary_stats: Dict[str, float], model_name: str):
        """Generates a radar chart for the aggregate metrics."""
        categories = list(summary_stats.keys())
        values = list(summary_stats.values())
        
        N = len(categories)
        if N < 3:
            return # Radar chart needs at least 3 dims
            
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        values += values[:1]
        angles += angles[:1]
        
        ax = plt.subplot(111, polar=True)
        plt.xticks(angles[:-1], categories, color='grey', size=8)
        
        ax.plot(angles, values, linewidth=1, linestyle='solid')
        ax.fill(angles, values, 'b', alpha=0.1)
        
        plt.title(f"Metric Performance - {model_name}")
        filename = f"radar_{model_name}.png"
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
