"""
Author: Jongsoo Ha
Date: May 15, 2025
Description:
    Conduct a one-way ANOVA to test whether mean self-reported stress differs
    among students with High, Moderate, and Low physical activity levels
    using the Students Mental Health dataset. Save results and plot in output/.
"""

import os

import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

def main():
    # Load the dataset
    file_path = 'data/students_mental_health_survey.csv'
    df = pd.read_csv(file_path)
    
    # Inspect available columns
    print("Columns in dataset:", df.columns.tolist())
    
    # Define column names (update if necessary)
    stress_col = 'Stress_Level'         # self-reported stress score
    activity_col = 'Physical_Activity'  # categorical: High, Moderate, Low
    
    # Drop missing values in relevant columns
    df_clean = df[[stress_col, activity_col]].dropna()
    
    # Separate stress scores by activity level
    levels = ['High', 'Moderate', 'Low']
    stress_by_group = [df_clean[df_clean[activity_col] == lvl][stress_col] for lvl in levels]
    
    # Perform one-way ANOVA
    f_stat, p_val = stats.f_oneway(*stress_by_group)
    
    # Prepare output directory and filenames with timestamp
    out_dir = 'output'
    os.makedirs(out_dir, exist_ok=True)
    results_file = os.path.join(out_dir, f'ANOVA_Stress_PhysicalActivity.txt')
    plot_file = os.path.join(out_dir, f'ANOVA_Stress_PhysicalActivity.png')
    
    # Write ANOVA results to text file
    with open(results_file, 'w') as f:
        f.write("One-Way ANOVA: Stress by Physical Activity Level\n")
        f.write(f"Groups tested: {levels}\n")
        f.write(f"F-statistic = {f_stat:.4f}\n")
        f.write(f"p-value     = {p_val:.4f}\n")
        if p_val < 0.05:
            f.write("Result: Statistically significant differences between groups (p < 0.05)\n")
        else:
            f.write("Result: No statistically significant differences between groups (p >= 0.05)\n")
    
    # Generate and save boxplot
    plt.figure(figsize=(8,5))
    df_clean.boxplot(column=stress_col, by=activity_col, grid=False)
    plt.title('Stress Level by Physical Activity')
    plt.suptitle('')
    plt.xlabel('Physical Activity Level')
    plt.ylabel('Self-Reported Stress Score')
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()
    
    # Print locations of outputs
    print(f"Results saved to: {results_file}")
    print(f"Boxplot saved to: {plot_file}")

if __name__ == "__main__":
    main()

