"""
Author: Jongsoo Ha
Date: May 14, 2025
Description: 
    Analyze the resting heart rate distribution from the Student Health Data dataset
    by fitting Normal and Gamma distributions using Maximum Likelihood Estimation (MLE).
    Compute 95% confidence intervals for the estimated parameters, visualize
    the fitted PDFs against the empirical histogram, and save outputs with timestamps.
"""

import os

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def main():
    # 1. Load the dataset
    file_path = 'data/student_health_data.csv'
    df = pd.read_csv(file_path)
    
    # 2. Extract resting heart rate
    hr_col = 'Heart_Rate'
    heart_rates = df[hr_col].dropna()
    n = len(heart_rates)
    
    # 3. Estimate Normal distribution parameters
    mu_mle = heart_rates.mean()
    sigma_mle = heart_rates.std(ddof=0)
    var_mle = sigma_mle ** 2
    
    # 4. Compute 95% CI for Normal parameters
    conf_level = 0.95
    alpha = 1 - conf_level
    z = stats.norm.ppf(1 - alpha/2)
    se_mu = sigma_mle / np.sqrt(n)
    ci_mu = (mu_mle - z*se_mu, mu_mle + z*se_mu)
    ci_var = (
        (n * var_mle) / stats.chi2.ppf(1 - alpha/2, df=n),
        (n * var_mle) / stats.chi2.ppf(alpha/2, df=n)
    )
    
    # 5. Estimate Gamma distribution parameters
    shape_mle, loc_mle, scale_mle = stats.gamma.fit(heart_rates, floc=0)
    
    # 6. Bootstrap CIs for Gamma parameters
    n_boot = 1000
    shape_bs, scale_bs = [], []
    for _ in range(n_boot):
        sample = heart_rates.sample(n, replace=True)
        sh, _, sc = stats.gamma.fit(sample, floc=0)
        shape_bs.append(sh)
        scale_bs.append(sc)
    ci_shape = np.percentile(shape_bs, [2.5, 97.5])
    ci_scale = np.percentile(scale_bs, [2.5, 97.5])
    
    # Prepare output directory and filenames
    out_dir = 'output'
    os.makedirs(out_dir, exist_ok=True)
    plot_file = os.path.join(out_dir, f'ParameterEstimation.png')
    results_file = os.path.join(out_dir, f'ParameterEstimation.txt')
    
    # 7. Visualization: save histogram with fitted PDFs
    plt.figure(figsize=(8,5))
    plt.hist(heart_rates, bins=30, density=True, alpha=0.6, edgecolor='black')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 200)
    plt.plot(x, stats.norm.pdf(x, mu_mle, sigma_mle), label='Normal Fit', linewidth=2)
    plt.plot(x, stats.gamma.pdf(x, shape_mle, loc_mle, scale_mle),
             label='Gamma Fit', linewidth=2)
    plt.title('Resting Heart Rate Distribution with Fitted PDFs')
    plt.xlabel('Resting Heart Rate (bpm)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()
    
    # 8. Write results to text file
    with open(results_file, 'w') as f:
        f.write("Normal Distribution MLE:\n")
        f.write(f"  mu (mean) = {mu_mle:.4f}\n")
        f.write(f"  sigma (std)  = {sigma_mle:.4f}\n")
        f.write(f"  95% CI for mean: [{ci_mu[0]:.4f}, {ci_mu[1]:.4f}]\n")
        f.write(f"  95% CI for variance: [{ci_var[0]:.4f}, {ci_var[1]:.4f}]\n\n")
        f.write("Gamma Distribution MLE:\n")
        f.write(f"  shape (k) = {shape_mle:.4f}\n")
        f.write(f"  scale (theta) = {scale_mle:.4f}\n")
        f.write(f"  95% CI for shape: [{ci_shape[0]:.4f}, {ci_shape[1]:.4f}]\n")
        f.write(f"  95% CI for scale: [{ci_scale[0]:.4f}, {ci_scale[1]:.4f}]\n")
    
    print(f"Plot saved to: {plot_file}")
    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()

