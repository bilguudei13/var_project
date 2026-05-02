import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import kurtosis, skew
import warnings
warnings.filterwarnings('ignore')

try:
    from arch import arch_model
except ImportError:
    print("Please install the 'arch' package to run GARCH models: pip install arch")
    sys.exit(1)

PROCESSED_DIR = "data/processed"
REPORT_DIR = "report/figures"

def plot_stylised_facts(series, name, show_plot=False):
    """
    Plots time series, ACF/PACF for raw and squared returns,
    and prints descriptive stats.
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f"Stylised Facts: {name}", fontsize=16)
    
    # 1. Time series plot
    axes[0, 0].plot(series, color='b', lw=0.8)
    axes[0, 0].set_title(f"Time Series: {name}")
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Histogram
    axes[0, 1].hist(series, bins=50, density=True, alpha=0.7, color='g')
    axes[0, 1].set_title("Histogram")
    
    # 3. ACF raw
    plot_acf(series, ax=axes[1, 0], title="ACF (Raw)", lags=40, zero=False, auto_ylims=True)
    
    # 4. PACF raw
    plot_pacf(series, ax=axes[1, 1], title="PACF (Raw)", lags=40, zero=False, auto_ylims=True)
    
    # 5. ACF squared
    plot_acf(series**2, ax=axes[2, 0], title="ACF (Squared)", lags=40, zero=False, auto_ylims=True)
    
    # 6. PACF squared
    plot_pacf(series**2, ax=axes[2, 1], title="PACF (Squared)", lags=40, zero=False, auto_ylims=True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    os.makedirs(REPORT_DIR, exist_ok=True)
    out_path = os.path.join(REPORT_DIR, f"stylised_facts_{name}.png")
    plt.savefig(out_path)
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # Descriptive Stats
    mean = series.mean()
    std = series.std()
    skw = skew(series.dropna())
    kurt = kurtosis(series.dropna(), fisher=True) # Excess kurtosis
    
    print(f"--- Descriptive Stats for {name} ---")
    print(f"Mean:            {mean:.6f}")
    print(f"Std Dev:         {std:.6f}")
    print(f"Skewness:        {skw:.6f}")
    print(f"Excess Kurtosis: {kurt:.6f}")
    if kurt > 1:
        print("  -> Fat tails detected (Excess Kurtosis > 1). Student-t is appropriate.")
    else:
        print("  -> No strong evidence of fat tails (Excess Kurtosis <= 1).")
    print("-" * 45)


def run_garch_pipeline(series, name):
    """
    Fits a GARCH(1,1) model with Student-t distribution on the series.
    Rescales series by 100 before fitting for numerical stability.
    """
    print(f"Fitting GARCH(1,1) with Student-t for {name}...")
    
    # Rescaling logic: series * 100 for better optimization 
    # (returns are small, which can cause optimizer issues)
    # We dynamically scale it based on standard deviation
    scale_factor = 1.0
    if series.std() < 0.1:
        scale_factor = 100.0
        
    rescaled_series = series * scale_factor
    
    # GARCH(1,1) with Student-t
    model = arch_model(rescaled_series, vol='Garch', p=1, q=1, dist='t')
    res = model.fit(disp='off')
    
    print(f"\n[GARCH Results for {name} (Scale={scale_factor})]")
    print(res.summary())
    print("\n" + "="*60 + "\n")
    return res

if __name__ == "__main__":
    file_path = os.path.join(PROCESSED_DIR, "risk_factors.csv")
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Run src/data/compute_returns.py first.")
        sys.exit(1)
        
    factors = pd.read_csv(file_path, index_col=0, parse_dates=True)
    print(f"Loaded {len(factors)} dates and {len(factors.columns)} risk factors.\n")
    
    for col in factors.columns:
        series = factors[col].dropna()
        
        # Step 2: Visual checks and descriptive stats
        plot_stylised_facts(series, col)
        
        # Step 1: Run GARCH pipeline on each series individually
        run_garch_pipeline(series, col)
        
    print(f"Stylised facts plots have been saved to {REPORT_DIR}/")