import numpy as np
import pandas as pd

def tuning_curve(s, r_max, s_max, sigma_f):
    return r_max * np.exp(-0.5 * ((s - s_max) / sigma_f)**2)

def simulate_tuning_data(n_points=30, noise_std=2.0, 
                         r_max=52, s_max=0, sigma_f=15, 
                         s_range=(-50, 50), seed=42):
    np.random.seed(seed)
    
    s = np.linspace(s_range[0], s_range[1], n_points)
    response_clean = tuning_curve(s, r_max, s_max, sigma_f)
    response_noisy = response_clean + np.random.normal(0, noise_std, size=s.shape)
    
    df = pd.DataFrame({
        "s": s,
        "response": response_noisy,
        "error": np.full_like(s, noise_std)
    })
    
    csv_file = "tuning_data.csv" 
    df.to_csv(csv_file, index=False)

    return df

