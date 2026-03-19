import numpy as np
import matplotlib.pyplot as plt
from filters import KalmanFilter1D

def generate_pulse_discharge(duration=20, pulses=15):
    """
    Simulates a Pulse Discharge Test (PDT) curve:
    Staircase voltage drop with relaxation recovery.
    """
    t = np.linspace(0, duration, 1000)
    v_start = 4.2
    v_end = 3.4
    
    # Base trend (declining voltage)
    v_base = np.linspace(v_start, v_end, 1000)
    
    # Add pulse dynamics (staircase)
    pulse_indices = np.linspace(0, 1000, pulses + 1, dtype=int)
    v_pdt = v_base.copy()
    
    for i in range(len(pulse_indices) - 1):
        idx_start = pulse_indices[i]
        idx_end = pulse_indices[i+1]
        
        # Pulse drop (current applied)
        drop_width = (idx_end - idx_start) // 10
        v_pdt[idx_start:idx_start+drop_width] -= 0.1
        
        # Relaxation phase (recovery)
        relax_idx = idx_start + drop_width
        recovery = 0.08 * (1 - np.exp(-(np.arange(idx_end - relax_idx) / 10)))
        v_pdt[relax_idx:idx_end] -= (0.1 - recovery)

    # Add realistic sensor noise (Experimental)
    noise = np.random.normal(0, 0.005, 1000)
    v_experimental = v_pdt + noise
    
    return t, v_experimental, v_pdt

def plot_kalman_validation():
    """
    Plots the Experimental vs Kalman Filtered data in the requested professional style.
    """
    t, v_raw, v_true = generate_pulse_discharge()
    
    # Apply Kalman Filter (BMS Implementation)
    kf = KalmanFilter1D(process_variance=1e-5, measurement_variance=2e-3)
    v_kalman = [kf.update(val) for val in v_raw]
    
    plt.figure(figsize=(14, 8))
    
    # 1. Raw Noisy Signal (Experimental)
    plt.plot(t, v_raw, color='black', alpha=0.3, label='Experimental (Noisy Sensor)', lw=1)
    
    # 2. Kalman Filtered Signal
    plt.plot(t, v_kalman, color='red', label='Kalman Filter (BMS Estimate)', lw=2.5)
    
    # 3. Ground Truth Physics (ECM Reference)
    plt.plot(t, v_true, color='blue', ls='--', alpha=0.5, label='Theoretical ECM Reference', lw=1.5)
    
    # Professional Styling
    plt.title("Kalman Filter Validation: Pulse Discharge Tracking", fontsize=16, fontweight='bold')
    plt.xlabel("Time [hours]", fontsize=12)
    plt.ylabel("Voltage [V]", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11, loc='upper right', frameon=True, shadow=True)
    
    plt.xlim(0, 20)
    plt.ylim(3.3, 4.3)
    
    plt.tight_layout()
    plt.savefig("kalman_validation.png", dpi=300)
    print("✅ Kalman Filter Validation Plot saved to kalman_validation.png")

if __name__ == "__main__":
    plot_kalman_validation()
