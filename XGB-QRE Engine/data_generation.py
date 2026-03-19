import pandas as pd
import numpy as np
from filters import apply_kalman_smoothing

def generate_battery_data(num_batteries=50, cycles_per_battery=3500):
    """
    Generates synthetic battery degradation data anchored to LFP chemistry characteristics.
    Incorporates multi-phase degradation, non-linear physics factors, and Kalman smoothing.
    """
    initial_capacity = 35.0  # Ah
    all_battery_data = []
    
    for b in range(num_batteries):
        capacity = initial_capacity
        battery_factor = np.random.uniform(0.9, 1.1)  # Manufacturing variance
        
        # Buffers for cycle-by-cycle metrics
        ambient_temps = []
        peak_temps = []
        dods = []
        discharge_cs = []
        charge_cs = []
        soh_history = []
        cap_history = []
        
        # Initial state
        temp_history_buffer = []
        dod_history_buffer = []
        window_size = 20
        
        for i in range(1, cycles_per_battery + 1):
            # Operational parameters
            ambient_temp = np.random.uniform(20, 35)
            peak_temp = ambient_temp + np.random.uniform(5, 15)
            dod = np.random.uniform(60, 95)
            discharge_c = np.random.uniform(1.0, 3.0)
            charge_c = np.random.uniform(0.5, 1.5)
            
            # 1. Physics-inspired acceleration factors
            temp_accel = np.exp((peak_temp - 25) / 10) 
            dod_stress = (dod / 100) ** 1.3
            c_rate_impact = (discharge_c / 2.0) ** 1.1
            
            base_degradation = 0.00004 * battery_factor
            stress = base_degradation * temp_accel * dod_stress * c_rate_impact
            
            # 2. Elite Multi-phase degradation logic
            soh_current = (capacity / initial_capacity) * 100
            if soh_current > 95:
                stress *= 0.6  # Ultra-slow early fade
            elif soh_current > 90:
                stress *= 0.7  # SEI stabilization
            elif soh_current < 85:
                stress *= 1.6  # Late phase "knee"
            
            # Apply degradation
            capacity -= stress * initial_capacity
            capacity = max(capacity, initial_capacity * 0.6)  # Hard floor
            
            soh = (capacity / initial_capacity) * 100
            
            # Store raw cycle data
            ambient_temps.append(ambient_temp)
            peak_temps.append(peak_temp)
            dods.append(dod)
            discharge_cs.append(discharge_c)
            charge_cs.append(charge_c)
            soh_history.append(soh)
            cap_history.append(capacity)
            
            if soh < 65:
                break
        
        # 3. Post-Process Battery Data (Add Noise & Kalman Filter)
        noise = np.random.normal(0, 0.00015, len(soh_history))
        soh_noisy = np.array(soh_history) + noise
        cap_noisy = np.array(cap_history) + (noise * initial_capacity)

        df_battery = pd.DataFrame({
            "Battery_ID": [b] * len(soh_noisy),
            "Cycle_Index": list(range(1, len(soh_noisy) + 1)),
            "Avg_Ambient_Temp": ambient_temps,
            "Peak_Cell_Temp": peak_temps,
            "Daily_DoD": dods,
            "Max_Discharge_C_Rate": discharge_cs,
            "Max_Charge_C_Rate": charge_cs,
            "State_of_Health": soh_noisy,
            "Present_Capacity": cap_noisy
        })
        
        # --- PRO UPGRADE: Signal Processing & Feature Engineering ---
        # Apply Kalman Smoothing
        df_battery["SOH_Filtered"] = apply_kalman_smoothing(df_battery["State_of_Health"], process_var=1e-7, measure_var=1e-4)
        df_battery["Cap_Filtered"] = apply_kalman_smoothing(df_battery["Present_Capacity"], process_var=1e-6, measure_var=1e-3)
        
        # Velocity Feature (Degradation Gradient)
        df_battery["SOH_Delta"] = df_battery["SOH_Filtered"].diff().fillna(0)
        
        # Rolling stats for thermal memory
        df_battery["Rolling_Avg_Temp"] = df_battery["Peak_Cell_Temp"].rolling(window=window_size, min_periods=1).mean()
        df_battery["Rolling_Avg_DoD"] = df_battery["Daily_DoD"].rolling(window=window_size, min_periods=1).mean()
        
        all_battery_data.append(df_battery)
                
    return pd.concat(all_battery_data, ignore_index=True)

def calculate_rul(df, eol_threshold=80):
    """
    Calculates Remaining Useful Life (RUL) based on the first occurrence of EOL.
    """
    df["Target_RUL_Cycles"] = np.nan
    
    for b_id in df["Battery_ID"].unique():
        mask = df["Battery_ID"] == b_id
        battery_data = df[mask].copy().reset_index(drop=True)
        
        eol_indices = battery_data[battery_data["State_of_Health"] <= eol_threshold].index
        
        if not eol_indices.empty:
            eol_cycle_local = eol_indices[0]
            battery_data["Target_RUL_Cycles"] = eol_cycle_local - battery_data.index
            
            global_indices = df[mask].index
            df.loc[global_indices, "Target_RUL_Cycles"] = battery_data["Target_RUL_Cycles"].values
            
    df = df.dropna(subset=['Target_RUL_Cycles']).copy()
    df = df[df["Target_RUL_Cycles"] >= 0]
    return df

if __name__ == "__main__":
    print("Generating LFP battery data with Kalman filtering...")
    raw_df = generate_battery_data(num_batteries=60)
    print(f"Cycles generated: {len(raw_df)}")
    
    processed_df = calculate_rul(raw_df)
    print(f"Cycles after RUL calculation: {len(processed_df)}")
    
    processed_df.to_csv("battery_dataset_final.csv", index=False)
    print("Data saved to battery_dataset_final.csv")
