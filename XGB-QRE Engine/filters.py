import numpy as np

class KalmanFilter1D:
    """
    Simple 1D Kalman Filter for smoothing noisy battery sensor data (SOH, Capacity).
    """
    def __init__(self, process_variance=1e-5, estimation_error=1.0, measurement_variance=1e-3):
        self.process_variance = process_variance
        self.estimation_error = estimation_error
        self.measurement_variance = measurement_variance
        self.current_estimate = None
        self.last_estimate = None

    def update(self, measurement):
        if self.current_estimate is None:
            self.current_estimate = measurement
            self.last_estimate = measurement
            return measurement

        # Prediction Phase
        predicted_estimate = self.last_estimate
        predicted_error = self.estimation_error + self.process_variance

        # Measurement Update Phase (Kalman Gain)
        kalman_gain = predicted_error / (predicted_error + self.measurement_variance)
        self.current_estimate = predicted_estimate + kalman_gain * (measurement - predicted_estimate)
        self.estimation_error = (1 - kalman_gain) * predicted_error

        self.last_estimate = self.current_estimate
        return self.current_estimate

def apply_kalman_smoothing(series, process_var=1e-6, measure_var=1e-4):
    """
    Applies Kalman filtering to a pandas series/numpy array.
    """
    kf = KalmanFilter1D(process_variance=process_var, measurement_variance=measure_var)
    smoothed = []
    for val in series:
        smoothed.append(kf.update(val))
    return np.array(smoothed)
