# # -*- coding: utf-8 -*-
# """
# Created on Sun May 25 11:41:57 2025

# @author: Shay
# """

# import numpy as np

# def expected_pi2_behavior(n_pulses):
#     """Ideal behavior for pi/2 pulses: oscillates between 0 and 1 as sin^2(n * pi/2)."""
#     return np.sin(n_pulses * np.pi / 2) ** 2

# def compute_error(measured, expected):
#     """Mean squared error between measured and expected outcomes."""
#     return np.mean((np.array(measured) - np.array(expected))**2)

# def calibrate_amplitude_once(amplitude, pulse_sequence_results, n_pulses_list, learning_rate=0.05):
#     """
#     Given current amplitude and measurement results, compute corrected amplitude.
    
#     pulse_sequence_results: list of measured qubit P(excited) after n pi/2 pulses.
#     n_pulses_list: list of how many pulses were used in each experiment.
#     """
#     expected = expected_pi2_behavior(np.array(n_pulses_list))
#     error = compute_error(pulse_sequence_results, expected)

#     # Estimate slope direction via finite difference (central diff)
#     delta = 0.01
#     amp_up = amplitude + delta
#     amp_down = amplitude - delta
    
#     # Simulate the expected results if amplitude were slightly up/down
#     # Here we fake sensitivity as linear just for the sake of illustration
#     # In real case, you'd run `run_sequence(amp_up)` and `run_sequence(amp_down)`
#     sim_up = np.clip(expected + 0.1 * delta, 0, 1)
#     sim_down = np.clip(expected - 0.1 * delta, 0, 1)
    
#     error_up = compute_error(pulse_sequence_results, sim_up)
#     error_down = compute_error(pulse_sequence_results, sim_down)

#     # Gradient descent step
#     gradient = (error_up - error_down) / (2 * delta)
#     new_amplitude = amplitude - learning_rate * gradient

#     return new_amplitude, error

# def calibrate_until_converged(initial_amplitude, get_measurement_data_fn,
#                                n_pulses_list, threshold=1e-3, max_iters=50):
#     """
#     Loop until the error is below the threshold. 
#     """
#     amplitude = initial_amplitude
#     for i in range(max_iters):
#         results, err = get_measurement_data_fn(amplitude)
#         amplitude, error = calibrate_amplitude_once(amplitude, results, n_pulses_list)
#         print(f"Iteration {i+1}: amplitude={amplitude:.4f}, error={error:.6f}")
#         if error < threshold:
#             break
#     return amplitude

import numpy as np

def expected_sigma_z(n_pulses, fidelity_0=1.0, fidelity_1=1.0):
    """
    Returns expected ⟨σ_z⟩ values distorted by readout fidelity.
    fidelity_0: P(measuring 0 | in 0)
    fidelity_1: P(measuring 1 | in 1)
    """
    ideal = -np.cos(n_pulses * np.pi / 2)
    contrast = fidelity_0 + fidelity_1 - 1
    bias = -fidelity_0 + fidelity_1
    return contrast * ideal + bias

def compute_weighted_error(measured, expected, errors):
    """Weighted MSE: each term weighted by 1/σ²."""
    weights = 1 / (np.array(errors)**2 + 1e-12)  # avoid div by 0
    return np.sum(weights * (np.array(measured) - np.array(expected))**2) / np.sum(weights)

def calibrate_amplitude_once(amplitude, get_measurement_data_fn, n_pulses_list,
                              learning_rate=0.05, max_step_pct=0.05,
                              fidelity_0=1.0, fidelity_1=1.0):
    """
    Adjust amplitude using real measurement gradients.
    """
    delta = 0.01 * amplitude  # small % perturbation

    # Get true data at nominal, up, and down amplitudes
    results, errors = get_measurement_data_fn(amplitude)
    results_up, errors_up = get_measurement_data_fn(amplitude + delta)
    results_down, errors_down = get_measurement_data_fn(amplitude - delta)

    expected = expected_sigma_z(n_pulses_list, fidelity_0, fidelity_1)

    # Compute weighted errors
    error = compute_weighted_error(results, expected, errors)
    error_up = compute_weighted_error(results_up, expected, errors_up)
    error_down = compute_weighted_error(results_down, expected, errors_down)

    # Estimate gradient
    gradient = (error_up - error_down) / (2 * delta)

    # Compute step, clamp to max_step_pct
    step = -learning_rate * gradient
    max_step = max_step_pct * abs(amplitude)
    step = np.clip(step, -max_step, max_step)

    new_amplitude = amplitude + step
    return new_amplitude, error


def calibrate_until_converged(initial_amplitude, get_measurement_data_fn,
                               n_pulses_list, threshold=1e-3, max_iters=50,
                               learning_rate=0.05, max_step_pct=0.05,
                               fidelity_0=1.0, fidelity_1=1.0):
    amplitude = initial_amplitude
    for i in range(max_iters):
        amplitude, error = calibrate_amplitude_once(
            amplitude, get_measurement_data_fn, n_pulses_list,
            learning_rate=learning_rate,
            max_step_pct=max_step_pct,
            fidelity_0=fidelity_0,
            fidelity_1=fidelity_1
        )
        print(f"Iteration {i+1}: amplitude={amplitude:.6f}, weighted_error={error:.6f}")
        if error < threshold:
            break
    return amplitude