import numpy as np
import pandas as pd
import numpy.typing as npt


def utility_vector(P: npt.NDArray[np.float64], h: npt.NDArray[np.float64], A: np.float64):
    utilities = np.zeros(len(P))
    sum_1 = np.sum(h * np.sqrt(P))
    for n in range(len(P)):
        gain_term = A * h[n] * np.sqrt(P[n]) * (sum_1 - h[n] * np.sqrt(P[n])) + sum_1 - h[n] * np.sqrt(P[n])
        penalty_term = np.power(h[n], 2) * P[n]
        utilities[n] = gain_term - penalty_term
    return utilities


def closed_form_p_star(P: npt.NDArray[np.float64], h: npt.NDArray[np.float64], P_max: npt.NDArray[np.float64], A: np.float64):
    optimal_power = []
    sum_1 = np.sum(h * np.sqrt(P))
    for i in range(len(P)):
        P_optimal = np.power(A/(2*h[i]) * (sum_1 - h[i] * np.sqrt(P[i])), 2)
        P_optimal = np.clip(P_optimal, 0, P_max[i])
        optimal_power.append(P_optimal)
    return np.array(optimal_power)


def find_appropriate_A(h: npt.NDArray[np.float64], P_max: npt.NDArray[np.float64]):
    sum_1 = np.sum(h * np.sqrt(P_max))
    A_vec = np.zeros(len(h))
    for i in range(len(h)):
        A_vec[i] = 2 *h[i] * np.sqrt(P_max[i])/(sum_1 - h[i] * np.sqrt(P_max[i]))
    A_max = np.min(A_vec)
    return A_max


def adaptive_best_response(h: npt.NDArray[np.float64], P_max: npt.NDArray[np.float64], epsilon: float, max_iteration: int = 10):
    A = find_appropriate_A(h, P_max)
    print(f"A: {A}")

    P = np.random.uniform(0.1, 1.0, len(P_max))  # random initial power
    found = False
    history_util = {}
    history_power = {}
    # A = np.float64(1.0)

    while not found and A > 1e-30:
        print(f"\n=== Testing A = {A} ===")

        P_current = P.copy()
        utility_history = [utility_vector(P_current, h, A)]
        power_history = [P_current.copy()]
        convergence = 0
        iteration = 0
        stop_due_to_negative = False

        while convergence == 0:
            print(iteration)
            iteration += 1
            P_new = closed_form_p_star(P_current, h, P_max, A)
            utilities_new = utility_vector(P_new, h, A)
            utility_history.append(utilities_new.copy())
            power_history.append(P_new.copy())

            if np.any(utilities_new < 0):
                print(f"⚠ Negative utility at iteration {iteration}, increasing A...")
                stop_due_to_negative = True
                break

            if np.all(np.abs(P_new - P_current) < epsilon) or iteration == max_iteration:
                convergence = 1
            else:
                P_current = P_new

        history_util[A] = np.array(utility_history)
        history_power[A] = np.array(power_history)

        if not stop_due_to_negative:
            found = True
            print(f"✅ Found feasible A = {A}, converged in {iteration} iterations.")
            df = pd.DataFrame({
                "Node": np.arange(1, len(P) + 1),
                "Final Power P*": P_new,
                "Final Utility": utilities_new,
                "Channel Gain h": h
            })
            print(df)
            return A, history_util, history_power, df

        A /= 2 
        # break

    print("Could not find a feasible A")
    return None, history_util, history_power, None
