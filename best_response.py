import numpy as np
import pandas as pd
import numpy.typing as npt


def utility_vector(P: npt.NDArray[np.float64], h: npt.NDArray[np.float64], A: np.float64, B: np.float64):
    utilities = np.zeros(len(P))
    for n in range(len(P)):
        gain_term = B * np.power(np.sum(h[n] * np.sqrt(P)), 2)
        penalty_term = A * np.sum(np.power(h[n], 2) * P)
        utilities[n] = gain_term - penalty_term
    return utilities


def closed_form_p_star(P: npt.NDArray[np.float64], h: npt.NDArray[np.float64], A: np.float64, B: np.float64):
    optimal_power = []
    sum_1 = np.sum(h * np.sqrt(P))
    print(f"sum_1: {sum_1}")
    for i in range(len(P)):
        P_optimal = np.power(B/(A-B) * (sum_1 - h[i] * np.sqrt(P[i])), 2)
        optimal_power.append(P_optimal)
    return np.array(optimal_power)


def find_appropriate_A_over_B(h: npt.NDArray[np.float64], P_max: npt.NDArray[np.float64]):
    sum_1 = np.sum(h * np.sqrt(P_max))
    c_vec = np.zeros(len(h))
    for i in range(len(h)):
        c_vec[i] = np.sqrt(P_max[i])/(sum_1 - h[i] * np.sqrt(P_max[i]))
    c_max = np.min(c_vec)
    return c_max/(c_max + 1)


def adaptive_best_response(h: npt.NDArray[np.float64], P_max: npt.NDArray[np.float64], epsilon: float, max_iteration: int = 10):
    A_over_B = find_appropriate_A_over_B(h, P_max)
    print(f"A_over_B: {A_over_B}")

    P = np.random.uniform(0.1, 1.0, len(P_max))  # random initial power
    found = False
    history_util = {}
    history_power = {}

    A = np.float64(2.0)
    B = np.float64(A_over_B)
    while not found and A < 1e15:
        print(f"\n=== Testing A = {A} and B = {B} ===")

        P_current = P.copy()
        utility_history = [utility_vector(P_current, h, A, B)]
        power_history = [P_current.copy()]
        convergence = 0
        iteration = 0
        stop_due_to_negative = False

        while convergence == 0:
            print(iteration)
            iteration += 1
            P_new = closed_form_p_star(P_current, h, A, B)
            utilities_new = utility_vector(P_new, h, A, B)
            utility_history.append(utilities_new.copy())
            power_history.append(P_new.copy())

            print(f"power: {P_new}")
            print(f"utility: {utilities_new}")

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
            print(f"✅ Found feasible A = {A}, B = {B}, converged in {iteration} iterations.")
            df = pd.DataFrame({
                "Node": np.arange(1, len(P) + 1),
                "Final Power P*": P_new,
                "Final Utility": utilities_new,
                "Channel Gain h": h
            })
            print(df)
            return A, B, history_util, history_power, df

        A += 1
        B = A_over_B * A

    print("Could not find a feasible A up to 1e15")
    return None, None, history_util, history_power, None
