import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import channel_modelling_3GPP as cm
import best_response as br

eta = 1
epsilon = 1e-5
N = 5
max_iteration = 100
receiver_coordinates = (0, 0, 0)
transmitter_coordinates = [
    (7.5, 3.0, 2.25),
    (12.0, 6.0, 3.0),
    (16.5, 8.25, 3.75),
    (21.0, 10.5, 4.5),
    (25.5, 12.0, 5.25)
]
P_max = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

P = np.random.uniform(0.1, 1, N)  # random initial power
h = np.array([cm.get_channel_gain(tx, receiver_coordinates, 2405e6) for tx in transmitter_coordinates])

print("Channel gain values are:", h)
print("Initial Power values are:", P)

A, B, utility_histories, power_histories, df = br.adaptive_best_response(
    h, P_max, epsilon, max_iteration
)

# ----- Plot Utility Behaviour -----
plt.figure(figsize=(10, 6))
for c_val, history in utility_histories.items():
    iterations = np.arange(history.shape[0])
    for node in range(history.shape[1]):
        plt.plot(iterations, history[:, node], label=f"c={c_val:.1e}, Node {node+1}")
plt.xlabel("Iteration")
plt.ylabel("Utility")
plt.title("Utility Behaviour for Different c_max Trials")
plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# ----- Plot Power Behaviour -----
plt.figure(figsize=(10, 6))
for c_val, history in power_histories.items():
    iterations = np.arange(history.shape[0])
    for node in range(history.shape[1]):
        plt.plot(iterations, history[:, node], label=f"c={c_val:.1e}, Node {node+1}")
plt.xlabel("Iteration")
plt.ylabel("Power")
plt.title("Power Behaviour for Different c_max Trials")
plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

print(utility_histories)
print(power_histories)

