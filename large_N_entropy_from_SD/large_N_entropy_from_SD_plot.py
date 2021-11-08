# Calculate the large N limit SYK entropy, S/N, by numerically solving the Schwinger Dyson equations,
# and then carrying out numerical integrations as suggested in Maldacena, Stanford
# 'Comments on the SYK model' Appendix G.

# Fit the linear part of S/N to read off the specific heat of the model

from large_N_entropy_from_SD import calculate_entropy
from matplotlib import pyplot as plt
import time

# Set variables for run
dt = 0.0001  # Time step - must divide t0 exactly.
q = 4  # Even integer ≥4. Number of fermions involved in random interactions in SYK model
iteration_length = 20
temp_range = 10 ** np.arange(-2, -1, 0.1)  # Set range of temperatures to calculate S/N for

SbyN = []

for temp in temp_range:

    runtime_start = time.time()

    entropy = calculate_entropy(dt, q, temp, iteration_length)
    SbyN.append(entropy)

    runtime_end = time.time()

    print("temp. = {}, entropy = {}, runtime = {}".format(temp, entropy, runtime_end-runtime_start))

# Set range of temperatures to pick out linear part of S/N
temp_start = 0
temp_stop = 4

# Fit linear part of S/N to read of specific heat. Expect intercept to be 0.2324
# and gradient (the specific heat) to be 0.396

poly = np.polyfit(temp_range[temp_start:temp_stop], SbyN[temp_start:temp_stop], 1)
line = np.poly1d(poly)
print(poly)

# Plot S/N, expected linear part of S/N, and straight line fit
plt.scatter(temp_range[:len(SbyN)], SbyN)

plt.plot(temp_range[0:10], 0.2324 + 0.396 * temp_range[0:10],
         color='purple', label="expected linear part of S/N")
plt.plot(temp_range[temp_start:temp_stop], line(temp_range[temp_start:temp_stop]),
         color='red', label="straight line fit")
plt.legend()
plt.show()






