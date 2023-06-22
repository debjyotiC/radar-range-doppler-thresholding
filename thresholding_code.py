import numpy as np

# Read matrix from NPZ file
range_doppler_features = np.load("data/umbc_indoor.npz", allow_pickle=True)

x_data, y_data = range_doppler_features['out_x'], range_doppler_features['out_y']

matrix = x_data[0] # read first data sample

configParameters = {'numDopplerBins': 16, 'numRangeBins': 256, 'rangeResolutionMeters': 0.146484375,
                    'rangeIdxToMeters': 0.146484375, 'dopplerResolutionMps': 0.1252347734553042, 'maxRange': 33.75,
                    'maxVelocity': 1.0018781876424336}

# Generate the range and doppler arrays for the plot
rangeArray = np.array(range(configParameters["numRangeBins"])) * configParameters["rangeIdxToMeters"]
dopplerArray = np.multiply(np.arange(-configParameters["numDopplerBins"] / 2, configParameters["numDopplerBins"] / 2),
                           configParameters["dopplerResolutionMps"])

# Create boolean masks for the specified conditions
y_mask = np.logical_or(dopplerArray < -0.2, dopplerArray > 0.2)
x_mask = rangeArray > (0.2 * rangeArray.max())

# Apply the masks to the matrix
masked_matrix = np.copy(matrix)
masked_matrix[y_mask, :] = 0
masked_matrix[:, x_mask] = 0

# Divide the unmasked segment into equal sections
num_sections = 4
section_length = masked_matrix.shape[0] // num_sections

# Calculate the maximum sum of squares and the ratio of Max/Mean for each section
max_sum_of_squares = []
ratios = []

for i in range(num_sections):
    section = masked_matrix[i * section_length: (i + 1) * section_length, :]
    section_sum_of_squares = np.sum(section ** 2)
    max_sum_of_squares.append(section_sum_of_squares)
    section_mean = np.mean(section)
    ratio = section_sum_of_squares / section_mean
    ratios.append(ratio)

# Compare the ratios with a predefined threshold
threshold = 10

# Check if any of the ratios exceed the threshold
exceed_threshold = any(ratio > threshold for ratio in ratios)

# Print the ratios and the result of the comparison
print("Ratios:", ratios)
print("Exceed threshold:", exceed_threshold)
