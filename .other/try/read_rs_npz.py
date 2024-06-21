import numpy as np

# Read npz
filepath = "/home/wuch/Downloads/arrays_rs.npz"

data = np.load(filepath)

# Print keys
print(data.files)

# Print shape of each array
for key in data.files:
    print(key, data[key].shape)

# Access arrays
print(data['a'])
print(data['b'])
