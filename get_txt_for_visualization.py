import numpy as np

data = np.load("dataset\\split6lanes\\Tile406.npy")
print(data.shape)

np.savetxt("testinf.txt", data, delimiter=",", fmt="%f")