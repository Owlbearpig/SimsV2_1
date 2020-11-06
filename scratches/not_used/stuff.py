import numpy as np

d1 = np.array([18.49, 18.42, 18.44, 18.40, 18.46, 18.47, 18.41, 18.42, 18.48, 18.40])

d2= np.array([11.38, 11.40, 11.52, 11.71, 11.83, 11.75, 11.75, 11.67, 11.56, 11.57])

d3 = np.array([5.43, 5.46, 5.49, 5.68, 5.82, 5.65, 5.51, 5.43, 5.54, 5.81])

d4 = np.array([8.79, 8.76, 8.84, 9.06, 9.04, 8.97, 8.69, 8.81, 8.86, 9.04])

d5 = np.array([48.53, 48.63, 48.49, 48.54, 48.48, 48.39, 48.45, 48.62, 48.47, 48.93])

d6 = np.array([26.84, 26.87, 26.86, 26.76, 26.87, 26.89, 26.80, 26.80, 26.76, 26.76])

d7 = np.array([19.90, 19.67, 19.62, 19.64, 19.73, 19.75, 19.84, 19.59, 19.59, 19.74])

d8 = np.array([8.51, 8.59, 8.67, 8.67, 8.62, 8.48, 8.44, 8.54, 8.59, 8.57])

for d in [d1, d2, d3, d4, d5, d6, d7, d8]:
    print(np.round(np.mean(d), 2))
    print(np.round(np.std(d), 2))
    print()

