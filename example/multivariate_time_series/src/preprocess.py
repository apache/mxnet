#read in the txt file
import pandas as pd
import numpy as np

#read in the data
df = pd.read_csv("../data/electricity.txt", sep=",", header = None)

#extract feature values
feature_df = df.iloc[:, :].astype(float)

#convert to numpy matrix
x = feature_df.as_matrix()

#save files
np.save("../data/electric.npy", x)
