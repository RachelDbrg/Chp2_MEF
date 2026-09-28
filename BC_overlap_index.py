import numpy as np
import pandas as pd


## Creates an empty vector to store the overlap indexes
overlap_indexes = []
TimeStep_values = []


## Load the simulations .csv 
#data = pd.read_csv("simulations/more_overlap.csv")
data = pd.read_csv("simulations/less_overlap.csv")

## Test to filter time steps
#data = data[data["TimeStep"] < 10]


for TimeStep, group in data.groupby("TimeStep"): 

    print("TimeStep=", TimeStep)

    ## Check for negative values - because it will scrap the computation of the square sroot
    #print("Any negative C?", (group["C"] < 0).any())
    #print("Any negative P?", (group["P"] < 0).any())

    ## Store the columns values into vectors
    C = group["C"].values
    P = group["P"].values

    ## Compute the numerator, ie the square root of the product of C and P 
    num = np.sum(np.sqrt(C * P))

    ## Compute the denominator, ie the product of square roots of C and P
    denom = np.sqrt(np.sum(C)) * np.sqrt(np.sum(P))

    ## Compute the overlap
    overlap = num/denom

    ## Get the value for the overlap (should be inbetween 0 and 1)
    print("Overlap index a time", TimeStep, "=", overlap)


    TimeStep_values.append(TimeStep)
    overlap_indexes.append(overlap)

print(overlap_indexes)

# ------------------------------------------------------------------------------------------------

## Produce a graph to check how overlap changes through time

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(TimeStep_values, overlap_indexes, marker='o')
plt.xlabel("TimeStep")
plt.ylabel("Overlap Index")
plt.title("Overlap Index Through Time")
plt.grid(True)

plt.tight_layout()

plt.savefig("overlap_plot.png", dpi=300)  # saves to file
plt.close()  # closes the figure (good practice in scripts)