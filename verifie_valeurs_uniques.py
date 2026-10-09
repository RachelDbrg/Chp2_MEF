import pandas as pd

df = pd.read_csv("/users/rdubourg/Documents/Chp2_MEF/simulations/Nelder_Mead/NM_SM1/iter_0000/PDE_output.csv")

vals = df[df["Field"] == "laplacien_Phi_C"]["Value"]

print(vals.head())
print(vals.describe())

print("\nValeurs uniques :")
for v in sorted(vals.unique()):
    print(v)