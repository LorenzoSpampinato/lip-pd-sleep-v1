import numpy as np
import pandas as pd

# Path to your .npz file
npz_file_path = r"D:\TESI\lid-data-samples\lid-data-samples\Results\Features\DYS\PD012\\PD012prova_all_feats.npz"

# Load the .npz file
npz_data = np.load(npz_file_path)

# Check the keys in the npz file
print("Keys in the npz file:", npz_data.files)

# Extract the data array (shape (6, 61, 870))
data = npz_data['data']
feats = npz_data['feats']  # Assuming feats contains feature labels
regions = npz_data['regions']  # Regions, if relevant for indexing

# Select one slice (e.g., first slice: data[0], which corresponds to the first of the 6 signals)
data_slice = data[0]  # Shape (61, 870)

# Transpose the slice to have epochs as rows and features as columns (shape becomes (870, 61))
data_transposed = data_slice.T  # Now it's (870, 61)

# Convert the transposed data to a DataFrame
df_data = pd.DataFrame(data_transposed, columns=feats)

# Print the DataFrame's structure
print(f"Shape of the DataFrame: {df_data.shape}")
print(df_data.head())  # Display the first few rows

# Optional: you can scroll through the DataFrame if you're working in a notebook environment
# In case you want to visualize it in a scrollable format
from IPython.display import display
display(df_data)