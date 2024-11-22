import pandas as pd

# Sample data (adjust with actual data)
data = {
    'price': [100, 102, 101, 105, 103, 106]
}
df = pd.DataFrame(data)

# Step 1: Compute the 'X' indicator (percentage change from previous price)
df['X'] = df['price'].pct_change()

# Set the first value of X to be 0
df['X'].iloc[0] = 0

# Define parameters for the algorithm
TP = 0.1      # Take Profit (10%)
SL = 0.05      # Stop Loss (5%)
W = 3          # Window size (look ahead W ticks)
MAX_OFFSET = len(df)  # Maximum offset (use the length of the data as the limit)

# Step 2: Define function to calculate the 'B' indicator for each row
def calculate_B(i, X, TP, SL, W, MAX_OFFSET):
    t = 1
    for j in range(i + 1, min(i + W, MAX_OFFSET)):  # Loop within window size or MAX_OFFSET
        t *= (1 + X[j])
        if t - 1 >= TP or t - 1 <= -SL:
            return t - 1  # Return the profit/loss if the threshold is met
    return t - 1  # If the loop completes, return the final value of t - 1

# Step 3: Apply the function to calculate the 'B' indicator for each index
df['B'] = df['X'].apply(lambda i: calculate_B(i, df['X'], TP, SL, W, MAX_OFFSET))

# Display the result
print(df)
