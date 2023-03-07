import csv
import math
import pandas as pd


# Open the CSV file
df = pd.read_csv("maze_lap3_samples_region.csv")

reward = df["reward"]
samples = df["samples"]

# Initialize variables
total_reward = 0
total_samples = 0

# Iterate over each row in the CSV file
for i in range(len(reward)):

    # Update the total reward and total samples
    total_reward += reward[i] * samples[i]
    total_samples += samples[i]

# Calculate the mean and standard deviation
mean_reward = total_reward / total_samples
variance = 0
for i in range(len(reward)):

    variance += samples[i] * ((reward[i] - mean_reward) ** 2)

std_deviation = math.sqrt(variance / total_samples)

# Print the results
print("Mean reward:", mean_reward)
print("Standard deviation:", std_deviation)
