import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("economic_indicators.csv")

print(df.head())

plt.figure(figsize=(10, 5))
plt.hist(df['GDP_per_capita'], bins=30, color='blue', alpha=0.7)
plt.title('Distribution of GDP per Capita')
plt.xlabel('GDP per Capita')
plt.ylabel('Frequency')
plt.show()