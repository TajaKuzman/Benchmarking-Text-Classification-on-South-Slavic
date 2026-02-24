import pandas as pd

results = pd.read_csv("results.csv")

print(results.head(3).to_markdown())

models = list(results["model"].unique())

print("List of unique models in the results:\n")
print(models)

tasks = list(results["task"].unique())

print("\nList of tasks:\n")

print(tasks)