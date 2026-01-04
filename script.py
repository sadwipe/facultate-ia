import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Load data
subjects = pd.read_csv("subject_user.csv", low_memory=False)
consumption = pd.read_csv("consumption_user.csv", low_memory=False)

# Select food columns
food_columns = [col for col in consumption.columns if col.endswith("_g")]

# Preprocess
food_data = consumption[food_columns].fillna(0)
transactions = food_data > 0  # bool matrix

# Descriptive statistics
num_individuals = transactions.shape[0]
num_food_items = transactions.shape[1]
item_frequency = transactions.sum().sort_values(ascending=False)

# FP-Growth
frequent_itemsets = fpgrowth(
    transactions,
    min_support=0.10,
    use_colnames=True
)

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.60
)

rules = rules[rules['lift'] > 1.2]
rules_sorted = rules.sort_values(by="lift", ascending=False)

rules_sorted.head(10)
