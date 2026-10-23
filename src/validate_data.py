import pandas as pd
from pathlib import Path

data_dir = Path("data/raw")

for file in data_dir.glob("*.csv"):
    df = pd.read_csv(file)

    print("\nDataset:", file.name)
    print("Shape:", df.shape)
    print("Missing values:", df.isnull().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())

# Load related datasets
customers = pd.read_csv("data/raw/customers.csv")
restaurants = pd.read_csv("data/raw/restaurants.csv")
orders = pd.read_csv("data/raw/orders.csv")
deliveries = pd.read_csv("data/raw/deliveries.csv")
drivers = pd.read_csv("data/raw/drivers.csv")
reviews = pd.read_csv("data/raw/reviews.csv")
payments = pd.read_csv("data/raw/payments.csv")

# Check whether order customer IDs exist
print(
    "Invalid customer IDs:",
    (~orders["customer_id"].isin(customers["customer_id"])).sum()
)

# Check whether order restaurant IDs exist
print(
    "Invalid restaurant IDs:",
    (~orders["restaurant_id"].isin(restaurants["restaurant_id"])).sum()
)

# Check whether delivery order IDs exist
print(
    "Invalid delivery order IDs:",
    (~deliveries["order_id"].isin(orders["order_id"])).sum()
)

# Check whether delivery driver IDs exist
print(
    "Invalid driver IDs:",
    (~deliveries["driver_id"].isin(drivers["driver_id"])).sum()
)

# Check whether review order IDs exist
print(
    "Invalid review order IDs:",
    (~reviews["order_id"].isin(orders["order_id"])).sum()
)

# Check whether payment order IDs exist
print(
    "Invalid payment order IDs:",
    (~payments["order_id"].isin(orders["order_id"])).sum()
)