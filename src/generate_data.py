import pandas as pd
import numpy as np
from pathlib import Path

# Make the results reproducible
np.random.seed(42)

# Create output folder
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Number of records
NUM_CUSTOMERS = 500
NUM_RESTAURANTS = 50
NUM_DRIVERS = 100
NUM_ORDERS = 3000

# -------------------------
# 1. CUSTOMERS
# -------------------------

customers = pd.DataFrame({
    "customer_id": range(1, NUM_CUSTOMERS + 1),
    "age": np.random.randint(18, 65, NUM_CUSTOMERS),
    "city": np.random.choice(
        ["Bengaluru", "Chennai", "Hyderabad", "Mumbai", "Delhi"],
        NUM_CUSTOMERS
    ),
    "signup_date": pd.to_datetime(
        np.random.choice(
            pd.date_range("2023-01-01", "2025-12-31"),
            NUM_CUSTOMERS
        )
    )
})

# -------------------------
# 2. RESTAURANTS
# -------------------------

restaurants = pd.DataFrame({
    "restaurant_id": range(1, NUM_RESTAURANTS + 1),
    "restaurant_name": [
        f"Restaurant_{i}" for i in range(1, NUM_RESTAURANTS + 1)
    ],
    "cuisine": np.random.choice(
        ["Indian", "Chinese", "Italian", "Mexican", "Fast Food"],
        NUM_RESTAURANTS
    ),
    "rating": np.round(
        np.random.uniform(2.5, 5.0, NUM_RESTAURANTS), 1
    ),
    "avg_prep_time_minutes": np.random.randint(
        10, 41, NUM_RESTAURANTS
    )
})

# -------------------------
# 3. DRIVERS
# -------------------------

drivers = pd.DataFrame({
    "driver_id": range(1, NUM_DRIVERS + 1),
    "experience_years": np.random.randint(0, 11, NUM_DRIVERS),
    "driver_rating": np.round(
        np.random.uniform(3.0, 5.0, NUM_DRIVERS), 1
    )
})

# -------------------------
# 4. ORDERS
# -------------------------

order_timestamps = pd.to_datetime(
    np.random.choice(
        pd.date_range(
            "2025-01-01",
            "2025-12-31 23:00:00",
            freq="h"
        ),
        NUM_ORDERS
    )
)

orders = pd.DataFrame({
    "order_id": range(1, NUM_ORDERS + 1),
    "customer_id": np.random.randint(
        1, NUM_CUSTOMERS + 1, NUM_ORDERS
    ),
    "restaurant_id": np.random.randint(
        1, NUM_RESTAURANTS + 1, NUM_ORDERS
    ),
    "order_timestamp": order_timestamps,
    "order_amount": np.round(
        np.random.uniform(100, 1500, NUM_ORDERS), 2
    ),
    "order_status": np.random.choice(
        ["Delivered", "Cancelled", "Preparing"],
        NUM_ORDERS,
        p=[0.88, 0.07, 0.05]
    )
})

# -------------------------
# 5. DELIVERIES
# -------------------------

deliveries = pd.DataFrame({
    "delivery_id": range(1, NUM_ORDERS + 1),
    "order_id": orders["order_id"],
    "driver_id": np.random.randint(
        1, NUM_DRIVERS + 1, NUM_ORDERS
    ),
    "distance_km": np.round(
        np.random.uniform(1, 20, NUM_ORDERS), 2
    ),
    "delivery_time_minutes": np.random.randint(
        10, 91, NUM_ORDERS
    ),
    "weather": np.random.choice(
        ["Clear", "Rainy", "Cloudy"],
        NUM_ORDERS
    ),
    "traffic": np.random.choice(
        ["Low", "Medium", "High"],
        NUM_ORDERS
    )
})

# -------------------------
# 6. REVIEWS
# -------------------------

positive_reviews = [
    "Great food and quick delivery",
    "Excellent taste and good packaging",
    "Loved the food and service",
    "Fresh food delivered on time",
    "Amazing experience"
]

negative_reviews = [
    "Food arrived late",
    "Food was cold",
    "Poor packaging and bad service",
    "Order was incorrect",
    "Very disappointing experience"
]

neutral_reviews = [
    "Food was okay",
    "Average service",
    "Delivery was acceptable",
    "Nothing special",
    "An ordinary experience"
]

review_texts = []
review_sentiments = []

for status in orders["order_status"]:
    sentiment = np.random.choice(
        ["Positive", "Negative", "Neutral"],
        p=[0.60, 0.25, 0.15]
    )

    if sentiment == "Positive":
        review_texts.append(np.random.choice(positive_reviews))
    elif sentiment == "Negative":
        review_texts.append(np.random.choice(negative_reviews))
    else:
        review_texts.append(np.random.choice(neutral_reviews))

    review_sentiments.append(sentiment)

reviews = pd.DataFrame({
    "review_id": range(1, NUM_ORDERS + 1),
    "order_id": orders["order_id"],
    "customer_id": orders["customer_id"],
    "review_text": review_texts,
    "sentiment": review_sentiments,
    "review_date": orders["order_timestamp"]
})

# -------------------------
# 7. PAYMENTS
# -------------------------

payments = pd.DataFrame({
    "payment_id": range(1, NUM_ORDERS + 1),
    "order_id": orders["order_id"],
    "payment_method": np.random.choice(
        ["UPI", "Card", "Cash", "Wallet"],
        NUM_ORDERS
    ),
    "tip_amount": np.round(
        np.random.uniform(0, 200, NUM_ORDERS), 2
    ),
    "payment_status": np.random.choice(
        ["Successful", "Failed"],
        NUM_ORDERS,
        p=[0.97, 0.03]
    )
})

# -------------------------
# SAVE ALL DATASETS
# -------------------------

datasets = {
    "customers": customers,
    "restaurants": restaurants,
    "orders": orders,
    "deliveries": deliveries,
    "drivers": drivers,
    "reviews": reviews,
    "payments": payments
}

for name, data in datasets.items():
    file_path = DATA_DIR / f"{name}.csv"
    data.to_csv(file_path, index=False)
    print(f"Saved {file_path} | Shape: {data.shape}")

print("\nAll seven datasets generated successfully!")