# Prompt Engineering: Uber Eats Synthetic Data Generation

## Persona
Act as an experienced data engineer specializing in synthetic data generation and analytics pipelines.

## Task
Design and generate synthetic datasets for an Uber Eats Operational Intelligence and Customer Analytics project.

## Context
The project aims to analyze customer behaviour, restaurant performance, order patterns, delivery efficiency, customer sentiment, and payment behaviour.

Generate seven related datasets:
1. Customers
2. Restaurants
3. Orders
4. Deliveries
5. Drivers
6. Reviews
7. Payments

The initial dataset sizes are:
- Customers: 500 records
- Restaurants: 50 records
- Drivers: 100 records
- Orders: 3,000 records
- Deliveries: 3,000 records
- Reviews: 3,000 records
- Payments: 3,000 records

Use realistic field values and appropriate data types. Ensure that customer, restaurant, driver, and order identifiers are consistent across related datasets.

The datasets should support exploratory data analysis, customer and restaurant segmentation, sentiment analysis, delivery-time prediction, and demand forecasting.

## Format
Provide executable Python code using Pandas, NumPy, and pathlib.

The code should:
- Generate all seven datasets.
- Save each dataset as a separate CSV file in `data/raw/`.
- Use meaningful column names.
- Maintain consistent IDs across datasets.
- Print the shape of each generated dataset.
- Include basic validation checks.

Also document any assumptions and limitations in the generated data.