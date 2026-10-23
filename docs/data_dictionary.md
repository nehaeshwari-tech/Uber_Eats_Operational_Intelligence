# Uber Eats Operational & Customer Intelligence System
## Data Dictionary

### 1. Customers — customers.csv
- `customer_id`: Unique customer identifier.
- `age`: Customer age.
- `city`: Customer's city.
- `signup_date`: Date the customer registered.

### 2. Restaurants — restaurants.csv
- `restaurant_id`: Unique restaurant identifier.
- `restaurant_name`: Restaurant name.
- `cuisine`: Restaurant cuisine category.
- `rating`: Restaurant rating.
- `avg_prep_time_minutes`: Average food preparation time in minutes.

### 3. Orders — orders.csv
- `order_id`: Unique order identifier.
- `customer_id`: Customer who placed the order.
- `restaurant_id`: Restaurant receiving the order.
- `order_timestamp`: Date and time of the order.
- `order_amount`: Order value.
- `order_status`: Current order status.

### 4. Deliveries — deliveries.csv
- `delivery_id`: Unique delivery identifier.
- `order_id`: Associated order.
- `driver_id`: Assigned driver.
- `distance_km`: Delivery distance in kilometres.
- `delivery_time_minutes`: Delivery duration in minutes.
- `weather`: Weather conditions.
- `traffic`: Traffic conditions.

### 5. Drivers — drivers.csv
- `driver_id`: Unique driver identifier.
- `experience_years`: Driver's years of experience.
- `driver_rating`: Driver rating.

### 6. Reviews — reviews.csv
- `review_id`: Unique review identifier.
- `order_id`: Associated order.
- `customer_id`: Customer who submitted the review.
- `review_text`: Written customer feedback.
- `sentiment`: Generated sentiment label.
- `review_date`: Date and time associated with the review.

### 7. Payments — payments.csv
- `payment_id`: Unique payment identifier.
- `order_id`: Associated order.
- `payment_method`: Payment method used.
- `tip_amount`: Tip amount.
- `payment_status`: Payment status.

## Relationships
- Customers connect to Orders through `customer_id`.
- Restaurants connect to Orders through `restaurant_id`.
- Orders connect to Deliveries through `order_id`.
- Drivers connect to Deliveries through `driver_id`.
- Orders connect to Reviews through `order_id`.
- Orders connect to Payments through `order_id`.

## Initial Data Quality Notes

The synthetic datasets have been generated with consistent entity IDs and basic relationships.

Further improvements are required to ensure that delivery, review, payment and order statuses are logically consistent. The datasets will be reviewed and refined before machine learning model development.