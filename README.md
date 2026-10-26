# Uber Eats Operational & Customer Intelligence System

## Project Overview
This project uses data analytics and machine learning to analyze customer behaviour, restaurant performance, delivery operations, customer sentiment, and order demand.

## Objectives
- Segment customers and restaurants.
- Analyze customer reviews using Natural Language Processing (NLP).
- Predict delivery time or tip amount using regression models.
- Forecast hourly order demand.
- Generate actionable business insights and recommendations.

## Technology Stack
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- NLP tools
- Time-series forecasting libraries

## Project Structure
- `data/raw/`: Original synthetic datasets
- `data/processed/`: Cleaned datasets
- `notebooks/`: Exploratory data analysis and experiments
- `src/`: Python scripts and machine learning pipelines
- `outputs/visualizations/`: Generated charts
- `docs/`: Project documentation

## Project Status
Day 1: Project setup and synthetic data generation.

## Customer and Restaurant Segmentation

### Overview
This phase of the Uber Eats Operational Intelligence project uses clustering techniques to identify customer ordering patterns and restaurant performance groups.

### Methodology
- **Feature Preparation:** Selected relevant numerical features for customer and restaurant segmentation.
- **Data Scaling:** Standardized numerical features using `StandardScaler`.
- **K-Means Clustering:** Used the Elbow Method and Silhouette Score to evaluate suitable cluster counts.
- **DBSCAN Clustering:** Compared parameter combinations to identify density-based clusters and potential outliers.
- **PCA Visualization:** Reduced the feature dimensions to two principal components for visualization.
- **Cluster Profiling:** Summarized customer behaviour and restaurant performance for interpretation.

### Business Insights and Recommendations

#### Customer Segmentation
Customer clusters can reveal differences in ordering frequency, spending patterns, weekend activity, and late-night orders. These insights can help inform targeted promotions, loyalty programmes, and customer-retention strategies.

#### Restaurant Segmentation
Restaurant clusters can reveal differences in revenue, order volume, ratings, cancellation rates, preparation times, and delivery times. These insights can help identify opportunities to improve restaurant operations.

#### Model Comparison
K-Means groups records into a predefined number of clusters, while DBSCAN identifies density-based groups and potential outliers. Comparing both approaches helps explore different patterns in customer behaviour and restaurant performance.

#### Recommendations
- Develop targeted promotions based on customer ordering and spending behaviour.
- Investigate restaurants with relatively high cancellation rates or long preparation times.
- Use restaurant ratings and order performance to identify operational improvement opportunities.
- Validate recommendations using the actual cluster profiles before implementation.

### Output Files
The processed segmentation datasets are saved in `data/processed/`:

- `customer_segments.csv` — Customer features with K-Means and DBSCAN cluster labels.
- `restaurant_segments.csv` — Restaurant features with K-Means and DBSCAN cluster labels.