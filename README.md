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

# Day 3: Customer Feedback Analysis Using NLP

## 1. Objective

The objective of Day 3 was to analyze customer reviews using Natural Language Processing (NLP) techniques, classify customer sentiment, compare machine learning models, and identify common customer complaints. The analysis helps understand customer satisfaction and identify areas where Uber Eats' delivery and restaurant services can improve.

## 2. Dataset Overview

The customer review dataset was loaded and examined to understand its structure and quality.

- **Total records:** 3,000
- **Total columns:** 6
- **Columns:** `review_id`, `order_id`, `customer_id`, `review_text`, `sentiment`, `review_date`

The dataset contains customer review text, sentiment labels, order and customer identifiers, and review dates.

## 3. Data Quality Assessment

The dataset was checked for missing values, duplicate records, duplicate review IDs, and empty review text.

### Data Quality Results

- Missing values: 0 across all columns.
- Duplicate rows: 0.
- Duplicate review IDs: 0.
- Empty or whitespace-only reviews: 0.

These checks confirmed that the dataset did not contain the identified data quality issues.

## 4. Sentiment Distribution Analysis

The distribution of sentiment labels was analyzed to understand the overall pattern of customer feedback.

| Sentiment | Number of Reviews | Percentage |
|---|---:|---:|
| Positive | 1,766 | 58.87% |
| Negative | 779 | 25.97% |
| Neutral | 455 | 15.17% |
| **Total** | **3,000** | **100% approximately** |

### Key Observations

- Positive reviews account for the largest share of customer feedback.
- Approximately one-quarter of the reviews are negative, indicating opportunities to improve the customer experience.
- Neutral reviews provide additional feedback that may help identify areas requiring attention.

## 5. Text Preprocessing

The review text was cleaned and normalized before applying machine learning techniques.

The preprocessing steps included:

1. Converting text to lowercase.
2. Removing URLs.
3. Removing punctuation and non-alphabetic characters.
4. Tokenizing the text into individual words.
5. Removing English stopwords while retaining important negation words such as `no`, `nor`, and `not`.
6. Removing very short tokens.
7. Applying lemmatization to normalize words.
8. Combining the cleaned tokens into processed review text.

The processed text was stored in a new column named `processed_review`.

The cleaned dataset was saved as:

`data/processed/cleaned_reviews.csv`

## 6. Feature Extraction Using TF-IDF

Term Frequency–Inverse Document Frequency (TF-IDF) was used to convert the processed review text into numerical features suitable for machine learning.

### Configuration

- **Technique:** TF-IDF Vectorization
- **Maximum features:** 3,000
- **N-gram range:** (1, 2), including individual words and two-word combinations
- **Number of reviews:** 3,000
- **Number of extracted features:** 54
- **TF-IDF matrix shape:** (3000, 54)

The extracted features included terms and phrases such as `acceptable`, `amazing`, `amazing experience`, `arrived`, and `arrived late`.

### Saved Artifacts

- `data/processed/reviews_tfidf.npz`
- `data/processed/tfidf_vectorizer.pkl`

## 7. Sentiment Classification Using Machine Learning

Two machine learning models were trained to classify customer reviews into Positive, Negative, and Neutral categories.

### Models Used

**1. Multinomial Naive Bayes**

Multinomial Naive Bayes is a probabilistic classification algorithm commonly used for text classification tasks. It estimates the probability of a sentiment class based on the observed text features.

**2. Linear Support Vector Machine (Linear SVM)**

Linear SVM identifies a decision boundary that separates sentiment classes in the feature space. It is commonly used for high-dimensional text classification problems.

### Training and Testing

- **Train-test split:** 80:20
- **Training samples:** 2,400
- **Testing samples:** 600
- **Random state:** 42
- **Stratification:** Applied to preserve the distribution of sentiment classes.

### Evaluation Metrics

The models were evaluated using:

- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1-score

### Model Evaluation Results

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-score |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 1.00 | 1.00 | 1.00 | 1.00 |
| Linear SVM | 1.00 | 1.00 | 1.00 | 1.00 |

Both models achieved 100% on the reported evaluation metrics for the current test split.

**Important observation:** These perfect scores are unusually high for a real-world sentiment classification task. The results should be validated further by checking repeated review text, potential overlap between similar reviews in the training and testing sets, and performance on genuinely unseen reviews. The reported scores should not be treated as proof of generalization until these checks are completed.

### Saved Artifacts

- `data/processed/model_evaluation_report.csv`
- `data/processed/naive_bayes_sentiment_model.pkl`
- `data/processed/svm_sentiment_model.pkl`

## 8. Confusion Matrix Visualization

Confusion matrices were generated for both trained models to compare actual sentiment labels with predicted labels.

The visualizations were saved as:

- `outputs/visualizations/naive_bayes_confusion_matrix.png`
- `outputs/visualizations/svm_confusion_matrix.png`

These plots help inspect classification performance across the three sentiment categories.

## 9. Sentiment Analysis Using VADER

VADER (Valence Aware Dictionary and sEntiment Reasoner) was applied to the original customer review text as a lexicon-based sentiment analysis approach.

The analysis involved:

1. Initializing the VADER sentiment analyzer.
2. Calculating sentiment scores for customer reviews.
3. Extracting the compound sentiment score.
4. Assigning sentiment labels based on compound score thresholds.

The classification rules used were:

- Compound score ≥ 0.05: Positive
- Compound score ≤ -0.05: Negative
- Compound score between -0.05 and 0.05: Neutral

VADER provides a rule-based approach that can be compared with the machine learning predictions. Its evaluation metrics should be documented separately if calculated.

## 10. Customer Complaint Analysis

Negative reviews were analyzed to identify frequently occurring complaint-related words and common problem areas.

### 10.1 Most Frequent Complaint Words

The processed text of negative reviews was analyzed using word-frequency counts. The 15 most frequent tokens were identified and visualized to highlight recurring complaint-related terms.

The visualization was saved as:

`outputs/visualizations/top_customer_complaint_words.png`

### 10.2 Complaint Category Classification

Negative reviews were assigned complaint categories using keyword-based rules.

The categories included:

- **Late Delivery:** Complaints related to delays, waiting, and delivery time.
- **Incorrect or Missing Order:** Complaints about incorrect orders, missing items, or wrong deliveries.
- **Food Quality:** Complaints involving cold food, stale food, taste, or food quality.
- **Packaging Issues:** Complaints about damaged packaging, spills, and leaks.
- **Customer Service:** Complaints involving service, support, staff, or responses.
- **High Prices:** Complaints about expensive items, costs, and overpriced orders.
- **Other:** Reviews that did not match the defined complaint categories.

A review may match multiple categories because the classification uses keyword matching. Consequently, the sum of category counts may exceed the total number of negative reviews.

### 10.3 Complaint Category Visualization

The distribution of complaint categories was visualized to make recurring service issues easier to identify.

The visualization was saved as:

`outputs/visualizations/customer_complaint_categories.png`

### 10.4 Saved Complaint Analysis Outputs

- `data/processed/reviews_with_sentiment_analysis.csv`
- `data/processed/categorized_customer_complaints.csv`
- `data/processed/complaint_category_summary.csv`
- `data/processed/sentiment_summary.csv`

## 11. Business Insights and Recommendations

The sentiment and complaint analysis can help identify opportunities to improve customer satisfaction.

1. **Monitor delivery performance:** Investigate recurring complaints involving late deliveries and waiting times.
2. **Improve order accuracy:** Review order preparation and verification processes if incorrect or missing items are frequently mentioned.
3. **Maintain food quality:** Work with restaurant partners to improve food preparation, packaging, and delivery practices.
4. **Strengthen packaging standards:** Investigate reports of spills, leaks, and damaged packaging.
5. **Improve customer support:** Review recurring complaints involving service quality and response times.
6. **Monitor pricing concerns:** Examine customer feedback about perceived high prices and value for money.
7. **Track sentiment over time:** Monitor positive, neutral, and negative feedback periodically to identify changes in customer satisfaction.

These recommendations are potential actions based on the analysis approach. Specific priorities should be selected after reviewing the actual complaint-category counts and representative reviews.

## 12. Limitations

- Keyword-based complaint categorization may misclassify reviews when words are used in different contexts.
- A single review may belong to multiple complaint categories.
- TF-IDF represents text using weighted terms and does not fully capture context, sarcasm, or complex language.
- VADER uses a predefined sentiment lexicon and may not interpret every customer review correctly.
- The 100% scores achieved by both machine learning models require further validation for possible duplicate or highly repetitive text patterns.
- Additional testing on unseen reviews is required before deploying the models in a production environment.

## 13. Day 3 Deliverables

The following deliverables were produced during the customer feedback analysis workflow:

- Dataset quality assessment.
- Sentiment distribution analysis.
- Cleaned and preprocessed review text.
- TF-IDF feature extraction.
- Multinomial Naive Bayes sentiment classifier.
- Linear SVM sentiment classifier.
- Model evaluation report.
- Confusion matrix visualizations.
- VADER-based sentiment scoring.
- Negative-review word-frequency analysis.
- Keyword-based complaint categorization.
- Complaint category visualization.
- Processed datasets, trained models, and supporting artifacts.

## 14. Conclusion

Day 3 focused on understanding customer feedback through NLP and sentiment analysis. The workflow covered data quality checks, text preprocessing, TF-IDF feature extraction, supervised sentiment classification, model evaluation, VADER sentiment scoring, and complaint analysis.

The results provide a foundation for identifying recurring customer concerns and developing data-driven service improvement strategies. Further validation of the unusually high classification scores is recommended before drawing conclusions about model performance on unseen data.