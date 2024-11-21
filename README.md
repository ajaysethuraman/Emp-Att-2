This project focuses on analyzing and predicting employee turnover using machine learning models. It involves exploring patterns in HR data, visualizing trends, engineering features, and training classifiers to identify factors contributing to employee retention or attrition.

Process Overview:

1. Data Exploration and Preprocessing:

- Loaded and inspected HR data to identify key features.
- Renamed columns for consistency and handled missing values.
- Merged similar categories in the department column for clarity.
- Encoded categorical variables (department and salary) into numerical formats using one-hot encoding.

2. Visualization:

- Created bar charts to examine turnover frequency by department and salary levels.
- Generated histograms to explore the distribution of numerical variables.

3. Feature Selection:

- Used Recursive Feature Elimination (RFE) to select the top 10 features contributing to the target variable (left).

4. Model Development:

- Split the dataset into training and testing sets.
- Trained the following models:
  - Logistic Regression: For baseline accuracy.
  - Random Forest Classifier: For robust feature importance evaluation.
  - Support Vector Machine (SVM): For comparison with other classifiers.

5. Model Evaluation:

- Assessed accuracy scores for each model.
- Generated confusion matrices and classification reports to evaluate model performance.
- Visualized confusion matrices for Logistic Regression and Random Forest models using heatmaps.

6. ROC Analysis:

- Calculated and plotted Receiver Operating Characteristic (ROC) curves to compare the models' true positive rates against false positive rates.

7. Feature Importance:

- Analyzed feature importance from the Random Forest model to understand the influence of individual factors like satisfaction level and time spent at the company.

8. Visualizations:
- Bar charts for department and salary-level turnover trends.
- ROC curves comparing Logistic Regression and Random Forest performance.
- Heatmaps for model-specific confusion matrices.

9. Insights:
- Feature importance rankings to identify critical drivers of employee turnover.
- Detailed performance metrics for each classifier.

The Python notebook contains the full implementation, including code, visualizations, and analysis, while the results highlight actionable insights for improving employee retention strategies.
