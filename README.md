# Exploratory-Data-Analysis-Templates
Templates for Exploratory Data Analysis

## PACE Framework for Data Analytics Projects

| Stage                      | Description                                           | Key Questions to Ask                                  |
|----------------------------|-------------------------------------------------------|------------------------------------------------------|
| Plan                       | Define project scope and informational needs         | - What are the project goals?<br>- What strategies are needed?<br>- What are the business impacts?          |
| Prepare for Success        | Identify necessary preparations for a successful project | - How can I acquire and clean the data?<br>- What insights can Exploratory Data Analysis (EDA) uncover?<br>- Which areas are worth pursuing in detail? |
| Analyze                    | Engage with the data and conduct in-depth analysis   | - How can I build and revise models?<br>- What relationships can I uncover?<br>- What statistical inferences can I make? |
| Execute                    | Share analysis results and collaborate with stakeholders | - How can I present findings effectively?<br>- What recommendations can I provide?<br>- How can I incorporate feedback? |



## Top Techniques

| Task                      | Technique                           | Code (where applicable)                                  | Description                                                                                                 |
|---------------------------|-------------------------------------|----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| Missing Value Imputation  | Multiple Imputation                 | N/A (requires specific library implementations)           | A technique to handle missing data by generating multiple imputed datasets to improve analysis.          |
| Outlier Detection         | Z-Score method                     | `from scipy import stats`<br>`z_scores = stats.zscore(df['numeric_column'])` | Identifies outliers using the Z-Score method based on standard deviations from the mean.                  |
|                           | Interquartile Range (IQR) method    | ```python                                    Q1 = df['numeric_column'].quantile(0.25)                                    Q3 = df['numeric_column'].quantile(0.75)                                    IQR = Q3 - Q1                                    outliers = df[(df['numeric_column'] < (Q1 - 1.5 * IQR)) | (df['numeric_column'] > (Q3 + 1.5 * IQR))]``` | Identifies outliers using the IQR method based on the range between the first and third quartiles.        |
| Feature Scaling           | Standardization (Z-Score scaling)   | `from sklearn.preprocessing import StandardScaler`<br>`scaler = StandardScaler()`<br>`scaled_data = scaler.fit_transform(df[['numeric_column']])` | Scales numeric features to have a mean of 0 and standard deviation of 1.                                   |
|                           | Min-Max scaling                    | `from sklearn.preprocessing import MinMaxScaler`<br>`scaler = MinMaxScaler()`<br>`scaled_data = scaler.fit_transform(df[['numeric_column']])` | Scales numeric features to a specified range, usually [0, 1].                                              |
| Dimensionality Reduction  | Principal Component Analysis (PCA) | `from sklearn.decomposition import PCA`<br>`pca = PCA(n_components=2)`<br>`principal_components = pca.fit_transform(df[['feature1', 'feature2']])` | Reduces the dimensionality of data while retaining important information using PCA.                       |
| Feature Selection         | Tree-Based Feature Importance      | `from sklearn.ensemble import RandomForestClassifier`<br>`model = RandomForestClassifier()`<br>`model.fit(X_train, y_train)`<br>`feature_importance = model.feature_importances_` | Determines the importance of features using tree-based models, like Random Forests.                       |
| Model Evaluation          | Cross-Validation                   | `from sklearn.model_selection import cross_val_score`<br>`scores = cross_val_score(model, X, y, cv=5)` | Evaluates model performance using cross-validation to mitigate overfitting.                                |
|                           | Grid Search and Hyperparameter Tuning | `from sklearn.model_selection import GridSearchCV`<br>`param_grid = {'param1': [value1, value2], 'param2': [value3, value4]}`<br>`grid_search = GridSearchCV(model, param_grid, cv=5)`<br>`grid_search.fit(X_train, y_train)` | Optimizes model hyperparameters using an exhaustive grid search.                                            |
| Model Performance Metrics | Metrics (e.g., accuracy, precision, recall, F1-score) | `from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score`<br>`y_pred = model.predict(X_test)`<br>`accuracy = accuracy_score(y_test, y_pred)`<br>`precision = precision_score(y_test, y_pred)`<br>`recall = recall_score(y_test, y_pred)`<br>`f1 = f1_score(y_test, y_pred)` | Calculates common performance metrics like accuracy, precision, recall, and F1-score for classification models. |
| Best Algorithm            | XGBoost                             | `from xgboost import XGBClassifier`<br>`model = XGBClassifier()`<br>`model.fit(X_train, y_train)`<br>`y_pred = model.predict(X_test)` | Utilizes the XGBoost algorithm, a powerful gradient boosting technique, for classification tasks.         |

