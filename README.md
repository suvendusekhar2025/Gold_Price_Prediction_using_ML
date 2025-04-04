# Gold_Price_Prediction_using_ML
Sure! Here's a well-structured and detailed description of the **Gold Price Prediction Using Machine Learning** project:

---

### **Project Title: Gold Price Prediction Using Machine Learning**

#### **Project Description:**

This project aims to build a predictive model for estimating gold prices using machine learning techniques. The motivation behind this project stems from the significance of gold as a valuable commodity and its impact on the global economy. Accurate gold price forecasting can assist investors, analysts, and policymakers in making informed decisions.

#### **Dataset:**

- The dataset used for this project is a CSV file consisting of **2290 rows** and **6 columns**.
- It contains historical gold price data with various features that potentially influence the target variable, which is the gold price.

#### **Technologies & Libraries Used:**

- **Python Libraries:**
  - **NumPy**: For numerical operations.
  - **Pandas**: For data manipulation and analysis.
  - **Matplotlib** and **Seaborn**: For data visualization.
- **Scikit-learn (sklearn)**:
  - `train_test_split` from `sklearn.model_selection`: For splitting the dataset into training and testing sets.
  - `RandomForestRegressor` from `sklearn.ensemble`: Used as the main regression model.
  - `r2_score` from `sklearn.metrics`: Used to evaluate the model’s performance as the target variable is continuous.

#### **Exploratory Data Analysis (EDA):**

- **Visualization**:
  - Various plots were generated using **Matplotlib** and **Seaborn** to understand the distribution of the data and relationships between features.
  - A **correlation matrix** was created to analyze the interdependence between features and the target variable.
  - A **heatmap** was plotted using Seaborn to visualize the correlation matrix, highlighting both positive and negative correlations.
  
#### **Model Implementation:**

- The dataset was split into training and testing sets using `train_test_split` (e.g., 80% training and 20% testing).
- A **Random Forest Regressor** was trained on the training data to learn patterns and predict the gold price.
- This model was chosen due to its robustness, ability to handle non-linear data, and high accuracy in regression tasks.

#### **Model Performance:**

- **Training Accuracy**: The model achieved a high accuracy of **99.83%** on the training data.
- **Testing Accuracy**: On unseen data, the model achieved an accuracy score of **98.93%**, indicating strong generalization ability.
- The **R² score (R-squared error)** was used as the performance metric, as it is suitable for evaluating regression models dealing with continuous values.

#### **Results Visualization:**

- A comparison plot was created using **Matplotlib** to show the difference between the **actual** gold prices and the **predicted** values on the test dataset. The close match between the two confirms the model’s effectiveness.

---
