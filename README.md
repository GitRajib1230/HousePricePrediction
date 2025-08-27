# House Price Prediction

# 🏠 House Price Prediction with Regression Models  

## 🔹 Project Overview & Importance  
Housing prices are critical for **buyers, sellers, lenders, and policymakers**, as they directly influence investments, mortgage lending, and housing policy. Accurately predicting prices provides an edge in decision-making and helps ensure market stability.  

From a data science perspective, this project highlights the complete predictive modeling pipeline: **data preprocessing, exploratory analysis, feature engineering, model training, hyperparameter tuning, evaluation, and interpretation**. It demonstrates how technical models can be translated into **business insights**.  

Dataset: **Ames Housing Dataset** (79 features across property characteristics, structural attributes, location, and sale conditions).  
Target variable: **SalePrice**  

---

## 🔹 Steps Taken  

### 1. Workspace Setup  
- Imported core libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`.  
- Modeling tools: `sklearn` (Linear, Ridge, Lasso, Random Forest, Stacking), `statsmodels`, `scipy`.  
- Installed `ace_tools` for additional utilities.  
- Purpose: ensure environment supports **cleaning, visualization, modeling, and statistical diagnostics**.  

---

### 2. Data Loading & Auditing  
- Loaded dataset (`hw_1_house_price_data.xlsx`) with **79 predictors + SalePrice**.  
- Checked dataset shape and indices.  
- **Reset index** → avoided mismatches during training/testing.  
- Initial inspection with `.head()`, `.info()`, `.describe()`.  

---

### 3. Data Exploration (EDA)  
- Identified **categorical vs numerical features**.  
- Corrected data type for `MSSubClass` (treated as categorical).  
- Plotted histograms, boxplots, and scatterplots to examine distributions.  
- Correlation analysis → strong predictors:  
  - **Overall Quality (OverallQual)**  
  - **Above Ground Living Area (GrLivArea)**  
  - **Garage Cars (GarageCars)**  
  - **Neighborhood** (location-based premium)  

---

### 4. Handling Missing Values  
- Checked missingness in all features.  
- Applied **targeted imputation strategies**:  
  - Numerical features → mean/median.  
  - Categorical features → mode or “None”.  
- Preserved as much data as possible instead of dropping rows.  

---

### 5. Feature Engineering  
- **Encoding**: One-hot encoding for categorical variables (e.g., `Neighborhood`, `BldgType`, `Exterior1st`).  
- **Log transformation**: Applied to skewed features (including `SalePrice`) to normalize distribution.  
- **Scaling**: Standardized numerical features with `StandardScaler`.  
- This ensured models handled features fairly and avoided bias toward larger-scale variables.  

---

### 6. Multicollinearity Check  
- Calculated **Variance Inflation Factor (VIF)** for predictors.  
- Identified and reduced redundancy from highly correlated variables.  
- Prevented unstable coefficients in regression models.  

---

### 7. Train-Test Split  
- Divided dataset into **training (80%) and testing (20%) sets**.  
- Ensured evaluation on unseen data.  

---

### 8. Model Training  
Trained multiple models to compare performance:  
1. **Linear Regression** → simple baseline.  
2. **Ridge Regression** → regularized model to prevent overfitting.  
3. **Lasso Regression** → performed feature selection by shrinking coefficients.  
4. **Random Forest Regressor** → captured non-linear interactions.  
5. **Stacking Regressor** → ensemble of multiple models.  

---

### 9. Hyperparameter Tuning  
- Used **GridSearchCV with cross-validation**.  
- Tuned:  
  - Ridge/Lasso → regularization strength (α).  
  - Random Forest → `n_estimators`, `max_depth`, `min_samples_split`.  
- Cross-validation ensured stable performance across folds.  

---

### 10. Model Evaluation  
Metrics applied:  
- **Mean Squared Error (MSE)** → prediction error magnitude.  
- **Root Mean Squared Error (RMSE)** → interpretable error in price units.  
- **R² and Adjusted R²** → proportion of variance explained.  

Findings:  
- **Ridge Regression** (MSE: 0.0175, Adj. R²: 0.883) → best balance of accuracy, interpretability, and generalization.  
- **Stacking Regression** (MSE: 0.0176, Adj. R²: 0.882) → highly accurate but less interpretable.  
- **Random Forest** → strong but more complex and less transparent.  
- **Lasso** → aided feature selection but slightly weaker predictive accuracy.  
- **Linear Regression** → weakest due to multicollinearity.  

---

### 11. Feature Importance (Ridge Regression)  
Top features influencing house prices:  
- **Positive impact** → Brick Face exterior, Overall Quality, desirable neighborhoods (Crawford, Stone Brook, North Ridge), Normal Sale condition, Gas Water Heating, Number of Rooms.  
- **Negative impact** → Townhouse type, Meadow Village neighborhood.  

---

### 12. Model Selection & Conclusion  
- **Ridge Regression chosen** as final model:  
  - Accurate (low error, high R²).  
  - Interpretable → coefficients explain how features affect price.  
  - Generalizable → regularization prevents overfitting.  

💡 **Business Insight**: Housing prices are driven not only by size and quality, but also by **location, sale conditions, and amenities**. For real estate stakeholders, this means **neighborhood reputation and construction quality matter as much as square footage**.  

---

## 🔹 Skills Demonstrated  
- Data Cleaning & Handling Missing Values  
- Exploratory Data Analysis (EDA) & Visualization  
- Feature Engineering (encoding, scaling, transformations)  
- Multicollinearity Diagnostics (VIF)  
- Regression Modeling (Linear, Ridge, Lasso)  
- Ensemble Learning (Random Forest, Stacking)  
- Hyperparameter Tuning (GridSearchCV, cross-validation)  
- Model Evaluation (MSE, RMSE, R², Adj. R²)  
- Feature Interpretation & Business Communication  

---

👉 This project demonstrates my ability to **carry out the full predictive modeling pipeline**, compare models, fine-tune them, and extract **actionable business insights** from technical results.  
