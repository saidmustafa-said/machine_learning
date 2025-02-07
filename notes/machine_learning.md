<!-- @format -->

Here's a detailed Markdown study guide explaining each step, why we did it, and the different models used. Let me know if you want more details or formatting tweaks! 🚀

---

### 🏡 House Price Prediction Study Guide

This guide walks through building and improving a model to predict house prices using machine learning. We'll explore:

- **Data exploration and preprocessing**
- **Decision Tree regression**
- **Model evaluation using Mean Absolute Error (MAE)**
- **Hyperparameter tuning (max_leaf_nodes)**
- **Using Random Forest for better accuracy**

## 📌 Part 1: Loading and Understanding the Data

Before training a model, we must load and understand the dataset.

```python
import pandas as pd

# Path to dataset
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

# Load data
home_data = pd.read_csv(iowa_file_path)

# Show summary statistics
home_data.describe()
```

### 🔹 Why This Step?

- `pd.read_csv()` loads the dataset.
- `.describe()` gives a summary of numerical data (mean, min, max, etc.), helping us understand the dataset.

---

## 📌 Part 2: Selecting the Target and Features

### **Target Variable (y)**

The variable we want to predict is **SalePrice**.

```python
y = home_data["SalePrice"]
```

### **Selecting Features (X)**

We choose relevant features that influence house prices:

```python
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_names]
```

### 🔹 Why These Features?

- **LotArea**: Larger lots may increase value.
- **YearBuilt**: Older homes might be less valuable unless renovated.
- **1stFlrSF & 2ndFlrSF**: Total floor space affects price.
- **FullBath, BedroomAbvGr, TotRmsAbvGrd**: More rooms and bathrooms generally mean higher prices.

---

## 📌 Part 3: Training a Decision Tree Model

A **Decision Tree Regressor** predicts house prices based on input features.

```python
from sklearn.tree import DecisionTreeRegressor

# Initialize model
iowa_model = DecisionTreeRegressor(random_state=1)

# Train the model
iowa_model.fit(X, y)

# Make predictions
predictions = iowa_model.predict(X)

# Show first few predictions
print(predictions[:5])
```

### 🔹 Why Decision Tree?

- **Easy to interpret**
- **Handles non-linear relationships**
- **Quick to train**

**But:** It may **overfit** (perform well on training data but poorly on new data).

---

## 📌 Part 4: Splitting Data for Better Evaluation

Instead of training on all data, we split into:

- **Training set** (used for learning)
- **Validation set** (used for testing)

```python
from sklearn.model_selection import train_test_split

# Split data into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
```

### 🔹 Why Split Data?

- **Prevents overfitting** (model memorizing data instead of generalizing)
- **Helps assess real-world performance**

---

## 📌 Part 5: Evaluating the Model with MAE

To measure accuracy, we use **Mean Absolute Error (MAE)**:

```python
from sklearn.metrics import mean_absolute_error

# Train the model on training data
iowa_model.fit(train_X, train_y)

# Predict on validation data
val_predictions = iowa_model.predict(val_X)

# Calculate MAE
val_mae = mean_absolute_error(val_y, val_predictions)
print(f"Validation MAE: {val_mae}")
```

### 🔹 Why MAE?

- Measures how far predictions are from actual prices.
- Lower MAE = Better model performance.

**Downside:** It treats all errors equally (no penalty for large errors).

---

## 📌 Part 6: Improving Decision Tree with Hyperparameter Tuning

We can **optimize max_leaf_nodes** to balance performance and overfitting.

```python
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae

# Test different values
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
mae_scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}

# Best tree size
best_tree_size = min(mae_scores, key=mae_scores.get)
print(f"Best tree size: {best_tree_size}")
```

```python
# Train final model with best max_leaf_nodes
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)
final_model.fit(X, y)
```

### 🔹 Why Tune max_leaf_nodes?

- **Too few leaves** → Underfitting (missing patterns).
- **Too many leaves** → Overfitting (memorizing data).

---

## 📌 Part 7: Using Random Forest for Better Accuracy

A **Random Forest** is a collection of multiple decision trees, improving accuracy.

```python
from sklearn.ensemble import RandomForestRegressor

# Initialize model
rf_model = RandomForestRegressor(random_state=1)

# Train model
rf_model.fit(train_X, train_y)

# Predict on validation data
rf_val_predictions = rf_model.predict(val_X)

# Calculate MAE
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
print(f"Validation MAE for Random Forest Model: {rf_val_mae}")
```

### 🔹 Why Use Random Forest?

✅ **Reduces overfitting** (averaging multiple trees prevents memorization).  
✅ **Handles complex patterns better than a single tree**.  
✅ **More stable and accurate predictions**.

---

## 📌 Comparison of Models

| Model               | Validation MAE (lower is better) | Strengths                          | Weaknesses                  |
| ------------------- | -------------------------------- | ---------------------------------- | --------------------------- |
| Decision Tree       | Higher                           | Simple, interpretable              | Overfits easily             |
| Tuned Decision Tree | Lower                            | Better than basic tree             | Still overfits sometimes    |
| Random Forest       | Lowest                           | Best accuracy, reduces overfitting | Slower, harder to interpret |

---

## 🎯 Final Thoughts

1. **Decision Trees** are great but prone to **overfitting**.
2. **Tuning max_leaf_nodes** helps balance bias vs. variance.
3. **Random Forests** give the **best accuracy** by combining multiple trees.

📌 **Key Takeaways:**

- **Always split data into training & validation sets**.
- **Use MAE to evaluate models**.
- **Experiment with different models for better results**.

---

🔹 **Next Steps:**  
Try improving further by:

- Adding more features.
- Using different algorithms (e.g., Gradient Boosting).
- Normalizing data for better performance.

Hope this helps! 🚀 Let me know if you need any refinements! 😊
