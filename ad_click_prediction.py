# ================================================
# 📊 Ad Click Prediction using Logistic Regression
# Beginner-Friendly Machine Learning Project
# Objective: Predict if users will click a banner ad
# ================================================

# STEP 0: IMPORT NECESSARY LIBRARIES

import pandas as pd  # Load and handle tabular data
import matplotlib.pyplot as plt  # Create basic plots like histograms
import seaborn as sns  # Enhance plots, good for data relationships
from sklearn.model_selection import train_test_split  # For splitting data into training/testing
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
from sklearn.metrics import confusion_matrix, classification_report  # Model evaluation tools


# STEP 1: LOAD THE DATA

# Define the path to your CSV dataset
dataset_path = r"C:\FAWAZUL\Bootcamp\DQLab\ml_beginner\ecommerce_banner_promo.csv"

# Load the dataset into a DataFrame (table format)
data = pd.read_csv(dataset_path)


# STEP 2: EXPLORE BASIC STRUCTURE OF THE DATA

print("\n[1] Data Preview")
print("First 5 rows:\n", data.head())  # See the top 5 rows of the data

print("\nDataset Information:")
print(data.info())  # See column names, data types, and null values

print("\nDescriptive Statistics:")
print(data.describe())  # View stats like mean, std, min, and max

print("\nDataset Dimensions (rows, columns):", data.shape)  # View size of dataset


# STEP 3: CORRELATION ANALYSIS (NUMERIC ONLY)

print("\n[2] Feature Correlation:")

# Select only columns with numeric data types (float64, int64)
numerical_data = data.select_dtypes(include=['float64', 'int64'])

# Compute and display correlation matrix (how features relate to each other)
# Values range from -1 (inverse) to +1 (direct correlation)
print(numerical_data.corr())


# STEP 4: EXPLORE TARGET LABEL BALANCE

print("\n[3] Target Label Distribution:")

# Group data by label (Clicked on Ad) and count how many of each
# Helps identify if the classes are balanced (e.g., 0 vs 1)
print(data.groupby("Clicked on Ad").size())


# STEP 5: VISUALIZE AGE DISTRIBUTION

print("\n[4a] Plotting Age Distribution...")

# Set plot size
plt.figure(figsize=(10, 5))

# Plot histogram of age distribution using number of unique ages as bins
plt.hist(data["Age"], bins=data["Age"].nunique())
plt.xlabel("User Age")
plt.ylabel("Frequency")
plt.title("User Age Distribution")
plt.tight_layout()  # Adjust spacing
plt.show()


# STEP 6: VISUALIZE PAIRWISE FEATURE RELATIONSHIPS

print("\n[4b] Generating Pairplot...")

# Use white background grid for all seaborn plots
sns.set_style("whitegrid")

# Pairplot creates multiple scatter plots and histograms
# Helps visualize the relationship between every pair of variables
sns.pairplot(data)
plt.show()


# STEP 7: CHECK FOR MISSING VALUES

print("\n[5] Missing Values Check:")

# Check for missing values in all cells of the table
# .isnull() returns True for missing values, .sum() counts them
total_missing = data.isnull().sum().sum()
print(f"Total Missing Values in Dataset: {total_missing}")


# STEP 8: SELECT FEATURES AND TARGET FOR MODELING

print("\n[6] Preparing Features and Labels...")

# X = all numerical features (except the label and text-based ones)
# These columns are removed because they are strings or not needed for prediction:
# - 'Ad Topic Line', 'City', 'Country', 'Timestamp' (non-numeric)
# - 'Clicked on Ad' (this is our target label)
X = data.drop(['Ad Topic Line', 'City', 'Country', 'Timestamp', 'Clicked on Ad'], axis=1)

# y = target column, the label we want to predict (0 or 1)
y = data['Clicked on Ad']


# STEP 9: SPLIT INTO TRAINING AND TESTING SETS

# Split data into 80% training and 20% testing
# random_state=42 ensures we get the same split every time
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# STEP 10: BUILD AND TRAIN THE MODEL

print("\nTraining Logistic Regression Model...")

# Create logistic regression classifier with higher max_iter to avoid convergence warning
# By default, max_iter=100, which may not be enough for some datasets
# 500 iterations should give the model more room to find the optimal weights
logreg = LogisticRegression(max_iter=500)

# Train the model using training features and labels
logreg.fit(X_train, y_train)

# Predict target values for test features
y_pred = logreg.predict(X_test)


# STEP 11: EVALUATE MODEL ACCURACY

# Score shows model accuracy: % of correct predictions
print("Training Accuracy:", logreg.score(X_train, y_train))  # Accuracy on known (training) data
print("Testing Accuracy :", logreg.score(X_test, y_test))    # Accuracy on unseen (test) data


# STEP 12: DETAILED EVALUATION USING METRICS

print("\n[7] Confusion Matrix & Classification Report:")

# Confusion matrix: Shows how many were correctly/incorrectly predicted
# - Top-left = true negatives
# - Bottom-right = true positives
# - Others = false predictions
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification report: Shows precision, recall, F1-score for each class
# Useful when classes are imbalanced
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
