
# 📊 Ad Click Prediction (Beginner Machine Learning Project)

This project is a beginner-level Machine Learning exercise using Python and scikit-learn.  
It predicts whether a user will click on a promotional banner ad based on their demographic data and internet behavior.

---

## 🧠 Objective

> Predict user behavior on an e-commerce website by building a logistic regression model that determines whether they click on an ad.

---

## 📂 Dataset Features

| Feature                  | Description                                           |
|--------------------------|-------------------------------------------------------|
| Daily Time Spent on Site | Time (minutes) the user spent on the site            |
| Age                      | Age of the user                                       |
| Area Income              | Average income in the user's area                     |
| Daily Internet Usage     | Average time spent on the internet per day (minutes) |
| Male                     | Whether the user is male (1) or not (0)              |
| Clicked on Ad            | Target variable: 1 if clicked, 0 if not              |

---

## 🛠 Tools Used

- Python 3
- Pandas, Matplotlib, Seaborn
- Scikit-learn (Logistic Regression)
- Jupyter Notebook / VS Code

---

## 📈 Model Workflow

1. Load and explore the dataset
2. Visualize data distribution
3. Check for missing values
4. Drop non-numeric features
5. Train-test split (80:20)
6. Build Logistic Regression model
7. Evaluate with accuracy, confusion matrix, and classification report

---

## 🧪 Result (Sample Output)

- **Training Accuracy**: ~88%
- **Testing Accuracy**: ~92%
- **Model**: Logistic Regression with `max_iter=500`

✅ Confusion Matrix and Classification Report are shown at the end of the script.

---

## 🚀 How to Run

1. Make sure you update the dataset path in `ad_click_prediction.py`
2. Run the script using:

```bash
python ad_click_prediction.py
```

Or run cell-by-cell in:

```bash
ad_click_prediction.ipynb
```

---

## 💡 Why `max_iter=500`?

Logistic Regression sometimes gives a warning when the model doesn't converge within the default 100 iterations.

This project uses:

```python
LogisticRegression(max_iter=500)
```

to give the model enough time to learn and avoid convergence warnings.

✅ No more errors. All set!

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---


## 🙏 Acknowledgment

This project is based on a beginner machine learning module from [DQLab](https://dqlab.id/), using the `ecommerce_banner_promo.csv` dataset provided in the course material.
