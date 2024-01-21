import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer


# Read data from local file
file_path = '/Users/songyafeng/6.2/CFL/Assignment/AI-generated-text-detection/gpt4_data_processed.csv'
df = pd.read_csv(file_path)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=0)

# Feature extraction
# vectorizer = CountVectorizer()
# vectorizer = TfidfVectorizer()
vectorizer = CountVectorizer(ngram_range=(3, 3))  # tri-gram features

X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Creating and training the logistic regression model with increased max_iter
model = LogisticRegression(max_iter=1000)  # Increase the number of iterations
model.fit(X_train_counts, y_train)

# Predicting the probabilities for the test set
# Get probabilities for the positive class (label=1)
probabilities = model.predict_proba(X_test_counts)[:, 1]

# Evaluating the model using AUROC
auroc = roc_auc_score(y_test, probabilities)
report = classification_report(y_test, model.predict(X_test_counts))

print(f"AUROC: {auroc}")
print(f"Classification Report:\n{report}")

# Extracting feature names and their corresponding weights
feature_names = vectorizer.get_feature_names_out()
feature_weights = model.coef_[0]

# Combining feature names and weights and sorting them
features_with_weights = list(zip(feature_names, feature_weights))
features_with_weights.sort(key=lambda x: x[1], reverse=True)

# Displaying top features for each class
top_features_for_class_1 = features_with_weights[:10]  # Top 10 features for class 1
top_features_for_class_0 = features_with_weights[-10:] # Top 10 features for class 0

print("Top features for class 1:", top_features_for_class_1)
print("Top features for class 0:", top_features_for_class_0)