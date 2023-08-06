import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt




# Loading the data from the csv file in the 'data' directory
data = pd.read_csv('../data/spam_ham_dataset.csv')

# Handling the null values
data = data.where(pd.notnull(data), '')

# Getting the names of our classes
target_names = ['ham', 'spam']  # Specify the class labels in the desired order

# Performing label encoding on the target column
data['Category'] = data['Category'].map({'ham': 0, 'spam': 1})

# Separating the data from the target classes into X and y
X = data['Message']
y = data['Category']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Transforming the text data into numeric vector features
vectorizer = TfidfVectorizer(min_df=0.0, max_df=0.9, strip_accents='ascii',stop_words='english')
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Train a Logistic Regression classifier
clf_lr = LogisticRegression(max_iter=500)
clf_lr.fit(X_train_features, y_train)
y_pred_lr = clf_lr.predict(X_test_features)

# Calculate accuracy score
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy Score:", accuracy_lr)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred_lr)

# Plot confusion matrix
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(np.arange(len(target_names)), labels=target_names)
plt.yticks(np.arange(len(target_names)), labels=target_names)
plt.title("Confusion Matrix")
plt.show()



# Prepare the sample text
sample_text = """

Enter the email you want to classify here !

"""

# Transform the sample text into features
sample_features = vectorizer.transform([sample_text])

# Make predictions on the sample features
sample_prediction = clf_lr.predict(sample_features)

# Interpret the prediction
predicted_class = target_names[sample_prediction[0]]

# Print the prediction
print("Predicted class:", predicted_class)


