import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Data
df = pd.read_csv('D:\sad\classificador/basedetreinamento.csv')
train_data, test_data, train_labels, test_labels = train_test_split(df['comentario'], df['sentimento'], test_size=0.2)

# Vectorize Data
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data)
X_test = vectorizer.transform(test_data)

# Train Model
model = MultinomialNB()
model.fit(X_train, train_labels)

# Make Predictions
predictions = model.predict(X_test)

# Evaluate Model
accuracy = accuracy_score(test_labels, predictions)
conf_matrix = confusion_matrix(test_labels, predictions)
class_report = classification_report(test_labels, predictions)
print(accuracy)
print(conf_matrix)
print(class_report)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save Model and Vectorizer
joblib.dump(model, 'classificador/modelo_naive_bayes.joblib')
joblib.dump(vectorizer, 'classificador/vetorizador.joblib')
