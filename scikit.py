# basic_scikit_learn_functions.py

import warnings
warnings.filterwarnings("ignore")

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

# 1. Load a Dataset - Iris
iris = datasets.load_iris()
X, y = iris.data, iris.target

# 2. Split the Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Linear Regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

# 5. K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
cluster_labels = kmeans.labels_

# 6. Support Vector Machine (SVM)
svm_classifier = SVC(kernel='linear', C=1)
svm_classifier.fit(X_train_scaled, y_train)
svm_predictions = svm_classifier.predict(X_test_scaled)

# 7. Decision Tree Classifier
tree_classifier = DecisionTreeClassifier(random_state=42)
tree_classifier.fit(X_train, y_train)
tree_predictions = tree_classifier.predict(X_test)

# 8. Evaluate Classifier Accuracy
accuracy_svm = accuracy_score(y_test, svm_predictions)
accuracy_tree = accuracy_score(y_test, tree_predictions)
print("SVM Classifier Accuracy:", accuracy_svm)
print("Decision Tree Classifier Accuracy:", accuracy_tree)

# 9. Mean Squared Error for Regression
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error for Linear Regression:", mse)

# 10. Principal Component Analysis (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 11. Text Vectorization - CountVectorizer
corpus = ["This is the first document.", "This document is the second document.", "And this is the third one."]
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(corpus).toarray()

# 12. Cross-Validation Score
cross_val_result = cross_val_score(svm_classifier, X, y, cv=3)
print("Cross-Validation Scores:", cross_val_result)

# 13. Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)

# 14. K-Nearest Neighbors Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train_scaled, y_train)
knn_predictions = knn_classifier.predict(X_test_scaled)

# 15. Naive Bayes Classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
nb_predictions = nb_classifier.predict(X_test)

# 16. Grid Search for Hyperparameter Tuning
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
grid_search = GridSearchCV(SVC(), param_grid, cv=3)
grid_search.fit(X_train_scaled, y_train)
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# 17. Pipeline for Preprocessing and Model Training
pipeline = make_pipeline(StandardScaler(), SVC())
pipeline.fit(X_train, y_train)
pipeline_accuracy = pipeline.score(X_test, y_test)
print("Accuracy with Pipeline:", pipeline_accuracy)
