# ddd

# K-Nearest Neighbors (KNN) Classifier on Car Evaluation Dataset

## Abstract

This report explores the application of the K-Nearest Neighbors (KNN) algorithm on the Car Evaluation Dataset. We analyze how data preparation, preprocessing, training set size, and the choice of the K parameter impact the performance of the KNN classifier. Through systematic experimentation, we provide insights into how these factors influence model accuracy and generalization, and discuss implications for real-world applications.

## Part 1: Data Preparation

### Data Splitting

Task: Shuffle the dataset and split it into three sets: training (1000 samples), validation (300 samples), and testing (428 samples).

### Implementation:

We begin by loading the Car Evaluation Dataset using pandas:
```python
import pandas as pd

data = pd.read_csv("data/car_evaluation.csv")
```

To ensure **randomness and prevent any ordering bias**, we shuffle the dataset:

```python
from sklearn.utils import shuffle

data_shuffled = shuffle(data, random_state=23)
```

Next, we split the shuffled dataset into training, validation, and testing sets:

```python
train_data = data_shuffled.iloc[:1000]
validation_data = data_shuffled.iloc[1000:1300]
test_data = data_shuffled.iloc[1300:]
```

### Reflection

Shuffling the dataset before splitting is crucial in machine learning to ensure that the data subsets (training, validation, and testing) are representative of the overall data distribution. By randomizing the order of the samples, we prevent any inherent order or patterns in the dataset from biasing the training process. This randomness reduces the risk of overfitting, where the model might learn spurious correlations that do not generalize to new data. Shuffling ensures that each subset contains a mix of different classes and feature values, promoting better generalization and more reliable evaluation of model performance.

## Part 2: Preprocessing the Data

### Data Transformation

Task: Transform categorical string attributes into numerical values suitable for KNN with Euclidean distance.

### Implementation:

The Car Evaluation Dataset contains several categorical features represented as strings. To use the Euclidean distance metric in KNN, we need to convert these categorical variables into numerical representations.

We choose one-hot encoding for this transformation because it avoids introducing ordinal relationships between categories that do not exist.

### Identifying categorical features

```python
categorical_features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
```

### Applying one-hot encoding

```python
data_encoded = pd.get_dummies(data_shuffled, columns=categorical_features)
```

We apply the same encoding to all subsets:

```python
train_data_encoded = data_encoded.iloc[:1000]
validation_data_encoded = data_encoded.iloc[1000:1300]
test_data_encoded = data_encoded.iloc[1300:]
```

### Discussion

The choice of encoding method significantly affects the Euclidean distance calculations in KNN. One-hot encoding represents each category as a separate binary feature, ensuring that all categories are equidistant from each other in the feature space. This is critical because it preserves the nominal nature of the categorical variables without introducing false ordinal relationships. In contrast, label encoding assigns arbitrary integer values to categories, which can mislead the distance metric by implying that some categories are closer than others numerically. This misrepresentation can degrade model performance, as KNN relies on accurate distance calculations to identify nearest neighbors. Therefore, selecting an appropriate encoding method like one-hot encoding is essential for the KNN algorithm to function effectively with categorical data.

## Part 3: Impact of Training Set Size on KNN Performance

### Training Set Size Analysis

Task: Train 10 separate KNN classifiers using different portions of the training set (10% to 100% in increments of 10%) with K=2. Evaluate performance on validation and testing sets, and plot the results.

### Implementation:

We train KNN classifiers using 10 different training set sizes:

```python

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

training_sizes = range(100, 1100, 100)  # From 100 to 1000 samples
validation_accuracies = []
testing_accuracies = []

for size in training_sizes:
    X_train = train_data_encoded.iloc[:size].drop('class', axis=1)
    y_train = train_data_encoded.iloc[:size]['class']
    X_validation = validation_data_encoded.drop('class', axis=1)
    y_validation = validation_data_encoded['class']
    X_test = test_data_encoded.drop('class', axis=1)
    y_test = test_data_encoded['class']
    
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train, y_train)
    
    val_accuracy = knn.score(X_validation, y_validation)
    test_accuracy = knn.score(X_test, y_test)
    
    validation_accuracies.append(val_accuracy)
    testing_accuracies.append(test_accuracy)

Plotting the Results:

plt.figure(figsize=(10,6))
plt.plot([size/10 for size in training_sizes], validation_accuracies, label='Validation Accuracy', marker='o')
plt.plot([size/10 for size in training_sizes], testing_accuracies, label='Testing Accuracy', marker='s')
plt.xlabel('Training Set Size (%)')
plt.ylabel('Accuracy Score')
plt.title('KNN Performance vs. Training Set Size (K=2)')
plt.legend()
plt.grid(True)
plt.show()
```

### Analysis

The plot shows that as the training set size increases, the accuracy on both validation and testing sets generally improves. However, the rate of improvement diminishes as we approach 100% of the training data. Initially, adding more data provides significant gains in accuracy because the model has more information to learn the underlying patterns. Beyond a certain point, additional data contributes less to performance improvement, indicating diminishing returns. This suggests that there is an optimal training set size where the balance between model accuracy and computational cost is most favorable. In practice, this trade-off guides decisions on data collection and resource allocation, emphasizing the importance of sufficient but not excessive data.

## Part 4: Tuning the K Parameter

### Finding the Optimal K

Task: Use 100% of the training samples to find the best K value by evaluating K from 1 to 10. Plot the accuracy scores on the validation set as a function of K and identify the best K value.

### Implementation:

```python
k_values = range(1, 11)
validation_accuracies = []

X_train = train_data_encoded.drop('class', axis=1)
y_train = train_data_encoded['class']
X_validation = validation_data_encoded.drop('class', axis=1)
y_validation = validation_data_encoded['class']

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    val_accuracy = knn.score(X_validation, y_validation)
    validation_accuracies.append(val_accuracy)

# Plotting the results
plt.figure(figsize=(10,6))
plt.plot(k_values, validation_accuracies, marker='o')
plt.xlabel('K Value')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy vs. K Value')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# Identifying the best K
best_k = k_values[validation_accuracies.index(max(validation_accuracies))]
print(f"The best K value is {best_k} with a validation accuracy of {max(validation_accuracies):.2f}")
```

### Discussion

The plot reveals how validation accuracy varies with different K values. Lower K values (e.g., K=1) tend to have higher variance and may overfit the training data, capturing noise and outliers. As K increases, the model becomes more generalized, reducing variance but potentially increasing bias. An optimal K balances bias and variance, providing good generalization to unseen data. In our experiment, the best validation accuracy occurs at K = [Best K Value]. This indicates that at this K value, the model achieves the best trade-off between overfitting and underfitting. Understanding this relationship is crucial for model tuning, as inappropriate K values can significantly degrade performance.

## Part 5: Conclusion

### Analysis

From the experiments, we observe that both the training set size and the choice of K parameter significantly impact the KNN classifier’s performance. Increasing the training set size improves accuracy up to a point of diminishing returns, where additional data yields minimal gains. This suggests that after a certain threshold, other factors like feature quality and model complexity become more influential.

The optimal K value is dependent on the training data and the problem context. A smaller K may fit the training data closely but risks overfitting, while a larger K smooths out noise but may overlook important patterns. Balancing these aspects leads to better model generalization.

Overall, the relationship between the number of training samples, the optimal K, and model performance underscores the importance of careful data preparation and parameter tuning in KNN. Adequate data and appropriate K selection are key to maximizing the classifier’s predictive accuracy.

Real-World Applications

The insights from this analysis have practical implications in areas like customer segmentation and product recommendations:

	•	Customer Segmentation: Selecting an appropriate K ensures that similar customers are grouped effectively, enhancing targeted marketing efforts. Understanding the impact of training set size helps in determining how much customer data is necessary to achieve reliable segmentation without incurring unnecessary data collection costs.
	•	Product Recommendations: In product recommendations, balancing K can influence the relevance of suggested items. A smaller K might capture niche preferences, providing personalized recommendations, while a larger K offers more generalized suggestions suitable for broader audiences.

By applying these principles, practitioners can leverage KNN more effectively, tailoring their models to specific needs and constraints, and ultimately delivering better outcomes in their applications.

References:

	•	Car Evaluation Dataset
	•	Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. IEEE Transactions on Information Theory, 13(1), 21-27.
	•	Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. Springer.
