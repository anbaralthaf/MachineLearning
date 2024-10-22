
---

# Week 06: Cost Functions

### **1. Definition and Importance**
- A **Cost function** measures how well or poorly a machine learning model predicts outcomes. It calculates the difference between the predicted and actual values and outputs a single number representing the "cost" or error.
- Types of cost functions:
  - **Regression Cost Functions**:
    - **Mean Squared Error (MSE)**: Measures the average of squared differences between predicted and actual values. This function heavily penalizes larger errors.
    - **Mean Absolute Error (MAE)**: Measures the average of absolute differences between predicted and actual values. It is less sensitive to large errors than MSE.
  - **Classification Cost Functions**:
    - **Binary Cross-Entropy (Log Loss)**: Used for classification models with two classes, it measures the difference between predicted probabilities and actual labels.
    - **Multi-Class Cross-Entropy**: An extension of binary cross-entropy for cases with more than two classes.

### **2. Gradient Descent**
- **Gradient Descent** is an optimization algorithm used to minimize the cost function by iteratively updating model parameters.
- The process involves:
  - **Initialization**: Start with an initial guess for the model's parameters (weights).
  - **Iteration**: Update the parameters in the direction that reduces the cost function. This is done using a small step, known as the learning rate.
  - **Convergence**: The process continues until the parameter updates become very small, indicating that the algorithm has converged to an optimal solution.

### **3. Common Cost Functions**
- **MSE** is typically used in regression problems. It penalizes larger errors more than smaller ones, making it sensitive to outliers.
- **Binary Cross-Entropy** is used in classification tasks where the output is a probability between 0 and 1. It increases the cost when the predicted probability differs significantly from the actual label, making it suitable for probability-based models.

### **4. Assignments**
- **Linear Regression Task**: Implement linear regression using gradient descent for a given dataset. Track the error over several iterations (epochs) to observe how the model improves.
- **Logistic Regression Task**: Apply logistic regression for classification, plot the error versus iteration to observe how the model's predictions improve over time.
- **Scikit-learn Task**: Use the scikit-learn library to explore how changing the intercepts and slopes in a regression model impacts the results.

---

# Week 07: Naïve Bayes Classifier

### **1. Bayes' Theorem**
- **Bayes' Theorem** helps update the probability estimate of an event based on new evidence. It calculates the probability of one event given the occurrence of another.
- Example: If you know that it often rains when it's cloudy, and you observe that it is cloudy today, Bayes’ theorem helps calculate the probability that it will rain.

### **2. Naïve Bayes Classifier**
- This classifier is based on Bayes' Theorem and makes the "naïve" assumption that all features are independent of each other. This simplifies the computation but may not always hold true in real-world data.
- The Naïve Bayes classifier has different variants:
  - **Multinomial Naïve Bayes**: Commonly used for text classification tasks like spam detection. It works well with data represented as word counts.
  - **Gaussian Naïve Bayes**: Used when the features follow a normal (Gaussian) distribution, making it suitable for continuous data.

### **3. Applications**
- **Text Classification**: Used in tasks like spam detection or sentiment analysis by classifying texts based on the frequency of words.
- **Medical Diagnosis**: Helps predict the likelihood of diseases based on observed symptoms.

### **4. Advantages and Limitations**
- **Advantages**:
  - Simple and easy to implement.
  - Works well with small datasets and can handle large feature sets.
- **Limitations**:
  - Assumes that features are independent, which is rarely true in real-world datasets.
  - If a category is not observed in the training set, the classifier might assign zero probability to that category. This issue can be mitigated using smoothing techniques.

### **5. Assignments**
- **Naïve Bayes Task**: Implement Naïve Bayes for text classification using scikit-learn. Preprocess text data (e.g., remove punctuation, convert to lowercase), extract features, and evaluate the model’s accuracy.

---

# Week 08: K-Nearest Neighbor (KNN) & ID3 Decision Tree

### **1. K-Nearest Neighbor (KNN)**
- **KNN** is a simple algorithm that stores training data and classifies new data points based on the majority vote of their nearest neighbors.
- Key steps:
  - Choose the number of neighbors (K).
  - Calculate the distance between the test data and all training data points (e.g., using Euclidean or Manhattan distance).
  - The label of the majority of the nearest neighbors determines the prediction.

### **2. ID3 Decision Tree**
- **ID3** is a decision tree algorithm that splits data based on features that provide the most information. It uses **entropy** to measure the disorder or impurity in the dataset and selects the feature with the highest **information gain** for each split.
  
### **3. Assignments**
- **KNN Task**: Implement KNN from scratch and classify a new point based on its distance from other points.
- **ID3 Task**: Build an ID3 Decision Tree and manually calculate information gain to choose the best feature for splitting.

---

# Week 09: Decision Trees (C4.5 and CART)

### **1. Decision Trees Overview**
- **C4.5** is an improved version of the ID3 algorithm. It can handle both continuous and discrete data and prunes the tree after it is built to improve its performance. It selects attributes based on **Gain Ratio**, which adjusts for biases that can occur when attributes have many values.
- **CART (Classification and Regression Trees)** is another decision tree algorithm that produces binary splits. It uses the **Gini Index** to measure impurity and selects splits that minimize this index.

### **2. Pruning**
- Pruning reduces the size of decision trees to prevent overfitting. Two common types are:
  - **Cost Complexity Pruning**: Involves balancing the tree's accuracy and complexity by pruning less significant branches.
  - **Reduced Error Pruning**: Prunes branches that don’t improve accuracy on a validation set.

### **3. Assignments**
- **C4.5 and CART Task**: Implement both algorithms to classify a dataset and visualize the decision tree. Experiment with pruning to avoid overfitting.

---

# Week 10: K-Means Clustering

### **1. Introduction to Clustering**
- **Clustering** is an unsupervised learning method that groups data points based on their similarity. **K-Means** is one of the most popular clustering algorithms.
  
### **2. Steps in K-Means Algorithm**
1. **Initialize**: Randomly select K data points as initial cluster centers (centroids).
2. **Assign**: Each data point is assigned to the nearest centroid.
3. **Update**: Recalculate the centroid of each cluster based on the points assigned to it.
4. **Repeat**: Repeat the assignment and update steps until the centroids stabilize.

### **3. Evaluating Clustering**
- The **Sum of Squared Errors (SSE)** measures how well data points fit within their clusters. Lower SSE indicates better clustering.
- The **Elbow Method** helps determine the optimal number of clusters by plotting SSE versus the number of clusters (K) and looking for the "elbow" point where adding more clusters does not significantly reduce SSE.

### **4. Assignments**
- **K-Means Task**: Perform K-Means clustering on a dataset, experiment with different values of K, and plot SSE vs. K to determine the optimal number of clusters.

---
