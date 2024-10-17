
---
# Week 06: Cost Functions

### **1. Definition and Importance**
- A **Cost function** measures how well or poorly a machine learning model predicts outcomes.
- It calculates the difference between the **predicted** and **actual values** and outputs a single number representing the "cost" or error.
- Types:
  - **Regression Cost Functions:**
    - **Mean Squared Error (MSE):** Measures the squared differences between predicted and actual values.
    - **Mean Absolute Error (MAE):** Measures the absolute differences between predicted and actual values.
  - **Classification Cost Functions:**
    - **Binary Cross-Entropy (Log Loss):** Measures the performance of a classification model with two classes.
    - **Multi-Class Cross-Entropy:** Used when there are more than two target classes.

### **2. Gradient Descent**
- **Gradient Descent** is an optimization algorithm used to minimize the cost function.
- Formula:  
  $ \( \theta_j = \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)})x_j^{(i)} \)  
  Where:
  - \( \theta_j \): Parameters (weights) of the model.
  - \( \alpha \): Learning rate.
  - \( h_{\theta}(x^{(i)}) \): Predicted value.
  - \( y^{(i)} \): Actual value.

### **3. Common Cost Functions**
- **Mean Squared Error (MSE):**  
  \( \text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 \)  
  Penalizes larger errors due to the square term.
  
- **Cross-Entropy (Binary):**  
  \( \text{Cross-Entropy} = -\frac{1}{n}\sum_{i=1}^{n}[y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)] \)  
  Measures the difference between two probability distributions.

### **4. Assignments**
- **Apply linear regression** using gradient descent for the dataset \( x = \{1, 2, 4, 3, 5\} \) and \( y = \{1, 3, 3, 2, 5\} \). Plot error vs. iteration for 4 epochs.
- **Apply logistic regression** for classification and plot error vs. iteration.
- Use **scikit-learn** to compare results for different values of intercepts and slopes in regression models.

---

# Week 08: Naïve Bayes Classifier

### **1. Bayes' Theorem**
- **Bayes' Theorem** provides a way to update the probability estimate of an event based on new evidence:
  
  \( P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)} \)
  
  Where:
  - \( P(A|B) \): Posterior probability (probability of A after knowing B).
  - \( P(B|A) \): Likelihood (probability of B given A).
  - \( P(A) \): Prior probability of A.
  - \( P(B) \): Probability of B.

### **2. Naïve Bayes Classifier**
- **Naïve Assumption**: Assumes all features are independent of each other.
- **Variants**:
  - **Multinomial Naïve Bayes**: Used for discrete data (e.g., text classification).
  - **Gaussian Naïve Bayes**: Used for continuous data assuming a normal distribution.

### **3. Applications**
- **Text Classification**: Spam detection, sentiment analysis.
- **Medical Diagnosis**: Predicting diseases based on symptoms.

### **4. Advantages and Limitations**
- **Advantages**:
  - Simple and easy to implement.
  - Works well with small datasets.
  - Particularly effective with text data.
  
- **Limitations**:
  - Assumption of independence may not hold in real-world data.
  - Limited in handling complex relationships between features.

### **5. Assignments**
- **Task**: Implement Naïve Bayes for text classification using **scikit-learn** and evaluate accuracy.

---

# Week 08: K-Nearest Neighbor (KNN) & ID3 Decision Tree

### **1. K-Nearest Neighbor (KNN)**
- **KNN** is a non-parametric, lazy learning algorithm.
- **Working**:
  - Stores training data and classifies a new data point based on the majority vote of its K nearest neighbors.
  - **Distance Metrics**:
    - **Euclidean Distance**:
      \( d(x, y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2} \)
    - **Manhattan Distance**:
      \( d(x, y) = \sum_{i=1}^{n} |x_i - y_i| \)
  
- **Choosing K**: Small values of K can lead to overfitting, while large K values can lead to underfitting.

### **2. ID3 Decision Tree**
- **ID3 (Iterative Dichotomiser 3)** builds a decision tree using a top-down, greedy approach.
- **Metric**: Information Gain (based on entropy).
  
  \( \text{Entropy}(S) = -p_1\log_2(p_1) - p_2\log_2(p_2) \)
  
  **Information Gain**:
  \( \text{Gain}(S, A) = \text{Entropy}(S) - \sum_{v \in A} \frac{|S_v|}{|S|} \cdot \text{Entropy}(S_v) \)

### **3. Assignments**
- Implement **KNN** from scratch and classify a new point.
- Build an **ID3 Decision Tree** and calculate information gain manually.

---

# Week 09: Decision Trees (C4.5 and CART)

### **1. Decision Trees Overview**
- **C4.5**: An extension of ID3 that handles both categorical and continuous data.
- **CART (Classification and Regression Trees)**: Produces binary splits using the **Gini Index**:
  
  \( \text{Gini}(S) = 1 - \sum_{i=1}^{c}(p_i)^2 \)

### **2. Pruning**
- **Pruning** reduces the size of the tree to avoid overfitting:
  - **Cost Complexity Pruning**.
  - **Reduced Error Pruning**.

### **3. Assignments**
- Implement **C4.5 and CART** algorithms to classify a dataset. Plot the decision tree graph.

---

# Week 10: K-Means Clustering

### **1. Introduction to Clustering**
- **Clustering** is an unsupervised learning technique that groups data points based on similarity.
- **K-Means Algorithm**:
  - Assigns each point to the nearest of **K** centroids (cluster centers).
  - Recomputes centroids until cluster assignments stabilize.

### **2. Steps in K-Means Algorithm**
1. Initialize **K** random points (centroids).
2. Assign each point to the nearest centroid based on **Euclidean Distance**.
3. Recalculate centroids.
4. Repeat until centroids no longer move.

### **3. Evaluating Clustering**
- **Sum of Squared Errors (SSE)**: Measures how well the data points fit into clusters.
  
  \( \text{SSE} = \sum_{i=1}^{K} \sum_{x_j \in C_i} (x_j - \mu_i)^2 \)

### **4. Assignments**
- Perform **K-Means clustering** with different values of **K** and plot SSE vs. K.

---
