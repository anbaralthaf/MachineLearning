
---

# Week 06: Cost Functions

### **1. Definition and Importance**
- A **Cost function** measures how well or poorly a machine learning model predicts outcomes.
- It calculates the difference between the **predicted** and **actual values** and outputs a single number representing the "cost" or error.
- Types:
  - **Regression Cost Functions:**
    - **Mean Squared Error (MSE):** Measures the squared differences between predicted and actual values.
      - **Formula:**
        \[
        \text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
        \]
        - \( n \): Number of observations.
        - \( y_i \): Actual value.
        - \( \hat{y}_i \): Predicted value.
    - **Mean Absolute Error (MAE):** Measures the absolute differences between predicted and actual values.
      - **Formula:**
        \[
        \text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
        \]
  - **Classification Cost Functions:**
    - **Binary Cross-Entropy (Log Loss):** Measures the performance of a classification model with two classes.
      - **Formula:**
        \[
        \text{Cross-Entropy} = -\frac{1}{n}\sum_{i=1}^{n}\left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
        \]
        - \( y_i \): Actual class label (0 or 1).
        - \( \hat{y}_i \): Predicted probability that \( y_i = 1 \).
    - **Multi-Class Cross-Entropy:** Used when there are more than two target classes.
      - **Formula:**
        \[
        \text{Cross-Entropy} = -\sum_{i=1}^{n}\sum_{k=1}^{K} y_{i,k} \log(\hat{y}_{i,k})
        \]
        - \( K \): Number of classes.
        - \( y_{i,k} \): Binary indicator (0 or 1) if class label \( k \) is the correct classification for observation \( i \).
        - \( \hat{y}_{i,k} \): Predicted probability that observation \( i \) is of class \( k \).

### **2. Gradient Descent**
- **Gradient Descent** is an optimization algorithm used to minimize the cost function.
- **Formula:**
  \[
  \theta_j = \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m}\left( h_{\theta}(x^{(i)}) - y^{(i)} \right) x_j^{(i)}
  \]
  - **Explanation of Terms:**
    - \( \theta_j \): Parameter \( j \) (weight) of the model.
    - \( \alpha \): Learning rate (step size).
    - \( m \): Number of training examples.
    - \( h_{\theta}(x^{(i)}) \): Predicted value for the \( i \)-th training example.
      - For linear regression, \( h_{\theta}(x^{(i)}) = \theta_0 + \theta_1 x_1^{(i)} + \theta_2 x_2^{(i)} + \dots + \theta_n x_n^{(i)} \).
    - \( y^{(i)} \): Actual value for the \( i \)-th training example.
    - \( x_j^{(i)} \): Feature \( j \) of the \( i \)-th training example.

- **Process:**
  - **Initialization:** Start with initial guesses for \( \theta_j \).
  - **Iteration:** Update \( \theta_j \) iteratively to minimize the cost function.
  - **Convergence:** Stop when changes in \( \theta_j \) are below a threshold.

### **3. Common Cost Functions**
- **Mean Squared Error (MSE):**
  \[
  \text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
  \]
  - **Interpretation:** Calculates the average of the squares of the errors between actual and predicted values.
  - **Properties:**
    - Penalizes larger errors more heavily due to squaring.
    - Sensitive to outliers.

- **Cross-Entropy (Binary):**
  \[
  \text{Cross-Entropy} = -\frac{1}{n}\sum_{i=1}^{n}\left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
  \]
  - **Interpretation:** Measures the performance of a classification model where the output is a probability value between 0 and 1.
  - **Properties:**
    - The cost increases as the predicted probability diverges from the actual label.
    - Particularly useful for models that output probabilities.

### **4. Assignments**
- **Apply linear regression** using gradient descent for the dataset \( x = \{1, 2, 4, 3, 5\} \) and \( y = \{1, 3, 3, 2, 5\} \). Plot error vs. iteration for 4 epochs.
  - **Solution Steps:**
    1. **Initialize Parameters:**
       - \( \theta_0 = 0 \), \( \theta_1 = 0 \).
    2. **Set Learning Rate:**
       - \( \alpha = 0.01 \).
    3. **Perform Gradient Descent:**
       - For each epoch (repeat 4 times):
         - **Compute Predictions:**
           \[
           \hat{y}_i = \theta_0 + \theta_1 x_i
           \]
         - **Compute Errors:**
           \[
           \text{Error}_i = \hat{y}_i - y_i
           \]
         - **Update Parameters:**
           \[
           \theta_0 = \theta_0 - \alpha \left( \frac{1}{n} \sum_{i=1}^{n} \text{Error}_i \right)
           \]
           \[
           \theta_1 = \theta_1 - \alpha \left( \frac{1}{n} \sum_{i=1}^{n} \text{Error}_i x_i \right)
           \]
         - **Compute Cost (MSE):**
           \[
           \text{MSE} = \frac{1}{n}\sum_{i=1}^{n} (\text{Error}_i)^2
           \]
         - **Plot MSE vs. Iteration:**
           - Record MSE after each epoch.

- **Apply logistic regression** for classification and plot error vs. iteration.
  - **Solution Steps:**
    1. **Initialize Parameters:**
       - \( \theta \) vector with zeros.
    2. **Set Learning Rate:**
       - \( \alpha = 0.01 \).
    3. **Perform Gradient Descent:**
       - For each iteration:
         - **Compute Predictions:**
           \[
           \hat{y}_i = \sigma(\theta^T x^{(i)})
           \]
           - Where \( \sigma(z) = \frac{1}{1 + e^{-z}} \) is the sigmoid function.
         - **Compute Errors:**
           \[
           \text{Error}_i = \hat{y}_i - y_i
           \]
         - **Update Parameters:**
           \[
           \theta = \theta - \alpha \left( \frac{1}{n} \sum_{i=1}^{n} \text{Error}_i x^{(i)} \right)
           \]
         - **Compute Cost (Cross-Entropy):**
           \[
           \text{Cross-Entropy} = -\frac{1}{n}\sum_{i=1}^{n}\left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
           \]
         - **Plot Cross-Entropy vs. Iteration:**
           - Record Cross-Entropy after each iteration.

- Use **scikit-learn** to compare results for different values of intercepts and slopes in regression models.
  - **Solution Steps:**
    1. **Import Libraries:**
       - `from sklearn.linear_model import LinearRegression`
    2. **Fit the Model:**
       - Create instances with different intercepts and slopes.
    3. **Predict and Evaluate:**
       - Use the `.predict()` method.
       - Calculate MSE for each model.
    4. **Compare Results:**
       - Analyze how changes in intercepts and slopes affect the predictions and error.

---

# Week 08: Naïve Bayes Classifier

### **1. Bayes' Theorem**
- **Bayes' Theorem** provides a way to update the probability estimate of an event based on new evidence:
  
  \[
  P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
  \]
  - **Definitions:**
    - \( P(A|B) \): Posterior probability (probability of event \( A \) occurring given that \( B \) is true).
    - \( P(B|A) \): Likelihood (probability of event \( B \) occurring given that \( A \) is true).
    - \( P(A) \): Prior probability of event \( A \) occurring.
    - \( P(B) \): Marginal probability of event \( B \) occurring.

- **Example:**
  - If \( P(\text{Rain}) = 0.3 \), \( P(\text{Cloudy}|\text{Rain}) = 0.8 \), and \( P(\text{Cloudy}) = 0.5 \), then:
    \[
    P(\text{Rain}|\text{Cloudy}) = \frac{P(\text{Cloudy}|\text{Rain}) \cdot P(\text{Rain})}{P(\text{Cloudy})} = \frac{0.8 \times 0.3}{0.5} = 0.48
    \]

### **2. Naïve Bayes Classifier**
- **Naïve Assumption**: Assumes all features are independent given the class label.
- **Classification Rule:**
  \[
  \hat{C} = \arg\max_C P(C) \prod_{i=1}^{n} P(x_i | C)
  \]
  - \( \hat{C} \): Predicted class.
  - \( P(C) \): Prior probability of class \( C \).
  - \( P(x_i | C) \): Likelihood of feature \( x_i \) given class \( C \).
  - \( n \): Number of features.

- **Variants**:
  - **Multinomial Naïve Bayes**:
    - Used for discrete data, like word counts.
    - **Probability Estimation:**
      \[
      P(x_i | C) = \frac{N_{C,x_i} + \alpha}{N_C + \alpha d}
      \]
      - \( N_{C,x_i} \): Count of feature \( x_i \) in class \( C \).
      - \( N_C \): Total count of all features in class \( C \).
      - \( \alpha \): Smoothing parameter (Laplace smoothing).
      - \( d \): Number of possible feature values.

  - **Gaussian Naïve Bayes**:
    - Assumes continuous features have a Gaussian distribution.
    - **Probability Density Function:**
      \[
      P(x_i | C) = \frac{1}{\sqrt{2\pi \sigma_C^2}} \exp\left( -\frac{(x_i - \mu_C)^2}{2\sigma_C^2} \right)
      \]
      - \( \mu_C \): Mean of feature \( x_i \) for class \( C \).
      - \( \sigma_C^2 \): Variance of feature \( x_i \) for class \( C \).

### **3. Applications**
- **Text Classification**:
  - **Spam Detection**: Classifying emails as spam or not spam using word frequencies.
  - **Sentiment Analysis**: Determining sentiment (positive/negative) from text data.

- **Medical Diagnosis**:
  - Predicting diseases based on symptoms by calculating the probability of disease given observed symptoms.

### **4. Advantages and Limitations**
- **Advantages**:
  - **Simplicity**: Easy to implement and understand.
  - **Efficiency**: Fast computation for both training and prediction.
  - **Scalability**: Performs well with a large number of features.
  - **Performs Well with Small Data**: Effective even with limited training data.

- **Limitations**:
  - **Feature Independence Assumption**: Unrealistic in many real-world scenarios.
  - **Zero Probability Problem**: If a category and feature value never occur together in the training set, the model assigns zero probability (addressed with smoothing techniques).
  - **Continuous Features**: Assumes normal distribution for continuous features, which may not always be the case.

### **5. Assignments**
- **Task**: Implement Naïve Bayes for text classification using **scikit-learn** and evaluate accuracy.
  - **Solution Steps:**
    1. **Data Preparation**:
       - Use a dataset like the SMS Spam Collection.
       - **Preprocess Text**:
         - Convert to lowercase.
         - Remove punctuation and stop words.
         - Tokenize text.
         - Apply stemming or lemmatization.
    2. **Feature Extraction**:
       - Use Bag of Words or TF-IDF vectorization to convert text into numerical features.
       - Example using `CountVectorizer` or `TfidfVectorizer` from `sklearn.feature_extraction.text`.
    3. **Model Training**:
       - Split data into training and test sets using `train_test_split`.
       - Instantiate the classifier:
         ```python
         from sklearn.naive_bayes import MultinomialNB
         model = MultinomialNB()
         ```
       - Train the model:
         ```python
         model.fit(X_train, y_train)
         ```
    4. **Evaluation**:
       - Predict on test set:
         ```python
         y_pred = model.predict(X_test)
         ```
       - Calculate accuracy:
         ```python
         from sklearn.metrics import accuracy_score
         accuracy = accuracy_score(y_test, y_pred)
         ```
       - Compute other metrics like precision, recall, and F1-score.
    5. **Result Interpretation**:
       - Analyze misclassified examples.
       - Discuss model performance and potential improvements.

---

# Week 08: K-Nearest Neighbor (KNN) & ID3 Decision Tree

### **1. K-Nearest Neighbor (KNN)**
- **KNN** is a non-parametric, lazy learning algorithm.
- **Working**:
  - **Algorithm Steps**:
    1. **Choose** the number of neighbors \( K \).
    2. **Compute** the distance between the test data and all training data points using a distance metric.
    3. **Sort** the distances and determine the \( K \) nearest neighbors.
    4. **Gather** the labels of the nearest neighbors.
    5. **Predict** the label by majority vote (classification) or averaging (regression).

- **Distance Metrics**:
  - **Euclidean Distance**:
    \[
    d(x, y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}
    \]
    - **Example Calculation**:
      - For two points \( x = (x_1, x_2) \) and \( y = (y_1, y_2) \):
        \[
        d(x, y) = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2}
        \]
  - **Manhattan Distance**:
    \[
    d(x, y) = \sum_{i=1}^{n} |x_i - y_i|
    \]
    - **Example Calculation**:
      - For two points \( x = (x_1, x_2) \) and \( y = (y_1, y_2) \):
        \[
        d(x, y) = |x_1 - y_1| + |x_2 - y_2|
        \]

- **Choosing K**:
  - **Rule of Thumb**:
    \[
    K = \sqrt{n}
    \]
    - Where \( n \) is the number of training samples.
  - **Cross-Validation**:
    - Use cross-validation to select the best \( K \) by evaluating model performance for different \( K \) values.

### **2. ID3 Decision Tree**
- **ID3 Algorithm**:
  - **Goal**: Build a decision tree that classifies data with minimal depth.
  - **Entropy Calculation**:
    \[
    \text{Entropy}(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)
    \]
    - **Example**:
      - If dataset \( S \) has 9 positive and 5 negative instances:
        \[
        p_{\text{positive}} = \frac{9}{14}, \quad p_{\text{negative}} = \frac{5}{14}
        \]
        \[
        \text{Entropy}(S) = -\left( \frac{9}{14} \log_2 \frac{9}{14} + \frac{5}{14} \log_2 \frac{5}{14} \right)
        \]
  - **Information Gain Calculation**:
    \[
    \text{Gain}(S, A) = \text{Entropy}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Entropy}(S_v)
    \]
    - **Example**:
      - For attribute \( A \) with values \( v \), calculate entropy for each subset \( S_v \) and compute the weighted average.

### **3. Assignments**
- **Implement KNN from scratch and classify a new point.**
  - **Solution Steps:**
    1. **Dataset**:
       - Example dataset:
         | Point ID | \( x \) | \( y \) | Class |
         |----------|---------|---------|-------|
         | 1        | 1       | 2       | A     |
         | 2        | 2       | 3       | A     |
         | 3        | 3       | 1       | B     |
         | 4        | 6       | 5       | B     |
    2. **New Point**:
       - \( x = 4 \), \( y = 3 \)
    3. **Compute Distances**:
       - Calculate Euclidean distances to all points.
    4. **Select K Neighbors**:
       - For \( K = 3 \), select the three closest points.
    5. **Classify**:
       - Majority class among neighbors is the predicted class.

- **Build an ID3 Decision Tree and calculate information gain manually.**
  - **Solution Steps:**
    1. **Dataset**:
       - Use a simple dataset with attributes like Weather (Sunny, Overcast, Rain) and Play (Yes, No).
    2. **Calculate Entropy of Dataset**:
       - Compute \( \text{Entropy}(S) \).
    3. **Calculate Entropy for Each Attribute**:
       - For attribute Weather:
         - For each value (Sunny, Overcast, Rain), compute \( \text{Entropy}(S_v) \).
    4. **Calculate Information Gain**:
       - Use the formula for \( \text{Gain}(S, A) \).
    5. **Select Best Attribute**:
       - Attribute with highest information gain becomes the root node.
    6. **Repeat for Subsets**:
       - Recursively apply steps 2–5 for each branch.

---

# Week 09: Decision Trees (C4.5 and CART)

### **1. Decision Trees Overview**
- **C4.5**:
  - **Improvements over ID3**:
    - Handles both continuous and discrete attributes.
    - Prunes trees after creation to improve accuracy.
    - Uses **Gain Ratio** to select attributes.
  - **Gain Ratio Calculation**:
    - **Split Information**:
      \[
      \text{SplitInfo}(A) = -\sum_{i=1}^{n} \frac{|S_i|}{|S|} \log_2 \left( \frac{|S_i|}{|S|} \right)
      \]
    - **Gain Ratio**:
      \[
      \text{GainRatio}(A) = \frac{\text{Gain}(S, A)}{\text{SplitInfo}(A)}
      \]

- **CART (Classification and Regression Trees)**:
  - **Gini Index Calculation**:
    \[
    \text{Gini}(S) = 1 - \sum_{i=1}^{c} (p_i)^2
    \]
    - **Example**:
      - If a node contains 10 samples with 4 in class A and 6 in class B:
        \[
        p_{\text{A}} = \frac{4}{10}, \quad p_{\text{B}} = \frac{6}{10}
        \]
        \[
        \text{Gini}(S) = 1 - \left( \left( \frac{4}{10} \right)^2 + \left( \frac{6}{10} \right)^2 \right ) = 0.48
        \]

### **2. Pruning**
- **Cost Complexity Pruning (CART)**:
  - Minimizes the cost complexity function:
    \[
    \text{CostComplexity}(T) = \text{MisclassificationCost}(T) + \alpha \times \text{NumberOfLeaves}(T)
    \]
    - \( \alpha \): Complexity parameter.
  - **Process**:
    - Increase \( \alpha \) to prune more branches.
    - Select \( \alpha \) that minimizes cross-validated error.

- **Reduced Error Pruning**:
  - **Process**:
    - Remove branches that do not improve accuracy on a validation set.
    - Simplifies the tree without significantly affecting performance.

### **3. Assignments**
- **Implement C4.5 and CART algorithms to classify a dataset. Plot the decision tree graph.**
  - **Solution Steps:**
    1. **Dataset**:
       - Use the Iris dataset or a similar dataset.
    2. **Implement C4.5**:
       - **Handle Continuous Attributes**:
         - Determine threshold values to split continuous attributes.
       - **Compute Gain Ratio**:
         - For each attribute, calculate Gain Ratio.
       - **Build Tree**:
         - Recursively split nodes using attributes with highest Gain Ratio.
       - **Prune Tree**:
         - Apply pruning techniques to avoid overfitting.
    3. **Implement CART**:
       - **Binary Splits**:
         - For each attribute, find the best split that minimizes Gini Index.
       - **Build Tree**:
         - Recursively create binary splits.
       - **Prune Tree**:
         - Use Cost Complexity Pruning.
    4. **Visualization**:
       - Use `Graphviz` or `sklearn.tree.plot_tree` to visualize.
         ```python
         from sklearn import tree
         tree.plot_tree(model)
         ```
    5. **Evaluation**:
       - Compare accuracy on test data.
       - Analyze differences between C4.5 and CART.

---

# Week 10: K-Means Clustering

### **1. Introduction to Clustering**
- **Clustering** groups data points so that points in the same group are more similar to each other than to those in other groups.
- **K-Means Algorithm**:
  - **Objective Function**:
    \[
    \text{Minimize} \sum_{k=1}^{K} \sum_{x_i \in C_k} \| x_i - \mu_k \|^2
    \]
    - \( \mu_k \): Centroid of cluster \( C_k \).
    - The goal is to minimize the within-cluster sum of squares (WCSS).

### **2. Steps in K-Means Algorithm**
1. **Initialization**:
   - Randomly select \( K \) data points as initial centroids.
2. **Assignment Step**:
   - For each data point \( x_i \), assign it to the nearest centroid \( \mu_k \) based on Euclidean distance:
     \[
     \text{Assign } x_i \text{ to cluster } C_k \text{ where } \| x_i - \mu_k \|^2 \text{ is minimized}
     \]
3. **Update Step**:
   - Recalculate centroids for each cluster:
     \[
     \mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i
     \]
4. **Convergence**:
   - Repeat steps 2 and 3 until centroids stabilize (no change in assignments or centroids).

### **3. Evaluating Clustering**
- **Sum of Squared Errors (SSE)**:
  \[
  \text{SSE} = \sum_{k=1}^{K} \sum_{x_i \in C_k} \| x_i - \mu_k \|^2
  \]
  - Lower SSE indicates better clustering.
- **Elbow Method**:
  - Plot SSE vs. \( K \).
  - The point where the rate of decrease sharply changes (elbow point) suggests a suitable \( K \).

### **4. Assignments**
- **Perform K-Means clustering with different values of \( K \) and plot SSE vs. \( K \).**
  - **Solution Steps:**
    1. **Dataset**:
       - Use a dataset like the Iris dataset without labels.
    2. **Run K-Means**:
       - For \( K \) ranging from 1 to 10:
         - Perform K-Means clustering.
         - Record SSE for each \( K \).
    3. **Plot SSE vs. \( K \)**:
       - Use `matplotlib` to create the plot.
         ```python
         import matplotlib.pyplot as plt
         plt.plot(K_values, SSE_values, 'bx-')
         plt.xlabel('Number of clusters K')
         plt.ylabel('Sum of squared errors (SSE)')
         plt.title('Elbow Method For Optimal K')
         plt.show()
         ```
    4. **Determine Optimal \( K \)**:
       - Identify the elbow point in the plot.
    5. **Interpret Results**:
       - Analyze cluster centroids.
       - Assign labels to clusters if possible.

---
