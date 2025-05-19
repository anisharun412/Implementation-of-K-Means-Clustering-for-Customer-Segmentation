# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess data: Import data, inspect it, and handle missing values if any.
2. Determine optimal clusters: Use the Elbow Method to identify the number of clusters by plotting WCSS against cluster numbers.
3. Fit the K-Means model: Apply K-Means with the chosen number of clusters to the selected features. Assign cluster labels to each data point.
4. Plot data points in a scatter plot, color-coded by cluster assignments for interpretation.
   
## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: 
RegisterNumber:  
Developed by: Arunsamy D
RegisterNumber:  212224240016
*/
```

DataSet

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
df = pd.read_csv("/content/Mall_Customers.csv")

# Check the data
print(df.head())
print(df.info())
print(df.isnull().sum())
```

Elbow Method

```python
features = df[['Annual Income (k$)', 'Spending Score (1-100)']]  #Features

# Elbow method
wcss = []
for i in range(1, 11):
    model = KMeans(n_clusters=i, init='k-means++', random_state=42)
    model.fit(features)
    wcss.append(model.inertia_)


plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

```

Kmeans Model

```python

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(features)


df['Cluster'] = clusters

group0 = df[df['Cluster'] == 0]
group1 = df[df['Cluster'] == 1]
group2 = df[df['Cluster'] == 2]
group3 = df[df['Cluster'] == 3]
group4 = df[df['Cluster'] == 4]

plt.figure(figsize=(8, 5))
plt.scatter(group0['Annual Income (k$)'], group0['Spending Score (1-100)'], c='red', label='Cluster 0')
plt.scatter(group1['Annual Income (k$)'], group1['Spending Score (1-100)'], c='black', label='Cluster 1')
plt.scatter(group2['Annual Income (k$)'], group2['Spending Score (1-100)'], c='blue', label='Cluster 2')
plt.scatter(group3['Annual Income (k$)'], group3['Spending Score (1-100)'], c='green', label='Cluster 3')
plt.scatter(group4['Annual Income (k$)'], group4['Spending Score (1-100)'], c='magenta', label='Cluster 4')

plt.legend()
plt.title("Customer Segments")
```

## Output:

### DataSet

![image](https://github.com/user-attachments/assets/3b7712bb-323d-4b61-aeb2-08a946995474)

### Elbow Method Graph

![download](https://github.com/user-attachments/assets/7f2cdfd0-0dc9-4c11-a0e1-3487ba1385dc)

### Kmeans Cluster

![download](https://github.com/user-attachments/assets/63b4d4f4-47a5-4391-b16f-a403fd3f7f76)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
