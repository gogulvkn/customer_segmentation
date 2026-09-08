# Customer Segmentation using K-Means Clustering

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)

## 📌 Project Overview

Customer segmentation is the process of dividing customers into meaningful groups based on their behavior and characteristics.

This project uses **K-Means Clustering**, an unsupervised machine learning algorithm, to identify different customer groups based on purchasing and engagement behavior.

The goal is to help businesses understand their customers and create more targeted marketing strategies.

🔗 **GitHub Repository:**
https://github.com/gogulvkn/customer_segmentation

---

## 🎯 Objectives

The main objectives of this project are:

* Analyze customer behavior
* Identify meaningful customer groups
* Apply K-Means clustering
* Determine suitable customer segments
* Visualize customer clusters
* Build a simple application for customer segmentation
* Provide a foundation for targeted marketing strategies

---

## 🤖 Machine Learning Approach

This project uses **K-Means Clustering**, an unsupervised machine learning algorithm.

K-Means works by:

1. Selecting the number of clusters (`K`)
2. Initializing cluster centroids
3. Assigning customers to the nearest centroid
4. Updating the centroid of each cluster
5. Repeating the process until the clusters stabilize

The final result assigns each customer to a cluster representing customers with similar behavior.

---

## 📊 Customer Features

The dataset contains customer behavioral features such as:

| Feature                     | Description                            |
| --------------------------- | -------------------------------------- |
| `AnnualIncome`              | Customer's annual income               |
| `TotalSpend`                | Total amount spent by the customer     |
| `PurchaseFrequencyPerMonth` | Average purchases per month            |
| `WebsiteVisitsPerMonth`     | Monthly website visits                 |
| `AppUsageHoursPerMonth`     | Monthly application usage              |
| `DiscountUsesPerYear`       | Number of discounts used annually      |
| `Returns`                   | Number of product returns              |
| `LastPurchaseDays`          | Number of days since the last purchase |

These features are used to identify similarities and differences between customers.

---

## 🔄 Project Workflow

```text
Customer Dataset
       │
       ▼
Data Loading
       │
       ▼
Data Cleaning
       │
       ▼
Exploratory Data Analysis
       │
       ▼
Feature Selection
       │
       ▼
Feature Scaling
       │
       ▼
Find Optimal Number of Clusters
       │
       ▼
K-Means Clustering
       │
       ▼
Customer Segments
       │
       ▼
Visualization & Analysis
       │
       ▼
Customer Segmentation Application
```

---

## 📈 Finding the Optimal K

The **Elbow Method** can be used to determine an appropriate number of clusters.

The method calculates the model's inertia for different values of `K`.

```python
from sklearn.cluster import KMeans

inertia = []

for k in range(2, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)
    inertia.append(model.inertia_)
```

The value of `K` can then be selected by examining where the reduction in inertia begins to slow down.

---

## 🧮 Feature Scaling

Because the dataset contains features with different numerical ranges, feature scaling is important before applying K-Means.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

Scaling prevents features with larger numerical values from dominating the clustering process.

---

## 🧠 K-Means Model

After selecting an appropriate number of clusters, the K-Means model is trained:

```python
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)
```

The resulting cluster labels are added to the customer dataset.

```python
df["Cluster"] = clusters
```

> The number of clusters can be changed based on the results of the Elbow Method and other evaluation techniques.

---

## 👥 Customer Segments

The clusters can be analyzed to understand the characteristics of each customer group.

For example, segments can be interpreted based on:

* Spending behavior
* Purchase frequency
* Website engagement
* App usage
* Discount usage
* Return behavior
* Recency of purchase

Possible business interpretations include:

| Segment                   | Possible Characteristics             | Possible Strategy                              |
| ------------------------- | ------------------------------------ | ---------------------------------------------- |
| High-Value Customers      | High spending and frequent purchases | Loyalty rewards and VIP offers                 |
| Regular Customers         | Consistent purchasing behavior       | Cross-selling and personalized recommendations |
| Low-Engagement Customers  | Low visits and low purchases         | Re-engagement campaigns                        |
| Discount-Driven Customers | Frequent use of discounts            | Targeted promotions                            |
| At-Risk Customers         | Long time since last purchase        | Win-back campaigns                             |

> Segment names should be assigned after examining the actual cluster statistics rather than assuming that a particular cluster number represents a specific customer type.

---

## 📁 Project Structure

```text
customer_segmentation/
│
├── app.py
│
├── customer_segmentation.ipynb
│
├── customer_segmentation_100k.csv
│
└── README.md
```

The repository currently contains the Streamlit application, Jupyter notebook, and 100K customer dataset.

---

## 💻 Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **Jupyter Notebook**
* **Streamlit**

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/gogulvkn/customer_segmentation.git
```

Move into the project directory:

```bash
cd customer_segmentation
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit jupyter
```

---

## ▶️ Run the Jupyter Notebook

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
customer_segmentation.ipynb
```

The notebook contains the data analysis and clustering workflow.

---

## 🚀 Run the Streamlit Application

Start the application with:

```bash
streamlit run app.py
```

The application can then be opened in your browser.

---

## 📊 Dataset

The project uses a customer dataset containing **100,000 customer records**.

The dataset is included in the repository as:

```text
customer_segmentation_100k.csv
```

---

## 💡 Business Applications

Customer segmentation can be used to support:

### Marketing

Create targeted campaigns for different customer groups.

### Customer Retention

Identify customers with decreasing engagement or long periods since their last purchase.

### Personalization

Provide different recommendations and offers based on customer behavior.

### Loyalty Programs

Identify high-value customers and provide rewards.

### Sales Optimization

Focus sales efforts on customer groups with higher purchasing potential.

---

## 📌 Key Benefits

This project demonstrates how unsupervised machine learning can be used to:

* Discover hidden patterns in customer behavior
* Group similar customers automatically
* Understand customer purchasing patterns
* Support data-driven marketing decisions
* Build a practical machine learning application

---

## 🔮 Future Improvements

Possible improvements include:

* Add Silhouette Score evaluation
* Compare K-Means with DBSCAN
* Compare K-Means with Hierarchical Clustering
* Add automated cluster profiling
* Add interactive Plotly visualizations
* Add customer-level predictions
* Add downloadable segmented customer data
* Deploy the Streamlit application
* Add model persistence using `joblib`
* Create a REST API using FastAPI
* Add automated retraining for new customer data

---

## 📸 Application

The project includes a Streamlit application for interacting with the customer segmentation model.

Run it with:

```bash
streamlit run app.py
```

---

## 🧑‍💻 Author

**kamatchinathan v**

GitHub:
https://github.com/gogulvkn

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📄 License

This project is intended for educational and demonstration purposes.
