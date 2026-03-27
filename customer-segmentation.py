# --- 1. Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# --- 2. Load Dataset ---
# Note: This uses the popular 'Mall Customers' dataset structure
# If loading from a local file, use: df = pd.read_csv('Mall_Customers.csv')
# For demonstration, we will create a synthetic dataset if file not present.
try:
    df = pd.read_csv('Mall_Customers.csv')
except FileNotFoundError:
    # Creating a synthetic dataset for demonstration
    np.random.seed(42)
    data = {
        'CustomerID': range(1, 201),
        'Genre': np.random.choice(['Male', 'Female'], 200),
        'Age': np.random.randint(18, 70, 200),
        'Annual Income (k$)': np.random.randint(15, 137, 200),
        'Spending Score (1-100)': np.random.randint(1, 100, 200)
    }
    df = pd.DataFrame(data)
    print("Synthetic dataset created as 'Mall_Customers.csv' was not found.")

# --- 3. Data Preprocessing ---
print("\n--- Data Overview ---")
print(df.head())
print(df.info())

# Check for nulls
print("\nMissing Values:\n", df.isnull().sum())

# Feature Selection: We use Annual Income and Spending Score for clustering
# Ideally, demographics like Age/Gender are used for profiling later.
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scaling the data (Crucial for Distance-based algorithms like K-Means/DBSCAN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 4. Model Training: K-Means Clustering ---

# 4a. Elbow Method to find optimal K
wcss = []
k_range = range(1, 11)
for i in k_range:
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(10, 5))
plt.plot(k_range, wcss, marker='o')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.savefig('elbow_plot.png')
plt.show()

# 4b. Training K-Means with optimal K (Let's assume K=5 based on typical Mall data)
optimal_k = 5
kmeans_model = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
y_kmeans = kmeans_model.fit_predict(X_scaled)

# Add cluster labels to original dataframe
df['Cluster_KMeans'] = y_kmeans

# --- 5. Model Training: DBSCAN ---
# DBSCAN is density-based. We need to tune epsilon and min_samples.
dbscan = DBSCAN(eps=0.5, min_samples=5)
y_dbscan = dbscan.fit_predict(X_scaled)
df['Cluster_DBSCAN'] = y_dbscan

# --- 6. Visualization ---

# 6a. Visualizing K-Means Clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(
    x='Annual Income (k$)', 
    y='Spending Score (1-100)', 
    data=df, 
    hue='Cluster_KMeans', 
    palette='Set1', 
    s=100
)
plt.title('Customer Segments (K-Means)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.savefig('kmeans_clusters.png')
plt.show()

# 6b. 3D Visualization using PCA (Optional advanced viz)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
plt.figure(figsize=(8,6))
plt.scatter(principal_components[:,0], principal_components[:,1], c=y_kmeans, cmap='rainbow')
plt.title('PCA Reduction of Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig('pca_clusters.png')
plt.show()

# --- 7. Evaluation & Segment Identification ---
# Silhouette Score
sil_score = silhouette_score(X_scaled, y_kmeans)
print(f"Silhouette Score for K-Means: {sil_score:.2f}")

# Analyzing Clusters
cluster_stats = df.groupby('Cluster_KMeans').mean()[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
print("\n--- Cluster Statistics (Mean Values) ---")
print(cluster_stats)

# Mapping segment names based on analysis
def label_segment(row):
    if row['Cluster_KMeans'] == 0: return 'Standard'
    elif row['Cluster_KMeans'] == 1: return 'High Income - Low Spending (Careful)'
    elif row['Cluster_KMeans'] == 2: return 'Low Income - High Spending (Target)'
    elif row['Cluster_KMeans'] == 3: return 'Low Income - Low Spending (Sensible)'
    else: return 'High Income - High Spending (Premium)'

df['Segment'] = df.apply(label_segment, axis=1)

# --- 8. Recommendation Engine (Simple Similarity-Based) ---
# We will build a function to recommend products based on the Segment

product_map = {
    'High Income - High Spending (Premium)': ['Luxury Watch', 'Designer Apparel', 'Latest Electronics'],
    'Low Income - High Spending (Target)': ['Budget Smartphones', 'Trendy Accessories', 'Sale Items'],
    'High Income - Low Spending (Careful)': ['Investment Plans', 'Premium Groceries', 'Books'],
    'Low Income - Low Spending (Sensible)': ['Daily Essentials', 'Discount Coupons', 'Home Utilities'],
    'Standard': ['Clothing', 'Skincare', 'Home Decor']
}

def get_recommendations(income, score):
    # Scale input
    input_data = scaler.transform([[income, score]])
    # Predict cluster
    cluster = kmeans_model.predict(input_data)[0]
    
    # Get segment name logic (Reusing logic from above)
    temp_row = {'Cluster_KMeans': cluster} # simplified for logic
    if cluster == 0: segment = 'Standard'
    elif cluster == 1: segment = 'High Income - Low Spending (Careful)'
    elif cluster == 2: segment = 'Low Income - High Spending (Target)'
    elif cluster == 3: segment = 'Low Income - Low Spending (Sensible)'
    else: segment = 'High Income - High Spending (Premium)'
        
    return product_map.get(segment, ["General Items"])

# Test Recommendation
test_income, test_score = 80, 90 # High Income, High Spending
print(f"\nRecommendation for Income={test_income}k, Score={test_score}:")
print(get_recommendations(test_income, test_score))

# Save processed data
df.to_csv('segmented_customers.csv', index=False)
print("\nProcessing Complete. File saved as 'segmented_customers.csv'")


