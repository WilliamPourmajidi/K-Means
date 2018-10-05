# K-means implementation for assignment 2
# William Pourmajidi

# Importing Required library for plotting numbers
import matplotlib.pyplot as plt
import seaborn as sns;
sns.set()  # for plot styling
# Importing Required library for working with arrays (key,value)
from sklearn.datasets.samples_generator import make_blobs

# Importing Required library for K-means
from sklearn.cluster import KMeans

X = np.array([[8, 4], [5, 4], [2, 4], [2, 6], [2, 8], [8, 6]])

kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
y_kmeans = kmeans.predict(X)
print(X)
print(y_kmeans)
plt.scatter(X[:, 0], X[:, 1]);
plt.show()
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.show()

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.show()
