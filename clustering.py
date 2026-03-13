from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# -----------------------------
# Define the documents
# -----------------------------
documents = [
    "Cats are known for their agility and grace", 
    "Dogs are often called man's best friend",
    "Some dogs are trained to assist people with disabilities",
    "The sun rises in the east and sets in the west",
    "Many cats enjoy climbing trees and chasing toys"
]

# -----------------------------
# Convert text documents into TF-IDF vectors
# -----------------------------
vectorizer = TfidfVectorizer(stop_words="english")

X = vectorizer.fit_transform(documents)

# -----------------------------
# Perform K-Means clustering
# -----------------------------
kmeans = KMeans(n_clusters=3, random_state=0)

kmeans.fit(X)

# -----------------------------
# Print cluster labels
# -----------------------------
print("Cluster Labels:")
print(kmeans.labels_)

# -----------------------------
# Show which document belongs to which cluster
# -----------------------------
for i, doc in enumerate(documents):
    print(f"Document {i+1} → Cluster {kmeans.labels_[i]}")
