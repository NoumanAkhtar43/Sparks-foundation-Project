import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
# Load the dataset
file_path = 'D:\Practice Tutorials\Python practice\SparksFoundation\dataset.csv'
data = pd.read_csv(file_path)
# Define the features and target
X = data[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = data["Species"]

# Create the decision tree classifier
clf = tree.DecisionTreeClassifier()

# Train the model
clf = clf.fit(X, y)

# Visualize the decision tree
plt.figure(figsize=(15,7))
tree.plot_tree(clf, feature_names=X.columns, class_names=["setosa", "versicolor", "virginica"], filled=True, precision=3, proportion=True);
plt.show()
