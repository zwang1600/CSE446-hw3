import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

# Define data points and labels
X = np.array([[1, 2], [1, 3], [2, 3], [3, 4], 
                [0, 0.5], [1, 0], [2, 1], [3, 0]])
y = np.array([-1, -1, -1, -1, 1, 1, 1, 1])

# Fit SVM model
clf = SVC(kernel='linear', C=10)
clf.fit(X, y)

# # Get hyperplane parameters
w = clf.coef_[0]
b = clf.intercept_[0]
# w = np.sum((y.reshape(-1, 1) * X) * np.array([1, 1]), axis=0)
# b = 1 - np.min(X.dot(w))
xx = np.linspace(-1, 4)
yy = (-w[0] * xx - b) / w[1]
margin = 1 / np.sqrt(np.sum(w ** 2))

# plt.plot([-10, 10], [0, 0], color='black')  # x-axis
# plt.plot([0, 0], [-10, 10], color='black')  # y-axis

# Plot data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50)

# Plot the support vectors
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='g')
for sv in clf.support_vectors_:
    plt.annotate(f'  ({sv[0]}, {sv[1]})', xy=sv)

# Plot hyperplanes and margin
plt.plot(xx, yy, 'k-', label= 'x^T w + b = 0 (Maximum-margin hyperplane)', color='r')
plt.plot(xx, yy + margin, 'k--', label='x^T w + b = -1', color='g')
plt.plot(xx, yy - margin, 'k--', label='x^T w + b = 1', color='b')


plt.xlim([-1, 4])
plt.ylim([-1, 5])
plt.legend(loc='best')
plt.show()