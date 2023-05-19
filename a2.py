# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.svm import SVC

# if __name__ == "__main__":
#     # Data points
#     X = np.array([[1, 2], [1, 3], [2, 3], [3, 4], 
#                   [0, 0.5], [1, 0], [2, 1], [3, 0]])
#     y = np.array([-1, -1, -1, -1, 1, 1, 1, 1])

#     # Fit SVM model using sklearn
#     clf = SVC(kernel='linear')
#     clf.fit(X, y)

#     # Get the separating hyperplane and margins
#     w = clf.coef_[0]
#     b = clf.intercept_
#     slope = -w[0] / w[1]
#     y_intercept = -b / w[1]
#     margin = 1 / np.sqrt(np.sum(w ** 2))

#     print(f'slope: {slope}')
#     print(f'y_intercept: {y_intercept}')

#     xx = np.linspace(0, 5)
#     yy = slope * xx + y_intercept
#     yy_down = yy - slope * margin
#     yy_up = yy + slope * margin

#     # Plot data points, support vectors, and the hyperplanes
#     plt.scatter(X[:, 0], X[:, 1], c=y)
#     plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='r')
#     for sv in clf.support_vectors_:
#         plt.annotate(f'  ({sv[0]}, {sv[1]})', xy=sv)
#     plt.plot(xx, yy, 'k-')
#     plt.plot(xx, yy_down, 'k--')
#     plt.plot(xx, yy_up, 'k--')
#     plt.show()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

# Define data points and labels
X = np.array([[1, 2], [1, 3], [2, 3], [3, 4], 
                [0, 0.5], [1, 0], [2, 1], [3, 0]])
y = np.array([-1, -1, -1, -1, 1, 1, 1, 1])

# Fit SVM model
clf = SVC(kernel='linear')
clf.fit(X, y)

# # Get hyperplane parameters
w = clf.coef_[0]
b = clf.intercept_[0]
# w = np.sum((y.reshape(-1, 1) * X) * np.array([1, 1]), axis=0)
# b = 1 - np.min(X.dot(w))
xx = np.linspace(-1, 4)
yy = (-w[0] * xx - b) / w[1]
margin = 1 / np.sqrt(np.sum(w ** 2))

# Plot data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50)

# Plot the support vectors
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='g')
for sv in clf.support_vectors_:
    plt.annotate(f'  ({sv[0]}, {sv[1]})', xy=sv)

# Plot hyperplanes and margin
# plt.plot(xx, (-w[0] * xx - b) / w[1], 'k-')
# plt.plot(xx, (-w[0] * xx - b + margin) / w[1], 'k--')
# plt.plot(xx, (-w[0] * xx - b - margin) / w[1], 'k--')
plt.plot(xx, yy, 'k-', label='Maximum-margin hyperplane')
plt.plot(xx, yy + margin, 'k--', label='Positive hyperplane')
plt.plot(xx, yy - margin, 'k--', label='Negative hyperplane')

plt.xlim([-1, 4])
plt.ylim([-1, 5])
plt.show()