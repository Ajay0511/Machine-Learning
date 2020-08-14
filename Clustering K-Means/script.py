import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
digits = datasets.load_digits()

print(digits.DESCR)

print(digits.data)

print(digits.target)

plt.gray()
plt.matshow(digits.images[100])
plt.show()
print(digits.target[100])

model = KMeans(n_clusters=10,random_state=12)

model.fit(digits.data)

fig = plt.figure(figsize=[8,3])

plt.suptitle("Cluster Centre Images")

for i in range(10):
  ax=fig.add_subplot(2,5,1+i)

  ax.imshow(model.cluster_centers_[i].reshape((8,8)),cmap=plt.cm.binary)

plt.show()

new_samples = np.array([
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,3.79,5.01,0.00,0.00,0.00,0.00,0.00,0.00,5.33,5.93,0.00,0.00,0.00,0.00,0.00,0.00,5.32,5.32,0.00,0.00,0.00,0.00,0.00,0.00,5.16,7.06,0.52,0.00,0.00,0.00,0.00,0.00,2.11,7.61,2.27,0.00,0.00,0.00,0.00,0.00,0.00,2.05,0.23,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,2.80,6.86,6.31,1.58,0.00,0.00,0.00,0.00,5.32,6.69,6.85,6.99,4.03,1.06,0.00,0.00,5.10,6.85,3.55,7.31,7.62,4.56,0.00,0.00,1.97,7.54,7.15,7.01,7.62,4.48,0.00,0.00,0.00,1.44,4.41,5.79,7.62,0.68,0.00,0.00,0.00,0.00,0.00,3.04,7.61,1.13,0.00,0.00,0.00,0.00,0.00,1.74,7.62,1.52],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.38,0.76,0.68,0.00,0.00,0.00,0.00,1.73,7.54,7.62,7.62,7.38,4.63,1.43,0.00,3.05,7.62,4.78,3.11,6.02,7.22,6.30,0.00,1.28,6.99,7.62,4.02,0.07,5.09,6.84,0.00,0.00,0.08,5.09,7.62,7.29,7.62,3.86,0.00,0.00,0.00,0.08,2.42,4.48,3.64,0.07,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.23,2.96,0.45,0.08,2.20,0.23,0.00,0.00,2.88,7.62,1.82,1.51,7.62,2.28,0.00,0.00,2.81,7.62,1.51,2.05,7.62,2.28,0.00,0.00,0.60,7.46,5.01,3.88,7.62,2.28,0.00,0.00,0.00,5.39,7.62,7.62,7.62,1.28,0.00,0.00,0.00,0.22,0.76,2.81,7.62,0.76,0.00,0.00,0.00,0.00,0.00,2.21,7.62,0.68,0.00]
])

new_labels=model.predict(new_samples)
for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(4, end='')
  elif new_labels[i] == 1:
    print(3, end='')
  elif new_labels[i] == 2:
    print(1, end='')
  elif new_labels[i] == 3:
    print(9, end='')
  elif new_labels[i] == 4:
    print(0, end='')
  elif new_labels[i] == 5:
    print(6, end='')
  elif new_labels[i] == 6:
    print(5, end='')
  elif new_labels[i] == 7:
    print(1, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(2, end='')
print(new_labels)
