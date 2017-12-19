import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits() #read the list

clf = svm.SVC(gamma=0.001, C=100) #classification

x,y = digits.data[:-10], digits.target[:-10] #deixem els ultims 10 per testing 
clf.fit(x,y) # entrenem

print('Prediction:',clf.predict(digits.data[-1])) #provem amb el penultim 

plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()