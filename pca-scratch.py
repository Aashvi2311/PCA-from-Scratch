import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
data = load_iris()

df = pd.DataFrame(data.data,columns=data.feature_names)
df_centered = df - df.mean()
cov_matrix = df_centered.cov()
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
sorted_index = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_index]
sorted_eigenvectors = eigenvectors[:,sorted_index]

k = 2

feature_vector = sorted_eigenvectors[:,:k]
final_data = np.dot(df_centered,feature_vector)
#print(final_data)


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
sklearn_result = pca.fit_transform(df)
#print(sklearn_result)

import matplotlib.pyplot as plt
plt.scatter(final_data[:,0],final_data[:,1],color='red')
plt.scatter(sklearn_result[:,0],sklearn_result[:,1])
plt.show()