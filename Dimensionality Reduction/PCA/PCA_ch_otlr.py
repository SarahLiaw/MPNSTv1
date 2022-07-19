# Check n comp (3) at particular variance, heatmap, check for outliers.
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

# Change path if not running locally.
data_path = '/home/sarahl/PycharmProjects/MPNST_v1/data_v1/MPNST_v1_concat.csv'
data = pd.read_csv(data_path)
id = list(data['ID'])
id_concat = [i[3:] for i in id]
print(id)
data = data.iloc[:, 1:-1]
# Cross-checking some stuff here.

y = data["Diagnosis"]
print(y)

le = LabelEncoder()
binary_encoded_y = pd.Series(le.fit_transform(y))

X = data.iloc[:, 1:]

scaler = StandardScaler()
scaler.fit(X)
scaled_data = scaler.transform(X)

pca = PCA(n_components=3)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)

label_colour_dict = {'plexiform': 'orange', 'healthy': 'green', 'mpnst': 'red'}

#color vector creation

cvec = [label_colour_dict[label] for label in y]

plt.figure(figsize=(8, 6))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=cvec, edgecolor=['none'], alpha=0.5)

# can comment this out if don't want labels
for lib in id_concat:
    i = id_concat.index(lib)
    labelpad = 0.01
    plt.text(x_pca[i, 0]+labelpad, x_pca[i, 1]+labelpad, lib, fontsize=7)

plt.xlabel('First principle component')
plt.ylabel('Second principle component')
# plt.show(block=True)
# plt.interactive(False)

print(pca.components_)
import seaborn as sns
plt.figure(figsize=(12,6))
df_comp = pd.DataFrame(pca.components_,columns=list(data.keys())[1:])
sns.heatmap(df_comp,cmap='plasma',)
plt.show(block=True)
plt.interactive(False)