import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

st.title("Iris KMeans Clustering")

iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['Species'] = pd.Categorical.from_codes(iris.target,iris.target_names)

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Species Distribition")
st.bar_chart(df["Species"].value_counts())

iris1=df.drop(columns=["Species"])

k=st.slider("Coose number of clusters (k)",2,5,3)
kmeans=KMeans(n_clusters=k,random_state=42)
clusters=kmeans.fit_predict(iris1)

df["Cluster"] = clusters

#plot: sepal
st.subheader("sepal cluster visualization")
fig1,ax1=plt.subplots()
ax1.scatter(df['sepal length (cm)'],df['sepal width (cm)'],c=clusters,cmap='viridis')
ax1.set_xlabel("sepal Length")
ax1.set_ylabel("Sepal width")
ax1.set_title("KMeans clusters: sepal")
st.pyplot(fig1)

st.subheader("petal True Species visualization")
species_map = {'setosa': 0, 'versicolor': 1, 'virginia': 2}
colors = df['Species'].map(species_map)
fig2,ax2=plt.subplots()
ax2.scatter(df['petal length (cm)'],df['petal width (cm)'],c=colors,cmap='Accent')
ax2.set_xlabel("Petal Length")
ax2.set_ylabel("Petal Width")
ax2.set_title("True Species Labels: Petal")
st.pyplot(fig2)

