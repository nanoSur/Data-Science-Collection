#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[41]:


iris=pd.read_csv('IRIS.csv')


# In[42]:


iris.shape


# In[43]:


iris.columns


# In[44]:


iris['species'].value_counts()


# In[45]:


#use 2D scatter plot
iris.plot(kind="scatter",x='sepal_length',y='sepal_width');
plt.show()


# In[150]:


gb=iris.groupby('species')
gb


# In[151]:


iris_setosa=gb.get_group('Iris-setosa')
iris_versicolor=gb.get_group('Iris-versicolor')
iris_virginca=gb.get_group('Iris-virginica')


# In[152]:


iris_setosa.plot(kind='scatter',x='sepal_length',y='sepal_width',color='r')
iris_versicolor.plot(kind='scatter',x='sepal_length',y='sepal_width',color='g')
iris_virginca.plot(kind='scatter',x='sepal_length',y='sepal_width',color='b')


# In[144]:


# better way of plotting is using seaborn

sns.set_style("whitegrid")
sns.FacetGrid(iris,hue="species",height=6)     .map(plt.scatter,"sepal_length","sepal_width")     .add_legend();
plt.show();


# In[50]:


#https://plotly.com/python/3d-scatter-plots/


# **Pair Plot method**

# In[51]:


#no of pairs = no of independent variables and pairs ie xC2
#in this case 4C2 =6 as (sl,sw),(sl,pl),(sw,pl),(pl,pw),(sl,pw),(sw,pw)


# In[52]:


plt.close()
sns.set_style("whitegrid");
sns.pairplot(iris,hue="species",height=4);
plt.show()


# In[53]:


#limitations:
#dimensionality of plot is less <=6 imagine 100C2 no of plots are too many so not useful 


# **Histogram, PDF , CDF**

# In[ ]:





# In[148]:


import numpy as np
iris_setosa = iris.loc[iris["species"] == "setosa"];
iris_virginica = iris.loc[iris["species"] == "virginica"];
iris_versicolor = iris.loc[iris["species"] == "versicolor"];
plt.show()


# In[155]:


sns.FacetGrid(iris,hue='species',height=10)     .map(sns.displot,"petal_length")     .add_legend()
plt.show()


# In[84]:


#univariat analysis


# **CDF**

# The CDF of a point in graph is the corresponding area under the curve of PDF
# Therefore:
# 1. Integration on PDF will give CDF
# 2. Differentiation on CDF will give PDF

# In[111]:


#1.Plot PDF


counts, bin_edges = np.histogram(iris_setosa['petal_length'], bins=10, density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);

#compute CDF
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,'k-')
plt.plot(bin_edges[1:],cdf)
plt.show()


# In[127]:


iris_setosa.columns[2]


# In[147]:


iris_virginica.head()


# In[154]:


counts, bin_edges = np.histogram(iris_setosa['petal_length'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)


# virginica
counts, bin_edges = np.histogram(iris_virginca['petal_length'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)


#versicolor
counts, bin_edges = np.histogram(iris_versicolor['petal_length'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)


plt.show();


# **Mean Variance SD**
# Variance-and Std Dev --Gives an insight to all the points that are spread across a range
# Variance=(1\n Sigma xi-mean val)

# In[166]:


print("Mean");
print(np.mean(iris_setosa['petal_length']))
print(np.mean(iris_versicolor['petal_length']))
print(np.mean(iris_virginca['petal_length']))
print("\n")

print("Standard Deviation");
print(np.std(iris_setosa['petal_length']))
print(np.std(iris_versicolor['petal_length']))
print(np.std(iris_virginca['petal_length']))


#problem with mean and variance is that  if any outliers are there then there is a drastic effect in the values


# **Median,Percentile,Quartile,IQR,MAD**

# In[168]:


print("Median");
print(np.median(iris_setosa['petal_length']))
print(np.median(iris_versicolor['petal_length']))
print(np.median(iris_virginca['petal_length']))


# In[173]:


#Quantiles are 25th,50th,75th,100th values in a list are the index and then corresponding values of it.  

print("Quantile");
print(np.percentile(iris_setosa['petal_length'],np.arange(0,101,25)))
print(np.percentile(iris_versicolor['petal_length'],np.arange(0,101,25)))
print(np.percentile(iris_virginca['petal_length'],np.arange(0,101,25)))


# In[174]:


print("90th Percentile");
print(np.percentile(iris_setosa['petal_length'],90))
print(np.percentile(iris_versicolor['petal_length'],90))
print(np.percentile(iris_virginca['petal_length'],90))


# In[175]:


print("MAD-Median Absolute Median")

from statsmodels import robust
print(robust.mad(iris_setosa['petal_length']))
print(robust.mad(iris_versicolor['petal_length']))
print(robust.mad(iris_virginca['petal_length']))


#  **Box Plot and Whiskers**

# In[183]:


#the concept of median percentile and quartile
sns.boxplot(x='species',y='petal_length',data=iris)
plt.show()

#corresponds to 25,50,75 percentiles
#whiskers = lines above boxes - can be drawn using min and max values but in seaborn uses formula 1.5*IQR


# In[181]:


#violin plots
sns.violinplot(x='species',y='petal_length',data=iris,size=10)
plt.show()


# **2D density plot**
# draw both bivariate and univariate KDEs:

# In[189]:


sns.jointplot(x='petal_length',y='petal_width',data=iris_setosa,kind="kde"); # kernel density estimation.
plt.show()


# In[ ]:




