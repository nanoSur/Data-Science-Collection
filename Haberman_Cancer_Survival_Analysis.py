#!/usr/bin/env python
# coding: utf-8

# **Haberman's Survival : Exploratory Data Analysis** \
# Haberman's survival dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer.

# In[146]:


import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# In[147]:


haberman_data=pd.read_csv('haberman.csv')


# In[148]:


haberman_data=haberman_data.rename(columns={'30':'Age','64':'Oper_Year','1':'axil_nodes','1.1':'Surv_status'});


# In[149]:


haberman_data.head()


# In[150]:


haberman_data.shape


# Age of patient at the time of operation (numerical)\
# Patient's year of operation (year - 1900, numerical)\
# Number of positive axillary nodes detected (numerical)\
# Survival status (class attribute) 1 = the patient survived 5 years or longer 2 = the patient died within 5 years

# In[151]:


gSurvival=haberman_data.groupby('Surv_status')

for survival_status,survival_data in gSurvival:
    print(survival_status);
    print(survival_data);


# In[152]:


survived_1=gSurvival.get_group(1)
survived_2=gSurvival.get_group(2)

# print(survived_1)
# print(survived_2.empty)


# In[153]:


survived_1.plot(kind='scatter',x='Age',y='axil_nodes',color='b')
plt.xlabel('Age of patient')
plt.ylabel('axillary nodes')
plt.show()


# In[154]:


survived_2.plot(kind='scatter',x='Age',y='axil_nodes',color='k')
plt.xlabel('Age of patient')
plt.ylabel('axillary nodes')
plt.show()


# In[155]:


axil_nodes_survived_null=survived_1[survived_1['axil_nodes']==0].value_counts()
axil_nodes_survived_not_null=survived_1[survived_1['axil_nodes']>0].value_counts()
print("No of auxillary nodes were not found=",axil_nodes_survived_null.count())
print("No of auxillary nodes were found but survived more than 5 years=",axil_nodes_survived_not_null.count())


# In[156]:


axil_nodes_null=survived_2[survived_2['axil_nodes']==0].value_counts()
axil_nodes_not_null=survived_2[survived_2['axil_nodes']>0].value_counts()
print("No of auxillary nodes were not found=",axil_nodes_null.count())
print("No of auxillary nodes were found but did not survive more than 5 years=",axil_nodes_not_null.count())


# In[157]:


sns.set_style("whitegrid")
sns.FacetGrid(haberman_data,hue="Surv_status",height=6)     .map(plt.scatter,"Age","axil_nodes")     .add_legend();
plt.show();


# As it can be seen from the scatter the patients who have died from Cancer have some dependency on axillary nodes and few have no nodes still haven't survived the operation.
# Lets see a pair plot to visualize the dependency on the other factors

# In[158]:


plt.close()
sns.set_style("whitegrid")
sns.pairplot(haberman_data,hue='Surv_status',height=4)
plt.show()


# In[159]:


survived_2['Age'].std()


# In[160]:


sns.FacetGrid(haberman_data,hue='Surv_status',height=8)    .map(sns.distplot,'axil_nodes')    .add_legend()
plt.show()


# **PDF CDF**

# In[161]:


#Plot PDF

counts,bin_edges=np.histogram(survived_1['axil_nodes'],bins=10,density=True)
pdf=counts/sum(counts)
cdf=np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf,'g-')
plt.show()

survived_1['axil_nodes'].max()


# In[162]:


counts,bin_edges=np.histogram(survived_2['axil_nodes'],bins=10,density=True)
pdf=counts/sum(counts)
cdf=np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf,'k--')
plt.show()

survived_2['axil_nodes'].max()


# In[179]:


status_yes=haberman_data.loc[haberman_data["Surv_status"]==1]
status_no=haberman_data.loc[haberman_data["Surv_status"]==2]

print("Survival Status : YES:")
print(status_yes.describe())
print("Survival Status : NO:")
print(status_no.describe())


# In[164]:


print("Mean");
print(np.mean(survived_1['Age']));
print(np.mean(survived_2['Age']));
print("*******")
print(np.mean(survived_1['axil_nodes']));
print(np.mean(survived_2['axil_nodes']));

print("\nStd Dev");
print(np.std(survived_1['Age']));
print(np.std(survived_2['Age']));
print("*******")
print(np.std(survived_1['axil_nodes']));
print(np.std(survived_2['axil_nodes']));


# In[165]:


#Median 
print("Median");
print(np.median(survived_1['Age']));
print(np.median(survived_2['Age']));
print("*******")
print(np.median(survived_1['axil_nodes']));
print(np.median(survived_2['axil_nodes']));


# In[180]:


print("Quartile")
print(np.percentile(survived_1['Age'],np.arange(0,101,25)));
print(np.percentile(survived_1['axil_nodes'],np.arange(0,101,25)));

print(np.percentile(survived_2['Age'],np.arange(0,101,25)));
print(np.percentile(survived_2['axil_nodes'],np.arange(0,101,25)));
print(np.percentile(survived_2['Oper_Year'],np.arange(0,101,25)));


# In[167]:


print("90th Percentile")
print(np.percentile(survived_1['Age'],90));
print(np.percentile(survived_1['axil_nodes'],90));

print(np.percentile(survived_2['Age'],90));
print(np.percentile(survived_2['axil_nodes'],90));


# In[168]:


print("MAD-Median Absolute Distribution");
from statsmodels import robust

print(robust.mad(survived_1['Age']));
print(robust.mad(survived_1['axil_nodes']));

print(robust.mad(survived_2['Age']));
print(robust.mad(survived_2['axil_nodes']));


# In[173]:


sns.boxplot(x='Surv_status',y='axil_nodes',data=haberman_data)
plt.show()

sns.boxplot(x='Surv_status',y='Age',data=haberman_data)
plt.show()

sns.boxplot(x='Surv_status',y='Oper_Year',data=haberman_data)
plt.show()


# In[174]:


sns.violinplot(x='Surv_status',y='axil_nodes',data=haberman_data)
plt.show()

sns.violinplot(x='Surv_status',y='Age',data=haberman_data)
plt.show()

sns.violinplot(x='Surv_status',y='Oper_Year',data=haberman_data)
plt.show()


# In[176]:


sns.jointplot(x='Surv_status',y='axil_nodes',data=haberman_data,kind='kde')
plt.show()

sns.jointplot(x='Surv_status',y='Age',data=haberman_data,kind='kde')
plt.show()

sns.jointplot(x='Surv_status',y='Oper_Year',data=haberman_data,kind='kde')
plt.show()


# **OBSERVATION**
# 
# 1. The type of classification is based on **Binary Classification** as it has two types of outputs: Survived/Not Survived
# 2. The data provided int the pubic forum has given a very few clear ideas about the dependency of the survival rate based on Age of the patient at the time of operation,Tear of operation and auxillary nodes.Therefore unbalanced dataset but it contains no null values.
# 3. From the plots(Survived):\
# The age survival range is between 42-60.\
# Auxillary nodes=0 have highest survival range.\
# Operation Year 60 has the most survival rate.
# 4. From the pplots(Survived Less):\
# The less survival age > 53 and ages>77 were not able to survive.\
# Auxillary Nodes >10 have not survived.\
# Operation Year from 58-64 has the most non-surviving rate.

# In[ ]:




