#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Madison Stulir
#ML HW1


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#load expression level data
exp=np.loadtxt("expression.txt")


# In[3]:


#check data size
print(len(exp))
len(exp[0])
exp.shape


# In[4]:


#load SNP data
SNP=np.loadtxt("SNPs.txt")


# In[5]:


#check data size
print(len(SNP))
len(SNP[0])


# In[6]:


#mean center the data

#initialize 2d array for mean centered data
SNP_centered=SNP
#mean centering function
center_function = lambda x: x - x.mean()

#apply function to each column of SNP data
for i in range(len(SNP)):
    SNP_centered[i]=center_function(SNP[i])


#check a mean to ensure it is now centered
SNP_centered[40].mean()


# In[7]:


# get LEU2 expression levels to work with for 5a

#make sure correct col using gene_list to identify gene name
genes=np.loadtxt("gene_list.txt", dtype=str)
print(len(genes))
genes[394]


# In[8]:


#extract LEU2 column from gene expression levels to serve as Y
LEU2exp=exp[:,394]


# In[9]:


# mean center LEU2 levels

LEU2_centered=center_function(LEU2exp)

#make sure it is centered
print(LEU2_centered.mean())


# In[10]:


#practice indexing for each column of SNP_centered

f=SNP_centered[:,0]
print(f.shape)
print(len(SNP_centered[0]))


# In[11]:


#univariate regression

beta=[]
for i in range(len(SNP_centered[0])): #for 1260 columns separately
    #X is a single column of the centered SNP data
    X = np.column_stack([np.ones(len(SNP_centered[:,i]), dtype=np.float32),SNP_centered[:,i]])
    #Y is the LEU2 expression data (centered)
    y = LEU2_centered
    #calculate the linear regression for only that SNP
    coeffs = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,y))
    # add to the list of beta values, only the slope, not the intercept
    beta=np.append(beta,coeffs[1])


# In[12]:


#plot the betas for univariate regression
x=np.arange(0, 1260, 1, dtype=int)
plt.scatter(x,beta,s=10)
plt.title('Univariate Regression of LEU2 expression on 1260 SNPs')
plt.ylabel('Beta')
plt.xlabel('SNP #')
plt.savefig('univariate.png')


# In[13]:


# multivariate ridge regression

X = SNP_centered
y = LEU2_centered
beta = np.dot(np.linalg.inv(np.dot(X.T,X)+(1/5)*np.identity(1260)),np.dot(X.T,y))

beta


# In[14]:


#plot the betas
x=np.arange(0, 1260, 1, dtype=int)
plt.scatter(x,beta,s=10)
plt.title(r'Multivariate Ridge Regression of LEU2 expression on 1260 SNPs, $\sigma_0^2=5$')
plt.ylabel('Beta')
plt.xlabel('SNP #')
plt.savefig('ridge5.png')


# In[15]:


#2nd ridge regression

X = SNP_centered
y = LEU2_centered
beta = np.dot(np.linalg.inv(np.dot(X.T,X)+(1/0.005)*np.identity(1260)),np.dot(X.T,y))

beta


# In[16]:


x=np.arange(0, 1260, 1, dtype=int)
plt.scatter(x,beta,s=10)
plt.title(r'Multivariate Ridge Regression of LEU2 expression on 1260 SNPs, $\sigma_0^2=0.005$')
plt.ylabel('Beta')
plt.xlabel('SNP #')
plt.savefig('ridge0.005.png')


# In[17]:


#which SNP has largest influence on LEU2 expression?

#take min and determine its index then obtain which SNP this index corresponds to

index=np.argmin(beta)
print(genes[index])
print(index)


# In[ ]:
