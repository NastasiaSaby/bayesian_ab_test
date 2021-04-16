# Databricks notebook source
# MAGIC %pip install pymc3

# COMMAND ----------

# MAGIC %md # Computation

# COMMAND ----------

# DBTITLE 1,Prepare data
import pandas as pd

data = {'visitor':  ['James', 'Mike', 'Tommy', 'Meguy', 'Mia'],
        'variant': ['A', 'B', 'B', 'A', 'B'],
        'conversion': [0, 1, 0, 1, 1],
        'revenue': [0.00, 22.80, 0.00, 12.90, 37.5],
        }

df = pd.DataFrame (data, columns = ['visitor', 'variant', 'conversion', 'revenue'])

df.head()

# COMMAND ----------


import matplotlib.pyplot as plt
import numpy as np
import pandas
import pymc3 as pm

def data_split(df):
    variant = df[df['variant']!='A']
    default = df[df['variant']=='A']

    conv_variant = df[(df['variant']!='A') & (df['conversion']>0 )]
    conv_default = df[(df['variant']=='A') & (df['conversion']>0)]

    default = default['conversion']
    variant = variant['conversion']
    rev_default = conv_default['revenue']
    rev_variant = conv_variant['revenue']
    return [default,variant,rev_default,rev_variant]
      
dataframe = df
observations = data_split(dataframe);

# COMMAND ----------

with pm.Model() as ab_model:
  #Conversion a priori
  a_prior=pm.distributions.continuous.Beta('conversionA', alpha=0.1, beta=0.1)
  b_prior=pm.distributions.continuous.Beta('conversionB', alpha=0.1, beta=0.1)

  #Revenues a priori
  rev_a_prior=pm.distributions.continuous.Gamma('revenueA', alpha=0.1, beta=0.1)
  rev_b_prior=pm.distributions.continuous.Gamma('revenueB', alpha=0.1, beta=0.1)
  
  #Compute the likelihood
  pm.Bernoulli('likelihoodA', a_prior, observed=observations[0])
  pm.Bernoulli('likelihoodB', b_prior, observed=observations[1])
  pm.Poisson('likelihoodRevenueA', rev_a_prior, observed=observations[2])
  pm.Poisson('likelihoodRevenueB', rev_b_prior, observed=observations[3])

  #Compute metrics
  conv_a = pm.Deterministic('conversion_A', a_prior)
  conv_b = pm.Deterministic('conversion_B', b_prior)
  
  conv_rev_a = pm.Deterministic('conversionRevenueA', a_prior*rev_a_prior)
  conv_rev_b = pm.Deterministic('conversionRevenueB', b_prior*rev_b_prior)

  pm.Deterministic('lift', b_prior - a_prior)
  pm.Deterministic('revenueLift', conv_rev_b - conv_rev_a)
    
  step = pm.Slice()
  trace = pm.sample(1000, step=step) 

# COMMAND ----------

# MAGIC %md # Visualisation

# COMMAND ----------

plt.figure(figsize=(20,5))
plt.subplot(1, 2,1)
plt.hist(trace['conversionRevenueA'][:], bins=35, histtype='stepfilled', 
color='#da6d75', label='Posterior of revenue A',alpha=0.5)
plt.legend()
plt.hist(trace['conversionRevenueB'][:], bins=35, histtype='stepfilled',
color='#52c4a8', label='Posterior of revenue B',alpha=0.5)
plt.legend()
plt.subplot(1, 2,2)
plt.hist(trace['conversion_A'][:], bins=35, histtype='stepfilled',
color='#da6d75', label='Posterior of conversion A',alpha=0.5)
plt.legend()
plt.hist(trace['conversion_B'][:], bins=35, histtype='stepfilled',
color='#52c4a8', label='Posterior of conversion B',alpha=0.5)
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md # Interpretation

# COMMAND ----------

# DBTITLE 0,Interpretation
difference_rev_B_A=trace['conversionRevenueB']-trace['conversionRevenueA']
difference_conversion_B_A=trace['conversionB']-trace['conversionA']
difference_rev_A_B=trace['conversionA']-trace['conversionB']
difference_conversion_A_B=trace['conversionA']-trace['conversionB']

# COMMAND ----------

print("Probability of 14% increase by group B revenue and conversion: ", 100*len(difference_rev_B_A[difference_rev_B_A>0.14])*1.0/len(difference_rev_B_A))
print("Probability of 100% increase by group B revenue and conversion: ", 100*len(difference_rev_B_A[difference_rev_B_A>1])*1.0/len(difference_rev_B_A))
print("_________________________________________________________________________________________________")
print("Probability of 14% increase by group A revenue and conversion: ", 100*len(difference_rev_A_B[difference_rev_A_B>0.14])*1.0/len(difference_rev_A_B))
print("Probability of 100% increase by group A revenue and conversion: ", 100*len(difference_rev_A_B[difference_rev_A_B>1])*1.0/len(difference_rev_A_B))
