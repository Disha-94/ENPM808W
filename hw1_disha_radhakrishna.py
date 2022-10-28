#!pip install pandas plotnine
#pip3 install matplotlib 
#pip3 install seaborn   
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotnine import *
import seaborn as sns

#read the csv file
nytData = pd.read_csv('data1/nyt1.csv')

#nytData.dropna(inplace=True)
#nytData.drop_duplicates(inplace=True)

print("Top 5 rows (head()): \n\n", nytData.head(), "\n\n")
print(nytData.info(), "\n\n")

#Frequency distribution of Impressions for each age group
ggplot(nytData, aes(x='Impressions', fill = 'Age_Group')) + \
    geom_bar(stat = 'count')

