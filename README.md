import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers
df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1) 
df = df.rename(columns={'v1':'label','v2':'Text'}) 
df['label_enc'] = df['label'].map({'ham':0,'spam':1}) 
df.head()
