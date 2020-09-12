import os
import glob
import requests
import numpy as np
import matplotlib.pyplot as plt

df_Test_normal = glob.glob(os.path.join('chest_xray/train/NORMAL','*.jpeg'))
df_Test_pneumonia = glob.glob(os.path.join('chest_xray/train/PNEUMONIA','*.jpeg'))


