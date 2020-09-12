import os
import glob
import requests
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

df_test_normal = glob.glob(os.path.join('chest_xray/train/NORMAL','*.jpeg'))
df_test_pneumonia = glob.glob(os.path.join('chest_xray/train/PNEUMONIA','*.jpeg'))

print('Cantidad de datos en test/NORMAL',len(df_test_normal))
print('Cantidad de datos en test/PNEUMONIA',len(df_test_normal))

figexp, ax = plt.subplots(4,2)
figexp.suptitle('Exploración base de datos.')
ax[0,0].set_title('Imágenes')
ax[0][0].imshow(io.imread(df_test_normal[np.random.randint(0,len(df_test_normal))]))
ax[0][0].axis('off')
ax[1][0].imshow(io.imread(df_test_pneumonia[np.random.randint(0,len(df_test_pneumonia))]))
ax[1][0].axis('off')
ax[2][0].imshow(io.imread(df_test_normal[np.random.randint(0,len(df_test_normal))]))
ax[2][0].axis('off')
ax[3][0].imshow(io.imread(df_test_pneumonia[np.random.randint(0,len(df_test_pneumonia))]))
ax[3][0].axis('off')
ax[0,1].set_title('Anotaciones')
ax[0][1].text(0.4,0.5,'SANO',bbox=dict(facecolor='green', alpha=0.5))
ax[0][1].axis('off')
ax[1][1].text(0.3,0.5,'PNEUMONIA',bbox=dict(facecolor='red', alpha=0.5))
ax[1][1].axis('off')
ax[2][1].text(0.4,0.5,'SANO', bbox=dict(facecolor='green', alpha=0.5))
ax[2][1].axis('off')
ax[3][1].text(0.3,0.5,'PNEUMONIA',bbox=dict(facecolor='red', alpha=0.5))
ax[3][1].axis('off')

plt.show()