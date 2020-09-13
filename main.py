import os
import glob
import requests
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

df_test_normal = glob.glob(os.path.join('chest_xray/train/NORMAL','*.jpeg'))
df_test_pneumonia = glob.glob(os.path.join('chest_xray/train/PNEUMONIA','*.jpeg'))


def organizar_datos(df):
    list_norm = []
    list_bact = []
    list_virus = []
    for i in range(0,len(df)):
        if df[i].find('virus')!=-1:
            list_virus.append(df[i])
        elif df[i].find('bacteria')!=-1:
            list_bact.append(df[i])
        else:
            list_norm.append(df[i])
    return list_norm, list_virus, list_bact

img_test_normal = organizar_datos(df_test_normal)[0]
img_test_virus = organizar_datos(df_test_pneumonia)[1]
img_test_bact = organizar_datos(df_test_pneumonia)[2]
print(len(img_test_bact),len(img_test_normal),len(img_test_virus))

figexp, ax = plt.subplots(4,2)
figexp.suptitle('Exploración base de datos.')
ax[0,0].set_title('Imágenes')
ax[0][0].imshow(io.imread(img_test_normal[np.random.randint(0,len(img_test_normal))]),cmap='gray')
ax[0][0].axis('off')
ax[1][0].imshow(io.imread(img_test_bact[np.random.randint(0,len(img_test_bact))]),cmap='gray')
ax[1][0].axis('off')
ax[2][0].imshow(io.imread(img_test_normal[np.random.randint(0,len(img_test_normal))]),cmap='gray')
ax[2][0].axis('off')
ax[3][0].imshow(io.imread(img_test_virus[np.random.randint(0,len(img_test_virus))]),cmap='gray')
ax[3][0].axis('off')
ax[0,1].set_title('Anotaciones')
ax[0][1].text(0.4,0.5,'SANO',bbox=dict(facecolor='green', alpha=0.5))
ax[0][1].axis('off')
ax[1][1].text(0.3,0.5,'NEUMONIA BACTERIANA',bbox=dict(facecolor='red', alpha=0.5))
ax[1][1].axis('off')
ax[2][1].text(0.4,0.5,'SANO', bbox=dict(facecolor='green', alpha=0.5))
ax[2][1].axis('off')
ax[3][1].text(0.3,0.5,'NEUMONIA VIRAL',bbox=dict(facecolor='red', alpha=0.5))
ax[3][1].axis('off')

plt.show()

print((io.imread(df_test_normal[np.random.randint(0,len(df_test_normal))])).dtype)