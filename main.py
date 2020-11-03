
#%%
import os
import glob
import time
import requests
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.transform import resize

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

df_img_train_normal = organizar_datos(df_test_normal)[0]
df_img_train_virus = organizar_datos(df_test_pneumonia)[1]
df_img_train_bact = organizar_datos(df_test_pneumonia)[2]

start_time = time.time()
def carga_imagenes(df_img):
    img = []
    for i in range(0,len(df_img)):
        img.append(io.imread(df_img[i]))
    return img

img_train_normal = carga_imagenes(df_img_train_normal)
img_train_virus = carga_imagenes(df_img_train_virus)
img_train_bact = carga_imagenes(df_img_train_bact)


#PRE-PROCESAMIENTO

# 1.   Reescalado de las imagenes -----> tamaño de imágenes: 600X600
def reescalado_img(image_data_array, size):
    image_data_array_reescaled = image_data_array
    for i in range(0,len(image_data_array)):
        image_data_array_reescaled[i] = resize(image_data_array[i], (size,size), anti_aliasing=True)
    return image_data_array_reescaled

sizer = 1024
img_train_normal = reescalado_img(img_train_normal,800)
img_train_virus = reescalado_img(img_train_virus,800)
img_train_bact = reescalado_img(img_train_bact,800)

end_time = time.time()
print('Tiempo: ', end_time-start_time,'s')


#%%
input('Press enter to continue...')
fig0, ax0 = plt.subplots(4,2)

ax0[0][0].imshow(img_train_normal[np.random.randint(0,len(img_train_normal))],'gray')
ax0[0][0].axis('off')
ax0[1][0].imshow(img_train_virus[np.random.randint(0,len(img_train_virus))],'gray')
ax0[1][0].axis('off')
ax0[2][0].imshow(img_train_virus[np.random.randint(0,len(img_train_virus))],'gray')
ax0[2][0].axis('off')
ax0[3][0].imshow(img_train_bact[np.random.randint(0,len(img_train_bact))],'gray')
ax0[3][0].axis('off')

ax0[0][1].text(0.1,0.5,'Normal',bbox={'facecolor': 'green'})
ax0[0][1].axis('off')
ax0[1][1].text(0.1,0.5,'Neumonia viral',bbox={'facecolor': 'red'})
ax0[1][1].axis('off')
ax0[2][1].text(0.1,0.5,'Neumonia viral',bbox={'facecolor': 'red'})
ax0[2][1].axis('off')
ax0[3][1].text(0.1,0.5,'Neumonia bacteriana',bbox={'facecolor': 'red'})
ax0[3][1].axis('off')

plt.show()
# %%
