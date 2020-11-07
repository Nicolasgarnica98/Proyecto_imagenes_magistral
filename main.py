
#%%
import os
import glob
import time
import requests
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.transform import resize
from skimage import feature
from skimage import morphology
from skimage import filters
from skimage.filters import threshold_otsu

#%%

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


sizer = 700
img_train_normal = reescalado_img(img_train_normal,sizer)
img_train_virus = reescalado_img(img_train_virus,sizer)
img_train_bact = reescalado_img(img_train_bact,sizer)


# 2.   Ecualización de histogramas de las imágenes
def ecualizacion_hist(image_data_array):
    eq_image_array = image_data_array
    for i in range(0,len(eq_image_array)):
        eq_image_array[i] = exposure.equalize_hist(image_data_array[i])
    return eq_image_array

img_train_normal = ecualizacion_hist(img_train_normal)
img_train_virus = ecualizacion_hist(img_train_virus)
img_train_bact = ecualizacion_hist(img_train_bact)

end_time = time.time()
print('Tiempo: ', end_time-start_time,'s')


#%% PLOT DE EXPLORACIÓN INICIAL
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
# %% PROCESAMIENTO_TEST

# 1.   Aplicación del detector de bordes de Canny.
img_prueba_sano = img_train_normal[np.random.randint(0,len(img_train_normal))]
img_prueba_pneu = img_train_bact[np.random.randint(0,len(img_train_bact))]
img_prueba_pneu2 = img_train_virus[np.random.randint(0,len(img_train_virus))]

ee = morphology.disk(1)
canny_sano_prueba = filters.rank.gradient(img_prueba_sano,ee)
canny_pneu_prueba = filters.rank.gradient(img_prueba_pneu,ee)
canny_pneu2_prueba = filters.rank.gradient(img_prueba_pneu2,ee)
print(canny_sano_prueba)

# 2.  Aplicación de Top-Hat para rescatar detalles claros pequeños

ee2 = np.ones((13,13))
th_sano_prueba = morphology.white_tophat(canny_sano_prueba,ee2)
th_pneu_prueba = morphology.white_tophat(canny_pneu_prueba,ee2)
th_pneu2_prueba = morphology.white_tophat(canny_pneu2_prueba,ee2)

otsu_sano_prueba = threshold_otsu(th_sano_prueba)
otsu_pneu_prueba = threshold_otsu(th_pneu_prueba)
otsu_pneu2_prueba = threshold_otsu(th_pneu2_prueba)

img1 = th_sano_prueba > otsu_sano_prueba
img2 = th_pneu_prueba > otsu_pneu_prueba
img3 = th_pneu2_prueba > otsu_pneu2_prueba

fig1, ax1 = plt.subplots(3,1)
ax1[0].imshow(img1,'gray')
ax1[1].imshow(img2,'gray')
ax1[2].imshow(img3,'gray')
plt.show()



# %%
