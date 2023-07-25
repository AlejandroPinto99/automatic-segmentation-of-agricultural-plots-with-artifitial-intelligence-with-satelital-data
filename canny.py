import cv2
from skimage import io
import netCDF4
import matplotlib.pyplot as plt

# Carga el archivo NetCDF
dataset = netCDF4.Dataset('./STAMARIA_S2_2023_cube.nc');

# Tomando una banda
copia = dataset.variables['B08'][0, 0:, 0:];
B0_4 = dataset.variables['B08'][0, 0:, 0:];

# Normaliza los valores del arreglo a un rango entre 0 y 255
datos_norm = cv2.normalize(B0_4, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U);

# Aplicar suavizado Gaussiano
gauss = cv2.GaussianBlur(datos_norm, (5,5), 0);

# Detectamos los bordes con Canny
canny = cv2.Canny(gauss, 50, 150)

# Buscamos los contornos
(contornos,_) = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Se dibujan los contornos sobre la imagen
cv2.drawContours(B0_4,contornos,-1,(255,0,0), 2)

# Mostrar los datos
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4));

ax1.imshow(canny)
ax1.set_title('Canny result');
ax2.imshow(copia)
ax2.set_title('With Contours');

cv2.waitKey(0);
