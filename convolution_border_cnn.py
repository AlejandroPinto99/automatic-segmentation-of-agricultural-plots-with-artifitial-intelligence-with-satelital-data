# Se importan los paquetes necesarios
import numpy as np
import geopandas as gpd
import xarray as xr
import rasterio
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from scipy.signal import convolve2d


# Metodo para la normalizacion de las bandas
def normalize_band(band):
    band = np.nan_to_num(band)
    band_min = band.min()
    band_max = band.max()
    band_range = band_max - band_min

    if band_range == 0:
        band_range = 1

    normalized_band = (band - band_min) / band_range
    return normalized_band


# Bloque de Convolución
def double_conv_block(x, n_filters):
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)

    return x


# Bloque de Reducción
def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.3)(p)

    return f, p


# Bloque de Muestra Ascendente
def upsample_block(x, conv_features, n_filters):
    # upsample
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate
    x = layers.concatenate([x, conv_features])
    # dropout
    x = layers.Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)

    return x

# Construccion de la Red Neuronal en base al modelo U-Net
def build_unet_model():
    # inputs
    inputs = layers.Input(shape=(128, 128, 7))

    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 64)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128)

    f3, p3 = downsample_block(p2, 256)

    # 5 - bottleneck
    bottleneck = double_conv_block(p3, 512)

    # decoder: expanding path - upsample

    u1 = upsample_block(bottleneck, f3, 256)
    u2 = upsample_block(u1, f2, 128)
    u3 = upsample_block(u2, f1, 64)

    # outputs
    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(u3)

    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

    return unet_model


# 1) Obtencion de las fuentes de datos

datacube = xr.open_dataset("STAMARIA_S2_2023_cube.nc", chunks=150)
datacube_zeeland = xr.open_dataset("zeeland_s2_mclouds_marea_cube.nc", chunks=150)
mask_zeeland = rasterio.open('europe_mask.tif')
mask_stmaria = rasterio.open('stamaria_mask.tif')
y_true_stamaria = mask_stmaria.read(1)
y_true_eu = mask_zeeland.read(1)
shapefile = gpd.read_file("fields.shp")
shapefile = shapefile.dropna(subset=['geometry'])


dates = datacube['time']
dates = dates[0:70]

dates_eu = datacube_zeeland['time']
dates_eu = dates_eu[0:5]

# Variables de apoyo

dc_shape = datacube["B01"].shape
lat_len = dc_shape[1]
lon_len = dc_shape[2]

dc_shape_eu = datacube_zeeland["B01"].shape
lat_len_eu = dc_shape_eu[1]
lon_len_eu = dc_shape_eu[2]

image_size = 128

datacube_subset_list = []
border_subset_list = []
all_boundaries_list = []

i = 0

# Creacion de nuevas bandas - Indices (Dependiendo de la zona a trabajar)
nir = datacube["B08"]
red = datacube["B04"]
blue = datacube["B02"]
green = datacube["B03"]

ndvi = (nir - red) / (nir + red)

ndwi = (green - nir) / (green + nir)

L = 0.5
savi = ((nir - red) / (nir + red + L)) * (1 + L)

datacube = datacube.assign(NDVI=ndvi)
datacube = datacube.assign(NDWI=ndwi)
datacube = datacube.assign(SAVI=savi)

cloud_mask = datacube['cloud_mask']

band_2 = datacube['B02']
mean_2 = np.nanmean(band_2)
masked_2 = np.where(np.isnan(cloud_mask), band_2, mean_2)
band_2.values = masked_2

band_3 = datacube['B03']
mean_3 = np.nanmean(band_3)
masked_3 = np.where(np.isnan(cloud_mask), band_3, mean_3)
band_3.values = masked_3

band_4 = datacube['B04']
mean_4 = np.nanmean(band_4)
masked_4 = np.where(np.isnan(cloud_mask), band_4, mean_4)
band_4.values = masked_4

band_8 = datacube['B08']
mean_8 = np.nanmean(band_8)
masked_8 = np.where(np.isnan(cloud_mask), band_8, mean_8)
band_8.values = masked_8

datacube['B02'] = band_2
datacube['B03'] = band_3
datacube['B04'] = band_4
datacube['B08'] = band_8

datacube = datacube.assign(ONES=lambda x: 1 * np.isnan((x.B02 * np.nan)))

# Obtención de los bordes de los campos
for date in dates:
    all_boundaries = np.full((datacube.dims['lat'], datacube.dims['lon']), False)
    for field_geo in shapefile.geometry:
        datacube_roi = datacube.sel(time=date).salem.roi(geometry=field_geo, other=0.0)

        o_fields_zero_one = datacube_roi['ONES']

        conv_1_kernel = np.ones((3, 3))
        big_field = convolve2d(o_fields_zero_one, conv_1_kernel, 'same')

        big_field_zero_one = (big_field > 0) * 1.0

        field_boundary = (big_field_zero_one - o_fields_zero_one) == 1

        all_boundaries = np.logical_or(all_boundaries, field_boundary)

    all_boundaries_list.append(all_boundaries)

# Nota: Los bordes generados anteriormente se almacenan en un archivo .npy para optmizar procesos
loaded_all_boundaries = np.load('all_boundaries.npy', allow_pickle=True)

# Creación de las imágenes de entrada con un tamaño de 128x128
var_to_drop = ["cloud_mask", "solar_zenith", "solar_azimuth", "view_zenith",
               "view_azimuth", "B01", "B05", "B06", "B07", "B09", "B8A", "B11", "B12"]
datacube = datacube.drop_vars(var_to_drop)
counting = 0
for date in dates:
    for lat in range(0, lat_len, image_size):
        lat_sup = lat + image_size
        if lat == image_size * (lat_len // image_size):
            lat_sup = lat_len
            continue
        for lon in range(0, lon_len, image_size):
            lon_sup = lon + image_size
            if lon == image_size * (lon_len // image_size):
                lon_sup = lon_len
                counting += 1
                continue
            subset = datacube.sel(time=date).isel(lat=slice(lat, lat_sup), lon=slice(lon, lon_sup))
            datacube_subset_list.append(subset)
            border_subset = loaded_all_boundaries[0, lat:lat_sup, lon:lon_sup]
            border_subset_list.append(border_subset)


# Apilando bandas para crear la entrada de datos para la CNN
input_field_list = []

for datacube_piece in datacube_subset_list:
    datacube_piece["B02"] = datacube_piece["B02"].fillna(0)
    band_b02 = datacube_piece["B02"].values
    band_b02 = normalize_band(band_b02)

    datacube_piece["B03"] = datacube_piece["B03"].fillna(0)
    band_b03 = datacube_piece["B03"].values
    band_b03 = normalize_band(band_b03)

    datacube_piece["B04"] = datacube_piece["B04"].fillna(0)
    band_b04 = datacube_piece["B04"].values
    band_b04 = normalize_band(band_b04)

    datacube_piece["B08"] = datacube_piece["B08"].fillna(0)
    band_b08 = datacube_piece["B08"].values
    band_b08 = normalize_band(band_b08)

    datacube_piece["NDVI"] = datacube_piece["NDVI"].fillna(0)
    band_ndvi = datacube_piece["NDVI"].values
    band_ndvi = normalize_band(band_ndvi)

    datacube_piece["NDWI"] = datacube_piece["NDWI"].fillna(0)
    band_ndwi = datacube_piece["NDWI"].values
    band_ndwi = normalize_band(band_ndwi)

    datacube_piece["SAVI"] = datacube_piece["SAVI"].fillna(0)
    band_savi = datacube_piece["SAVI"].values
    band_savi = normalize_band(band_savi)

    input_field = np.dstack((band_b02, band_b03, band_b04, band_b08, band_ndvi, band_ndwi, band_savi))

    input_field_list.append(input_field)

input_field_list = np.array(input_field_list)

# Creando respectiva lista de máscaras para entrenamiento
input_border_list = []

for border in border_subset_list:
    input_border_list.append(border)

input_border_list = np.array(input_border_list)

# Creando los sets de entrenamiento y prueba
train_images, test_images, train_masks, test_masks = train_test_split(input_field_list, input_border_list, test_size=0.2, random_state=42)

# Construcción y compilación del modelo
model = build_unet_model()
model.compile(optimizer=tf.keras.optimizers.Adam(), loss="binary_crossentropy", metrics="accuracy")

# Entrenamiento del modelo
history = model.fit(x=train_images, y=train_masks, validation_data=(test_images, test_masks), epochs=10, batch_size=32)

# Generación de predicciones
predicted_masks = []
tests_images_lst = []
test_masks_lst = []

for i in range(100):
    image_test = test_images[i]
    mask_test = test_masks[i]
    image_test = np.expand_dims(image_test, axis=0)
    tests_images_lst.append(image_test)
    test_masks_lst.append(mask_test)

    predicted_mask = model.predict(image_test)
    predicted_mask = np.squeeze(predicted_mask)

    predicted_masks.append(predicted_mask)

tests_images_lst = np.array(tests_images_lst)
test_masks_lst = np.array(test_masks_lst)
predicted_masks = np.array(predicted_masks)

np.save('images_zelanda.npy', tests_images_lst)
np.save('predicciones_zelanda.npy', predicted_masks)
np.save('mascaras_zelanda.npy', test_masks_lst)


