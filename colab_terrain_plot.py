# Colab script to read NASA topography and plot 2D terrain
# 1) Install packages
# 2) Mount Google Drive
# 3) Load the .tif file and plot it

# Cell 1 - Install packages and mount drive
# ----------------------------------------
# Run the following commands in a Colab cell:
# !pip install rasterio matplotlib --quiet
# from google.colab import drive
# drive.mount('/content/drive')

# Cell 2 - Load and plot the terrain
# ----------------------------------
# After mounting, run this code in another Colab cell:

import rasterio
import matplotlib.pyplot as plt

# Path to the TIFF file on your Google Drive
# Update the path if your file is stored elsewhere
TIFF_PATH = '/content/drive/MyDrive/Green-Swamp/NASAtopo/NASAtopo.tif'

with rasterio.open(TIFF_PATH) as src:
    terrain = src.read(1)
    extent = (
        src.bounds.left,
        src.bounds.right,
        src.bounds.bottom,
        src.bounds.top,
    )

plt.figure(figsize=(8, 6))
plt.imshow(terrain, extent=extent, cmap='terrain')
plt.colorbar(label='Elevation (m)')
plt.title('Green Swamp Terrain')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
