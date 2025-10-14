import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

tree = ET.parse("C:\\Users\\user\\2025_machine_learning\\Week_6\\O-A0038-003.xml")
root = tree.getroot()

namespace = {"cwa": "urn:cwa:gov:tw:cwacommon:0.1"}
content = root.find(".//cwa:Content", namespace).text

# Split the content by commas and convert to a NumPy array of floats
lines = content.splitlines()
values = []
for line in lines:
    line = line.strip()
    if line:
        values.extend([float(x) for x in line.split(',')])
values = np.array(values)

# Reshape the array to 120*67 grid(latitude*longitude)
grid = values.reshape((120, 67))

# Grid information
lon_start, la_start = 120.0, 21.88
lon_res, la_res = 0.03, 0.03
rows, cols = grid.shape

# Construct the latitude and longitude arrays
lons = lon_start + np.arange(cols) * lon_res
lons = np.round(lons, 2)
las = la_start + np.arange(rows) * la_res
las = np.round(las, 2)

# Regression dataset(經度, 緯度, value)
reg_data = []
for i in range(rows):
    for j in range(cols):
        if grid[i, j] != -999:
            reg_data.append((lons[j], las[i], grid[i, j]))

reg_df = pd.DataFrame(reg_data, columns=['Longitude', 'Latitude', 'Value'])

# Classification dataset(經度, 緯度, class)
# Valid != -999 ->1, Invalid == -999 ->0
class_data = []
for i in range(rows):
    for j in range(cols):
        label = 0 if grid[i, j] == -999 else 1
        class_data.append((lons[j], las[i], label))

class_df = pd.DataFrame(class_data, columns=['Longitude', 'Latitude', 'Class'])

reg_df.to_csv('regression_dataset.csv', index=False)
class_df.to_csv('classification_dataset.csv', index=False)