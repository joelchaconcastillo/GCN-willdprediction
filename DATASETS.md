# Wildfire Datasets - Comprehensive Guide

This document provides detailed information about state-of-the-art wildfire datasets that can be used with this framework.

## Global Datasets

### 1. FIRMS (Fire Information for Resource Management System)

**Provider**: NASA

**Coverage**: Global

**Temporal Range**: 2000-present (MODIS), 2012-present (VIIRS)

**Spatial Resolution**: 
- MODIS: 1km
- VIIRS: 375m

**Data Variables**:
- Fire detection confidence
- Brightness temperature
- Fire radiative power (FRP)
- Detection date/time
- Latitude/longitude

**Access**:
- Website: https://firms.modaps.eosdis.nasa.gov/
- API: Available for automated downloads
- Formats: CSV, Shapefile, KML, WMS

**How to Use**:
```python
import pandas as pd
from wildfire_gnn.data import WildfireDataset

# Download FIRMS data as CSV
df = pd.read_csv('firms_data.csv')

# Process coordinates
coords = df[['latitude', 'longitude']].values

# Extract features (brightness, FRP, confidence, etc.)
features = df[['bright_ti4', 'frp', 'confidence']].values

# Create temporal sequences
# ... your processing code ...

dataset = WildfireDataset(
    coordinates=coords,
    features=features,
    targets=targets,
    time_steps=7,
    prediction_horizon=1
)
```

---

### 2. GWIS (Global Wildfire Information System)

**Provider**: Copernicus Emergency Management Service

**Coverage**: Global

**Temporal Range**: 2003-present

**Data Types**:
- Fire danger forecast (1-10 day)
- Burnt area analysis
- Fire emissions
- Fire weather indices

**Data Variables**:
- Fire Weather Index (FWI)
- Burnt area (hectares)
- CO2 and PM2.5 emissions
- Fire danger classes

**Access**:
- Website: https://gwis.jrc.ec.europa.eu/
- API: REST API available
- Formats: NetCDF, GeoTIFF, CSV

**Features for ML**:
- Temperature
- Precipitation
- Wind speed
- Relative humidity
- Fine Fuel Moisture Code (FFMC)
- Duff Moisture Code (DMC)
- Drought Code (DC)
- Initial Spread Index (ISI)
- Buildup Index (BUI)
- Fire Weather Index (FWI)

---

### 3. EFFIS (European Forest Fire Information System)

**Provider**: European Commission Joint Research Centre

**Coverage**: European countries, Mediterranean basin

**Temporal Range**: 2000-present

**Spatial Resolution**: Variable (from 250m to 1km)

**Data Types**:
- Current situation (active fires)
- Burnt areas
- Fire danger forecast
- Historical fire database

**Data Variables**:
- Burnt area (hectares)
- Fire start/end dates
- Fire causes
- Weather conditions
- Vegetation type

**Access**:
- Website: https://effis.jrc.ec.europa.eu/
- Data portal: https://effis.jrc.ec.europa.eu/applications/data-and-services
- Formats: Shapefile, KML, CSV

---

## Regional Datasets (United States)

### 4. MTBS (Monitoring Trends in Burn Severity)

**Provider**: USGS & USDA Forest Service

**Coverage**: United States

**Temporal Range**: 1984-present

**Spatial Resolution**: 30m (Landsat)

**Data Products**:
- Burn severity maps (dNBR, RdNBR)
- Fire perimeters
- Pre- and post-fire imagery
- Composite burn index

**Data Variables**:
- Differenced Normalized Burn Ratio (dNBR)
- Relativized dNBR (RdNBR)
- Burn severity classes (Unburned to High)
- Acres burned
- Fire name, date, cause

**Access**:
- Website: https://www.mtbs.gov/
- Direct Download: https://www.mtbs.gov/direct-download
- Formats: GeoTIFF, Shapefile

**Use Case**: Training models for burn severity prediction

---

### 5. FPA-FOD (Fire Program Analysis - Fire Occurrence Database)

**Provider**: USFS Forest Service

**Coverage**: United States

**Temporal Range**: 1992-2018

**Records**: 2.3+ million wildfires

**Data Variables**:
- Fire discovery date
- Fire size (acres)
- Fire cause (13 categories)
- State, county
- Latitude/longitude
- Containment date
- Owner (federal, state, private, etc.)

**Access**:
- Website: https://www.fs.usda.gov/rds/archive/Catalog/RDS-2013-0009.6
- Format: SQLite database, Shapefile

**Use Case**: Historical fire occurrence modeling, risk assessment

```python
import sqlite3
import pandas as pd

# Load FPA-FOD database
conn = sqlite3.connect('FPA_FOD_20170508.sqlite')
df = pd.read_sql_query("""
    SELECT 
        FIRE_YEAR, DISCOVERY_DATE, STAT_CAUSE_DESCR,
        FIRE_SIZE, LATITUDE, LONGITUDE
    FROM Fires
    WHERE FIRE_SIZE > 0
""", conn)
```

---

### 6. FIRED (Fire Event Delineation)

**Provider**: NASA

**Coverage**: Global (focus on CONUS, Alaska)

**Temporal Range**: 2001-2020

**Spatial Resolution**: 500m

**Data Products**:
- Individual fire events
- Daily fire progression
- Final fire perimeters
- Fire duration

**Unique Features**:
- Tracks individual fire growth day-by-day
- Links fire pixels into events
- Provides fire ignition locations

**Access**:
- Website: https://www.earthdata.nasa.gov/
- Format: GeoTIFF, CSV

---

### 7. CalFire (California Fire Perimeters)

**Provider**: California Department of Forestry and Fire Protection

**Coverage**: California

**Temporal Range**: 1878-present (comprehensive from 2000+)

**Data Variables**:
- Fire perimeters
- Fire name
- Start/end dates
- Acres burned
- Cause
- Agency response

**Access**:
- Website: https://www.fire.ca.gov/
- Data: https://gis.data.ca.gov/
- Format: Shapefile, GeoJSON

---

## Auxiliary Environmental Data

### 8. Weather Data

**ERA5 (ECMWF Reanalysis)**
- Provider: European Centre for Medium-Range Weather Forecasts
- Coverage: Global, 1950-present
- Resolution: 0.25° (~31km)
- Variables: Temperature, precipitation, wind, humidity, pressure
- Access: https://cds.climate.copernicus.eu/

**NOAA Weather Data**
- Provider: National Oceanic and Atmospheric Administration
- Various datasets (GHCN, ISD, etc.)
- Access: https://www.ncdc.noaa.gov/

**Gridded Meteorological Data**
```python
# Example: Loading ERA5 data with xarray
import xarray as xr

ds = xr.open_dataset('era5_temperature.nc')
temperature = ds['t2m']  # 2-meter temperature
```

---

### 9. Vegetation Indices

**MODIS Vegetation Indices (MOD13)**
- Provider: NASA
- Resolution: 250m, 500m, 1km
- Temporal: 16-day composite
- Variables: NDVI, EVI
- Access: https://lpdaac.usgs.gov/

**Landsat NDVI**
- Provider: USGS
- Resolution: 30m
- Temporal: 16-day revisit
- Access: https://earthexplorer.usgs.gov/

**Fuel Moisture**
- Live Fuel Moisture: LFMC (from MODIS, Landsat)
- Dead Fuel Moisture: From weather models

---

### 10. Topographic Data

**SRTM (Shuttle Radar Topography Mission)**
- Resolution: 30m (US), 90m (global)
- Variables: Elevation
- Derived: Slope, aspect, hillshade
- Access: https://www.usgs.gov/centers/eros/

**Derived Topographic Variables**:
```python
import rasterio
from rasterio.plot import show
import numpy as np

# Load DEM
with rasterio.open('dem.tif') as src:
    elevation = src.read(1)

# Calculate slope
dx, dy = np.gradient(elevation)
slope = np.arctan(np.sqrt(dx**2 + dy**2))

# Calculate aspect
aspect = np.arctan2(-dx, dy)
```

---

### 11. Land Cover

**ESA CCI Land Cover**
- Provider: European Space Agency
- Coverage: Global
- Resolution: 300m
- Temporal: Annual (1992-present)
- Classes: 22 land cover types
- Access: https://www.esa-landcover-cci.org/

**NLCD (National Land Cover Database) - USA**
- Provider: USGS
- Coverage: United States
- Resolution: 30m
- Temporal: Every 3-5 years
- Access: https://www.mrlc.gov/

---

## Data Integration Strategy

### Combining Multiple Datasets

```python
import numpy as np
import pandas as pd
from wildfire_gnn.data import WildfireDataset

# 1. Fire occurrence from FIRMS
firms_df = pd.read_csv('firms_active_fires.csv')

# 2. Weather data from ERA5
weather_df = pd.read_csv('era5_weather.csv')

# 3. Vegetation from MODIS
ndvi_df = pd.read_csv('modis_ndvi.csv')

# 4. Topography from SRTM
topo_df = pd.read_csv('srtm_elevation_slope.csv')

# Merge on location and date
merged_df = firms_df.merge(weather_df, on=['lat', 'lon', 'date']) \
                    .merge(ndvi_df, on=['lat', 'lon', 'date']) \
                    .merge(topo_df, on=['lat', 'lon'])

# Extract features
feature_columns = [
    'temperature', 'humidity', 'wind_speed',
    'ndvi', 'elevation', 'slope', 'aspect'
]
features = merged_df[feature_columns].values

# Create spatial-temporal sequences
# ... processing code ...

# Create dataset
dataset = WildfireDataset(
    coordinates=coords,
    features=features,
    targets=targets,
    graph_type='knn',
    k=8
)
```

---

## Recommended Dataset Combinations

### For Fire Occurrence Prediction:
1. **Fire History**: FPA-FOD or FIRMS historical
2. **Weather**: ERA5 (temperature, humidity, wind)
3. **Vegetation**: MODIS NDVI
4. **Topography**: SRTM elevation, slope
5. **Season**: Day of year, month

### For Fire Spread Prediction:
1. **Active Fires**: FIRMS real-time
2. **Weather**: High-resolution forecasts
3. **Fuel**: Vegetation type, fuel moisture
4. **Topography**: Slope, aspect
5. **Wind**: Detailed wind fields

### For Burn Severity Prediction:
1. **Pre-fire Conditions**: NDVI, fuel load
2. **Fire Characteristics**: Duration, intensity (FRP)
3. **Weather During Fire**: Temperature, wind
4. **Topography**: Slope, aspect
5. **Validation**: MTBS burn severity maps

---

## Data Processing Tips

### 1. Temporal Alignment
```python
# Resample to daily frequency
df_daily = df.groupby(['lat', 'lon', pd.Grouper(key='date', freq='D')]).mean()
```

### 2. Spatial Aggregation
```python
# Grid data to regular spatial resolution
from scipy.interpolate import griddata

points = df[['lon', 'lat']].values
values = df['temperature'].values
grid_lon, grid_lat = np.meshgrid(
    np.linspace(lon_min, lon_max, 100),
    np.linspace(lat_min, lat_max, 100)
)
grid_temp = griddata(points, values, (grid_lon, grid_lat), method='linear')
```

### 3. Missing Data Handling
```python
# Forward fill for short gaps
df['temperature'].fillna(method='ffill', limit=3, inplace=True)

# Interpolate for longer gaps
df['temperature'].interpolate(method='linear', inplace=True)
```

### 4. Normalization
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)
```

---

## Citation Requirements

When using these datasets, please cite appropriately:

**FIRMS**: 
- NASA FIRMS. (2023). Fire Information for Resource Management System. https://firms.modaps.eosdis.nasa.gov/

**MTBS**:
- Eidenshink, J., et al. (2007). A project for monitoring trends in burn severity. Fire Ecology, 3(1), 3-21.

**FPA-FOD**:
- Short, Karen C. (2021). Spatial wildfire occurrence data for the United States, 1992-2018 [FPA_FOD_20210617]. Forest Service Research Data Archive.

**ERA5**:
- Hersbach, H., et al. (2020). The ERA5 global reanalysis. Quarterly Journal of the Royal Meteorological Society, 146(730), 1999-2049.

---

## Additional Resources

- **NASA Earthdata**: https://earthdata.nasa.gov/
- **Google Earth Engine**: Pre-processed datasets available
- **Kaggle Wildfire Datasets**: Community-contributed datasets
- **LANDFIRE**: US-specific fuel and vegetation data

For questions about specific datasets, consult their respective documentation or contact the data providers.
