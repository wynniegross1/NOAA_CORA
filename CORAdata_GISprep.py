# %%
import numpy as np
import s3fs
import xarray as xr
from scipy.interpolate import griddata
from tqdm import tqdm

# Parameters
S3_PATH = "noaa-nos-cora-pds/V1.1/assimilated/native_grid/fort.63_2017.nc"
LON_TAMPA, LAT_TAMPA = -82.45, 27.95
START, END = "2017-09-10T00:00:00", "2017-09-10T23:00:00"

# Connect to S3
fs = s3fs.S3FileSystem(anon=True)
with fs.open(S3_PATH, 'rb') as file:
    with xr.open_dataset(file, engine="h5netcdf", chunks={}) as ds_raw:
        # ⏳ Time slice first to avoid loading everything
        ds = ds_raw.sel(time=slice(START, END))

        # Mesh coordinates
        x = ds['x'].values
        y = ds['y'].values

        # Regular grid around Tampa (25x25)
        lon_grid = np.linspace(-82.7, -82.2, 25) #grid size 25x25, can make bigger
        lat_grid = np.linspace(27.7, 28.2, 25)
        lon2d, lat2d = np.meshgrid(lon_grid, lat_grid)

        # Fast interpolation loop with progress bar
        zeta_all = ds['zeta']
        grid_stack = []

        for t in tqdm(zeta_all.time, desc="Interpolating hourly slices"):
            z_slice = zeta_all.sel(time=t).compute()
            interpolated = griddata((x, y), z_slice, (lon2d, lat2d), method="nearest")
            grid_stack.append(interpolated)

        # Create DataArray cube
        cube = xr.DataArray(
            np.array(grid_stack),
            coords={"time": zeta_all.time.values, "lat": lat_grid, "lon": lon_grid},
            dims=["time", "lat", "lon"],
            name="water_level"
        )

# Wrap and export to NetCDF
ds_out = xr.Dataset({"water_level": cube})
ds_out.attrs["title"] = "Tampa Water Levels - Hurricane Irma"
ds_out.attrs["description"] = "Interpolated hourly water levels on regular grid for ArcGIS voxel visualization"
ds_out.attrs["source"] = "NOAA CORA V1.1 via S3"

ds_out.to_netcdf("C:/Users/wyn14279/Documents/irma_tampa_voxel.nc")
print("Done! Saved: irma_tampa_voxel.nc (25×25 grid, ArcGIS-ready)")

# %%



