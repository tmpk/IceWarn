import os
import glob
import shutil 
import xarray 


def date_to_string(year, month, day):
    year = str(year)
    month = '0' + str(month) if month < 10 else str(month)
    day = '0' + str(day) if day < 10 else str(day)
    return year, month, day

def make_directory(year, month, day):
    """ Creates directory into which files are moved """
    year, month, day = date_to_string(year, month, day)
    directory = f'./data/Radiosonde/lindenberg/{year}/{month}/{day}/'
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Directory created at: ", directory)
    return directory

def merge_files(filedirectory):
    if len(os.listdir(filedirectory)) == 0:
        # Empty directory. Removing directory:
        os.rmdir(filedirectory)
    else:
        print("Merging files...")
        filepaths = filedirectory + '*.nc'
        ds = xarray.open_mfdataset(filepaths,combine='by_coords',concat_dim="time")
        merged_filepath = filedirectory + 'dataset.nc'
        ds.to_netcdf(merged_filepath)
        print("Files merged.")
    return

for y in [2021]:
    for m in range(1,13):
        for d in range(1, 32):
            directory = make_directory(y, m, d)
            #merge_files(directory)
            year, month, day = date_to_string(y, m, d)
            files = glob.glob(f'data/Radiosonde/lindenberg/{y}/*{y}{month}{day}*')
            for file in files:
                shutil.move(file, directory)
            