# encoding: utf-8
"""
Script for downloading (ceilometer) files from the CEDA archive using ftp. 
Code has been modified from https://github.com/cedadev/opendap-python-example

Older data (up to and including parts of 2021) is stored in the database differently from newer. Use 
download_files() and merge_files() for this data. More recent data is stored 
such that all measurements for a given date is in a single file.

See the readmes at 
    https://dap.ceda.ac.uk/thredds/fileServer//badc/eprofile/data/
and (e.g)
    https://dap.ceda.ac.uk/thredds/fileServer//badc/eprofile/data/daily_files/norway/andoya/met-norway-lufft-chm15k_A/
for more information
"""

# Import standard libraries
import os
import sys
import datetime
import requests
import time
import ftplib

# Import third-party libraries
from cryptography import x509
from multiprocessing.pool import ThreadPool
from cryptography.hazmat.backends import default_backend
from contrail.security.onlineca.client import OnlineCaClient
from xml.dom import minidom
from urllib.request import urlopen
from urllib.request import urlretrieve
from bs4 import BeautifulSoup 
import xarray 
from netCDF4 import Dataset

requests.packages.urllib3.disable_warnings() 

CERTS_DIR = os.path.expanduser('~/.certs')
if not os.path.isdir(CERTS_DIR):
    os.makedirs(CERTS_DIR)

TRUSTROOTS_DIR = os.path.join(CERTS_DIR, 'ca-trustroots')
CREDENTIALS_FILE_PATH = os.path.join(CERTS_DIR, 'credentials.pem')

TRUSTROOTS_SERVICE = 'https://slcs.ceda.ac.uk/onlineca/trustroots/'
CERT_SERVICE = 'https://slcs.ceda.ac.uk/onlineca/certificate/'

# Directory into which files are downloaded
def make_directory(year, month, day):
    year, month, day = date_to_string(year, month, day)
    #directory = f'./data/Ceilometer/lindenberg/{year}/{month}/{day}/'
    directory = f'./data/Ceilometer/andoya/{year}/{month}/{day}/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def cert_is_valid(cert_file, min_lifetime=0):
    """
    Returns boolean - True if the certificate is in date.
    Optional argument min_lifetime is the number of seconds
    which must remain.

    :param cert_file: certificate file path.
    :param min_lifetime: minimum lifetime (seconds)
    :return: boolean
    """
    try:
        with open(cert_file, 'rb') as f:
            crt_data = f.read()
    except IOError:
        return False

    try:
        cert = x509.load_pem_x509_certificate(crt_data, default_backend())
    except ValueError:
        return False

    now = datetime.datetime.now()

    return (cert.not_valid_before <= now
            and cert.not_valid_after > now + datetime.timedelta(0, min_lifetime))


def setup_credentials():
    """
    Download and create required credentials files.

    Return True if credentials were set up.
    Return False is credentials were already set up.

    :param force: boolean
    :return: boolean
    """

    # Test for DODS_FILE and only re-get credentials if it doesn't
    # exist AND `force` is True AND certificate is in-date.
    # if cert_is_valid(CREDENTIALS_FILE_PATH):
    #     #print('[INFO] Security credentials already set up.')
    #     return False

    # Get CEDA username and password from environment variables
    username = os.environ['CEDA_USERNAME']
    password = os.environ['CEDA_PASSWORD']

    onlineca_client = OnlineCaClient()
    onlineca_client.ca_cert_dir = TRUSTROOTS_DIR

    # Set up trust roots
    trustroots = onlineca_client.get_trustroots(
        TRUSTROOTS_SERVICE,
        bootstrap=True,
        write_to_ca_cert_dir=True)

    # Write certificate credentials file
    key_pair, certs = onlineca_client.get_certificate(
        username,
        password,
        CERT_SERVICE,
        pem_out_filepath=CREDENTIALS_FILE_PATH)

    print('[INFO] Security credentials set up.')
    return True

def get_file_links(archive_url): 
      
    # create response object 
    r = requests.get(archive_url) 
      
    # create beautiful-soup object 
    soup = BeautifulSoup(r.content,'html5lib') 
      
    # find all links on web-page 
    links = soup.findAll('a') 
  
    # filter the link sending with .nc
    file_links = [archive_url + link['href'] for link in links if link['href'].endswith('nc')] 
  
    return file_links 

def date_to_string(year, month, day):
    year = str(year)
    month = '0' + str(month) if month < 10 else str(month)
    day = '0' + str(day) if day < 10 else str(day)
    return year, month, day

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

def download_files(year, month, day, directory):
    
    username = os.environ['CEDA_USERNAME']
    password = os.environ['CEDA_PASSWORD']

    year, month, day = date_to_string(year, month, day)
    domain = "ftp.ceda.ac.uk"
    #archive = f"/badc/eprofile/data/germany/lindenberg/dwd-jenoptick-chm15k-nimbus_0/{year}/{month}/{day}/"
    archive = f"/badc/eprofile/data/norway/andoya/met-norway-jenoptick-chm15k-nimbus_A/{year}/{month}/{day}/"
    ftp = ftplib.FTP(domain, username, password)

    try:
        ftp.cwd(archive)
    except ftplib.all_errors as e:
        errorcode_string = str(e).split(None, 1)[0]
        if errorcode_string == 550:
            print("No data for specified date. Continuing with next date...")
        else: print("Caught exception: ", e)
        return

    filenames = ftp.nlst()
    ftp.quit()

    def download(file_name):
        ftp = ftplib.FTP(domain, username, password)
        ftp.cwd(archive)
        #try:
        #    ftp.cwd(archive)
        #except ftplib.all_errors as e:
        #    errorcode_string = str(e).split(None, 1)[0]
        #    if errorcode_string == 550:
        #        print("No data for specified date. Continuing with next date...")
        #    else: print("Caught exception: ", e)
        #    return

        file_path = directory + file_name
        ftp.retrbinary("RETR %s" % file_name, open(file_path, "wb").write)
        ftp.quit()
        
    print("Downloading " + str(len(filenames)) + " files...")
    start = time.time()
    fns = ThreadPool(10).imap_unordered(download, filenames)
    i=1
    for fn in fns:
        print(i,"/",len(filenames))
        i+=1
    end = time.time()
    print("Download finished in " + str(end-start) + "seconds")

def download_daily_files(year, month, day, directory):
    # function for downlading file from given "year", "month", "day" from 
    # the CEDA folder /badc/eprofile/data/daily_files/ into "directory"
    username = os.environ['CEDA_USERNAME']
    password = os.environ['CEDA_PASSWORD']

    year, month, day = date_to_string(year, month, day)
    domain = "ftp.ceda.ac.uk"
    archive = f"/badc/eprofile/data/daily_files/norway/andoya/met-norway-lufft-chm15k_A/{year}"
    ftp = ftplib.FTP(domain, username, password)
    
    try:
        ftp.cwd(archive)
    except ftplib.all_errors as e:
        errorcode_string = str(e).split(None, 1)[0]
        if errorcode_string == 550:
            print("No data in archive for specified year.")
        else: print("Caught exception: ", e)
        return

    file_name = f'L2_0-20000-0-01010_A{year}{month}{day}.nc'
    file_path = directory + 'dataset.nc'
    print("Downloading file: ",file_name)
    start = time.time()
    ftp.retrbinary("RETR %s" % file_name, open(file_path, "wb").write)
    end = time.time()
    print("Download finished in " + str(end-start) + "seconds")
    ftp.quit()

def main(year, month, day):
    directory = make_directory(year, month, day)
    download_daily_files(year, month, day, directory)
    #download_files(year, month, day, directory)
    #merge_files(directory)

main(2021, 1, 25)