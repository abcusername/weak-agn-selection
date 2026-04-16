import numpy as np
from ztfquery import lightcurve
import csv

# Load RA and DEC values
ra, dec = np.genfromtxt("../Sample.txt", unpack=True, usecols=(0, 1))

# Prompt for IRSA credentials
username = input('Enter Username for IRSA:\t')
password = input('Enter Password for IRSA:\t')

# Counter for skipped files
count = 0

for i in range(len(ra)):
    print(f"Processing RA: {ra[i]}, DEC: {dec[i]}")

    try:
        # Download light curve data
        data = lightcurve.LCQuery.download_data(
            circle=[ra[i], dec[i], 0.0028],
            bandname="r",
            auth=[username, password]
        )

        # Check if the returned data is an XML error response
        if isinstance(data, str) and data.startswith('<?xml'):
            print(f"Error page received for RA: {ra[i]}, DEC: {dec[i]} - {data[:200]}...")  # Print a snippet of the error
            count += 1
            continue

        # Debugging: Print columns of downloaded data
        print(f"Columns in data for RA: {ra[i]}, DEC: {dec[i]}: {data.columns}")

        # Check if required columns exist (case-sensitive)
        required_columns = ['mjd', 'mag', 'magerr', 'catflags', 'ra', 'dec', 'oid']
        available_columns = set(data.columns)
        
        print(f"Available columns: {available_columns}")

        # Verify columns are available
        if not all(col in available_columns for col in required_columns):
            print(f"Missing required columns for RA: {ra[i]}, DEC: {dec[i]}")
            count += 1
            continue

        # Extract data using correct columns
        date_init = data['mjd']
        mag_init = data['mag']
        magerr_init = data['magerr']
        catflag = data['catflags']
        oid = data['oid']

        # Filter for good data points (catflag == 0)
        good_id = np.where(catflag == 0)[0]
        if len(good_id) == 0:
            print(f"No good data points for RA: {ra[i]}, DEC: {dec[i]}")
            count += 1
            continue

        # Apply the filter
        date_init = date_init.iloc[good_id]
        mag_init = mag_init.iloc[good_id]
        magerr_init = magerr_init.iloc[good_id]
        oid = oid.iloc[good_id]

        # Handle insufficient data
        if len(date_init) <= 10:
            print(f"Insufficient data points for RA: {ra[i]}, DEC: {dec[i]}")
            count += 1
            continue

        # Save to CSV
        filename = f"J{ra[i]:.5f}_{dec[i]:.5f}_r.csv"
        with open(filename, 'w+', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["oid", "catflag", "mjd", "mag", "magerr"])
            csvwriter.writerows(zip(oid, catflag, date_init, mag_init, magerr_init))
        print(f"Light curve for RA: {ra[i]}, DEC: {dec[i]} saved as {filename}.")

    except Exception as e:
        print(f"Error downloading data for RA: {ra[i]}, DEC: {dec[i]} - {e}")
        count += 1

# Summary of skipped files
print(f"Total skipped files: {count}")
