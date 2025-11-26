import pandas as pd
import io
import requests

def download_paper_dataset():
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query.csv"

    # Set parameters as described in the paper
    params = {
        'starttime': '1900-01-01',
        'endtime': '2021-10-31',
        'minmagnitude': 5,          # Paper uses mag > 5
        'minlatitude': 30.259,      # Paper: 30.259 North
        'maxlatitude': 45.614,      # Paper: 45.614 North
        'minlongitude': 129.111,    # Paper: 129.111 East
        'maxlongitude': 146.074,    # Paper: 146.074 East
        'orderby': 'time'           # Ensure chronological order
    }

    print("Downloading data from USGS (this might take a few seconds)...")
    response = requests.get(url, params=params)

    if response.status_code == 200:
        df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))

        output_file = 'data/japan_earthquakes_paper.csv'
        df.to_csv(output_file, index=False)
        print(f"Success! Downloaded {len(df)} earthquakes.")
        print(f"Saved to: {output_file}")
        return output_file
    else:
        print("Error downloading data:", response.status_code)
        return None


if __name__ == "__main__":
    csv_path = download_paper_dataset()