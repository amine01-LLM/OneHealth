import pandas as pd
import requests
import time

def fetch_nasa_monthly(lat, lon, start_year, end_year):
    """
    Fetches monthly climate data from NASA POWER API.
    """
    url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    params = {
        "start": start_year,
        "end": end_year,
        "latitude": lat,
        "longitude": lon,
        "community": "RE",
        "parameters": "PRECTOTCORR,T2M,RH2M",
        "format": "JSON"
    }
    
    try:
        response = requests.get(url, params=params, timeout=60)
        if response.status_code == 200:
            return response.json()['properties']['parameter']
        else:
            print(f"API Error {response.status_code} for {lat}, {lon}")
    except Exception as e:
        print(f"Connection error: {e}")
    return None

def main():
    df_health = pd.read_csv('DRC_Health_Final.csv')
    
    # 2. Identify unique locations to minimize API calls
    unique_locs = df_health[['PROV', 'LAT', 'LON']].drop_duplicates().dropna()
    
    weather_results = []
    
    print(f"Starting weather fetch for {len(unique_locs)} provinces...")
    
    for _, row in unique_locs.iterrows():
        print(f"Fetching data for {row['PROV']}...")
        
        # We fetch the entire 2006-2017 range in one call per location
        data = fetch_nasa_monthly(row['LAT'], row['LON'], "2006", "2017")
        
        if data:
            for param, values in data.items():
                for date_key, val in values.items():
                    # NASA date_key format is 'YYYYMM'
                    weather_results.append({
                        'PROV': row['PROV'],
                        'ANNEE': int(date_key[:4]),
                        'MOIS': int(date_key[4:]),
                        'PARAM': param,
                        'VALUE': val
                    })
        
        # Respect the API rate limit
        time.sleep(1)

    # 3. Pivot weather data to have columns for each parameter
    df_weather = pd.DataFrame(weather_results)
    df_weather = df_weather.pivot_table(
        index=['PROV', 'ANNEE', 'MOIS'], 
        columns='PARAM', 
        values='VALUE'
    ).reset_index()

    # 4. Merge weather data back into health data
    # We join on PROV, ANNEE, and MOIS
    final_dataset = pd.merge(df_health, df_weather, on=['PROV', 'ANNEE', 'MOIS'], how='left')

    # 5. Save the Master Dataset
    final_dataset.to_csv('DRC_Health_Weather_Master.csv', index=False)
    print("\n--- Integration Success ---")
    print(f"Final file 'DRC_Health_Weather_Master.csv' created with {len(final_dataset)} rows.")

if __name__ == "__main__":
    main()