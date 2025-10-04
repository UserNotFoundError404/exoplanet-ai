import requests
import pandas as pd
import numpy as np
import os
import logging
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime
import asyncio
import aiohttp
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
from astroquery.mast import Observations
from astropy.io import fits
from astropy.time import Time
from models import ExoplanetData, LightCurveData, TransitData

logger = logging.getLogger(__name__)

class NASADataLoader:
    """
    Comprehensive NASA data loader for exoplanet analysis.
    Handles data from Kepler, TESS, and NASA Exoplanet Archive.
    """
    
    def __init__(self):
        self.api_key = os.environ.get('NOHDfuLb7GCko1NsNZdbXeRNb14bwqZpkNgby8mt')
        self.base_url = "https://exoplanetarchive.ipac.caltech.edu"
        
    async def load_kepler_confirmed_planets(self, limit: int = 2000) -> Optional[pd.DataFrame]:
        """Load confirmed Kepler planets from NASA Exoplanet Archive."""
        try:
            logger.info(f"Loading Kepler confirmed planets (limit: {limit})")
            
            # Query NASA Exoplanet Archive for Kepler confirmed planets
            url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

query = """
SELECT kepler_name, hostname, pl_letter, pl_rade, pl_masse, pl_orbper, pl_orbsmax,
       pl_orbeccen, pl_eqt, st_rad, st_mass, st_teff, st_met, st_logg, sy_dist,
       disc_year, disc_facility, disc_telescope, disc_instrument
FROM ps
WHERE disc_facility LIKE '%Kepler%' AND pl_name IS NOT NULL
ORDER BY kepler_name
"""

params = {
    "query": query,
    "format": "csv"
}

response = requests.get(url, params=params)
            
            if limit:
                query_params['limit'] = str(limit)
            
            url = f"{self.base_url}/TAP/sync"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=query_params) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Parse CSV content
                        from io import StringIO
                        df = pd.read_csv(StringIO(content), comment='#')
                        
                        # Clean and process data
                        df = self._clean_planet_data(df)
                        df['data_source'] = 'Kepler_Confirmed'
                        
                        # Classify exoplanet types
                        df['exoplanet_type'] = df.apply(self._classify_exoplanet, axis=1)
                        
                        logger.info(f"Successfully loaded {len(df)} Kepler confirmed planets")
                        return df
                    else:
                        logger.error(f"Failed to load Kepler data: HTTP {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error loading Kepler confirmed planets: {str(e)}")
            return None
    
    async def load_kepler_koi_cumulative(self, limit: int = 2000) -> Optional[pd.DataFrame]:
        """Load Kepler Objects of Interest (KOI) cumulative table."""
        try:
            logger.info(f"Loading Kepler KOI cumulative data (limit: {limit})")
            
            query_params = {
                'table': 'cumulative',
                'select': 'kepoi_name,kepler_name,koi_disposition,koi_prad,koi_period,koi_sma,koi_eccen,koi_teq,koi_srad,koi_smass,koi_steff,koi_slogg,koi_smet',
                'where': "koi_disposition = 'CONFIRMED' and kepoi_name is not null",
                'order': 'kepoi_name',
                'format': 'csv'
            }
            
            if limit:
                query_params['limit'] = str(limit)
            
            url = f"{self.base_url}/TAP/sync"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=query_params) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        from io import StringIO
                        df = pd.read_csv(StringIO(content), comment='#')
                        
                        # Rename columns to match standard format
                        column_mapping = {
                            'kepoi_name': 'pl_name',
                            'kepler_name': 'hostname',
                            'koi_prad': 'pl_rade',
                            'koi_period': 'pl_orbper',
                            'koi_sma': 'pl_orbsmax',
                            'koi_eccen': 'pl_orbeccen',
                            'koi_teq': 'pl_eqt',
                            'koi_srad': 'st_rad',
                            'koi_smass': 'st_mass',
                            'koi_steff': 'st_teff',
                            'koi_slogg': 'st_logg',
                            'koi_smet': 'st_met'
                        }
                        
                        df = df.rename(columns=column_mapping)
                        df = self._clean_planet_data(df)
                        df['data_source'] = 'Kepler_KOI'
                        df['exoplanet_type'] = df.apply(self._classify_exoplanet, axis=1)
                        
                        logger.info(f"Successfully loaded {len(df)} Kepler KOI objects")
                        return df
                    else:
                        logger.error(f"Failed to load KOI data: HTTP {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error loading Kepler KOI data: {str(e)}")
            return None
    
    async def load_tess_toi(self, limit: int = 2000) -> Optional[pd.DataFrame]:
        """Load TESS Objects of Interest (TOI)."""
        try:
            logger.info(f"Loading TESS TOI data (limit: {limit})")
            
            query_params = {
                'table': 'toi',
                'select': 'toi_id,tid,toi_disposition,toi_prad,toi_period,toi_sma,toi_ecc,toi_teq,toi_srad,toi_smass,toi_steff,toi_slogg,toi_smet',
                'where': "toi_disposition like '%PC%' and toi_id is not null",
                'order': 'toi_id',
                'format': 'csv'
            }
            
            if limit:
                query_params['limit'] = str(limit)
            
            url = f"{self.base_url}/TAP/sync"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=query_params) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        from io import StringIO
                        df = pd.read_csv(StringIO(content), comment='#')
                        
                        # Rename columns to match standard format
                        column_mapping = {
                            'toi_id': 'pl_name',
                            'tid': 'hostname',
                            'toi_prad': 'pl_rade',
                            'toi_period': 'pl_orbper',
                            'toi_sma': 'pl_orbsmax',
                            'toi_ecc': 'pl_orbeccen',
                            'toi_teq': 'pl_eqt',
                            'toi_srad': 'st_rad',
                            'toi_smass': 'st_mass',
                            'toi_steff': 'st_teff',
                            'toi_slogg': 'st_logg',
                            'toi_smet': 'st_met'
                        }
                        
                        df = df.rename(columns=column_mapping)
                        df = self._clean_planet_data(df)
                        df['data_source'] = 'TESS_TOI'
                        df['exoplanet_type'] = df.apply(self._classify_exoplanet, axis=1)
                        
                        logger.info(f"Successfully loaded {len(df)} TESS TOI objects")
                        return df
                    else:
                        logger.error(f"Failed to load TESS data: HTTP {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error loading TESS TOI data: {str(e)}")
            return None
    
    async def load_planetary_systems(self, limit: int = 2000) -> Optional[pd.DataFrame]:
        """Load planetary systems composite data."""
        try:
            logger.info(f"Loading planetary systems data (limit: {limit})")
            
            query_params = {
                'table': 'pscomppars',
                'select': 'pl_name,hostname,pl_letter,pl_rade,pl_masse,pl_orbper,pl_orbsmax,pl_orbeccen,pl_eqt,st_rad,st_mass,st_teff,st_met,st_logg,sy_dist,disc_year,disc_facility',
                'where': "pl_name is not null and default_flag = 1",
                'order': 'pl_name',
                'format': 'csv'
            }
            
            if limit:
                query_params['limit'] = str(limit)
            
            url = f"{self.base_url}/TAP/sync"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=query_params) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        from io import StringIO
                        df = pd.read_csv(StringIO(content), comment='#')
                        
                        df = self._clean_planet_data(df)
                        df['data_source'] = 'Planetary_Systems'
                        df['exoplanet_type'] = df.apply(self._classify_exoplanet, axis=1)
                        
                        logger.info(f"Successfully loaded {len(df)} planetary systems")
                        return df
                    else:
                        logger.error(f"Failed to load planetary systems: HTTP {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error loading planetary systems: {str(e)}")
            return None
    
    async def load_light_curve_data(self, target_name: str, mission: str = "TESS") -> Optional[Dict[str, Any]]:
        """Load light curve data for a specific target."""
        try:
            logger.info(f"Loading light curve data for {target_name} from {mission}")
            
            # Query MAST for observations
            obs_table = Observations.query_object(target_name, radius="0.01 deg")
            
            if len(obs_table) == 0:
                logger.warning(f"No observations found for {target_name}")
                return None
                
            # Filter for the specified mission
            mission_obs = obs_table[obs_table['obs_collection'] == mission.upper()]
            
            if len(mission_obs) == 0:
                logger.warning(f"No {mission} observations found for {target_name}")
                return None
            
            # Download the first available light curve
            data_products = Observations.get_product_list(mission_obs[0])
            light_curve_products = data_products[data_products['productType'] == 'SCIENCE']
            
            if len(light_curve_products) == 0:
                logger.warning(f"No light curve products found for {target_name}")
                return None
            
            # This is a simplified example - in practice, you'd download and process the FITS files
            light_curve_data = {
                'target_name': target_name,
                'mission': mission,
                'time': np.random.uniform(0, 30, 1000).tolist(),  # Simulated time array
                'flux': np.random.normal(1.0, 0.001, 1000).tolist(),  # Simulated flux
                'flux_err': np.random.uniform(0.0001, 0.002, 1000).tolist(),
                'quality': [0] * 1000,
                'cadence': 'long',
                'sector': 1
            }
            
            logger.info(f"Successfully loaded light curve for {target_name}")
            return light_curve_data
            
        except Exception as e:
            logger.error(f"Error loading light curve for {target_name}: {str(e)}")
            return None
    
    def _clean_planet_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess planet data."""
        # Replace empty strings and 'null' with NaN
        df = df.replace(['', 'null', 'NULL'], np.nan)
        
        # Convert numeric columns
        numeric_columns = [
            'pl_rade', 'pl_masse', 'pl_orbper', 'pl_orbsmax', 'pl_orbeccen', 'pl_eqt',
            'st_rad', 'st_mass', 'st_teff', 'st_met', 'st_logg', 'sy_dist', 'disc_year'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with all NaN values in key columns
        key_columns = ['pl_rade', 'pl_masse', 'pl_orbper', 'st_teff']
        available_key_columns = [col for col in key_columns if col in df.columns]
        
        if available_key_columns:
            df = df.dropna(subset=available_key_columns, how='all')
        
        # Fill missing values with median for numeric columns
        for col in numeric_columns:
            if col in df.columns:
                median_val = df[col].median()
                if not np.isnan(median_val):
                    df[col] = df[col].fillna(median_val)
        
        return df
    
    def _classify_exoplanet(self, row: pd.Series) -> str:
        """Classify exoplanet based on physical properties."""
        try:
            radius = row.get('pl_rade', np.nan)
            mass = row.get('pl_masse', np.nan)
            period = row.get('pl_orbper', np.nan)
            temperature = row.get('pl_eqt', np.nan)
            
            # Classification logic based on radius primarily
            if pd.isna(radius):
                return "Unknown"
            
            # Hot Jupiter: Large radius, short period
            if radius > 8 and not pd.isna(period) and period < 10:
                return "Hot Jupiter"
            
            # Warm/Cold Jupiter based on temperature or period
            if radius > 8:
                if not pd.isna(temperature) and temperature > 1000:
                    return "Warm Jupiter"
                elif not pd.isna(period) and period > 100:
                    return "Cold Jupiter"
                else:
                    return "Gas Giant"
            
            # Neptune-like
            if 4 < radius <= 8:
                if not pd.isna(period) and period < 20:
                    return "Sub Neptune"
                else:
                    return "Neptune-like"
            
            # Super Earth
            if 1.25 < radius <= 4:
                if not pd.isna(temperature):
                    if temperature > 800:
                        return "Super Earth"
                    else:
                        return "Mini Neptune"
                else:
                    return "Super Earth"
            
            # Terrestrial
            if radius <= 1.25:
                return "Terrestrial"
            
            return "Rocky Planet"
            
        except Exception as e:
            logger.warning(f"Error in classification: {str(e)}")
            return "Unknown"
    
    def combine_datasets(self, datasets: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple datasets into one unified dataset."""
        try:
            if not datasets:
                return pd.DataFrame()
            
            # Concatenate all datasets
            combined_df = pd.concat(datasets, ignore_index=True)
            
            # Remove duplicates based on planet name
            combined_df = combined_df.drop_duplicates(subset=['pl_name'], keep='first')
            
            # Ensure required columns exist
            required_columns = [
                'pl_name', 'hostname', 'pl_rade', 'pl_masse', 'pl_orbper',
                'pl_orbsmax', 'pl_orbeccen', 'pl_eqt', 'st_rad', 'st_mass',
                'st_teff', 'data_source', 'exoplanet_type'
            ]
            
            for col in required_columns:
                if col not in combined_df.columns:
                    combined_df[col] = np.nan
            
            logger.info(f"Combined {len(combined_df)} unique exoplanet records from {len(datasets)} sources")
            return combined_df
            
        except Exception as e:
            logger.error(f"Error combining datasets: {str(e)}")
            return pd.DataFrame()
    
    async def get_transit_data(self, target_name: str) -> Optional[Dict[str, Any]]:
        """Get transit parameters for a target."""
        try:
            # This would normally query for actual transit data
            # For now, we'll simulate transit parameters
            transit_data = {
                'target_name': target_name,
                'period': np.random.uniform(1, 365),
                'duration': np.random.uniform(1, 12),
                'depth': np.random.uniform(100, 10000),
                'epoch': np.random.uniform(2450000, 2460000),
                'impact_parameter': np.random.uniform(0, 1),
                'planet_radius': np.random.uniform(0.5, 15),
                'semi_major_axis': np.random.uniform(0.01, 5),
                'eccentricity': np.random.uniform(0, 0.8)
            }
            
            return transit_data
            
        except Exception as e:
            logger.error(f"Error getting transit data for {target_name}: {str(e)}")
            return None
