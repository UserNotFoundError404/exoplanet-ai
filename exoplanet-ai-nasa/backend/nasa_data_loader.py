import os
import logging
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from io import StringIO
from typing import Optional, List, Dict, Any
from astroquery.mast import Observations
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NASADataLoader:
    """
    NASA Exoplanet Archive loader for Kepler, TESS, and Planetary Systems datasets.
    """

    def __init__(self):
        self.base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

    # ==========================================================
    # Utility Functions
    # ==========================================================

    async def _fetch_tap(self, query: str, limit: int = 2000) -> Optional[pd.DataFrame]:
        """Perform async TAP query and return DataFrame."""
        try:
            params = {"query": query.strip(), "format": "csv"}
            if limit:
                params["limit"] = str(limit)

            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"TAP query failed: HTTP {response.status}")
                        return None
                    text = await response.text()
                    df = pd.read_csv(StringIO(text), comment="#")
                    if df.empty:
                        logger.warning("Query returned no rows.")
                        return None
                    return df
        except Exception as e:
            logger.error(f"TAP query error: {e}")
            return None

    def _clean_planet_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess planet data."""
        df = df.replace(["", "null", "NULL"], np.nan)

        numeric_columns = [
            "pl_rade", "pl_masse", "pl_orbper", "pl_orbsmax", "pl_orbeccen", "pl_eqt",
            "st_rad", "st_mass", "st_teff", "st_met", "st_logg", "sy_dist", "disc_year"
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        key_columns = ["pl_rade", "pl_masse", "pl_orbper", "st_teff"]
        df = df.dropna(subset=[c for c in key_columns if c in df.columns], how="all")

        for col in numeric_columns:
            if col in df.columns:
                median = df[col].median()
                if not np.isnan(median):
                    df[col] = df[col].fillna(median)

        return df

    def _classify_exoplanet(self, row: pd.Series) -> str:
        """Classify planet by radius and orbital properties."""
        try:
            radius = row.get("pl_rade", np.nan)
            period = row.get("pl_orbper", np.nan)
            temperature = row.get("pl_eqt", np.nan)

            if pd.isna(radius):
                return "Unknown"

            if radius > 8:
                if not pd.isna(period) and period < 10:
                    return "Hot Jupiter"
                if not pd.isna(temperature) and temperature > 1000:
                    return "Warm Jupiter"
                return "Gas Giant"
            elif 4 < radius <= 8:
                return "Neptune-like"
            elif 1.25 < radius <= 4:
                return "Super Earth"
            elif radius <= 1.25:
                return "Terrestrial"
            return "Unknown"
        except Exception:
            return "Unknown"

    def combine_datasets(self, datasets: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple datasets into one unified DataFrame."""
        if not datasets:
            return pd.DataFrame()

        combined_df = pd.concat(datasets, ignore_index=True)

        if "pl_name" in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=["pl_name"], keep="first")

        required_columns = [
            "pl_name", "hostname", "pl_rade", "pl_masse", "pl_orbper", "pl_orbsmax",
            "pl_orbeccen", "pl_eqt", "st_rad", "st_mass", "st_teff",
            "data_source", "exoplanet_type"
        ]
        for col in required_columns:
            if col not in combined_df.columns:
                combined_df[col] = np.nan

        logger.info(f"Combined {len(combined_df)} records from {len(datasets)} datasets.")
        return combined_df

    # ==========================================================
    # Individual Dataset Loaders
    # ==========================================================

    async def load_kepler_confirmed_planets(self, limit: int = 2000) -> Optional[pd.DataFrame]:
        """Load confirmed Kepler planets from PS table."""
        logger.info("Loading Kepler confirmed planets...")
        query = """
        SELECT pl_name, hostname, pl_letter, pl_rade, pl_masse, pl_orbper, pl_orbsmax,
               pl_orbeccen, pl_eqt, st_rad, st_mass, st_teff, st_met, st_logg, sy_dist,
               disc_year, disc_facility
        FROM ps
        WHERE disc_facility LIKE '%Kepler%' AND pl_name IS NOT NULL
        ORDER BY pl_name
        """
        df = await self._fetch_tap(query, limit)
        if df is None:
            return None
        df = self._clean_planet_data(df)
        df["data_source"] = "Kepler_Confirmed"
        df["exoplanet_type"] = df.apply(self._classify_exoplanet, axis=1)
        return df

    async def load_kepler_koi_cumulative(self, limit: int = 2000) -> Optional[pd.DataFrame]:
        """Load Kepler KOI cumulative confirmed planets."""
        logger.info("Loading Kepler KOI cumulative data...")
        query = """
        SELECT kepoi_name, kepler_name, koi_disposition, koi_prad, koi_period, koi_sma, 
               koi_eccen, koi_teq, koi_srad, koi_smass, koi_steff, koi_slogg, koi_smet
        FROM cumulative
        WHERE koi_disposition = 'CONFIRMED'
        ORDER BY kepoi_name
        """
        df = await self._fetch_tap(query, limit)
        if df is None:
            return None

        mapping = {
            "kepoi_name": "pl_name",
            "kepler_name": "hostname",
            "koi_prad": "pl_rade",
            "koi_period": "pl_orbper",
            "koi_sma": "pl_orbsmax",
            "koi_eccen": "pl_orbeccen",
            "koi_teq": "pl_eqt",
            "koi_srad": "st_rad",
            "koi_smass": "st_mass",
            "koi_steff": "st_teff",
            "koi_slogg": "st_logg",
            "koi_smet": "st_met",
        }
        df = df.rename(columns=mapping)
        df = self._clean_planet_data(df)
        df["data_source"] = "Kepler_KOI"
        df["exoplanet_type"] = df.apply(self._classify_exoplanet, axis=1)
        return df

    async def load_tess_confirmed_planets(self, limit: int = 2000) -> Optional[pd.DataFrame]:
        """Load confirmed TESS planets from pscomppars (tested query)."""
        logger.info("Loading confirmed TESS planets from pscomppars...")
        query = f"""
        SELECT TOP {limit}
            pl_name, hostname, pl_rade, pl_masse, pl_orbper,
            disc_year, disc_facility, disc_telescope, st_teff, st_mass, st_rad, st_logg, st_met
        FROM pscomppars
        WHERE (UPPER(disc_facility) LIKE '%TESS%' OR UPPER(disc_telescope) LIKE '%TESS%')
          AND pl_name IS NOT NULL
          AND default_flag = 1
        ORDER BY pl_name
        """.strip()

        params = {"query": query, "format": "csv"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    logger.info(f"TESS request URL: {response.url}")
                    if response.status != 200:
                        logger.error(f"TESS query failed: HTTP {response.status}")
                        return None
                    text = await response.text()
                    if "pl_name" not in text:
                        logger.error("TESS query returned no CSV header — likely no matches.")
                        return None
                    df = pd.read_csv(StringIO(text), comment="#")
                    if df.empty:
                        logger.warning("TESS confirmed planets: 0 rows.")
                        return None
            df = self._clean_planet_data(df)
            df["data_source"] = "TESS_Confirmed"
            df["exoplanet_type"] = df.apply(self._classify_exoplanet, axis=1)
            logger.info(f"✅ Loaded {len(df)} TESS confirmed planets.")
            return df
        except Exception as e:
            logger.error(f"Error loading TESS confirmed planets: {e}")
            return None


    async def load_planetary_systems(self, limit: int = 2000) -> Optional[pd.DataFrame]:
        """Load planetary systems composite (tested query)."""
        logger.info("Loading Planetary Systems composite from pscomppars...")
        query = f"""
        SELECT TOP {limit}
            pl_name, hostname, pl_rade, pl_masse, pl_orbper, pl_orbsmax,
            pl_orbeccen, pl_eqt, st_rad, st_mass, st_teff, st_met, st_logg,
            sy_dist, disc_year, disc_facility
        FROM pscomppars
        WHERE pl_name IS NOT NULL AND default_flag = 1
        ORDER BY pl_name
        """.strip()

        params = {"query": query, "format": "csv"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    logger.info(f"PS request URL: {response.url}")
                    if response.status != 200:
                        logger.error(f"Planetary Systems query failed: HTTP {response.status}")
                        return None
                    text = await response.text()
                    if "pl_name" not in text:
                        logger.error("Planetary Systems returned no CSV header.")
                        return None
                    df = pd.read_csv(StringIO(text), comment="#")
                    if df.empty:
                        logger.warning("Planetary Systems query returned 0 rows.")
                        return None
            df = self._clean_planet_data(df)
            df["data_source"] = "Planetary_Systems"
            df["exoplanet_type"] = df.apply(self._classify_exoplanet, axis=1)
            logger.info(f"✅ Loaded {len(df)} Planetary Systems entries.")
            return df
        except Exception as e:
            logger.error(f"Error loading Planetary Systems: {e}")
            return None



    # ==========================================================
    # Combined Loader
    # ==========================================================

    async def load_all_data(self, limit: int = 2000) -> pd.DataFrame:
        """Load and combine all NASA datasets."""
        logger.info("Loading all NASA datasets (Kepler, TESS, PS)...")
        try:
            tasks = [
                self.load_kepler_confirmed_planets(limit),
                self.load_kepler_koi_cumulative(limit),
                self.load_tess_confirmed_planets(limit),
                self.load_planetary_systems(limit)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            datasets = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]
            if not datasets:
                logger.error("No valid datasets loaded.")
                return pd.DataFrame()

            combined = self.combine_datasets(datasets)
            logger.info(
                f"Loaded total of {len(combined)} planets from {len(datasets)} sources: "
                f"{combined['data_source'].unique().tolist()}"
            )
            return combined
        except Exception as e:
            logger.error(f"Error loading all data: {e}")
            return pd.DataFrame()

    # ==========================================================
    # Helper
    # ==========================================================

    def summarize_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Return dataset overview."""
        return {
            "total_records": len(df),
            "sources": df["data_source"].unique().tolist() if "data_source" in df.columns else [],
            "columns": df.columns.tolist(),
            "planet_types": (
                df["exoplanet_type"].value_counts().to_dict()
                if "exoplanet_type" in df.columns
                else {}
            ),
        }
