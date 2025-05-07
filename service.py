from typing import List
from datetime import datetime
from pathlib import Path
import pickle
import pandas as pd

from fractal.loaders.base_loader import LoaderType
from fractal.loaders.thegraph.uniswap_v3 import (
    UniswapV3EthereumPoolHourDataLoader, UniswapV3EthereumPoolMinuteDataLoader
)
from fractal.loaders.binance import BinanceHourPriceLoader, BinanceMinutePriceLoader
from fractal.loaders.structs import PriceHistory, PoolHistory

from fractal.core.base import Observation
from fractal.core.entities import UniswapV3LPGlobalState


def get_observations(
    pool_data: PoolHistory,
    price_data: PriceHistory,
    start_time: datetime = None,
    end_time: datetime = None
) -> List[Observation]:
    """
    Get observations from the pool and price data for the TauResetStrategy.

    Returns:
        List[Observation]: The observation list for TauResetStrategy.
    """
    observations_df: pd.DataFrame = pool_data.join(price_data)
    observations_df = observations_df.dropna()
    observations_df = observations_df.loc[start_time:end_time]
    if start_time is None:
        start_time = observations_df.index.min()
    if end_time is None:
        end_time = observations_df.index.max()
    observations_df = observations_df[observations_df.tvl > 0]
    observations_df = observations_df.sort_index()
    return [
        Observation(
            timestamp=timestamp,
            states={
                'UNISWAP_V3': UniswapV3LPGlobalState(price=price, tvl=tvls, volume=volume, fees=fees, liquidity=liquidity),
            }
        ) for timestamp, (tvls, volume, fees, liquidity, price) in observations_df.iterrows()
    ]


def build_observations(
    ticker: str,
    pool_address: str,
    api_key: str,
    start_time: datetime = None,
    end_time: datetime = None,
    fidelity: str = 'hour',
) -> List[Observation]:
    """
    Build observations for the TauResetStrategy from the given start and end time.
    Uses caching to avoid expensive API calls when possible.
    """
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)

    # Creating cache filename
    start_str = start_time.strftime('%Y%m%d') if start_time else "none"
    end_str = end_time.strftime('%Y%m%d') if end_time else "none"
    cache_filename = f"observations_{ticker}_{pool_address}_{fidelity}_{start_str}_{end_str}.pkl"
    cache_path = cache_dir / cache_filename

    # Check if cache exists
    if cache_path.exists():
        print(f"Loading cached observations from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    # If no cache, load data
    print(f"Fetching observations for {ticker} from {start_time} to {end_time}...")
    try:
        if fidelity == 'hour':
            pool_data: PoolHistory = UniswapV3EthereumPoolHourDataLoader(
                api_key, pool_address, loader_type=LoaderType.CSV).read(with_run=True)
            binance_prices: PriceHistory = BinanceHourPriceLoader(
                ticker, loader_type=LoaderType.CSV).read(with_run=True)
        elif fidelity == 'minute':
            pool_data: PoolHistory = UniswapV3EthereumPoolMinuteDataLoader(
                api_key, pool_address, loader_type=LoaderType.CSV).read(with_run=True)
            binance_prices: PriceHistory = BinanceMinutePriceLoader(
                ticker,
                loader_type=LoaderType.CSV,
                start_time=start_time,
                end_time=end_time
            ).read(with_run=True)
        else:
            raise ValueError("Fidelity must be either 'hour' or 'minute'.")

        observations = get_observations(pool_data, binance_prices, start_time, end_time)

        # Save to cache
        print(f"Caching {len(observations)} observations to {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(observations, f)

        return observations

    except Exception as e:
        print(f"Error fetching observations: {e}")
        if cache_path.exists():
            print(f"Falling back to cached observations")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        raise