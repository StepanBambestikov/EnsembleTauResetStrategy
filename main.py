import os
from datetime import datetime, UTC
from pathlib import Path
import pickle

from fractal.loaders.base_loader import LoaderType
from fractal.loaders.thegraph import EthereumUniswapV3Loader

from plot import plot_history_with_pnl
from service import build_observations
from tau_agregator_strategy.avr_tau import TauResetPredict
from tau_agregator_strategy.oracle_strategy import OracleStrategy
from tau_agregator_strategy.tau_ensemble_strategy import TauEnsembleParams, TauEnsembleStrategy


#THE_GRAPH_API_KEY = os.getenv('THE_GRAPH_API_KEY')
THE_GRAPH_API_KEY = '279f1b788bdf80ec5532277e82d82ca7'

if __name__ == '__main__':
    # Set up
    ticker: str = 'ETHUSDT'
    pool_address: str = '0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8'

    token0_decimals, token1_decimals = EthereumUniswapV3Loader(
        THE_GRAPH_API_KEY, loader_type=LoaderType.CSV).get_pool_decimals(pool_address)

    start_time = datetime(2025, 1, 5, tzinfo=UTC)
    end_time = datetime(2025, 4, 5, tzinfo=UTC)
    fidelity = 'hour'

    observations = build_observations(ticker, pool_address, THE_GRAPH_API_KEY, start_time, end_time, fidelity=fidelity)

    observation0 = observations[0]

    initial_balance = 1_000_000
    ensemble_params: TauEnsembleParams = TauEnsembleParams(
        TAU=10,  # Tau value is the same as in the original strategy
        INITIAL_BALANCE=initial_balance
    )

    TauEnsembleStrategy.token0_decimals = token0_decimals
    TauEnsembleStrategy.token1_decimals = token1_decimals
    TauEnsembleStrategy.tick_spacing = 60
    ensemble_strategy: TauEnsembleStrategy = TauEnsembleStrategy(
        oracul_strategy=OracleStrategy(),
        debug=True,
        params=ensemble_params
    )

    small = TauResetPredict(tau=5)
    big = TauResetPredict(tau=30)

    ensemble_strategy.add_model("tau_big", big.avr_tau_reset_predictor, 1)
    ensemble_strategy.add_model("tau_small", small.avr_tau_reset_predictor, 1)

    entities = ensemble_strategy.get_all_available_entities().keys()
    assert all(entity in observation0.states for entity in entities)

    # Running the ensemble strategy
    print("Starting ensemble strategy (with one base model)...")
    ensemble_result = ensemble_strategy.run(observations)
    print("Ensemble strategy results:")
    print(ensemble_result.get_default_metrics())

    with open('ensemble_entity_history.pkl', 'wb') as f:
        pickle.dump(ensemble_strategy.entity_history, f)

    file_name = 'tau_ensemble_result.csv'
    ensemble_result.to_dataframe().to_csv(file_name)
    print(ensemble_result.to_dataframe().iloc[-1])  # show the last state of the strategy

    print("\nDone! Results saved to files tau_ensemble_result.csv and tau_strategy_result.csv")

    # Visualize results
    plot_history_with_pnl(file_name, initial_balance, ensemble_strategy.model_weights)