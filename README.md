# Adaptive Ensemble Strategy for Liquidity Range Management

This project extends the tau-reset liquidity strategy for Uniswap V3 by implementing an ensemble-based approach to optimize liquidity provision. The strategy uses multiple independent predictive models working together to maximize profitability. This project is built on top of the [fractal-defi repository](https://github.com/Logarithm-Labs/fractal-defi.git). The original tau-reset strategy was taken from this repository, and all backtesting was conducted using its infrastructure. The core framework, data loading mechanisms, and backtesting environment are leveraged from fractal-defi, with our enhancements focused on the ensemble model approach.

## Overview

The project has dynamic weighting mechanism which adjusts the importance of individual models within the ensemble using gradient descent with delayed feedback. This adaptive approach allows the strategy to automatically identify and leverage the most effective prediction models in different market conditions.

## Implementation Details

Both `main.py` and the Jupyter notebook implement an ensemble of two classical tau-reset strategies with different tau parameters:

The ensemble dynamically weights these strategies based on their performance through gradient descent optimization, automatically adjusting the influence of each strategy over time as market conditions evolve.

## Documentation

A detailed description of the approach, methodology, and mathematical foundations can be found in the PDF file included in this repository (doc/strategy_doc.pdf).

## Getting Started

1. Clone this repository
2. Install the required dependencies using `pip install -r requirements.txt`
3. Set up your environment variables (API keys, etc.)
4. Run `python main.py` to execute the strategy with the ensemble approach
5. Alternatively, use the Jupyter notebook for an interactive exploration of the strategy's behavior

## Usage

You can run the strategy using either:
- The standalone Python script (`main.py`) for automated execution
- The Jupyter notebook for interactive exploration, visualization, and experimentation with the ensemble parameters
