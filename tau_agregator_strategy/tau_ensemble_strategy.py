import copy
from typing import Dict, List, Any

from fractal.core.base import (Action, ActionToTake, BaseStrategy,
                               NamedEntity)
from fractal.core.entities import UniswapV3LPConfig, UniswapV3LPEntity

from tau_agregator_strategy.structs import TauEnsembleParams, ModelWeight, ModelPrediction


class TauEnsembleStrategy(BaseStrategy):
    """
    Adaptive Ensemble Strategy for Liquidity Range Management

    This strategy manages an ensemble of models, each providing
    predictions about optimal rebalancing decisions, range centers, and
    range widths. The ensemble combines these predictions using a weighted
    average, and the weights are updated using a gradient descent method with
    delayed feedback.
    """

    # Pool-specific constants
    token0_decimals: int = -1
    token1_decimals: int = -1
    tick_spacing: int = -1
    entity_history: list = []

    def __init__(self, oracul_strategy, params: TauEnsembleParams, debug: bool = False, *args, **kwargs):
        self._params: TauEnsembleParams = None  # Set for type hinting
        assert self.token0_decimals != -1 and self.token1_decimals != -1 and self.tick_spacing != -1
        super().__init__(params=params, debug=debug, *args, **kwargs)

        # Model weights and history initialization
        self.models: Dict[str, Any] = {}
        self.model_weights: Dict[str, ModelWeight] = {}

        self.oracul_strategy = oracul_strategy

        # History tracking
        self.prediction_history: List[Dict[str, ModelPrediction]] = []
        self.decision_history: List[ModelPrediction] = []

        # State tracking
        self.deposited_initial_funds = False


    def set_up(self):
        """Register Uniswap V3 LP entity"""
        self.register_entity(NamedEntity(
            entity_name='UNISWAP_V3',
            entity=UniswapV3LPEntity(
                UniswapV3LPConfig(
                    token0_decimals=self.token0_decimals,
                    token1_decimals=self.token1_decimals
                )
            )
        ))
        assert isinstance(self.get_entity('UNISWAP_V3'), UniswapV3LPEntity)

    def add_model(self, name: str, model_fn,
                  initial_weight: float = 1.0):
        """Add a new model to the ensemble"""
        self.models[name] = model_fn
        self.model_weights[name] = ModelWeight(alpha=initial_weight)

    def _aggregate_predictions(self, predictions: Dict[str, ModelPrediction]) -> ModelPrediction:
        """Aggregate predictions from all models using weighted average"""
        weights = {name: self.model_weights[name].alpha for name in predictions.keys()}
        weight_sum = sum(weights.values())

        r_weighted = sum(weights[name] * pred.r for name, pred in predictions.items()) / weight_sum
        c_weighted = sum(weights[name] * pred.c for name, pred in predictions.items()) / weight_sum
        w_weighted = sum(weights[name] * pred.w for name, pred in predictions.items()) / weight_sum

        return ModelPrediction(r=r_weighted, c=c_weighted, w=w_weighted)

    def _update_weights(self, learning_rate: float = 0.01):
        """
        Update model weights using gradient descent method.

        Implements the algorithm from the document:
        1. Calculate optimal solution for past period
        2. Calculate gradients for model weights
        3. Update weights via gradient descent
        4. Normalize weights relative to mean

        Args:
            learning_rate: Learning rate for gradient descent
        """
        if len(self.prediction_history) - self.oracul_strategy.future_window < 2:
            return  # Not enough history for updates


        # Get latest predictions and ensemble decision
        last_predictions = self.prediction_history[-self.oracul_strategy.future_window]
        ensemble_decision = self.decision_history[-self.oracul_strategy.future_window]

        # For this example, we'll assume the optimal decision is the current decision
        # In a real implementation, the optimal decision should be calculated based on
        # historical data analysis and profit maximization
        optimal_decision = self.oracul_strategy.oracle_predictor(self.entity_history, self._params.TAU, self.decision_history)
        if optimal_decision is None:
            return

        deltas = {}
        for model_name, model_pred in last_predictions.items():
            # Gradients for each component (r, c, w)
            # ∂L/∂α_j = 2(K^(t) - K_opt^(t)) · (k_j^(t) - K^(t))/∑α_i^(t)
            weight_sum = sum(weight.alpha for weight in self.model_weights.values())

            grad_r = 2 * (ensemble_decision.r - optimal_decision.r) * (model_pred.r - ensemble_decision.r) / weight_sum
            grad_c = 2 * (ensemble_decision.c - optimal_decision.c) * (model_pred.c - ensemble_decision.c) / weight_sum
            grad_w = 2 * (ensemble_decision.w - optimal_decision.w) * (model_pred.w - ensemble_decision.w) / weight_sum

            grad_total = grad_r + grad_c + grad_w
            deltas[model_name] = -learning_rate * grad_total

        avg_delta = sum(deltas.values()) / len(deltas)

        for model_name, delta in deltas.items():
            adj_delta = delta - avg_delta

            self.model_weights[model_name].history.append(self.model_weights[model_name].alpha)
            self.model_weights[model_name].alpha += adj_delta
            self.model_weights[model_name].alpha = max(0.01, self.model_weights[model_name].alpha)

        self._debug(f"Updated model weights: {[(name, weight.alpha) for name, weight in self.model_weights.items()]}")

    def predict(self) -> List[ActionToTake]:
        """
        Main strategy logic. Collects predictions from all models,
        aggregates them, and makes a decision about the action.
        """
        uniswap_entity: UniswapV3LPEntity = self.get_entity('UNISWAP_V3')
        self.entity_history.append(copy.deepcopy(uniswap_entity))
        self.oracul_strategy.add_history_element(copy.deepcopy(uniswap_entity))

        current_predictions = {
            name: model_fn(self.entity_history, self._params.TAU, self.decision_history)
            for name, model_fn in self.models.items()
        }
        self.prediction_history.append(current_predictions)

        ensemble_decision = self._aggregate_predictions(current_predictions)
        self.decision_history.append(ensemble_decision)

        if len(self.prediction_history) > 1:
            self._update_weights()

        if ensemble_decision.r > 0.5:  # Threshold for rebalancing
            if not uniswap_entity.is_position and not self.deposited_initial_funds:
                self._debug("No active position. Depositing initial funds...")
                self.deposited_initial_funds = True
                return self._deposit_to_lp()

            # Need to rebalance if new center
            if abs(ensemble_decision.c - self.decision_history[-2].c) > 0.1:
                self._debug(
                    f"Ensemble decided to rebalance with center={ensemble_decision.c:.4f}, width={ensemble_decision.w:.4f}")
                return self._rebalance(ensemble_decision.c, ensemble_decision.w)
            return []
        else:
            if uniswap_entity.internal_state.liquidity > 0:
                self._debug("Liquidity withdrawn from current range.")
                return [ActionToTake(entity_name='UNISWAP_V3', action=Action(action='close_position', args={}))]

        self._debug("Ensemble decided to maintain current position")
        return []

    def _deposit_to_lp(self) -> List[ActionToTake]:
        """Deposit funds into Uniswap LP"""
        return [ActionToTake(
            entity_name='UNISWAP_V3',
            action=Action(action='deposit', args={'amount_in_notional': self._params.INITIAL_BALANCE})
        )]

    def _rebalance(self, center_price: float, width: float) -> List[ActionToTake]:
        """
        Redistribute liquidity to a new range centered around the given price
        with the specified width.
        """
        actions = []
        entity: UniswapV3LPEntity = self.get_entity('UNISWAP_V3')

        # Step 1: Withdraw liquidity from current range
        if entity.internal_state.liquidity > 0:
            actions.append(
                ActionToTake(entity_name='UNISWAP_V3', action=Action(action='close_position', args={}))
            )
            self._debug("Liquidity withdrawn from current range.")

        # Step 2: Calculate new range boundaries based on center and width
        tick_spacing = self.tick_spacing
        price_lower = center_price * 1.0001 ** (-width * tick_spacing)
        price_upper = center_price * 1.0001 ** (width * tick_spacing)

        # Step 3: Open new position with new range
        delegate_get_cash = lambda obj: obj.get_entity('UNISWAP_V3').internal_state.cash
        actions.append(ActionToTake(
            entity_name='UNISWAP_V3',
            action=Action(
                action='open_position',
                args={
                    'amount_in_notional': delegate_get_cash,  # Allocate all available funds
                    'price_lower': price_lower,
                    'price_upper': price_upper
                }
            )
        ))
        self._debug(f"New position opened with range [{price_lower:.4f}, {price_upper:.4f}].")
        return actions