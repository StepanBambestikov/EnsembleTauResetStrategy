import numpy as np
from tau_agregator_strategy.structs import ModelPrediction


def avr_tau_reset_predictor(entity_history, TAU, decision_history) -> ModelPrediction:

    # Extract the current price from the observation
    entity = entity_history[-1]
    current_price = entity.global_state.price

    high_volatility = False

    # Example: checking volatility for the last N periods, this example is completely fictitious
    # and is used only to test cases of absence of LP position
    if len(entity_history) > 5:
        recent_prices = [entity_history[-i].global_state.price for i in range(1, 5)]
        price_volatility = np.std(recent_prices) / np.mean(recent_prices)
        high_volatility = price_volatility > 0.001

    if not entity.is_position:
        if high_volatility:
            return ModelPrediction(r=0.0, c=current_price, w=TAU)
        else:
            return ModelPrediction(r=1.0, c=current_price, w=TAU)

    lower_bound, upper_bound = entity.internal_state.price_lower, entity.internal_state.price_upper
    rebalance_needed = current_price < lower_bound or current_price > upper_bound

    if rebalance_needed:
        if high_volatility:
            return ModelPrediction(r=0.0, c=current_price,
                                   w=0.0)
        else:
            return ModelPrediction(r=1.0, c=current_price, w=TAU)
    else:
        return ModelPrediction(r=1.0, c=decision_history[-1].c, w=TAU)