import numpy as np
from typing import Tuple
from tau_agregator_strategy.structs import ModelPrediction


class OracleStrategy:
    """
    Oracle strategy that uses information about future prices to minimize impermanent loss (IL).
    Algorithm:
    1. When position is closed - find a moment in the future with similar price (for minimal IL)
    2. When position is open - wait for the found moment and then decide on closing
    """

    def __init__(self, history):
        # Complete trading history, including future data
        self.history = history

        # Tracking target moments for IL minimization
        self.target_index = None  # Future index for position closing
        self.position_open_price = None  # Position opening price
        self.position_open_index = None  # Position opening index

        # Strategy parameters
        self.future_window = 11  # Prediction window (in periods)
        self.price_similarity_threshold = 0.01  # Threshold for determining "similar" price (1%)
        self.min_position_duration = 5  # Minimum position duration
        self.balance_ratio = 0.3  # Target ratio for liquidity balancing

    def oracle_predictor(self, entity_history, TAU, decision_history) -> ModelPrediction:
        """
        Predicts optimal action based on knowledge of future prices,
        minimizing impermanent loss.
        """
        if len(entity_history) < 2:
            # Not enough history for decision making
            return ModelPrediction(r=0.0, c=0.0, w=TAU)

        # Extract current entity and price
        current_entity = entity_history[-1]
        current_price = current_entity.global_state.price

        has_position = current_entity.is_position

        # Get current index in history and future prices
        current_index = len(entity_history) - 1
        future_prices = self._get_future_prices(current_index, self.future_window)

        # If no data about future prices, use base strategy
        if future_prices is None or len(future_prices) < 3:
            if has_position:
                return ModelPrediction(r=1.0, c=decision_history[-1].c if decision_history else current_price, w=TAU)
            else:
                return ModelPrediction(r=0.0, c=current_price, w=TAU)

        # State 1: Position is closed - looking for opening moment
        if not has_position:
            # Find moment in future with similar price (to minimize IL)
            future_similar_indices = self._find_similar_prices(current_price, future_prices)

            if future_similar_indices and future_similar_indices[-1] >= self.min_position_duration:
                # Found suitable future moment - opening position
                self.target_index = current_index + future_similar_indices[-1]
                self.position_open_price = current_price
                self.position_open_index = current_index

                # Determine position center for liquidity balancing
                optimal_center, optimal_width = self._calculate_balanced_center(
                    current_price, future_prices, TAU)

                if optimal_center is None:
                    # Did not find suitable moment - not opening position
                    return ModelPrediction(r=0.0, c=current_price, w=TAU)

                return ModelPrediction(r=1.0, c=optimal_center, w=optimal_width)
            else:
                # Did not find suitable moment - not opening position
                return ModelPrediction(r=0.0, c=current_price, w=TAU)

        # State 2: Position is open - check if closing moment has arrived
        else:
            if self.target_index is None:
                self.position_open_price = current_price
                self.position_open_index = current_index

                # Find moment in future with similar price
                future_similar_indices = self._find_similar_prices(current_price, future_prices)
                if future_similar_indices and future_similar_indices[0] >= self.min_position_duration:
                    self.target_index = current_index + future_similar_indices[0]
                else:
                    # Did not find suitable moment - set temporary target index
                    self.target_index = current_index + 20

            # Check if we reached the target moment
            if current_index >= self.target_index:
                # Reached target moment - find next moment with similar price
                future_similar_indices = self._find_similar_prices(self.position_open_price, future_prices)

                if future_similar_indices and future_similar_indices[0] >= self.min_position_duration:
                    self.target_index = current_index + future_similar_indices[0]
                    return ModelPrediction(r=1.0, c=decision_history[-1].c, w=TAU)
                else:
                    self.target_index = None
                    self.position_open_price = None
                    self.position_open_index = None
                    return ModelPrediction(r=0.0, c=current_price, w=TAU)
            else:
                # Have not reached target moment yet - hold position
                # Check if price is outside range
                lower_bound, upper_bound = current_entity.internal_state.price_lower, current_entity.internal_state.price_upper
                price_outside_range = current_price < lower_bound or current_price > upper_bound

                if price_outside_range and False:
                    optimal_center, optimal_width = self._calculate_balanced_center(
                        current_price, future_prices, TAU)
                    return ModelPrediction(r=1.0, c=optimal_center, w=optimal_width)
                else:
                    return ModelPrediction(r=1.0, c=decision_history[-1].c, w=TAU)

    def _get_future_prices(self, current_index, window_size):
        """
        Extracts future prices from complete trading history.
        """
        if self.history is None or len(self.history) <= current_index + 1:
            return None

        future_prices = []
        max_index = min(current_index + window_size, len(self.history) - 1)

        for i in range(current_index + 1, max_index + 1):
            if i < len(self.history):
                future_prices.append(self.history[i].global_state.price)

        return future_prices if future_prices else None

    def _find_similar_prices(self, reference_price, prices, threshold=None):
        """
        Finds indices of prices that are within a given threshold of the reference price.
        """
        if threshold is None:
            threshold = self.price_similarity_threshold

        similar_indices = []

        for i, price in enumerate(prices):
            price_diff = abs(price - reference_price) / reference_price
            if price_diff < threshold:
                similar_indices.append(i)

        return similar_indices

    def _calculate_balanced_center(self, current_price, future_prices, TAU, tick_spacing=60):
        """
        Calculates optimal center and width of position based on future prices,
        considering possible asymmetry in price movement up and down.

        Args:
            current_price: Current price
            future_prices: List of future prices
            TAU: Base parameter for range width
            tick_spacing: Tick step (default 60)

        Returns:
            Tuple[float, float]: (optimal_center, optimal_width) or (None, None),
                                 if it's impossible to create a suitable range
        """
        if not future_prices or len(future_prices) < 2:
            return None, None

        similar_indices = self._find_similar_prices(current_price, future_prices)

        if not similar_indices:
            target_index = len(future_prices) - 1
        else:
            target_index = max(similar_indices)

        relevant_prices = future_prices[:target_index + 1]

        if not relevant_prices:
            return None, None

        lower_bound = min(relevant_prices)
        upper_bound = max(relevant_prices)

        # Adapt range so current price is inside
        if current_price < lower_bound:
            lower_bound = current_price * 0.95
        if current_price > upper_bound:
            upper_bound = current_price * 1.05

        # Calculate range center as geometric mean of boundaries
        # (optimal for logarithmic scales, as in Uniswap V3)
        center = np.sqrt(lower_bound * upper_bound)

        width_lower = abs(np.log(lower_bound / center) / (np.log(1.0001) * tick_spacing))
        width_upper = abs(np.log(upper_bound / center) / (np.log(1.0001) * tick_spacing))

        width = np.ceil(max(width_lower, width_upper))

        # Check if range is too wide
        max_width = TAU * 2  # Maximum width
        if width > max_width:
            # Instead of simply limiting width, try to shift center for better coverage
            # First find average price and take it as center
            avg_price = np.mean(relevant_prices)

            temp_lower = avg_price * (1.0001 ** (-max_width * tick_spacing))
            temp_upper = avg_price * (1.0001 ** (max_width * tick_spacing))

            if current_price >= temp_lower and current_price <= temp_upper:
                center = avg_price
                width = max_width
            else:
                if current_price < temp_lower:
                    shift_factor = current_price / temp_lower
                    center = avg_price * shift_factor
                elif current_price > temp_upper:
                    shift_factor = current_price / temp_upper
                    center = avg_price * shift_factor

                new_lower = center * (1.0001 ** (-max_width * tick_spacing))
                new_upper = center * (1.0001 ** (max_width * tick_spacing))

                if new_lower > lower_bound or new_upper < upper_bound:
                    center = current_price
                    width = max_width
                else:
                    width = max_width

        final_lower = center * (1.0001 ** (-width * tick_spacing))
        final_upper = center * (1.0001 ** (width * tick_spacing))

        if current_price < final_lower or current_price > final_upper:
            return None, None

        return center, width