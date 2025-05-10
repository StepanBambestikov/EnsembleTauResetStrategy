import pandas as pd
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_history_with_pnl(file_name, initial_balance, model_weights):
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np
    from matplotlib.gridspec import GridSpec
    import matplotlib.colors as mcolors

    df = pd.read_csv(file_name)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # When there's no LP position at the beginning, net_balance values are zero, which isn't accurate
    if df.loc[0, 'net_balance'] < 0.01:
        i = 0
        while df.loc[i, 'net_balance'] < 0.01 and i < len(df) - 1:
            df.loc[i, 'net_balance'] = initial_balance
            i += 1

    df['pnl_percent'] = (df['net_balance'] / initial_balance - 1) * 100

    # Create figure with GridSpec for 3 subplots (price+ranges, weights, pnl)
    fig = plt.figure(figsize=(15, 14))
    gs = GridSpec(3, 1, height_ratios=[2, 1, 1])

    # First subplot: Price and liquidity ranges
    ax1 = fig.add_subplot(gs[0])

    ax1.plot(df['timestamp'], df['UNISWAP_V3_price'], label='Price', color='blue', linewidth=2)

    mask = (df['UNISWAP_V3_price_lower'] > 0) & (df['UNISWAP_V3_price_upper'] > 0)

    unique_ranges = []
    last_lower = None
    last_upper = None
    range_start_idx = None

    for i, row in df.iterrows():
        if mask[i]:
            if last_lower != row['UNISWAP_V3_price_lower'] or last_upper != row['UNISWAP_V3_price_upper']:
                if range_start_idx is not None:
                    unique_ranges.append({
                        'start_idx': range_start_idx,
                        'end_idx': i - 1,
                        'lower': last_lower,
                        'upper': last_upper
                    })

                range_start_idx = i
                last_lower = row['UNISWAP_V3_price_lower']
                last_upper = row['UNISWAP_V3_price_upper']
        if not mask[i]:
            if range_start_idx is not None:
                unique_ranges.append({
                    'start_idx': range_start_idx,
                    'end_idx': i - 1,
                    'lower': last_lower,
                    'upper': last_upper
                })
                range_start_idx = None

    if range_start_idx is not None:
        unique_ranges.append({
            'start_idx': range_start_idx,
            'end_idx': df[mask].index[-1],
            'lower': last_lower,
            'upper': last_upper
        })

    for range_info in unique_ranges:
        start_time = df.loc[range_info['start_idx'], 'timestamp']
        end_time = df.loc[range_info['end_idx'], 'timestamp']
        lower_price = range_info['lower']
        upper_price = range_info['upper']

        ax1.plot([start_time, start_time], [lower_price, upper_price],
                 color='purple', linewidth=2)

        ax1.plot([end_time, end_time], [lower_price, upper_price],
                 color='purple', linewidth=2)

        ax1.plot([start_time, end_time], [lower_price, lower_price],
                 color='green', linewidth=1.5, linestyle='-')
        ax1.plot([start_time, end_time], [upper_price, upper_price],
                 color='red', linewidth=1.5, linestyle='-')

        ax1.fill_between([start_time, end_time],
                         [lower_price, lower_price],
                         [upper_price, upper_price],
                         color='yellow', alpha=0.2)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax1.set_xticklabels([])

    ax1.grid(True, linestyle='--', alpha=0.7)

    ax1.legend(['Price', 'Range boundaries'])
    ax1.set_title('Price dynamics and liquidity ranges')
    ax1.set_ylabel('Price')

    # Second subplot: Model weights
    ax3 = fig.add_subplot(gs[1], sharex=ax1)

    # Process model weights based on the structure:
    # model_weights[model_name].history contains weight history
    if model_weights is not None:
        # Extract model names and their weight histories
        weight_data = {}
        max_history_length = 0

        # Extract weight history for each model
        for model_name, model_weight in model_weights.items():
            if hasattr(model_weight, 'history'):
                weight_data[model_name] = model_weight.history
                max_history_length = max(max_history_length, len(model_weight.history))

        # Create a DataFrame with weight histories
        if weight_data and max_history_length > 0:
            # Create evenly spaced timestamps across the trading period
            # If trading period is shorter than weight history, use all timestamps
            if len(df) < max_history_length:
                timestamps = df['timestamp'].tolist()
                # Extend timestamps if needed
                last_timestamp = timestamps[-1]
                time_delta = (timestamps[-1] - timestamps[-2]) if len(timestamps) > 1 else pd.Timedelta(minutes=1)
                for i in range(len(df), max_history_length):
                    last_timestamp = last_timestamp + time_delta
                    timestamps.append(last_timestamp)
            else:
                # Sample timestamps evenly across trading period
                total_period = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
                time_step = total_period / (max_history_length - 1) if max_history_length > 1 else pd.Timedelta(
                    minutes=1)
                timestamps = [df['timestamp'].iloc[0] + time_step * i for i in range(max_history_length)]

            # Create DataFrame with timestamps and weight histories
            weights_df = pd.DataFrame({'timestamp': timestamps[:max_history_length]})

            # Add weight history for each model
            for model_name, history in weight_data.items():
                # Pad or truncate history as needed
                if len(history) < max_history_length:
                    # Pad with last value
                    padded_history = history + [history[-1]] * (max_history_length - len(history))
                    weights_df[model_name] = padded_history
                else:
                    # Use history up to max_history_length
                    weights_df[model_name] = history[:max_history_length]

            # Get weight column names (excluding timestamp)
            weight_cols = [col for col in weights_df.columns if col != 'timestamp']

            # Generate colors for each model
            colors = list(mcolors.TABLEAU_COLORS.values())[:len(weight_cols)]
            if len(weight_cols) > len(colors):
                # If more models than colors, cycle through colors
                colors = colors * (len(weight_cols) // len(colors) + 1)

            # Plot each model's weight
            for i, col in enumerate(weight_cols):
                ax3.plot(weights_df['timestamp'], weights_df[col],
                         label=f"{col}", color=colors[i], linewidth=2)

                # Add markers at first and last points
                ax3.plot(weights_df['timestamp'].iloc[0], weights_df[col].iloc[0],
                         marker='o', color=colors[i], markersize=6)
                ax3.plot(weights_df['timestamp'].iloc[-1], weights_df[col].iloc[-1],
                         marker='s', color=colors[i], markersize=6)

            # Add horizontal line at 0.5 as reference
            ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

            # Add text annotations for final weights
            for i, col in enumerate(weight_cols):
                ax3.annotate(f"{col}: {weights_df[col].iloc[-1]:.3f}",
                             xy=(weights_df['timestamp'].iloc[-1], weights_df[col].iloc[-1]),
                             xytext=(10, 0), textcoords='offset points',
                             color=colors[i], fontweight='bold')

            min_weight = min([weights_df[col].min() for col in weight_cols])
            max_weight = max([weights_df[col].max() for col in weight_cols])
            padding = (max_weight - min_weight) * 0.1  # 10% отступ
            ax3.set_ylim(min_weight - padding, max_weight + padding)

            #ax3.set_ylim(0, 1.05)  # Set y limits for weights (typically 0-1)
            ax3.legend(loc='upper right', title='Model Weights')
            ax3.set_ylabel('Weight Value')
            ax3.grid(True, linestyle='--', alpha=0.7)
            ax3.set_xticklabels([])
        else:
            ax3.text(0.5, 0.5, 'No weight history available',
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, 'No weight data available',
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax3.transAxes)

    # Third subplot: PnL percentage
    ax2 = fig.add_subplot(gs[2], sharex=ax1)

    ax2.plot(df['timestamp'], df['pnl_percent'], label='PnL (%)', color='green', linewidth=2)
    ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)

    ax2.fill_between(df['timestamp'], df['pnl_percent'], 0,
                     where=(df['pnl_percent'] >= 0), color='green', alpha=0.3)
    ax2.fill_between(df['timestamp'], df['pnl_percent'], 0,
                     where=(df['pnl_percent'] < 0), color='red', alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    plt.xticks(rotation=45)

    ax2.grid(True, linestyle='--', alpha=0.7)

    ax2.legend(['PnL (%)'])
    ax2.set_ylabel('Balance change (%)')
    ax2.set_xlabel('Time')

    final_pnl = df['pnl_percent'].iloc[-1]
    ax2.annotate(f'Final PnL: {final_pnl:.2f}%',
                 xy=(df['timestamp'].iloc[-1], final_pnl),
                 xytext=(df['timestamp'].iloc[-1], final_pnl + 0.5),
                 fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.7))

    max_drawdown = (df['net_balance'].min() / initial_balance - 1) * 100
    max_profit = (df['net_balance'].max() / initial_balance - 1) * 100

    textstr = f'Max drawdown: {max_drawdown:.2f}%\nMax profit: {max_profit:.2f}%'
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax2.text(0.02, 0.05, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='bottom', bbox=props)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)

    plt.savefig('price_liquidity_weights_and_pnl.png', dpi=300)

    plt.show()

    print("Profitability statistics:")
    print(f"Initial balance: {initial_balance:.2f}")
    print(f"Final balance: {df['net_balance'].iloc[-1]:.2f}")
    print(f"Change: {final_pnl:.2f}%")
    print(f"Maximum drawdown: {max_drawdown:.2f}%")
    print(f"Maximum profit: {max_profit:.2f}%")
    print(f"Balance volatility: {df['pnl_percent'].std():.2f}%")

    if 'UNISWAP_V3_fees' in df.columns:
        total_fees = df['UNISWAP_V3_fees'].iloc[-1] - df['UNISWAP_V3_fees'].iloc[0]
        print(f"Total fees earned: {total_fees:.2f}")
