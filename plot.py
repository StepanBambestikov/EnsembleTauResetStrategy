import pandas as pd
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_history_with_pnl(file_name, initial_balance):
    df = pd.read_csv(file_name)

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # When there's no LP position at the beginning, net_balance values are zero, which isn't accurate
    if df.loc[0, 'net_balance'] < 0.01:
        i = 0
        while df.loc[i, 'net_balance'] < 0.01 and i < len(df) - 1:
            df.loc[i, 'net_balance'] = initial_balance
            i += 1

    df['pnl_percent'] = (df['net_balance'] / initial_balance - 1) * 100

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 1, height_ratios=[2, 1])

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

    ax2 = fig.add_subplot(gs[1], sharex=ax1)

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

    plt.savefig('price_liquidity_and_pnl.png', dpi=300)

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