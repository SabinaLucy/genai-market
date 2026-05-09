from __future__ import annotations

import numpy  as np
import pandas as pd
import yfinance as yf


TRANSACTION_COST = 0.0005   # 0.05% round-trip per switch
HYSTERESIS_IN    = 2        # consecutive CRISIS signals to enter cash
HYSTERESIS_OUT   = 3        # consecutive non-CRISIS signals to exit cash
RISK_FREE_RATE   = 0.04     # annualised, used for Sharpe ratio


def download_spy(start: str, end: str) -> pd.DataFrame:
    """
    Download SPY daily close prices via yfinance for the given date range.

    Returns DataFrame with columns: spy_close, spy_return.
    Timezone info is stripped for consistent index alignment.
    """
    raw = yf.download('SPY', start=start, end=end, progress=False)
    spy = raw[['Close']].copy()
    spy.columns = ['spy_close']
    spy.index   = pd.to_datetime(spy.index)
    if spy.index.tz is not None:
        spy.index = spy.index.tz_localize(None)
    spy['spy_return'] = spy['spy_close'].pct_change()
    return spy


def run_naive_backtest(
    spy_returns   : pd.Series,
    crisis_signal : pd.Series,
    tx_cost       : float = TRANSACTION_COST,
) -> tuple[pd.DataFrame, int]:
    """
    Naive regime backtest: switch to cash on any single CRISIS signal,
    return to SPY on the next non-CRISIS day.

    A 0.05% round-trip transaction cost is applied on every switching day.

    Parameters
    ----------
    spy_returns   : daily SPY returns aligned to trading calendar
    crisis_signal : pd.Series of regime labels indexed by date
    tx_cost       : round-trip transaction cost per switch

    Returns
    -------
    (result_df, n_trades)
        result_df columns: spy_return, strategy_return, in_cash,
                           spy_cum, strategy_cum
    """
    in_cash    = False
    strat_rets = []
    n_trades   = 0

    for date, spy_ret in spy_returns.items():
        if pd.isna(spy_ret):
            strat_rets.append(np.nan)
            continue

        regime    = crisis_signal.get(date, 'LOW')
        is_crisis = regime == 'CRISIS'

        if is_crisis and not in_cash:
            in_cash   = True
            n_trades += 1
            ret       = -tx_cost
        elif not is_crisis and in_cash:
            in_cash   = False
            n_trades += 1
            ret       = spy_ret - tx_cost
        elif in_cash:
            ret = 0.0
        else:
            ret = spy_ret

        strat_rets.append(ret)

    result = pd.DataFrame({
        'spy_return'     : spy_returns.values,
        'strategy_return': strat_rets,
        'in_cash'        : [(crisis_signal.get(d, 'LOW') == 'CRISIS')
                            for d in spy_returns.index],
    }, index=spy_returns.index).dropna()

    result['spy_cum']      = (1 + result['spy_return']).cumprod()
    result['strategy_cum'] = (1 + result['strategy_return']).cumprod()

    return result, n_trades


def run_hysteresis_backtest(
    spy_returns   : pd.Series,
    crisis_signal : pd.Series,
    n_in          : int   = HYSTERESIS_IN,
    n_out         : int   = HYSTERESIS_OUT,
    tx_cost       : float = TRANSACTION_COST,
) -> tuple[pd.DataFrame, int]:
    """
    Hysteresis regime backtest: requires n_in consecutive CRISIS signals
    before switching to cash, and n_out consecutive non-CRISIS signals
    before returning to SPY.

    Eliminates whipsaw trading caused by single noisy regime predictions.
    A 0.05% round-trip transaction cost is applied on every switching day.

    Parameters
    ----------
    spy_returns   : daily SPY returns aligned to trading calendar
    crisis_signal : pd.Series of regime labels indexed by date
    n_in          : consecutive CRISIS signals required to switch to cash
    n_out         : consecutive non-CRISIS signals required to return to SPY
    tx_cost       : round-trip transaction cost per switch

    Returns
    -------
    (result_df, n_trades)
        result_df columns: spy_return, strategy_return, position,
                           spy_cum, strategy_cum
    """
    in_cash       = False
    crisis_streak = 0
    clear_streak  = 0
    strat_rets    = []
    position_log  = []
    n_trades      = 0

    for date, spy_ret in spy_returns.items():
        if pd.isna(spy_ret):
            strat_rets.append(np.nan)
            position_log.append(np.nan)
            continue

        regime    = crisis_signal.get(date, 'LOW')
        is_crisis = regime == 'CRISIS'

        if is_crisis:
            crisis_streak += 1
            clear_streak   = 0
        else:
            clear_streak  += 1
            crisis_streak  = 0

        if not in_cash and crisis_streak >= n_in:
            in_cash   = True
            n_trades += 1
            ret       = -tx_cost
        elif in_cash and clear_streak >= n_out:
            in_cash   = False
            n_trades += 1
            ret       = spy_ret - tx_cost
        elif in_cash:
            ret = 0.0
        else:
            ret = spy_ret

        strat_rets.append(ret)
        position_log.append(0 if in_cash else 1)

    result = pd.DataFrame({
        'spy_return'     : spy_returns.values,
        'strategy_return': strat_rets,
        'position'       : position_log,
    }, index=spy_returns.index).dropna()

    result['spy_cum']      = (1 + result['spy_return']).cumprod()
    result['strategy_cum'] = (1 + result['strategy_return']).cumprod()

    return result, n_trades


def compute_backtest_stats(
    df           : pd.DataFrame,
    ret_col      : str,
    cum_col      : str,
    rf_annual    : float = RISK_FREE_RATE,
    trading_days : int   = 252,
) -> dict:
    """
    Compute annualised performance statistics for a backtest result DataFrame.

    Metrics: total return, annualised return, annualised volatility,
    Sharpe ratio (excess return over risk-free rate), maximum drawdown.

    Parameters
    ----------
    df           : backtest result DataFrame from run_naive_backtest or
                   run_hysteresis_backtest
    ret_col      : column name for daily returns
    cum_col      : column name for cumulative wealth
    rf_annual    : annualised risk-free rate
    trading_days : trading days per year for annualisation

    Returns
    -------
    dict with keys: total_return, ann_return, ann_volatility,
                    sharpe_ratio, max_drawdown
    """
    rets     = df[ret_col].dropna()
    rf_daily = (1 + rf_annual) ** (1 / trading_days) - 1

    excess  = rets - rf_daily
    sharpe  = float(excess.mean() / (excess.std() + 1e-10) * np.sqrt(trading_days))

    cum     = df[cum_col].dropna()
    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / roll_max
    max_dd   = float(drawdown.min())

    total_ret = float(cum.iloc[-1] - 1)
    ann_ret   = float(cum.iloc[-1] ** (trading_days / len(rets)) - 1)
    ann_vol   = float(rets.std() * np.sqrt(trading_days))

    return {
        'total_return'  : total_ret,
        'ann_return'    : ann_ret,
        'ann_volatility': ann_vol,
        'sharpe_ratio'  : sharpe,
        'max_drawdown'  : max_dd,
    }


def run_full_backtest(
    test_df       : 'pd.DataFrame',
    regime_signal : pd.Series,
    val_end       : str,
    tx_cost       : float = TRANSACTION_COST,
    n_in          : int   = HYSTERESIS_IN,
    n_out         : int   = HYSTERESIS_OUT,
    rf_annual     : float = RISK_FREE_RATE,
) -> dict:
    """
    End-to-end backtest pipeline called by Phase 9 API at startup.

    Downloads SPY, aligns with regime signal, runs both naive and
    hysteresis strategies, and returns all stats in a single dict.

    Parameters
    ----------
    test_df       : test split DataFrame with DatetimeIndex
    regime_signal : pd.Series of regime labels from XGBoost classifier
    val_end       : validation end date string (e.g. '2021-12-31')
    tx_cost       : round-trip transaction cost
    n_in          : hysteresis entry threshold
    n_out         : hysteresis exit threshold
    rf_annual     : annualised risk-free rate

    Returns
    -------
    dict with keys: spy_stats, naive_stats, hyst_stats,
                    naive_trades, hyst_trades, naive_df, hyst_df
    """
    test_start = (pd.Timestamp(val_end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    test_end   = test_df.index[-1].strftime('%Y-%m-%d')

    spy          = download_spy(test_start, test_end)
    common_dates = spy.index.intersection(regime_signal.index)
    spy_aligned  = spy.loc[common_dates]
    sig_aligned  = regime_signal.loc[common_dates]

    naive_df, naive_trades = run_naive_backtest(
        spy_aligned['spy_return'], sig_aligned, tx_cost
    )
    hyst_df, hyst_trades = run_hysteresis_backtest(
        spy_aligned['spy_return'], sig_aligned, n_in, n_out, tx_cost
    )

    spy_stats   = compute_backtest_stats(naive_df, 'spy_return',      'spy_cum',      rf_annual)
    naive_stats = compute_backtest_stats(naive_df, 'strategy_return', 'strategy_cum', rf_annual)
    hyst_stats  = compute_backtest_stats(hyst_df,  'strategy_return', 'strategy_cum', rf_annual)

    return {
        'spy_stats'   : spy_stats,
        'naive_stats' : naive_stats,
        'hyst_stats'  : hyst_stats,
        'naive_trades': naive_trades,
        'hyst_trades' : hyst_trades,
        'naive_df'    : naive_df,
        'hyst_df'     : hyst_df,
    }
