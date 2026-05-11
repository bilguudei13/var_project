import pandas as pd
import os

pnl = pd.read_csv('data/processed/total_portfolio_pnl.csv', index_col=0, parse_dates=True)
var = pd.read_csv('outputs/tables/backtest_copula.csv', index_col=0, parse_dates=True)

df = pnl.join(var, how='inner')
df['loss'] = -df['pnl_total']
df['new_exception'] = (df['loss'] > df['VaR']).astype(int)

summary = df.groupby(df.index.year).agg(
    observations=('new_exception', 'count'),
    actual_exceptions=('new_exception', 'sum'),
    old_exceptions=('exception', 'sum')
)
summary['actual_exception_rate'] = summary['actual_exceptions'] / summary['observations']

overall = pd.DataFrame({
    'observations': [df['new_exception'].count()],
    'actual_exceptions': [df['new_exception'].sum()],
    'old_exceptions': [df['exception'].sum()],
    'actual_exception_rate': [df['new_exception'].mean()]
}, index=['Total'])

final_summary = pd.concat([summary, overall])

out_path = 'outputs/tables/actual_pnl_vs_var_exceptions.csv'
final_summary.to_csv(out_path)

print('Total observations:', overall['observations'][0])
print('Total actual exceptions:', overall['actual_exceptions'][0])
print('Total old exceptions:', overall['old_exceptions'][0])
print(f'Actual Exception rate: {overall["actual_exception_rate"][0]:.2%}')
print('File created at:', out_path)
