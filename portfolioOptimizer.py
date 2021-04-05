import numpy as np
import datetime as dt
import pandas_datareader.data as web
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
tickers = ['BAC', 'TSLA','EA', 'ATVI', 'GS', 'AMZN', 'KO']
start = dt.datetime(2014, 1, 1)
end = dt.datetime(2020, 11, 20)
number_of_portfolios = 5000
risk_free_rate = 0

returns = pd.DataFrame()
for ticker in tickers:
  data = web.DataReader(ticker, 'yahoo', start, end)
  data = pd.DataFrame(data)
  data[ticker] = data['Adj Close'].pct_change()
  if returns.empty:
    returns = data[[ticker]]
  else:
    returns = returns.join(data[[ticker]], how = 'outer')
portfolio_returns = []
portfolio_risks = []
sharpe_ratios = []
portfolio_weights = []

for portfolio in range(number_of_portfolios):
  weights = np.random.random_sample(len(tickers))
  weights = np.round(weights/np.sum(weights), 4)
  portfolio_weights.append(weights)
  annualized_return = np.sum(returns.mean()*weights)*252
  portfolio_returns.append(annualized_return)
  matrix_covariance = returns.cov()*252
  portfolio_variance = np.dot(weights.T, np.dot(matrix_covariance, weights))
  portfolio_standard_deviation = np.sqrt(portfolio_variance)
  portfolio_risks.append(portfolio_standard_deviation)
  sharpe_ratio = (annualized_return-risk_free_rate)/portfolio_standard_deviation
  sharpe_ratios.append(sharpe_ratio)

portfolio_returns = np.array(portfolio_returns)
portfolio_risks = np.array(portfolio_risks)
sharpe_ratios = np.array(sharpe_ratios)
portfolio_metrics = [portfolio_returns, portfolio_risks, sharpe_ratios, portfolio_weights]

portfolios_df = pd.DataFrame(portfolio_metrics).T
portfolios_df.columns = ['Return', 'Risk', 'Sharpe', 'Weights']

min_risk = portfolios_df.iloc[portfolios_df['Risk'].astype(float).idxmin()]
max_return = portfolios_df.iloc[portfolios_df['Return'].astype(float).idxmax()]
max_sharpe = portfolios_df.iloc[portfolios_df['Sharpe'].astype(float).idxmax()]
print('Lowest Risk')
print(min_risk)
print(tickers)
print('')

print('Highest Return')
print(max_return)
print(tickers)
print('')

print('Highest sharpe')
print(max_sharpe)
print(tickers)
print('')

plt.figure(figsize=(10,5))
plt.scatter(portfolio_risks, portfolio_returns, c= (portfolio_returns/portfolio_risks))
plt.title('Portfolio Optimization', fontsize=20)
plt.xlabel('Volatility', fontsize=20)
plt.ylabel('Return', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.colorbar(label='Sharpe Ratio')
