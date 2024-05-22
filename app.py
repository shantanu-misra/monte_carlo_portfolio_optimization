import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from flask import Flask, render_template, request, send_file, session
import io
import pickle
import os
from flask_session import Session

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configure server-side session
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './.flask_session/'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
Session(app)


def get_stock_data(stock_symbols, start_date, end_date):
    """
    Fetches historical stock data for the given symbols and date range.

    Args:
        stock_symbols (list): List of stock symbols.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        DataFrame: Adjusted close prices of the stocks.
    """
    data = {}
    for symbol in stock_symbols:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        data[symbol] = stock_data['Adj Close']
    return pd.DataFrame(data)


def calculate_statistics(data):
    """
    Calculates annualized returns, total returns, standard deviations, and correlation matrix.

    Args:
        data (DataFrame): DataFrame containing historical stock prices.

    Returns:
        tuple: Annualized returns, total returns, standard deviations, and correlation matrix.
    """
    daily_returns = data.pct_change().dropna()
    annualized_returns = daily_returns.mean() * 252
    total_returns = (data.iloc[-1] / data.iloc[0]) - 1
    std_devs = daily_returns.std() * np.sqrt(252)
    correlation_matrix = daily_returns.corr()
    return annualized_returns, total_returns, std_devs, correlation_matrix


def monte_carlo_simulation(returns, std_devs, correlation_matrix, num_portfolios=10000, min_stocks=2):
    """
    Performs Monte Carlo simulation to generate portfolio weights, returns, risks, Sharpe ratios, and VaR.

    Args:
        returns (Series): Annualized returns of the stocks.
        std_devs (Series): Annualized standard deviations of the stocks.
        correlation_matrix (DataFrame): Correlation matrix of the stock returns.
        num_portfolios (int): Number of portfolios to simulate.
        min_stocks (int): Minimum number of stocks in a portfolio.

    Returns:
        list: List of tuples containing portfolio weights, returns, risks, Sharpe ratios, and VaR.
    """
    results = []
    num_stocks = len(returns)
    risk_free_rate = 0.01
    for _ in range(num_portfolios):
        num_active_stocks = np.random.randint(min_stocks, num_stocks + 1)
        active_stocks = np.random.choice(num_stocks, num_active_stocks, replace=False)
        weights = np.zeros(num_stocks)
        random_weights = np.random.random(num_active_stocks)
        random_weights /= np.sum(random_weights)
        weights[active_stocks] = random_weights
        portfolio_return = np.dot(weights, returns)
        portfolio_std_dev = np.sqrt(
            np.dot(weights.T, np.dot(correlation_matrix * np.outer(std_devs, std_devs), weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev
        var_95 = np.percentile(portfolio_return, 5)
        results.append((weights, portfolio_return, portfolio_std_dev, sharpe_ratio, var_95))
    return results


def optimize_portfolios(results, stock_symbols, total_returns):
    """
    Identifies the minimum risk and maximum return portfolios from the simulation results.

    Args:
        results (list): List of tuples containing portfolio weights, returns, risks, Sharpe ratios, and VaR.
        stock_symbols (list): List of stock symbols.
        total_returns (Series): Total returns of the stocks over the period.

    Returns:
        tuple: DataFrame of all portfolios, minimum risk portfolio, and maximum return portfolio.
    """
    portfolios = pd.DataFrame(results, columns=['Weights', 'Return', 'Risk', 'SharpeRatio', 'VaR_95'])
    min_risk_portfolio = portfolios.loc[portfolios['Risk'].idxmin()]
    max_return_portfolio = portfolios.loc[portfolios['Return'].idxmax()]

    min_risk_total_return = np.dot(min_risk_portfolio['Weights'], total_returns)
    max_return_total_return = np.dot(max_return_portfolio['Weights'], total_returns)

    min_risk_weights = {stock: f"{weight:.2%}" for stock, weight in zip(stock_symbols, min_risk_portfolio['Weights'])}
    max_return_weights = {stock: f"{weight:.2%}" for stock, weight in
                          zip(stock_symbols, max_return_portfolio['Weights'])}

    min_risk_portfolio_display = {
        "Stocks": min_risk_weights,
        "TotalReturn": f"{min_risk_total_return:.2%}",
        "Return": f"{min_risk_portfolio['Return']:.2%} (annualized)",
        "Risk": f"{min_risk_portfolio['Risk']:.2%}",
        "SharpeRatio": f"{min_risk_portfolio['SharpeRatio']:.2f}",
        "VaR_95": f"{min_risk_portfolio['VaR_95']:.2%}"
    }
    max_return_portfolio_display = {
        "Stocks": max_return_weights,
        "TotalReturn": f"{max_return_total_return:.2%}",
        "Return": f"{max_return_portfolio['Return']:.2%} (annualized)",
        "Risk": f"{max_return_portfolio['Risk']:.2%}",
        "SharpeRatio": f"{max_return_portfolio['SharpeRatio']:.2f}",
        "VaR_95": f"{max_return_portfolio['VaR_95']:.2%}"
    }

    return portfolios, min_risk_portfolio_display, max_return_portfolio_display


def plot_efficient_frontier(portfolios):
    """
    Plots the efficient frontier for the portfolios.

    Args:
        portfolios (DataFrame): DataFrame containing portfolio data.

    Returns:
        BytesIO: Buffer containing the plot image.
    """
    plt.figure()
    plt.scatter(portfolios['Risk'], portfolios['Return'], c=portfolios['SharpeRatio'], marker='o')
    plt.title('Efficient Frontier')
    plt.xlabel('Risk')
    plt.ylabel('Return')
    plt.colorbar(label='Sharpe ratio')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock_symbols = request.form['stocks'].split()
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        data = get_stock_data(stock_symbols, start_date, end_date)
        annualized_returns, total_returns, std_devs, correlation_matrix = calculate_statistics(data)
        num_portfolios = 100000
        min_stocks = 5
        results = monte_carlo_simulation(annualized_returns, std_devs, correlation_matrix,
                                         num_portfolios=num_portfolios, min_stocks=min_stocks)
        portfolios, min_risk_portfolio_display, max_return_portfolio_display = optimize_portfolios(results,
                                                                                                   stock_symbols,
                                                                                                   total_returns)

        session['portfolios'] = pickle.dumps(portfolios)
        session['min_risk_portfolio_display'] = min_risk_portfolio_display
        session['max_return_portfolio_display'] = max_return_portfolio_display
        session['metadata'] = {
            "num_portfolios": num_portfolios,
            "min_stocks": min_stocks,
            "num_stocks": len(stock_symbols)
        }

        return render_template('results.html',
                               min_risk=min_risk_portfolio_display,
                               max_return=max_return_portfolio_display,
                               metadata=session['metadata'],
                               plot_url='/plot.png')

    return render_template('index.html')


@app.route('/plot.png')
def plot_png():
    portfolios = pickle.loads(session['portfolios'])
    buf = plot_efficient_frontier(portfolios)
    return send_file(buf, mimetype='image/png')


if __name__ == '__main__':
    if not os.path.exists(app.config['SESSION_FILE_DIR']):
        os.makedirs(app.config['SESSION_FILE_DIR'])
    app.run(debug=True)