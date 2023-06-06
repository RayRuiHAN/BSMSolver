# BSMSolver
The BSMSolver class is designed to calculate the price, Greeks (Delta, Gamma, Vega, Theta, Rho), and implied volatility of European options using the Black-Scholes-Merton (BSM) model.

* asset_price is the current price of the underlying asset.
* strike_price is the strike price of the option.
* time_to_maturity is the time to the option's expiry, in years.
* call_put is a flag to specify the option type. Use 1 for a call option, and 0 for a put option.
* volatility is the annualized standard deviation of the asset's return.
* risk_free_rate is the risk-free rate (e.g. the return on a government bond).

After instantiating the BSMSolver class, you can get the option's price and Greeks by accessing the greeks attribute. For example, bsm.greeks['price'] gives the option price, and bsm.greeks['delta'] gives the Delta of the option.

You can calculate the implied volatility using a market price with the implied_volatility method. For example, bsm.implied_volatility(target_price=5) calculates the implied volatility assuming the option's market price is 5.

Run the following test code
```python
if __name__ == '__main__':
    bsm = BSMSolver(asset_price=4815, strike_price=4500, time_to_maturity=0.0877, risk_free_rate=0, volatility=0.26, call_put=1)

    data = {
        'd1': [bsm.d1],
        'd2': [bsm.d2],
        'Price': [bsm.greeks['price']],
        'Delta': [bsm.greeks['delta']],
        'Gamma': [bsm.greeks['gamma']],
        'Theta': [bsm.greeks['theta']],
        'Vega': [bsm.greeks['vega']],
        'Rho': [bsm.greeks['rho']],
        'Implied Volatility': [bsm.implied_volatility(target_price=352.404034)]
    }

    df = pd.DataFrame(data)
    print(df)
```

and you should get:

|    d1    |    d2    |   Price   |   Delta   |   Gamma   |    Theta    |    Vega    |    Rho    | Implied Volatility |
|----------|----------|-----------|-----------|-----------|-------------|------------|------------|--------------------|
| 0.917218 | 0.840221 | 352.404034| 0.820486  | 0.000707  | -553.689713 | 373.527598 | 315.565187 |        0.26        |
