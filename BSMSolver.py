import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Optional

class BSMSolver:
    def __init__(self, 
                 asset_price: float, 
                 strike_price: float, 
                 time_to_maturity: float, 
                 risk_free_rate: float, 
                 volatility: Optional[float]=0.5, 
                 call_put: Optional[int]=1) -> None:
        self.asset_price = asset_price
        self.strike_price = strike_price
        self.time_to_maturity = time_to_maturity
        self.volatility = volatility
        self.risk_free_rate = risk_free_rate
        self.call_put = call_put

        self.d1 = self.calculate_d1()
        self.d2 = self.calculate_d2()
        self.greeks = self.calculate_greeks()

    def calculate_d1(self) -> float:
        return (np.log(self.asset_price / self.strike_price) + 
                (self.risk_free_rate + 0.5 * self.volatility ** 2) * self.time_to_maturity) / (
                self.volatility * np.sqrt(self.time_to_maturity))

    def calculate_d2(self) -> float:
        return self.d1 - self.volatility * np.sqrt(self.time_to_maturity)

    def calculate_greeks(self) -> dict:
        g = {}
        ttm = self.time_to_maturity
        sr_ttm = np.sqrt(ttm)
        ap = self.asset_price
        sp = self.strike_price
        rf = self.risk_free_rate
        vol = self.volatility

        d1 = self.d1
        d2 = self.d2

        p1 = norm.pdf(d1)
        c1 = norm.cdf(d1)
        c2 = norm.cdf(d2)

        g['gamma'] = p1 / (ap * vol * sr_ttm)
        g['vega'] = ap * p1 * sr_ttm

        if self.call_put == 1: # Call
            g['price'] = c1 * ap - c2 * sp * np.exp(-rf * ttm)
            g['delta'] = c1
            g['theta'] = -1 * (ap * p1 * vol) / (2 * sr_ttm) - rf * sp * np.exp(-rf * ttm) * c2
            g['rho'] = ttm * sp * np.exp(-rf * ttm) * c2
        else: # Put
            g['price'] = sp * np.exp(-rf * ttm) - ap + c1 * ap - c2 * sp * np.exp(-rf * ttm)
            g['delta'] = c1 - 1
            g['theta'] = -1 * (ap * p1 * vol) / (2 * sr_ttm) + rf * sp * np.exp(-rf * ttm) * (1 - c2)
            g['rho'] = -ttm * sp * np.exp(-rf * ttm) * (1 - c2)

        return g

    def implied_volatility(self, target_price: float, max_iterations: int = 100, tolerance: float = 1e-5) -> float:
        volatility = 0.5  # Initial guess
        for _ in range(max_iterations):
            self.volatility = volatility
            self.d1 = self.calculate_d1()
            self.d2 = self.calculate_d2()
            self.greeks = self.calculate_greeks()
            
            price = self.greeks['price']
            vega = self.greeks['vega']

            diff = target_price - price
            if abs(diff) < tolerance:
                return volatility
            volatility += diff / vega  # Newton-Raphson update

        return volatility  # Returns the final estimate



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
