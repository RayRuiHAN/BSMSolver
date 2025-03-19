import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Optional

class BSMSolver:
    def __init__(
        self,
        asset_price: float,
        strike_price: float,
        time_to_maturity: float,
        risk_free_rate: float,
        volatility: Optional[float] = 0.5,
        call_put: Optional[int] = 1
    ) -> None:
        """
        Black-Scholes-Merton期权定价模型求解器
        
        参数:
        asset_price (float): 标的资产当前价格
        strike_price (float): 行权价格
        time_to_maturity (float): 到期时间（年）
        risk_free_rate (float): 无风险利率
        volatility (Optional[float]): 波动率，默认为0.5
        call_put (Optional[int]): 期权类型，1表示看涨，-1表示看跌，默认为1
        """
        # 参数验证
        if time_to_maturity <= 0:
            raise ValueError("到期时间必须大于0")
        if asset_price < 0 or strike_price < 0:
            raise ValueError("资产价格和行权价格不能为负数")
        if call_put not in (1, -1):
            raise ValueError("期权类型必须为1（看涨）或-1（看跌）")

        self.asset_price = asset_price
        self.strike_price = strike_price
        self.time_to_maturity = time_to_maturity
        self._volatility = volatility
        self.risk_free_rate = risk_free_rate
        self.call_put = call_put
        self._d1: Optional[float] = None
        self._d2: Optional[float] = None
        self._greeks: Optional[dict] = None

    @property
    def volatility(self) -> float:
        """波动率属性（带缓存管理）"""
        return self._volatility

    @volatility.setter
    def volatility(self, value: float) -> None:
        """波动率设置器（自动清除缓存）"""
        self._volatility = value
        self._d1 = None
        self._d2 = None
        self._greeks = None

    @property
    def d1(self) -> float:
        """计算并返回d1值"""
        if self._d1 is None:
            self._d1 = self._calculate_d1()
        return self._d1

    @property
    def d2(self) -> float:
        """计算并返回d2值"""
        if self._d2 is None:
            self._d2 = self._calculate_d2()
        return self._d2

    @property
    def greeks(self) -> dict:
        """计算并返回所有希腊字母"""
        if self._greeks is None:
            self._greeks = self._calculate_greeks()
        return self._greeks

    def _calculate_d1(self) -> float:
        """计算d1值（内部方法）"""
        return (
            np.log(self.asset_price / self.strike_price) +
            (self.risk_free_rate + 0.5 * self.volatility ** 2) * self.time_to_maturity
        ) / (self.volatility * np.sqrt(self.time_to_maturity))

    def _calculate_d2(self) -> float:
        """计算d2值（内部方法）"""
        return self.d1 - self.volatility * np.sqrt(self.time_to_maturity)

    def _calculate_greeks(self) -> dict:
        """计算所有希腊字母（内部方法）"""
        g = {}
        sqrt_ttm = np.sqrt(self.time_to_maturity)
        discount = np.exp(-self.risk_free_rate * self.time_to_maturity)
        
        d1 = self.d1
        d2 = self.d2
        
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)

        # 公共希腊字母
        g['gamma'] = pdf_d1 / (self.asset_price * self.volatility * sqrt_ttm)
        g['vega'] = self.asset_price * pdf_d1 * sqrt_ttm

        # 期权特定计算
        if self.call_put == 1:
            # 看涨期权
            g['price'] = (
                cdf_d1 * self.asset_price -
                cdf_d2 * self.strike_price * discount
            )
            g['delta'] = cdf_d1
            g['theta'] = (
                -self.asset_price * pdf_d1 * self.volatility / (2 * sqrt_ttm) -
                self.risk_free_rate * self.strike_price * discount * cdf_d2
            )
            g['rho'] = self.time_to_maturity * self.strike_price * discount * cdf_d2
        else:
            # 看跌期权
            g['price'] = (
                self.strike_price * discount -
                self.asset_price +
                cdf_d1 * self.asset_price -
                cdf_d2 * self.strike_price * discount
            )
            g['delta'] = cdf_d1 - 1
            g['theta'] = (
                -self.asset_price * pdf_d1 * self.volatility / (2 * sqrt_ttm) +
                self.risk_free_rate * self.strike_price * discount * (1 - cdf_d2)
            )
            g['rho'] = -self.time_to_maturity * self.strike_price * discount * (1 - cdf_d2)

        return g

    def implied_volatility(
        self,
        target_price: float,
        max_iterations: int = 100,
        tolerance: float = 1e-5
    ) -> float:
        """
        使用牛顿-拉夫森方法计算隐含波动率
        
        参数:
        target_price (float): 目标期权价格
        max_iterations (int): 最大迭代次数，默认为100
        tolerance (float): 收敛容差，默认为1e-5
        
        返回:
        float: 隐含波动率估计值
        """
        volatility = 0.5  # 使用0.5作为初始猜测
        
        for _ in range(max_iterations):
            self.volatility = volatility
            price = self.greeks['price']
            vega = self.greeks['vega']

            # 处理vega接近零的情况
            if np.isclose(vega, 0):
                return volatility

            diff = target_price - price
            if abs(diff) < tolerance:
                return volatility

            # 牛顿-拉夫森更新
            volatility += diff / vega

        return volatility  # 返回最终估计值（即使未收敛）

if __name__ == '__main__':
    bsm = BSMSolver(
        asset_price=4815,
        strike_price=4500,
        time_to_maturity=0.0877,
        risk_free_rate=0,
        volatility=0.26,
        call_put=1
    )

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
