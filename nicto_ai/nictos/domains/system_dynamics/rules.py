import numpy as np
from nicto_ai.nictos.core.rules import GenericRule
from nicto_ai.nictos.core.types import GenericType, Effect


class ExponentialGrowthRule(GenericRule):
    """dX/dt = r * X"""

    def __init__(self, stock_id: int, rate: float = 0.1):
        super().__init__(name="exponential_growth")
        self.stock_id = stock_id
        self.rate = rate

    def apply(self, entities: list[GenericType], dt: float) -> list[Effect]:
        for ent in entities:
            if ent.id == self.stock_id and "value" in ent.attributes:
                delta = self.rate * ent.attributes["value"] * dt
                return [Effect(entity_id=ent.id, attr="value", delta=delta)]
        return []


class LotkaVolterraRule(GenericRule):
    """Predator–prey model:

        dprey/dt   = alpha * prey - beta * prey * pred
        dpred/dt   = delta * prey * pred - gamma * pred

    Expects two stocks with ``attributes["value"]`` = population.
    """

    def __init__(self, prey_id: int, pred_id: int, alpha: float = 1.0, beta: float = 0.1, delta: float = 0.075, gamma: float = 1.5):
        super().__init__(name="lotka_volterra")
        self.prey_id = prey_id
        self.pred_id = pred_id
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma

    def apply(self, entities: list[GenericType], dt: float) -> list[Effect]:
        prey_val = pred_val = None
        for ent in entities:
            if ent.id == self.prey_id and "value" in ent.attributes:
                prey_val = ent.attributes["value"]
            elif ent.id == self.pred_id and "value" in ent.attributes:
                pred_val = ent.attributes["value"]
        if prey_val is None or pred_val is None:
            return []
        prey_val = prey_val or 1e-10
        pred_val = pred_val or 1e-10
        d_prey = (self.alpha * prey_val - self.beta * prey_val * pred_val) * dt
        d_pred = (self.delta * prey_val * pred_val - self.gamma * pred_val) * dt
        return [
            Effect(entity_id=self.prey_id, attr="value", delta=d_prey),
            Effect(entity_id=self.pred_id, attr="value", delta=d_pred),
        ]


class SIRRule(GenericRule):
    """Susceptible–Infectious–Recovered epidemic model:

        dS/dt = -beta * S * I / N
        dI/dt =  beta * S * I / N - gamma * I
        dR/dt =  gamma * I

    Expects 3 stocks — S, I, R — each with ``attributes["value"]``.
    """

    def __init__(self, S_id: int, I_id: int, R_id: int, beta: float = 0.3, gamma: float = 0.1):
        super().__init__(name="sir")
        self.S_id = S_id
        self.I_id = I_id
        self.R_id = R_id
        self.beta = beta
        self.gamma = gamma

    def apply(self, entities: list[GenericType], dt: float) -> list[Effect]:
        vals = {e.id: e.attributes.get("value", 0.0) for e in entities}
        S = vals.get(self.S_id, 0.0)
        I = vals.get(self.I_id, 0.0)
        R = vals.get(self.R_id, 0.0)
        N = S + I + R
        if N < 1e-10:
            return []
        dS = (-self.beta * S * I / N) * dt
        dI = (self.beta * S * I / N - self.gamma * I) * dt
        dR = (self.gamma * I) * dt
        return [
            Effect(entity_id=self.S_id, attr="value", delta=dS),
            Effect(entity_id=self.I_id, attr="value", delta=dI),
            Effect(entity_id=self.R_id, attr="value", delta=dR),
        ]
