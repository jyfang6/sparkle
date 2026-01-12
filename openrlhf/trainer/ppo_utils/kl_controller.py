import numpy as np


# class AdaptiveKLController:
#     """
#     Adaptive KL controller described in the paper:
#     https://arxiv.org/pdf/1909.08593.pdf
#     """

#     def __init__(self, init_kl_coef, target, horizon):
#         self.value = init_kl_coef
#         self.target = target
#         self.horizon = horizon

#     def update(self, current, n_steps):
#         target = self.target
#         proportional_error = np.clip(current / target - 1, -0.2, 0.2)
#         mult = 1 + proportional_error * n_steps / self.horizon
#         self.value *= mult

class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon
        self.min_beta, self.max_beta = 0.05, 8.0

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        # mult = 1 + proportional_error * n_steps / self.horizon
        # self.value *= mult
        mult = 1.0 + proportional_error * min(n_steps, self.horizon) / self.horizon
        self.value = float(np.clip(self.value * mult, self.min_beta, self.max_beta))


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass
