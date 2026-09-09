"""
Truncated Quantile Critics (TQC) PyTorch implementation.
Based on Kuznetsov et al. (ICML 2020):
'Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics'
"""

from .actor import Actor

__all__ = ["Actor"]
