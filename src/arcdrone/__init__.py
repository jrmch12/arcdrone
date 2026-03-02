"""
Reinforcement Learning Control of the ARC drone

This package provides:
- RL training capabilities using Brax and MJX
- Hardware control
- Utilities for logging, configuration, and data processing
"""

__version__ = "0.1.0"
__author__ = "Jose Mares"

# Import main modules for easy access



__all__ = ["__version__", "__author__"]

from .controller.rl.task.velocity_mode.arcdrone import ARCDroneRL_Vel
from .controller.rl.task.landing_mode.arcdrone import ARCDroneRL_Landing
from .controller.rl.task.hover_mode.arcdrone import ARCDroneRL_Hover

from .controller.sitt.sitt import train as sitt_train
from .controller.sitt.sitt import networks as sitt_networks
