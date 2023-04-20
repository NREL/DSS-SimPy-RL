"""Type alias shared by reward-related code.
Code adopted from https://github.com/HumanCompatibleAI/imitation.git
"""

from typing import Callable

import numpy as np

RewardFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
