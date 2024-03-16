import threading
from typing import List, Optional, Union, Dict, Tuple, SupportsFloat
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

from welford import Welford


class MetricsTracker:
    """
    Author: Matthijs van der Lende (email: m.r.van.der.lende@student.rug.nl).
    Thread-safe object for recording metrics. Slight abuse of the Singleton pattern similar
    to how a logging object is designed.
    NOTE: Thread-safety means that if you want, you can run each agent concurrently using threads
    and use this object. When an agent running on a thread records a reward/result all other agent threads
    will be blocked. You can also simply train each agent sequentially in a single thread.
    You may also choose to implement the Singleton pattern to ensure only one instance of MetricsTracker exists
    at a time.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._return_aggr = Welford()
        self._return_history: Dict[str, tuple] = defaultdict(lambda: ([], []))

    def plot(self, x_axis_label='Episodes', y_axis_label='Average Return', title="Return History") -> None:
        """
        Plot the metrics to a matplotlib figure.
        """
        with self._lock:
            fig, ax = plt.subplots(figsize=(10, 8))

            for agent_id, (mean_returns, var_returns) in self._return_history.items():
                x_return = np.linspace(0, len(mean_returns), len(mean_returns), endpoint=False)
                ax.plot(x_return, mean_returns, label=f'{agent_id} agent')
                ax.fill_between(x_return,
                                mean_returns - np.sqrt(var_returns) * 0.1,
                                mean_returns + np.sqrt(var_returns) * 0.1,
                                alpha=0.2)

            ax.set_title(title)
            ax.set_xlabel(x_axis_label)
            ax.set_ylabel(y_axis_label)
            ax.legend()
            ax.grid(True)

            plt.tight_layout()
            plt.show()
            plt.savefig('../plots/result.png')

    def record_return(self, agent_id: str, return_val: Union[float, int, SupportsFloat]) -> None:
        """
        Record a return value for a specific agent.

        :param agent_id: The identifier of the agent.
        :param return_val: The return value to record.
        """
        with self._lock:
            self._return_aggr.update_aggr(return_val)
            mean_returns, variance_returns = self._return_history[agent_id]
            mean, var = self._return_aggr.get_curr_mean_variance()
            mean_returns.append(mean)
            variance_returns.append(var)

    def latest_return(self, agent_id: str) -> Optional[Union[float, int]]:
        """
        Get the latest recorded return value for a specific agent.

        :param agent_id: The identifier of the agent.
        :return: The latest recorded return value for the agent, or None if no return has been recorded.
        """
        with self._lock:
            return_values = self._return_history.get(agent_id)
            if return_values:
                return return_values[-1]
            else:
                return None

    def clear(self) -> None:
        """
        Clear the recorded metrics (loss and return history) for all agents.
        """
        with self._lock:
            self._return_history.clear()

    @property
    def return_history(self) -> dict[str, tuple]:
        """
        Get the history of return values for all agents.

        :return: A dictionary containing the return history for each agent.
        """
        with self._lock:
            return self._return_history
