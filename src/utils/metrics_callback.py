import numpy as np
from stable_baselines3.common.callbacks import EvalCallback

class MetricsCallback(EvalCallback):
    def __init__(self, eval_env, **kwargs):
        super().__init__(eval_env, **kwargs)
        # Initialize counters and storage for metrics
        self.avg_node_occupancy = []
        self.message_channel_occupancy = []
        self.empty_nodes = []

    def _log_success_callback(self, locals_, globals_):
        super()._log_success_callback(locals_, globals_)
        if locals_["done"]:
            info = locals_['info']
            # Store the metrics for the episode
            self.avg_node_occupancy.append(info.get("avg_node_occupancy"))
            self.message_channel_occupancy.append(info.get("message_channel_occupancy"))
            if info.get("is_success", False):
                self.empty_nodes.append(info.get("empty_nodes", 0))

    def _store_metrics(self):
        mean_avg_node_occupancy = np.mean(self.avg_node_occupancy) if self.avg_node_occupancy else 0
        mean_message_channel_occupancy = np.mean(self.message_channel_occupancy) if self.message_channel_occupancy else 0
        mean_empty_nodes = np.mean(self.empty_nodes) if self.empty_nodes else 0
        self.logger.record("eval/avg_node_occupancy", mean_avg_node_occupancy)
        self.logger.record("eval/message_channel_occupancy", mean_message_channel_occupancy)
        self.logger.record("eval/empty_nodes", mean_empty_nodes)
        if(self.verbose > 0):
            print(f"Avg Node Occupancy: {mean_avg_node_occupancy:.2f}%")
            print(f"Avg Message Channel Occupancy: {mean_message_channel_occupancy:.2f}%")
            print(f"Avg Empty Nodes: {mean_empty_nodes:.2f}%")
        self.avg_node_occupancy.clear()
        self.message_channel_occupancy.clear()
        self.empty_nodes.clear()

    def _on_step(self) -> np.bool:
        """Called at each step."""
        super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self._store_metrics()
        return True