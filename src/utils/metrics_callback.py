import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from env.cades_env import TerminationCause

class MetricsCallback(EvalCallback):
    def __init__(self, eval_env, **kwargs):
        super().__init__(eval_env, **kwargs)
        # Initialize episode count for evaluation cycle
        self.episode_count = 0
        # Initialize counters and storage for metrics
        self.avg_node_occupancy = []
        self.message_channel_occupancy = []
        self.empty_nodes = []
        # For each termination cause, add an entry to the termination_cause dictionary
        self.termination_cause = {str(cause): 0 for cause in TerminationCause}

    def _log_success_callback(self, locals_, globals_):
        super()._log_success_callback(locals_, globals_)
        if locals_["done"]:
            self.episode_count += 1
            info = locals_['info']
            # Store the metrics for the episode
            self.avg_node_occupancy.append(info.get("avg_node_occupancy"))
            self.message_channel_occupancy.append(info.get("message_channel_occupancy"))
            # Store the termination cause
            self.termination_cause[info.get("termination_cause")] += 1
            if info.get("is_success", False):
                self.empty_nodes.append(info.get("empty_nodes", 0))

    def _store_metrics(self):
        mean_avg_node_occupancy = np.mean(self.avg_node_occupancy) if self.avg_node_occupancy else 0
        mean_message_channel_occupancy = np.mean(self.message_channel_occupancy) if self.message_channel_occupancy else 0
        mean_empty_nodes = np.mean(self.empty_nodes) if self.empty_nodes else 0
        termination_cause_means = {cause: count / self.episode_count * 100 for cause, count in self.termination_cause.items()}
        # Log the other metrics
        self.logger.record("eval/avg_node_occupancy", mean_avg_node_occupancy)
        self.logger.record("eval/message_channel_occupancy", mean_message_channel_occupancy)
        self.logger.record("eval/empty_nodes", mean_empty_nodes)
        for cause, cause_mean in termination_cause_means.items():
            self.logger.record(f"eval/{cause}", cause_mean)
            if(self.verbose > 0):
                print(f"{cause}: {cause_mean:.2f}%")
        if(self.verbose > 0):
            print(f"Avg Node Occupancy: {mean_avg_node_occupancy:.2f}%")
            print(f"Avg Message Channel Occupancy: {mean_message_channel_occupancy:.2f}%")
            print(f"Avg Empty Nodes: {mean_empty_nodes:.2f}%")
        # Clear the metrics for the next evaluation cycle
        self.avg_node_occupancy.clear()
        self.message_channel_occupancy.clear()
        self.empty_nodes.clear()
        self.termination_cause = {cause: 0 for cause in self.termination_cause.keys()}

    def _on_step(self) -> np.bool:
        """Called at each step."""
        super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self._store_metrics()
            self.episode_count = 0
        return True