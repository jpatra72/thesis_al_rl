from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        for idx, done in enumerate(self.locals['dones']):
            if done:
                model_performance = self.locals['infos'][idx]['model_performance']
                model_entropy = self.locals['infos'][idx]['model_entropy']
                # queried_count = self.locals['infos'][idx]['queried_count']
                labelled_count = self.locals['infos'][idx]['labelled_count']
                self.logger.record('rollout/model_performance_test', model_performance['test_data'])
                self.logger.record('rollout/model_performance_learn', model_performance['learned_data'])
                self.logger.record('rollout/model_entropy_test', model_entropy['test_data'])
                self.logger.record('rollout/model_entropy_learn', model_entropy['learn_data'])
                # self.logger.record('rollout/queried_count', queried_count)
                self.logger.record('rollout/labelled_count', labelled_count)
            return True
