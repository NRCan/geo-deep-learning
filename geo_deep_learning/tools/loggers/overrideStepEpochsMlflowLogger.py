from lightning.pytorch.loggers import MLFlowLogger

class OverrideStepEpochsMlflowLogger(MLFlowLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def log_metrics(self, metrics, step=None):
        if step is not None:
            step = int(step)
        super().log_metrics(metrics, step)