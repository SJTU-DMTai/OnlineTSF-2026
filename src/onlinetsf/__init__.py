"""Online forecasting components, imported on first use."""

from importlib import import_module


_EXPORTS = {
    'SlidingWindowDataset': '.data',
    'load_benchmark_dataset': '.data',
    'ADWINDetector': '.drift',
    'KSWINDetector': '.drift',
    'PageHinkleyDetector': '.drift',
    'DLinearForecastBackbone': '.forecasting',
    'LinearForecastBackbone': '.forecasting',
    'LSTMForecastBackbone': '.forecasting',
    'PatchTSTForecastBackbone': '.forecasting',
    'TCNForecastBackbone': '.forecasting',
    'TimeTCNForecastBackbone': '.forecasting',
    'FeedbackEvent': '.online',
    'ForecastEmission': '.online',
    'MethodFeedback': '.online',
    'OnlineExecutor': '.online',
    'OnlineMethod': '.online',
    'OnlineMetrics': '.online',
    'OnlineRun': '.online',
    'OnlineStep': '.online',
    'AdaptiveConv1d': '.methods',
    'DSOFMethod': '.methods',
    'DSOFModel': '.methods',
    'FSNetMethod': '.methods',
    'FSNetTCN': '.methods',
    'OGDMethod': '.methods',
    'OneNetEnsemble': '.methods',
    'OneNetMethod': '.methods',
    'UnderCaliMethod': '.methods',
    'ControlledMethod': '.decision.control',
    'Policy': '.decision.actions',
    'AdaptiveMethod': '.decision.runtime',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
