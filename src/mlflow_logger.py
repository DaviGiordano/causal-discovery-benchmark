from typing import Any, Dict
from flatten_dict import flatten
import logging
import pathlib
import mlflow

logger = logging.getLogger(__name__)


class MLflowLogger:
    def __init__(self, experiment_name: str, mlflow_tracking_uri: str):
        """
        Initialize MLflow logger with experiment name.

        Args:
            experiment_name: Name of the MLflow experiment
        """
        self.experiment_name = experiment_name
        self.mlflow_tracking_uri = mlflow_tracking_uri

        if self.mlflow_tracking_uri:
            self._configure_mlfow(self.mlflow_tracking_uri)
        else:
            self._configure_mlfow("./mlruns")

        mlflow.set_experiment(experiment_name)

    def log_run(
        self,
        run_name: str,
        params: Dict,
        metrics: Dict,
        artifacts_dir: str = None,
    ) -> None:
        """Log a single experiment run to MLflow."""
        try:
            with mlflow.start_run(run_name=run_name):

                self._log_params(params=params)
                self._log_metrics(metrics)
                if artifacts_dir:
                    self._log_plot_images(artifacts_dir)

                logger.info(f"Successfully logged run '{run_name}' to MLflow")

        except Exception as e:
            logger.error(f"Failed to log run to MLflow: {str(e)}")
            raise

    def _log_params(
        self,
        params: Dict[str, Any],
    ) -> None:
        """Log params."""
        for key, value in flatten(params, reducer="dot").items():
            mlflow.log_param(key, value)

    def _log_metrics(self, metrics_results: Dict[str, Any]) -> None:
        """Flatten a nested metrics dict and log each metric."""
        for key, value in flatten(metrics_results, reducer="dot").items():
            if not isinstance(value, (str, int, float)):
                continue  # Must be str, int or float
            mlflow.log_metric(key, value)

    def _log_plot_images(self, artifacts_dir: pathlib.Path) -> None:
        """Log PNG plot images from the specified directory."""
        if artifacts_dir.exists():
            try:
                output_log_fpath = f"{str(artifacts_dir)}/output.log"
                mlflow.log_artifact(output_log_fpath)
            except:
                logging.error(f"Failed to log {output_log_fpath}")
            for ext in ["png", "txt", "json"]:
                for file in pathlib.Path(artifacts_dir).glob(f"*.{ext}"):
                    try:
                        mlflow.log_artifact(str(file))
                        logger.info(f"Logged file: {file.name}")
                    except Exception as e:
                        logger.error(f"Failed to log file {file.name}: {str(e)}")

    def _configure_mlfow(self, uri: str):
        """Configure MLflow tracking URI, falling back to local storage if needed."""
        try:
            mlflow.set_tracking_uri(uri)
            logger.info(f"MLflow tracking URI set to: {uri}")
            return
        except Exception as e:
            logger.warning(f"Failed to set MLflow tracking URI ({uri}): {str(e)}")
            raise
