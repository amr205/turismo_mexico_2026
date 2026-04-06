from .xgboost_model import XGBoostForecaster
from .mlp_model import MLPForecaster
from .gru_model import GRUForecaster
from .cnn_gru_model import CNNGRUForecaster
from .res_cnn_gru_model import ResCNNGRUForecaster

MODEL_REGISTRY = {
    "xgboost": XGBoostForecaster,
    "mlp": MLPForecaster,
    "gru": GRUForecaster,
    "cnn_gru": CNNGRUForecaster,
    "res_cnn_gru": ResCNNGRUForecaster,
}


def get_model(model_type: str, model_params: dict) -> "XGBoostForecaster":
    """Instancia un modelo por nombre usando el registro de modelos."""
    if model_type not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Modelo desconocido: '{model_type}'. Disponibles: {available}")
    return MODEL_REGISTRY[model_type](model_params)


def get_model_class(model_type: str):
    """Retorna la clase del modelo para llamar a .load() desde evaluate/interpret."""
    if model_type not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Modelo desconocido: '{model_type}'. Disponibles: {available}")
    return MODEL_REGISTRY[model_type]
