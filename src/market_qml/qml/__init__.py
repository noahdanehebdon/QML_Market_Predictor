"""Utilities for quantum machine learning experiments."""

from market_qml.qml.encoding import (
    ANGLE_MAX,
    ANGLE_MIN,
    AngleEncodingConfig,
    AngleEncodingResult,
    angle_encode_dataset,
    angle_encode_features,
    angle_encoding_operations,
    scale_value_to_angle,
)
from market_qml.qml.interface import (
    BaseQMLModel,
    QMLDataset,
    QMLModelConfig,
    QMLModelResult,
    QMLTrainValidation,
    build_qml_train_validation,
)
from market_qml.qml.feature_map import (
    QuantumFeatureMapConfig,
    QuantumFeatureMapResult,
    QuantumFeatureMapSplitResult,
    QuantumKernelFeatureMap,
    feature_map_operations,
    fidelity_kernel,
    save_feature_map_split,
)
from market_qml.qml.vqc import (
    MODEL_NAME as VQC_MODEL_NAME,
    VQCResult,
    VariationalQuantumClassifier,
    train_vqc,
)

__all__ = [
    "ANGLE_MAX",
    "ANGLE_MIN",
    "AngleEncodingConfig",
    "AngleEncodingResult",
    "BaseQMLModel",
    "QMLDataset",
    "QMLModelConfig",
    "QMLModelResult",
    "QMLTrainValidation",
    "QuantumFeatureMapConfig",
    "QuantumFeatureMapResult",
    "QuantumFeatureMapSplitResult",
    "QuantumKernelFeatureMap",
    "VQC_MODEL_NAME",
    "VQCResult",
    "VariationalQuantumClassifier",
    "angle_encode_dataset",
    "angle_encode_features",
    "angle_encoding_operations",
    "build_qml_train_validation",
    "feature_map_operations",
    "fidelity_kernel",
    "save_feature_map_split",
    "scale_value_to_angle",
    "train_vqc",
]
