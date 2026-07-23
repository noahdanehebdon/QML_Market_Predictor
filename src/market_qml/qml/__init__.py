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
from market_qml.qml.feature_map import (
    QuantumFeatureMapConfig,
    QuantumFeatureMapResult,
    QuantumFeatureMapSplitResult,
    QuantumKernelFeatureMap,
    feature_map_operations,
    fidelity_kernel,
    save_feature_map_split,
)
from market_qml.qml.interface import (
    BaseQMLModel,
    QMLDataset,
    QMLModelConfig,
    QMLModelResult,
    QMLTrainValidation,
    build_qml_train_validation,
)
from market_qml.qml.qcnn import (
    MODEL_NAME as QCNN_MODEL_NAME,
)
from market_qml.qml.qcnn import (
    QCNNResult,
    QuantumConvolutionalClassifier,
    train_qcnn,
)
from market_qml.qml.qcnn_blocks import (
    QCNNArchitecture,
    build_qcnn_architecture,
    execute_qcnn_architecture,
    initialize_qcnn_parameters,
    pooling_block,
    two_qubit_convolution,
)
from market_qml.qml.qsvm import (
    MODEL_NAME as QSVM_MODEL_NAME,
)
from market_qml.qml.qsvm import (
    QSVMResult,
    QuantumKernelSVM,
    train_qsvm,
)
from market_qml.qml.vqc import (
    MODEL_NAME as VQC_MODEL_NAME,
)
from market_qml.qml.vqc import (
    VariationalQuantumClassifier,
    VQCResult,
    train_vqc,
)

__all__ = [
    "ANGLE_MAX",
    "ANGLE_MIN",
    "QCNN_MODEL_NAME",
    "QSVM_MODEL_NAME",
    "VQC_MODEL_NAME",
    "AngleEncodingConfig",
    "AngleEncodingResult",
    "BaseQMLModel",
    "QCNNArchitecture",
    "QCNNResult",
    "QMLDataset",
    "QMLModelConfig",
    "QMLModelResult",
    "QMLTrainValidation",
    "QSVMResult",
    "QuantumConvolutionalClassifier",
    "QuantumFeatureMapConfig",
    "QuantumFeatureMapResult",
    "QuantumFeatureMapSplitResult",
    "QuantumKernelFeatureMap",
    "QuantumKernelSVM",
    "VQCResult",
    "VariationalQuantumClassifier",
    "angle_encode_dataset",
    "angle_encode_features",
    "angle_encoding_operations",
    "build_qcnn_architecture",
    "build_qml_train_validation",
    "execute_qcnn_architecture",
    "feature_map_operations",
    "fidelity_kernel",
    "initialize_qcnn_parameters",
    "pooling_block",
    "save_feature_map_split",
    "scale_value_to_angle",
    "train_qcnn",
    "train_qsvm",
    "train_vqc",
    "two_qubit_convolution",
]
