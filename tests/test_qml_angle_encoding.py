from math import atan, pi

import pandas as pd
import pytest

from market_qml.qml.encoding import (
    ANGLE_MAX,
    ANGLE_MIN,
    AngleEncodingConfig,
    angle_encode_dataset,
    angle_encode_features,
    scale_value_to_angle,
)
from market_qml.qml.interface import QMLDataset


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            f"pca_{index:02d}": [float(index), -float(index)]
            for index in range(8)
        }
    )


def test_angle_encode_features_outputs_eight_angle_columns_and_operations():
    result = angle_encode_features(_features())

    assert result.angles.shape == (2, 8)
    assert result.angles.columns.tolist() == [
        f"theta_{index:02d}" for index in range(8)
    ]
    assert result.operations["qubit"].tolist() == list(range(8))
    assert result.operations["gate"].unique().tolist() == ["ry"]
    assert result.operations["feature_column"].tolist() == [
        f"pca_{index:02d}" for index in range(8)
    ]
    assert result.operations["angle_column"].tolist() == result.angles.columns.tolist()


def test_angle_encode_features_scales_values_to_valid_angle_range():
    result = angle_encode_features(_features())

    assert result.angles.min().min() >= ANGLE_MIN
    assert result.angles.max().max() <= ANGLE_MAX
    assert result.angles.loc[0, "theta_00"] == pytest.approx(0.0)
    assert result.angles.loc[0, "theta_01"] == pytest.approx(2 * atan(1.0))
    assert result.angles.loc[1, "theta_07"] == pytest.approx(2 * atan(-7.0))
    assert scale_value_to_angle(10_000.0) < pi
    assert scale_value_to_angle(-10_000.0) > -pi


def test_angle_encode_features_supports_grouped_pca_columns_and_config():
    features = pd.DataFrame(
        {
            f"group_{index}_pca_00": [float(index)]
            for index in range(8)
        }
    )

    result = angle_encode_features(
        features,
        config=AngleEncodingConfig(n_qubits=8, angle_prefix="angle", gate="rx"),
    )

    assert result.angles.columns.tolist() == [
        f"angle_{index:02d}" for index in range(8)
    ]
    assert result.operations["gate"].unique().tolist() == ["rx"]


def test_angle_encode_dataset_preserves_targets_and_metadata():
    dataset = QMLDataset(
        X=_features(),
        y=pd.Series([0, 1]),
        metadata=pd.DataFrame({"symbol": ["AAPL", "MSFT"]}),
    )

    encoded = angle_encode_dataset(dataset)

    assert encoded.X.shape == (2, 8)
    assert encoded.y.tolist() == [0, 1]
    assert encoded.metadata["symbol"].tolist() == ["AAPL", "MSFT"]


def test_angle_encode_features_validates_shape_and_values():
    with pytest.raises(ValueError, match="exactly 8 feature columns"):
        angle_encode_features(_features().drop(columns=["pca_07"]))

    bad = _features().astype(object)
    bad.loc[0, "pca_00"] = "not-a-number"
    with pytest.raises(ValueError, match="non-numeric"):
        angle_encode_features(bad)

    with pytest.raises(ValueError, match="positive"):
        angle_encode_features(_features(), config=AngleEncodingConfig(n_qubits=0))
