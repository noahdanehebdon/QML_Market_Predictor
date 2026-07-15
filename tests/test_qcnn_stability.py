import json

import pandas as pd
import pytest

from market_qml.qml.interface import build_qml_train_validation
from market_qml.qml.qcnn_stability import (
    evaluate_qcnn_stability,
    render_qcnn_stability_report,
    save_qcnn_stability_result,
)


def _sample() -> pd.DataFrame:
    rows = []
    for role, targets in {
        "train": [0, 1, 0, 1, 0, 1, 0, 1],
        "validation": [0, 1, 0, 1],
    }.items():
        for index, target in enumerate(targets):
            row = {
                "symbol": f"SYM{index}",
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
                "split_id": 0,
                "sample_role": role,
                "target": target,
                "forward_return_5d": 0.02 if target else -0.01,
                "forward_excess_return_5d": 0.01 if target else -0.02,
            }
            for component in range(8):
                row[f"pca_{component:02d}"] = target + index * 0.02 + component * 0.03
            rows.append(row)
    return pd.DataFrame(rows)


def test_qcnn_stability_grid_tracks_gradients_samples_and_selects_config():
    result = evaluate_qcnn_stability(
        build_qml_train_validation(_sample(), split_id=0),
        initialization_scales=[0.01, 0.1],
        learning_rates=[0.02],
        train_sample_sizes=[4, 8],
        max_iter=2,
        batch_size=4,
        random_state=7,
    )

    assert len(result.results) == 4
    assert set(result.results["initialization_scale"]) == {0.01, 0.1}
    assert set(result.results["train_sample_size"]) == {4, 8}
    assert result.results["median_gradient_norm"].notna().all()
    assert result.results["validation_log_loss"].notna().all()
    assert result.results["failure_modes"].notna().all()
    assert len(result.optimization_history) == 8
    assert set(result.optimization_history.columns) >= {
        "gradient_norm",
        "step_norm",
        "parameter_norm",
    }
    assert result.best_config["config_id"] == result.results.iloc[0]["config_id"]


def test_qcnn_stability_artifacts_document_config_and_failure_modes(tmp_path):
    result = evaluate_qcnn_stability(
        build_qml_train_validation(_sample(), split_id=0),
        initialization_scales=[0.05],
        learning_rates=[0.02],
        train_sample_sizes=[8],
        max_iter=2,
    )

    paths = save_qcnn_stability_result(result, output_dir=tmp_path)
    saved = json.loads(paths["best_config"].read_text(encoding="utf-8"))
    report = paths["report"].read_text(encoding="utf-8")

    assert all(path.exists() for path in paths.values())
    assert saved == result.best_config
    assert "Known failure modes" in report
    assert render_qcnn_stability_report(result) == report


@pytest.mark.parametrize(
    ("scales", "rates", "sizes", "message"),
    [
        ([], [0.1], [4], "initialization_scales"),
        ([0.1], [0.0], [4], "learning_rates"),
        ([0.1], [0.1], [3], "train_sample_sizes"),
        ([0.1], [0.1], [10], "available training rows"),
    ],
)
def test_qcnn_stability_rejects_invalid_grids(scales, rates, sizes, message):
    with pytest.raises(ValueError, match=message):
        evaluate_qcnn_stability(
            build_qml_train_validation(_sample(), split_id=0),
            initialization_scales=scales,
            learning_rates=rates,
            train_sample_sizes=sizes,
            max_iter=1,
        )
