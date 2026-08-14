import pandas as pd

from scripts.build_definitive_qml_comparison import _only_classical_predictions


def test_classical_report_lane_excludes_qml_rows():
    predictions = pd.DataFrame(
        {
            "model_name": ["linear_svm", "rbf_svm", "qsvm_tuned", "vqc"],
            "y_score": [0.1, 0.2, 0.3, 0.4],
        }
    )

    filtered = _only_classical_predictions(predictions)

    assert filtered["model_name"].tolist() == ["linear_svm", "rbf_svm"]
