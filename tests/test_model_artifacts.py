import json
import pickle
from types import SimpleNamespace

import pandas as pd
from sklearn.linear_model import LogisticRegression

from market_qml.models.artifacts import save_artifact_manifest, save_model_artifact
from market_qml.models.preprocessing import fit_preprocessor


def test_model_artifact_records_lineage_and_integrity(tmp_path):
    X = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    model = LogisticRegression().fit(X, [0, 0, 1, 1])
    metadata = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02"])})
    record = save_model_artifact(
        root=tmp_path,
        model_name="logistic_regression",
        split_id=3,
        model=model,
        preprocessor=fit_preprocessor(X),
        pca=None,
        result=object(),
        train_metadata=metadata,
        validation_metadata=metadata.assign(
            date=pd.to_datetime(["2024-01-03", "2024-01-04"])
        ),
        target_column="outperform_spy_5d",
        run_config={"models": ["logistic_regression"]},
        git_sha="abc123",
    )
    manifest_path = save_artifact_manifest(tmp_path, [record])

    assert record["artifact_id"] == "logistic_regression-split-003"
    assert record["git_sha"] == "abc123"
    assert record["train_range"] == {
        "start": "2024-01-01",
        "end": "2024-01-02",
        "rows": 2,
    }
    assert len(record["files"]["model"]["sha256"]) == 64
    model_path = tmp_path / "logistic_regression" / "split_003" / "model.pkl"
    with model_path.open("rb") as file:
        assert isinstance(pickle.load(file), LogisticRegression)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifacts"][0]["artifact_id"] == record["artifact_id"]


def test_model_artifact_saves_pca_and_explicit_qml_weights(tmp_path):
    metadata = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"])})
    record = save_model_artifact(
        root=tmp_path,
        model_name="vqc",
        split_id=0,
        model=SimpleNamespace(weights_=pd.Series([0.1, 0.2]).to_numpy()),
        preprocessor={"kind": "preprocessor"},
        pca={"kind": "pca"},
        result=object(),
        train_metadata=metadata,
        validation_metadata=metadata,
        target_column="outperform_spy_5d",
        run_config={},
        git_sha="abc123",
    )

    assert "pca" in record["files"]
    assert record["qml_parameters"] == {"weights": [0.1, 0.2]}
