from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from market_qml.utils.statistics import (
    absolute_correlation_matrix,
    safe_correlation,
)


def test_safe_correlation_reports_degenerate_inputs_without_runtime_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        constant = safe_correlation([1, 1, 1], [1, 2, 3])
        perfect = safe_correlation([1, 2, 3], [3, 2, 1])
        ranked = safe_correlation([10, 20, 30], [1, 4, 2], method="spearman")

    assert np.isnan(constant)
    assert perfect == -1.0
    assert ranked == 0.5


def test_absolute_correlation_matrix_uses_zero_for_undefined_pairs():
    frame = pd.DataFrame(
        {
            "constant": [2.0, 2.0, 2.0],
            "left": [1.0, 2.0, 3.0],
            "right": [3.0, 2.0, 1.0],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = absolute_correlation_matrix(frame)

    assert result.loc["constant", "constant"] == 0.0
    assert result.loc["constant", "left"] == 0.0
    assert result.loc["left", "left"] == 1.0
    assert result.loc["left", "right"] == 1.0
