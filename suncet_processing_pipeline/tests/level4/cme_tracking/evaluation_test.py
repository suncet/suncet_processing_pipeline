import numpy as np

from suncet_processing_pipeline.level4.cme_tracking.evaluation import (
    evaluate_front,
    finite_overlap_metrics,
)


def test_finite_overlap_keeps_missing_prediction_in_coverage() -> None:
    metrics = finite_overlap_metrics(
        np.array([1.0, np.nan, 4.0]),
        np.array([2.0, 3.0, 4.0]),
    )

    assert metrics.sample_count == 2
    assert metrics.truth_count == 3
    assert metrics.coverage_fraction == 2 / 3
    assert metrics.bias == -0.5
    assert metrics.root_mean_square_error == np.sqrt(0.5)


def test_front_evaluation_reports_frame_and_angle_errors() -> None:
    truth = np.array([[1.0, 2.0], [2.0, 3.0]])
    predicted = truth + np.array([[0.0, 1.0], [-1.0, 0.0]])

    result = evaluate_front(predicted, truth)

    np.testing.assert_allclose(result.per_frame_rmse, np.sqrt(0.5))
    np.testing.assert_allclose(result.per_angle_rmse, np.sqrt(0.5))
    assert result.all_samples.coverage_fraction == 1.0
