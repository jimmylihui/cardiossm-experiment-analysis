"""In-memory mathematical fixtures only; no study observations or outputs."""
import unittest
import numpy as np
from cardiossm_analysis.temporal import rolling_average, change_pairs
from cardiossm_analysis.calibration import offset_hr, select_qa_other_patients
from cardiossm_analysis.metrics import pearson, patient_metrics, aggregate_patient_values
from cardiossm_analysis.events import event_metrics
from cardiossm_analysis.bootstrap import paired_interval, candidate_interval
from cardiossm_analysis.scales import scale_metrics
from cardiossm_analysis.features import heldout_screen


class TemporalTests(unittest.TestCase):
    def test_past_support_and_startup(self):
        t = np.arange(5) * 120.
        x = np.arange(5, dtype=float)
        self.assertTrue(np.allclose(rolling_average(x, t, 240), [0, .5, 1.5, 2.5, 3.5]))

    def test_future_perturbation_cannot_change_prefix(self):
        t = np.arange(30) * 120.
        x = np.sin(np.arange(30))
        changed = x.copy(); changed[20:] += 1000
        np.testing.assert_array_equal(rolling_average(x, t)[:20], rolling_average(changed, t)[:20])

    def test_gap_resets_mean(self):
        t = np.array([0., 120., 4000.])
        self.assertEqual(rolling_average(np.array([1., 3., 20.]), t)[-1], 20.)

    def test_reading_endpoints_are_excluded(self):
        t = np.arange(20) * 120.
        read = np.zeros(20, bool); read[9] = True
        a, b = change_pairs(t, np.ones(20, bool), read)
        self.assertTrue(all(not (i <= 9 <= j) for i, j in zip(a, b)))

    def test_invalid_time_rejected(self):
        with self.assertRaises(ValueError):
            rolling_average([1., 2.], [0., 0.])


class MetricTests(unittest.TestCase):
    def test_constant_correlation_missing(self):
        self.assertTrue(np.isnan(pearson(np.ones(10), np.arange(10))))

    def test_aggregation_is_fisher_then_median(self):
        z = np.arctanh([.1, .5, .9])
        self.assertAlmostEqual(aggregate_patient_values(z, "z_seg"), .5)

    def test_exact_prediction_and_missing_segments(self):
        t = np.arange(100) * 120.
        y = 80 + 8 * np.sin(np.arange(100) / 5)
        read = np.zeros(100, bool); read[0] = True
        r = patient_metrics(t, y, y.copy(), np.ones(100, bool), read)
        self.assertEqual(r['mae'], 0.)
        self.assertAlmostEqual(r['r_test'], 1.)
        self.assertEqual(r['concordance'], 1.)

    def test_event_ranking(self):
        t = np.arange(300) * 120.
        y = 80 + 20 * np.sin(np.arange(300) / 5)
        r = event_metrics(t, y, y.copy(), np.ones(300, bool), np.zeros(300, bool))
        self.assertTrue(r['eligible'])
        self.assertAlmostEqual(r['auroc'], 1.)
        self.assertAlmostEqual(r['average_precision'], 1.)


class AssemblyTests(unittest.TestCase):
    def test_reading_is_predicted_before_filter_update(self):
        t = np.arange(20) * 120.
        y = np.full(20, 80.); y[10] = 140.
        cal = np.arange(20) < 5
        read = np.zeros(20, bool); read[10] = True
        _, p = offset_hr(t, y, np.full(20, 80.), np.zeros(20), np.zeros(20), cal, read)
        self.assertLess(p[10], 81.)
        self.assertGreater(p[11], 130.)

    def test_no_reading_baseline_delta_equals_hr_delta(self):
        t = np.arange(100) * 120.
        h = np.sin(np.arange(100)); y = 80 + h
        cal = np.arange(100) < 20
        read = np.zeros(100, bool); read[20] = True; read[60] = True
        _, p = offset_hr(t, y, np.full(100, 80.), np.zeros(100), h, cal, read)
        a, b = change_pairs(t, ~cal, read)
        np.testing.assert_allclose(p[b] - p[a], h[b] - h[a], atol=1e-12)

    def test_heldout_row_cannot_select_its_own_qa(self):
        e = np.array([[1., 8.], [2., 4.], [8., 1.]])
        first = select_qa_other_patients(e)[0]
        e[0] = [10000., 0.]
        self.assertEqual(first, select_qa_other_patients(e)[0])


class UncertaintyTests(unittest.TestCase):
    def test_identical_pairs_have_zero_interval(self):
        x = np.linspace(1, 3, 12)
        r = paired_interval(x, x, "mae", resamples=200)
        self.assertEqual((r['difference'], r['ci95_low'], r['ci95_high']), (0., 0., 0.))

    def test_missing_pairs_use_common_support(self):
        r = paired_interval([1., 2., np.nan, 100.], [1., 2., 4., np.nan], "mae", resamples=100)
        self.assertEqual(r['valid_pairs'], 2)
        self.assertEqual(r['difference'], 0.)

    def test_candidate_bootstrap_matches_explicit_identity_exclusion(self):
        values = np.array([[1., 7.], [2., 8.], [4., 6.], [3., 9.]])
        errors = np.array([[1., 9.], [3., 2.], [5., 1.], [2., 4.]])
        result = candidate_interval(values, errors, values + 1, errors, "mae", resamples=200, seed=7)
        self.assertAlmostEqual(result['difference'], -1.)
        self.assertAlmostEqual(result['ci95_low'], -1.)
        self.assertAlmostEqual(result['ci95_high'], -1.)
        self.assertGreater(result['valid_resamples'], 0)

    def test_scales_preserve_identical_predictions(self):
        t = np.arange(200) * 10.
        y = np.sin(np.arange(200) / 9)
        rows = scale_metrics(t, y, y, np.ones(200, bool), [10, 120, 600])
        self.assertTrue(all(r['mae'] == 0 for r in rows))
        with self.assertRaises(ValueError):
            scale_metrics(t, y, y, np.ones(200, bool), [2])

    def test_feature_selection_ignores_later_changes(self):
        t = np.arange(900) * 120.
        y = 80 + 8 * np.sin(np.arange(900) / 50)
        x = np.column_stack([y, np.cos(np.arange(900))])
        first = heldout_screen(t, y, x, null_resamples=3)
        x[400:, 0] *= -1
        second = heldout_screen(t, y, x, null_resamples=3)
        self.assertEqual(first['selected_feature'], second['selected_feature'])
        self.assertAlmostEqual(first['training_r'], second['training_r'])


if __name__ == '__main__':
    unittest.main()
