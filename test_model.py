import unittest

from model import (
    commercial_risk_product,
    get_previous_period_options,
    predict_numeric_series,
    shift_period,
    target_period_options,
    validate_prediction_periods,
)


AVAILABLE = [f"{year}.{quarter}/4" for year in range(2021, 2027) for quarter in range(1, 5)]


class CommercialRiskProductTest(unittest.TestCase):
    def test_balanced_pressures_outweigh_one_sided_pressure(self):
        self.assertLess(
            commercial_risk_product(20, 100, 100),
            commercial_risk_product(70, 70, 70),
        )

    def test_rejects_out_of_range_pressures(self):
        with self.assertRaises(ValueError):
            commercial_risk_product(101, 70, 70)


class PredictionPeriodTest(unittest.TestCase):
    def test_first_period_is_not_a_target(self):
        self.assertNotIn("2021.1/4", target_period_options(AVAILABLE))

    def test_2021_second_quarter_has_one_previous_option(self):
        options = get_previous_period_options("2021.2/4", AVAILABLE)
        self.assertEqual([item["period"] for item in options], ["2021.1/4"])

    def test_shift_period_across_year_boundary(self):
        self.assertEqual(shift_period("2022.1/4", -1), "2021.4/4")
        self.assertEqual(shift_period("2022.1/4", -4), "2021.1/4")
        self.assertEqual(shift_period("2025.4/4", 1), "2026.1/4")

    def test_four_previous_options(self):
        periods = [
            item["period"]
            for item in get_previous_period_options("2025.2/4", AVAILABLE)
        ]
        self.assertEqual(periods, ["2025.1/4", "2024.4/4", "2024.3/4", "2024.2/4"])

    def test_rejects_target_future_and_more_than_four_quarters_old(self):
        for invalid in ("2025.2/4", "2025.3/4", "2024.1/4"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_prediction_periods("2025.2/4", [invalid], AVAILABLE)

    def test_non_contiguous_periods_use_real_time_distance(self):
        value = predict_numeric_series(
            [("2024.2/4", 10), ("2024.4/4", 30), ("2025.1/4", 40)],
            "2025.2/4",
        )
        self.assertAlmostEqual(value, 50.0)

    def test_one_period_keeps_latest_observation(self):
        self.assertEqual(predict_numeric_series([("2025.1/4", 100)], "2025.2/4"), 100.0)

    def test_prediction_validation_requires_two_periods(self):
        with self.assertRaises(ValueError):
            validate_prediction_periods("2025.2/4", ["2025.1/4"], AVAILABLE)

    def test_target_value_cannot_leak_into_prediction(self):
        history = [("2024.4/4", 80), ("2025.1/4", 100)]
        expected = predict_numeric_series(history, "2025.2/4")
        self.assertEqual(expected, predict_numeric_series(history, "2025.2/4"))
        with self.assertRaises(ValueError):
            validate_prediction_periods("2025.2/4", ["2025.2/4"], AVAILABLE)


if __name__ == "__main__":
    unittest.main()
