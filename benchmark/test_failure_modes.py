from __future__ import annotations

import unittest

from benchmark.failure_harness import run_mock_failure_checks


class ValidationFailureModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.results = run_mock_failure_checks()

    def assert_passed(self, name: str) -> None:
        self.assertIn(name, self.results)
        self.assertTrue(self.results[name]["passed"], self.results[name])

    def test_dailymed_timeout_falls_back(self) -> None:
        self.assert_passed("dailymed_timeout")

    def test_openfda_timeout_is_insufficient(self) -> None:
        self.assert_passed("openfda_timeout")

    def test_malformed_spl_xml_falls_back(self) -> None:
        self.assert_passed("malformed_spl_xml")

    def test_empty_safety_sections_are_insufficient(self) -> None:
        self.assert_passed("empty_safety_sections")


if __name__ == "__main__":
    unittest.main()
