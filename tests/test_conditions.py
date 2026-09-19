import unittest

import numpy as np

from scegflow_energy import build_condition_tensors, parse_condition


class ConditionTests(unittest.TestCase):
    def test_combination_parser_preserves_parenthesized_plus(self) -> None:
        parsed = parse_condition("(+)-JQ1_1.0+Belinostat_0.1")
        self.assertEqual(parsed.drugs, ("(+)-JQ1", "Belinostat"))
        self.assertEqual(parsed.doses, (1.0, 0.1))

    def test_condition_tensor_padding(self) -> None:
        features = {"A": np.ones(4, dtype=np.float32), "B": np.zeros(4, dtype=np.float32)}
        drug, doses, mask = build_condition_tensors(["A_1.0", "A_1.0+B_0.1"], features)
        self.assertEqual(drug.shape, (2, 2, 4))
        self.assertEqual(doses.shape, (2, 2))
        self.assertTrue(np.allclose(doses.numpy(), [[1.0, 1.0], [1.0, 0.1]]))
        self.assertEqual(mask.tolist(), [[True, False], [True, True]])


if __name__ == "__main__":
    unittest.main()
