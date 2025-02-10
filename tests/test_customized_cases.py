"""We encourage you to create your own test cases, which helps
you confirm the correctness of your implementation.

If you are interested, you can write your own tests in this file
and share them with us by including this file in your submission.
Please make the tests "pytest compatible" by starting each test
function name with prefix "test_".

We appreciate it if you can share your tests, which can help
improve this course and the assignment. However, please note that
this part is voluntary -- you will not get more scores by sharing
test cases, and conversely, will not get fewer scores if you do
not share.
"""

from typing import List, Dict

import numpy as np
import pytest
import auto_diff as ad
from logistic_regression import softmax_loss


def check_compute_output(
    node: ad.Node, input_values: List[np.ndarray], expected_output: np.ndarray
) -> None:
    output = node.op.compute(node, input_values)
    np.testing.assert_allclose(actual=output, desired=expected_output)


def check_evaluator_output(
    evaluator: ad.Evaluator,
    input_values: Dict[ad.Node, np.ndarray],
    expected_outputs: List[np.ndarray],
) -> None:
    output_values = evaluator.run(input_values)
    assert len(output_values) == len(expected_outputs)
    for output_val, expected_val in zip(output_values, expected_outputs):
        np.testing.assert_allclose(actual=output_val, desired=expected_val)


def test_softmax():
    X = ad.Variable("X")
    y_hot = ad.Variable("y_hot")
    y = softmax_loss(X, y_hot, 2)

    evaluator = ad.Evaluator(eval_nodes=[y])

    check_evaluator_output(
        evaluator,
        input_values={
            X: np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
            y_hot: np.array([[0, 1, 0], [0, 0, 1]]),
        },
        expected_outputs=[np.array(0.9076059644443804)],
    )


if __name__ == "__main__":
    test_softmax()
