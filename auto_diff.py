# from attr import attrs
from typing import Any, Dict, List
from collections import defaultdict

import numpy as np


def ndarray_wrap(x):
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)


class Node:
    """Node in a computational graph.

    Fields
    ------
    inputs: List[Node]
        The list of input nodes to this node.

    op: Op
        The op of this node.

    attrs: Dict[str, Any]
        The attribute dictionary of this node.
        E.g. "constant" is the constant operand of add_by_const.

    name: str
        Name of the node for debugging purposes.
    """

    inputs: List["Node"]
    op: "Op"
    attrs: Dict[str, Any]
    name: str

    def __init__(
        self, inputs: List["Node"], op: "Op", attrs: Dict[str, Any] = {}, name: str = ""
    ) -> None:
        self.inputs = inputs
        self.op = op
        self.attrs = attrs
        self.name = name

    def __add__(self, other):
        if isinstance(other, Node):
            return add(self, other)
        else:
            assert isinstance(other, (int, float))
            return add_by_const(self, other)

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (-1) * self + other

    def __mul__(self, other):
        if isinstance(other, Node):
            return mul(self, other)
        else:
            assert isinstance(other, (int, float))
            return mul_by_const(self, other)

    def __truediv__(self, other):
        if isinstance(other, Node):
            return div(self, other)
        else:
            assert isinstance(other, (int, float))
            return div_by_const(self, other)

    # Allow left-hand-side add and multiplication.
    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        """Allow printing the node name."""
        return self.name

    def __getattr__(self, attr_name: str) -> Any:
        if attr_name in self.attrs:
            return self.attrs[attr_name]
        raise KeyError(f"Attribute {attr_name} does not exist in node {self}")

    __repr__ = __str__


class Variable(Node):
    """A variable node with given name."""

    def __init__(self, name: str) -> None:
        super().__init__(inputs=[], op=placeholder, name=name)


class Op:
    """The class of operations performed on nodes."""

    def __call__(self, *kwargs) -> Node:
        """Create a new node with this current op.

        Returns
        -------
        The created new node.
        """
        raise NotImplementedError

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Compute the output value of the given node with its input
        node values given.

        Parameters
        ----------
        node: Node
            The node whose value is to be computed

        input_values: List[np.ndarray]
            The input values of the given node.

        Returns
        -------
        output: np.ndarray
            The computed output value of the node.
        """
        raise NotImplementedError

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given a node and its output gradient node, compute partial
        adjoints with regards to each input node.

        Parameters
        ----------
        node: Node
            The node whose inputs' partial adjoints are to be computed.

        output_grad: Node
            The output gradient with regard to given node.

        Returns
        -------
        input_grads: List[Node]
            The list of partial gradients with regard to each input of the node.
        """
        raise NotImplementedError


class PlaceholderOp(Op):
    """The placeholder op to denote computational graph input nodes."""

    def __call__(self, name: str) -> Node:
        return Node(inputs=[], op=self, name=name)

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        raise RuntimeError(
            "Placeholder nodes have no inputs, and there values cannot be computed."
        )

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        raise RuntimeError("Placeholder nodes have no inputs.")


class AddOp(Op):
    """Op to element-wise add two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}+{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise addition of input values."""
        assert len(input_values) == 2
        return ndarray_wrap(input_values[0] + input_values[1])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of add node, return partial adjoint to each input."""
        return [output_grad, output_grad]


class AddByConstOp(Op):
    """Op to element-wise add a node by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}+{const_val})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise addition of the input value and the constant."""
        assert len(input_values) == 1
        return ndarray_wrap(input_values[0] + node.constant)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of add node, return partial adjoint to the input."""
        return [output_grad]


class MulOp(Op):
    """Op to element-wise multiply two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}*{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise multiplication of input values."""
        assert len(input_values) == 2
        return ndarray_wrap(input_values[0] * input_values[1])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of multiplication node, return partial adjoint to each input."""
        assert len(node.inputs) == 2
        return [output_grad * node.inputs[1], output_grad * node.inputs[0]]


class MulByConstOp(Op):
    """Op to element-wise multiply a node by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}*{const_val})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise multiplication of the input value and the constant."""
        assert len(input_values) == 1
        return ndarray_wrap(input_values[0] * node.constant)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of multiplication node, return partial adjoint to the input."""
        return [output_grad * node.constant]


class DivOp(Op):
    """Op to element-wise divide two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}/{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise division of input values."""
        assert len(input_values) == 2
        return ndarray_wrap(input_values[0] / input_values[1])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of division node, return partial adjoint to each input."""
        return [
            output_grad / node.inputs[1],
            -1 * output_grad * node.inputs[0] / (node.inputs[1] * node.inputs[1]),
        ]


class DivByConstOp(Op):
    """Op to element-wise divide a nodes by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}/{const_val})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise division of the input value and the constant."""
        return ndarray_wrap(input_values[0] / node.constant)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of division node, return partial adjoint to the input."""
        return [output_grad / node.constant]


class Exponential(Op):
    """Exponentiation op of one node"""

    def __call__(self, node_A: Node) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            name=f"(exp({node_A.name}))",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        return np.exp(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [output_grad * node.input[0]]


class Logarithm(Op):
    """Exponentiation op of one node"""

    def __call__(self, node_A: Node) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            name=f"(log({node_A.name}))",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        return np.log(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [output_grad / node.inputs[0]]


class Summation(Op):
    """Exponentiation op of many nodes"""

    def __call__(self, node: Node) -> Node:
        return Node(
            inputs=[node],
            op=self,
            name=f"(summation({node.name}))",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        s = np.sum(input_values[0])
        return s

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [ones_like(node.inputs[0]) * output_grad]


class MatMulOp(Op):
    """Matrix multiplication op of two nodes."""

    def __call__(
        self, node_A: Node, node_B: Node, trans_A: bool = False, trans_B: bool = False
    ) -> Node:
        """Create a matrix multiplication node.

        Parameters
        ----------
        node_A: Node
            The lhs matrix.
        node_B: Node
            The rhs matrix
        trans_A: bool
            A boolean flag denoting whether to transpose A before multiplication.
        trans_B: bool
            A boolean flag denoting whether to transpose B before multiplication.

        Returns
        -------
        result: Node
            The node of the matrix multiplication.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={"trans_A": trans_A, "trans_B": trans_B},
            name=f"({node_A.name + ('.T' if trans_A else '')}@{node_B.name + ('.T' if trans_B else '')})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the matrix multiplication result of input values.

        Note
        ----
        For this assignment, you can assume the matmul only works for 2d matrices.
        That being said, the test cases guarantee that input values are
        always 2d numpy.ndarray.
        """
        res = (input_values[0].T if node.attrs["trans_A"] else input_values[0]) @ (
            input_values[1].T if node.attrs["trans_B"] else input_values[1]
        )
        return ndarray_wrap(res)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of matmul node, return partial adjoint to each input.

        Note
        ----
        - Same as the `compute` method, you can assume that the input are 2d matrices.
        However, it would be a good exercise to think about how to handle
        more general cases, i.e., when input can be either 1d vectors,
        2d matrices, or multi-dim tensors.
        - You may want to look up some materials for the gradients of matmul.


        X = Zbar * W^T
        W = X^T * Zbar

        Keep in mind that given a trans_A, we are not given the transposed matrix, but the OG matrix
        We can make sure that given transposes, make sure the dimensions match, and that is probably the correct answer

        X = m x n
        W = n x r
        Z = m x r

        To be honest, a lot of the intution is generated by deepseek
        """
        trans_A = node.attrs["trans_A"]
        trans_B = node.attrs["trans_B"]

        if trans_A and trans_B:
            return [
                matmul(node.inputs[1], output_grad, trans_A=True, trans_B=True),
                matmul(output_grad, node.inputs[0], trans_A=True, trans_B=True),
            ]
        elif trans_A:
            return [
                matmul(node.inputs[1], output_grad, trans_A=False, trans_B=True),
                matmul(node.inputs[0], output_grad, trans_A=False, trans_B=True),
            ]
        elif trans_B:
            return [
                matmul(output_grad, node.inputs[1], trans_A=False, trans_B=False),
                matmul(output_grad, node.inputs[0], trans_A=True, trans_B=False),
            ]
        else:
            return [
                matmul(output_grad, node.inputs[1], trans_A=False, trans_B=True),
                matmul(node.inputs[0], output_grad, trans_A=True, trans_B=False),
            ]


class Shrink(Op):
    def __call__(self, node_A: Node) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            name=f"Shrink({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        return np.sum(input_values[0], axis=1, keepdims=True)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        # matrix and gradient is m x n
        # vector is a n x 1
        # output grad should have the same dimensions as node.input[1]

        # ones_mat = ones_like(node.inputs[0])
        return broadcast(output_grad, node.inputs[0])


class Broadcast(Op):
    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"Broadcast({node_A.name},{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return an all-zero tensor with the same shape as input."""
        assert len(input_values) == 2
        return np.broadcast_to(input_values[0], input_values[1].shape)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        # matrix and gradient is m x n
        # vector is a n x 1
        # output grad should have the same dimensions as node.input[1]

        n_vector = zeros_like(node.inputs[0])
        m_vector = matmul(node.inputs[1], n_vector)
        m_vector1 = ones_like(m_vector)

        return matmul(output_grad, m_vector1, trans_A=True)


class ZerosLikeOp(Op):
    """Zeros-like op that returns an all-zero array with the same shape as the input."""

    def __call__(self, node_A: Node) -> Node:
        return Node(inputs=[node_A], op=self, name=f"ZerosLike({node_A.name})")

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return an all-zero tensor with the same shape as input."""
        assert len(input_values) == 1
        return np.zeros(input_values[0].shape)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [zeros_like(node.inputs[0])]


class OnesLikeOp(Op):
    """Ones-like op that returns an all-one array with the same shape as the input."""

    def __call__(self, node_A: Node) -> Node:
        return Node(inputs=[node_A], op=self, name=f"OnesLike({node_A.name})")

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return an all-one tensor with the same shape as input."""
        assert len(input_values) == 1
        return np.ones(input_values[0].shape)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [zeros_like(node.inputs[0])]


# Create global instances of ops.
# Your implementation should just use these instances, rather than creating new instances.
placeholder = PlaceholderOp()
add = AddOp()
mul = MulOp()
div = DivOp()
add_by_const = AddByConstOp()
mul_by_const = MulByConstOp()
div_by_const = DivByConstOp()
matmul = MatMulOp()
zeros_like = ZerosLikeOp()
broadcast = Broadcast()
shrink = Shrink()
log = Logarithm()
ones_like = OnesLikeOp()
exp = Exponential()
summation = Summation()


"""
Citation for pseudocode
https://en.wikipedia.org/wiki/Topological_sorting
"""


def topological_sort(sink_node: Node) -> List[Node]:
    """
    Assumption: The graph is acyclic

    Returns a topological_sort of the eval_nodes/sink_nodes
    Sorted via input nodes to the output nodes
    """
    result = []
    visited = set()

    def dfs(node):
        if node in visited:
            return result

        # don't need tmp mark because acyclic guaranteed
        for neigh in node.inputs:
            dfs(neigh)

        visited.add(node)
        result.append(node)
        return result

    return dfs(sink_node)


class Evaluator:
    """The node evaluator that computes the values of nodes in a computational graph."""

    eval_nodes: List[Node]

    def __init__(self, eval_nodes: List[Node]) -> None:
        """Constructor, which takes the list of nodes to evaluate in the computational graph.

        Parameters
        ----------
        eval_nodes: List[Node]
            The list of nodes whose values are to be computed.
        """
        self.eval_nodes = eval_nodes

    def run(self, input_values: Dict[Node, np.ndarray]) -> List[np.ndarray]:
        """Computes values of nodes in `eval_nodes` field with
        the computational graph input values given by the `input_values` dict.

        Parameters
        ----------
        input_values: Dict[Node, np.ndarray]
            The dictionary providing the values for input nodes of the
            computational graph.
            Throw ValueError when the value of any needed input node is
            not given in the dictionary.

        Returns
        -------
        eval_values: List[np.ndarray]
            The list of values for nodes in `eval_nodes` field.
        """
        order: List[Node] = [
            node
            for output_node in self.eval_nodes
            for node in topological_sort(output_node)
        ]
        nodeToVal = {}
        for node, value in input_values.items():
            nodeToVal[node] = value
        for node in order:
            # Used in the derivation of multiple eval_nodes
            # or it is an input node
            # Safe to ignore
            if node in nodeToVal:
                continue
            # The node is an operator
            output_value = node.op.compute(
                node, [nodeToVal[input_node] for input_node in node.inputs]
            )
            nodeToVal[node] = output_value
        return [nodeToVal[node] for node in self.eval_nodes]


def gradients(output_node: Node, nodes: List[Node]) -> List[Node]:
    """Construct the backward computational graph, which takes gradient
    of given output node with respect to each node in input list.
    Return the list of gradient nodes, one for each node in the input list.

    Parameters
    ----------
    output_node: Node
        The output node to take gradient of, whose gradient is 1.

    nodes: List[Node]
        The list of nodes to take gradient with regard to.

    Returns
    -------
    grad_nodes: List[Node]
        A list of gradient nodes, one for each input nodes respectively.
    """
    node_to_grad = defaultdict(list)
    node_to_grad[output_node] = [ones_like(output_node)]
    adjoints = {}

    for i in reversed(topological_sort(output_node)):
        adjoints[i] = sum(node_to_grad[i])
        if isinstance(i, Variable):
            continue

        i_grads = i.op.gradient(i, adjoints[i])

        for k, grad in zip(i.inputs, i_grads):
            node_to_grad[k].append(grad)

    return [adjoints[i] for i in nodes]
