import pytest

import aesara
import aesara.sparse as sparse
import aesara.tensor as at


@pytest.mark.parametrize("is_sparse", [False, True])
# @pytest.mark.parametrize("is_sparse", [False])
class TestTensorMethods:
    """checks that dense and sparse tensors have the same interface"""

    @staticmethod
    def _get_x(is_sparse):
        x = at.dmatrix("x")
        if is_sparse:
            x = sparse.csr_from_dense(x)
        return x

    @staticmethod
    def _get_xy(is_sparse):
        x = at.lmatrix("x")
        y = at.lmatrix("y")
        if is_sparse:
            x = sparse.csr_from_dense(x)
            y = sparse.csr_from_dense(y)
        return x, y

    @pytest.mark.parametrize(
        "method",
        [
            "__abs__",
            "__neg__",
            "__ceil__",
            "__floor__",
            "__trunc__",
            "transpose",
            "any",
            "all",
            "flatten",
            "ravel",
            "arccos",
            "arcsin",
            "arctan",
            "arccosh",
            "arcsinh",
            "arctanh",
            "ceil",
            "cos",
            "cosh",
            "deg2rad",
            "exp",
            "exp2",
            "expm1",
            "floor",
            "log",
            "log10",
            "log1p",
            "log2",
            "rad2deg",
            "sin",
            "sinh",
            "sqrt",
            "tan",
            "tanh",
            "copy",
            "sum",
            "prod",
            "mean",
            "var",
            "std",
            "min",
            "max",
            "argmin",
            "argmax",
            "nonzero",
            "nonzero_values",
            "argsort",
            "conj",
            "round",
            "trace",
            "zeros_like",
            "ones_like",
            "cumsum",
            "cumprod",
            "ptp",
            "squeeze",
            "diagonal",
        ],
    )
    def test_unary(self, is_sparse, method):
        x = self._get_x(is_sparse)
        method_to_call = getattr(x, method)
        z = method_to_call()
        f = aesara.function([x], z, on_unused_input="ignore")
        print(f([[1.1, 0.0, 2.0], [-1.0, 0.0, 0.0]]))

    @pytest.mark.parametrize(
        "method",
        [
            "__lt__",
            "__le__",
            "__gt__",
            "__ge__",
            "__and__",
            "__or__",
            "__xor__",
            "__add__",
            "__sub__",
            "__mul__",
            "__pow__",
            "__mod__",
            "__divmod__",
            "__truediv__",
            "__floordiv__",
        ],
    )
    def test_binary(self, is_sparse, method):
        x, y = self._get_xy(is_sparse)
        method_to_call = getattr(x, method)
        z = method_to_call(y)
        f = aesara.function([x, y], z)
        f(
            [[1, 0, 2], [-1, 0, 0]],
            [[1, 1, 2], [1, 4, 1]],
        )

    def test_reshape(self, is_sparse):
        x = self._get_x(is_sparse)
        z = x.reshape((3, 2))
        f = aesara.function([x], z)
        f([[1.1, 0.0, 2.0], [-1.0, 0.0, 0.0]])

    def test_dimshuffle(self, is_sparse):
        x = self._get_x(is_sparse)
        z = x.dimshuffle((1, 0))
        f = aesara.function([x], z)
        f([[1.1, 0.0, 2.0], [-1.0, 0.0, 0.0]])

    def test_getitem(self, is_sparse):
        x = self._get_x(is_sparse)
        z = x[:, :2]
        f = aesara.function([x], z)
        f([[1.1, 0.0, 2.0], [-1.0, 0.0, 0.0]])

    def test_dot(self, is_sparse):
        x, y = self._get_xy(is_sparse)
        z = x.__dot__(y)
        f = aesara.function([x, y], z)
        f(
            [[1, 0, 2], [-1, 0, 0]],
            [[-1], [2], [1]],
        )

    def test_repeat(self, is_sparse):
        x = self._get_x(is_sparse)
        z = x.repeat(2, axis=1)
        f = aesara.function([x], z)
        f([[1.1, 0.0, 2.0], [-1.0, 0.0, 0.0]])
