import numpy as np
import pytest

from mlcombine.core.tensor import UnifiedTensor


@pytest.fixture
def t2d():
    return UnifiedTensor(np.array([[1.0, 2.0], [3.0, 4.0]]))


@pytest.fixture
def t1d():
    return UnifiedTensor(np.array([1.0, 2.0, 3.0]))


class TestCreation:
    def test_from_numpy(self):
        t = UnifiedTensor(np.array([1, 2, 3]))
        assert isinstance(t, UnifiedTensor)
        assert "NumpyAdapter" in repr(t)

    def test_numpy_roundtrip(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        t = UnifiedTensor(a)
        assert np.array_equal(t.numpy(), a)

    def test_copy_independence(self, t2d):
        c = t2d.copy()
        c_np = c.numpy()
        c_np[0, 0] = 999.0
        assert t2d.numpy()[0, 0] == 1.0


class TestArithmetic:
    def test_add(self, t2d):
        result = (t2d + 10).numpy()
        assert np.array_equal(result, [[11, 12], [13, 14]])

    def test_sub(self, t2d):
        result = (t2d - 1).numpy()
        assert np.array_equal(result, [[0, 1], [2, 3]])

    def test_mul(self, t2d):
        result = (t2d * 2).numpy()
        assert np.array_equal(result, [[2, 4], [6, 8]])

    def test_div(self, t2d):
        result = (t2d / 2).numpy()
        assert np.array_equal(result, [[0.5, 1], [1.5, 2]])

    def test_neg(self, t2d):
        result = (-t2d).numpy()
        assert np.array_equal(result, [[-1, -2], [-3, -4]])

    def test_matmul(self, t2d):
        result = (t2d @ t2d).numpy()
        assert np.array_equal(result, [[7, 10], [15, 22]])

    def test_pow(self, t2d):
        result = (t2d**2).numpy()
        assert np.array_equal(result, [[1, 4], [9, 16]])


class TestUnaryMath:
    def test_abs(self, t2d):
        t = UnifiedTensor(np.array([[-1.0, -2.0], [3.0, 4.0]]))
        result = t.abs().numpy()
        assert np.array_equal(result, [[1, 2], [3, 4]])

    def test_sqrt(self, t2d):
        result = t2d.sqrt().numpy()
        assert np.allclose(result, [[1, 1.41421356], [1.73205081, 2]])

    def test_exp(self, t2d):
        result = t2d.exp().numpy()
        assert np.allclose(result[0, 0], np.e)

    def test_log(self, t2d):
        result = t2d.log().numpy()
        assert np.allclose(result, [[0, 0.69314718], [1.09861229, 1.38629436]])

    def test_clip(self, t2d):
        result = t2d.clip(2.0, 3.0).numpy()
        assert np.array_equal(result, [[2, 2], [3, 3]])


class TestReductions:
    def test_sum(self, t2d):
        assert t2d.sum().item() == 10.0

    def test_sum_dim(self, t2d):
        result = t2d.sum(dim=0).numpy()
        assert np.array_equal(result, [4, 6])

    def test_mean(self, t2d):
        assert t2d.mean().item() == 2.5

    def test_mean_dim(self, t2d):
        result = t2d.mean(dim=0).numpy()
        assert np.array_equal(result, [2, 3])

    def test_min_max(self, t2d):
        assert t2d.min().item() == 1.0
        assert t2d.max().item() == 4.0

    def test_argmin_argmax(self):
        t = UnifiedTensor(np.array([3, 1, 4, 1, 5]))
        assert t.argmin().item() == 1
        assert t.argmax().item() == 4


class TestShapeOps:
    def test_reshape(self, t2d):
        result = t2d.reshape(4).numpy()
        assert np.array_equal(result, [1, 2, 3, 4])

    def test_transpose(self, t2d):
        result = t2d.transpose().numpy()
        assert np.array_equal(result, [[1, 3], [2, 4]])

    def test_flatten(self, t2d):
        result = t2d.flatten().numpy()
        assert np.array_equal(result, [1, 2, 3, 4])

    def test_squeeze(self):
        t = UnifiedTensor(np.array([[[1, 2, 3]]]))
        result = t.squeeze().numpy()
        assert np.array_equal(result, [1, 2, 3])

    def test_unsqueeze(self, t1d):
        result = t1d.unsqueeze(0).numpy()
        assert result.shape == (1, 3)


class TestProperties:
    def test_shape(self, t2d):
        assert t2d.shape == (2, 2)

    def test_ndim(self, t2d):
        assert t2d.ndim == 2

    def test_dtype(self, t2d):
        assert t2d.dtype is not None

    def test_device(self, t2d):
        assert t2d.device == "cpu"

    def test_T(self, t2d):
        result = t2d.T.numpy()
        assert np.array_equal(result, [[1, 3], [2, 4]])

    def test_len(self):
        t = UnifiedTensor(np.array([[1, 2], [3, 4], [5, 6]]))
        assert len(t) == 3

    def test_repr(self, t2d):
        r = repr(t2d)
        assert "UnifiedTensor" in r
        assert "NumpyAdapter" in r


class TestArrayProtocol:
    def test_np_asarray(self, t2d):
        a = np.asarray(t2d)
        assert np.array_equal(a, [[1, 2], [3, 4]])

    def test_np_array_with_dtype(self, t2d):
        a = np.array(t2d, dtype=np.int32)
        assert a.dtype == np.int32
        assert np.array_equal(a, [[1, 2], [3, 4]])


class TestUtility:
    def test_tolist(self, t2d):
        assert t2d.tolist() == [[1.0, 2.0], [3.0, 4.0]]

    def test_item_scalar(self):
        t = UnifiedTensor(np.array(42.0))
        assert t.item() == 42.0
