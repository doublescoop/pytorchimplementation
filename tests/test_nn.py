import minitorch
from hypothesis import given
from .strategies import tensors, assert_close
import pytest
import numpy as np


@pytest.mark.task4_3
def test_tile():
    tensor = minitorch.tensor(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20],
            [21, 22, 23, 24],
        ]
    )
    tensor = tensor.view(1, 1, tensor.shape[0], tensor.shape[1])
    kernel_size = (2, 2)

    tiled_tensor, new_height, new_width = minitorch.nn.tile(tensor, kernel_size)

    assert new_height == 3
    assert new_width == 2
    assert tiled_tensor.shape[0:4] == (1, 1, new_height, new_width)
    assert tiled_tensor.shape[-1] == 2 * 2

    # get the sum of the cells of each tile as a numpy array for
    # later assertion.
    sums_of_each_tile = tiled_tensor.sum(4)._tensor._storage

    # directly compute expected sums via visual inspection of `tensor`
    # declaration given the 2,2 kernel size.
    expected_sums_of_each_tile = [
        # top left 2x2 tile
        (1 + 2 + 5 + 6),  # 14
        (3 + 4 + 7 + 8),  # 22
        (9 + 10 + 13 + 14),  # 46
        (11 + 12 + 15 + 16),  # 54
        (17 + 18 + 21 + 22),  # 78
        # bottom right 2x2 tile
        (19 + 20 + 23 + 24),  # 86
    ]

    # assert that the expected sums of each tile equal the desired,
    # agnostic to the _order of the tiles_ in the tiled tensor. In order to be
    # agnostic to order, sort the expected and actual sums before asserting.
    assert np.array_equal(
        np.sort(sums_of_each_tile), np.sort(expected_sums_of_each_tile)
    )


@pytest.mark.task4_3
@given(tensors(shape=(1, 1, 4, 4)))
def test_avg(t):
    out = minitorch.avgpool2d(t, (2, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(2)]) / 4.0
    )

    out = minitorch.avgpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(1)]) / 2.0
    )

    out = minitorch.avgpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(1) for j in range(2)]) / 2.0
    )
    minitorch.grad_check(lambda t: minitorch.avgpool2d(t, (2, 2)), t)


@pytest.mark.task4_4
@given(tensors(shape=(2, 3, 4)))
def test_max(t):
    # TODO: Implement for Task 4.4.
    out = minitorch.nn.max(t, 2)  # 3rd dim
    assert out[0, 0, 0] == max(t[0, 0, i] for i in range(4))
    out = minitorch.nn.max(t, 1)
    assert out[0, 0, 0] == max(t[0, i, 0] for i in range(3))
    out = minitorch.nn.max(t, 0)
    assert out[0, 0, 0] == max(t[i, 0, 0] for i in range(2))
    rand_tensor = minitorch.rand(t.shape) * 1e-5
    t = t + rand_tensor
    minitorch.grad_check(lambda t: minitorch.nn.max(t, 2), t)
    # raise NotImplementedError("Need to implement for Task 4.4")


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_max_pool(t):
    out = minitorch.maxpool2d(t, (2, 2))
    print(out)
    print(t)
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(2)])
    )

    out = minitorch.maxpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(1)])
    )

    out = minitorch.maxpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(1) for j in range(2)])
    )


@pytest.mark.task4_4
@given(tensors())
def test_drop(t):
    q = minitorch.dropout(t, 0.0)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]
    q = minitorch.dropout(t, 1.0)
    assert q[q._tensor.sample()] == 0.0
    q = minitorch.dropout(t, 1.0, ignore=True)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_softmax(t):
    q = minitorch.softmax(t, 3)
    x = q.sum(dim=3)
    assert_close(x[0, 0, 0, 0], 1.0)

    q = minitorch.softmax(t, 1)
    x = q.sum(dim=1)
    assert_close(x[0, 0, 0, 0], 1.0)

    minitorch.grad_check(lambda a: minitorch.softmax(a, dim=2), t)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_log_softmax(t):
    q = minitorch.softmax(t, 3)
    q2 = minitorch.logsoftmax(t, 3).exp()
    for i in q._tensor.indices():
        assert_close(q[i], q2[i])

    minitorch.grad_check(lambda a: minitorch.logsoftmax(a, dim=2), t)
