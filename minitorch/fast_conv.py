# from inspect import indentsize
import numpy as np
from .tensor_data import to_index, index_to_position, broadcast_index
from .tensor_functions import Function
from numba import njit, prange


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


@njit(parallel=True)
def tensor_conv1d(
    out,
    out_shape,
    out_strides,
    out_size,
    input,
    input_shape,
    input_strides,
    weight,
    weight_shape,
    weight_strides,
    reverse,
):
    """
    1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (array): storage for `input` tensor.
        input_shape (array): shape for `input` tensor.
        input_strides (array): strides for `input` tensor.
        weight (array): storage for `input` tensor.
        weight_shape (array): shape for `input` tensor.
        weight_strides (array): strides for `input` tensor.
        reverse (bool): anchor weight at left or right
    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    s1 = input_strides
    s2 = weight_strides
    # TODO: Implement for Task 4.1.

    # if (input_shape[-1] % weight_shape[0]) != 0:
    #     for p in range(int(input_shape[-1] % weight_shape[0])):
    #         input[width+p] = 0

    for p in prange(out_size):
        out_index = np.zeros(3, np.int32)
        in_index = np.zeros(3, np.int32)
        w_index = np.zeros(3, np.int32)
        to_index(p, out_shape, out_index)
        out_pos = index_to_position(out_index, out_strides)

        acc = 0.0
        for k in range(kw):  # for every batch # weight_shape[2]
            tmp = out_index[2] + k
            if reverse:
                tmp = tmp - kw + 1
            for j in range(in_channels):  # weight_shape[1]

                if tmp < width:
                    in_index[0], in_index[1], in_index[2] = out_index[0], j, tmp
                    w_index[0], w_index[1], w_index[2] = out_index[1], j, k

                    in_pos = index_to_position(in_index, s1)
                    w_pos = index_to_position(w_index, s2)
                    acc += input[in_pos] * weight[w_pos]

        out[out_pos] = acc


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx, input, weight):
        """
        Compute a 1D Convolution

        Args:
            ctx : Context
            input (:class:`Tensor`) : batch x in_channel x h x w
            weight (:class:`Tensor`) : out_channel x in_channel x kh x kw

        Returns:
            (:class:`Tensor`) : batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


@njit(parallel=True, fastmath=True)
def tensor_conv2d(
    out,
    out_shape,
    out_strides,
    out_size,
    input,
    input_shape,
    input_strides,
    weight,
    weight_shape,
    weight_strides,
    reverse,
):
    """
    2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (array): storage for `input` tensor.
        input_shape (array): shape for `input` tensor.
        input_strides (array): strides for `input` tensor.
        weight (array): storage for `input` tensor.
        weight_shape (array): shape for `input` tensor.
        weight_strides (array): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right
    """
    batch_, out_channels, _, _ = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides
    # inners
    # s10, s11, s12, s13 = s1[0], s1[1], s1[2], s1[3]
    # s20, s21, s22, s23 = s2[0], s2[1], s2[2], s2[3]

    for p in prange(out_size):
        out_index = np.zeros(4, np.int32)
        in_index = np.zeros(4, np.int32)
        w_index = np.zeros(4, np.int32)
        out_pos = index_to_position(out_index, out_strides)
        to_index(p, out_shape, out_index)
        acc = 0.0

        for i in range(in_channels):
            for j in range(kh):
                for k in range(kw):
                    # set the region
                    tmp_h = out_index[2] + j
                    tmp_w = out_index[3] + k
                    if reverse:
                        # region start from bottom right when reversed(backward)
                        tmp_h = tmp_h - kh + 1
                        tmp_w = tmp_w - kw + 1
                    if tmp_h < j and tmp_w < k:
                        in_index[0], in_index[1], in_index[2], in_index[3] = (
                            # (batch, C_in, h, w)
                            out_index[0],
                            i,
                            tmp_h,
                            tmp_w,
                        )
                        w_index[0], w_index[1], w_index[2], w_index[3] = (
                            # (C_out, C_in, h, w)
                            out_index[1],
                            i,
                            j,
                            k,
                        )

                        in_pos = index_to_position(in_index, s1)
                        w_pos = index_to_position(w_index, s2)
                        acc += input[in_pos] * weight[w_pos]

        out[out_pos] = acc

    # TODO: Implement for Task 4.2.


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx, input, weight):
        """
        Compute a 2D Convolution

        Args:
            ctx : Context
            input (:class:`Tensor`) : batch x in_channel x h x w
            weight (:class:`Tensor`) : out_channel x in_channel x kh x kw

        Returns:
            (:class:`Tensor`) : batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
