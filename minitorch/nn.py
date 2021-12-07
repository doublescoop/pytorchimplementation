from numpy import multiply
from .fast_ops import FastOps
from .tensor_functions import rand, Function
from . import operators


def tile(input, kernel):
    """
    Reshape an image tensor for 2D pooling

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        (:class:`Tensor`, int, int) : Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    print("original_hw:", height, width)
    print("kernelsize:", kh, kw)
    new_height = height // kh
    new_width = width // kw
    print("n_h:", new_height, "n_w:", new_width)

    ## somehow this only change the view to new_width but not height...?
    # input = input.contiguous().view(batch, channel, new_height, new_width, kh * kw)
    # print('input:', input.shape )
    # write down 2*4*4

    input = input.contiguous().view(batch, channel, height, new_width, kw)
    input = input.permute(0, 1, 3, 2, 4)
    input = input.contiguous().view(batch, channel, new_width, new_height, kh * kw)
    input = input.permute(0, 1, 3, 2, 4)

    return input, new_height, new_width
    # raise NotImplementedError("Need to implement for Task 4.3")


def avgpool2d(input, kernel):
    """
    Tiled average pooling 2D

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        :class:`Tensor` : pooled tensor
    """
    batch, channel, height, width = input.shape
    # TODO: Implement for Task 4.3.

    input = tile(input, kernel)[0]
    input = input.mean(4)  # mean by kh*kw
    input = input.view(batch, channel, input.shape[2], input.shape[3])
    return input
    # raise NotImplementedError("Need to implement for Task 4.3")


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input, dim):
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx, input, dim):
        "Forward of max should be max reduction"

        # TODO: Implement for Task 4.4.
        res = max_reduce(input, dim)
        ctx.save_for_backward(input, dim)
        return res
        # raise NotImplementedError("Need to implement for Task 4.4")

    @staticmethod
    def backward(ctx, grad_output):
        "Backward of max should be argmax (see above)"

        # TODO: Implement for Task 4.4.
        input, dim = ctx.saved_values
        res = argmax(input, dim)
        return grad_output * res
        # raise NotImplementedError("Need to implement for Task 4.4")


max = Max.apply


def softmax(input, dim):
    r"""
    Compute the softmax as a tensor.

    .. math::

        z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply softmax

    Returns:
        :class:`Tensor` : softmax tensor
    """
    # TODO: Implement for Task 4.4.
    input = input.exp()
    sumoverdim = input.sum(dim)
    input = input / sumoverdim
    return input
    # raise NotImplementedError("Need to implement for Task 4.4")


def logsoftmax(input, dim):
    r"""
    Compute the log of the softmax as a tensor.

    .. math::

        z_i = x_i - \log \sum_i e^{x_i}

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply log-softmax

    Returns:
        :class:`Tensor` : log of softmax tensor
    """
    # TODO: Implement for Task 4.4.

    m = max(input, dim)
    x = input - m
    x = x.exp().sum(dim).log()
    input = input - x - m
    return input
    # raise NotImplementedError("Need to implement for Task 4.4")


def maxpool2d(input, kernel):
    """
    Tiled max pooling 2D

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        :class:`Tensor` : pooled tensor
    """
    batch, channel, height, width = input.shape
    # TODO: Implement for Task 4.4.
    input = tile(input, kernel)[0]
    input = max(input, 4)
    input = input.view(batch, channel, int(height / kernel[0]), int(width / kernel[1]))
    return input
    # raise NotImplementedError("Need to implement for Task 4.4")


def dropout(input, rate, ignore=False):
    """
    Dropout positions based on random noise.

    Args:
        input (:class:`Tensor`): input tensor
        rate (float): probability [0, 1) of dropping out each position
        ignore (bool): skip dropout, i.e. do nothing at all

    Returns:
        :class:`Tensor` : tensor with randoom positions dropped out
    """
    # TODO: Implement for Task 4.4.
    if not ignore:

        drop_n = rand(input.shape) < (1 - rate)
        res = multiply(input, drop_n)
        return res

    else:
        return input
    # raise NotImplementedError("Need to implement for Task 4.4")
    # got the idea from: https://stackoverflow.com/a/54170758
