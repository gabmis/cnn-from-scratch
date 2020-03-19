import numpy as np


def convolve2d(batch, filtr, full=True):
    """Performs a 2D convolution using a strided view of the image.

    We assume a convolution an odd kernel size, a  "stride" of 0
    and we add padding to preserve the shape of the input.

    Our numpy.ndarray's are stored as contiguous memory blocks.
    Strides are the number of bytes to move in order to increase
    the index by 1 in a given dimension.
    For example if a (3, 3) float (8 bytes) array has strides (24, 8), it means:
        - element at index [1, 0] is 24 bytes further than element at index [0, 0]
        - element at index [0, 1] is 8 bytes further than element at index [0, 0]

    In our case we want to vectorize the convolution computation.
    In order to do so we first create a view of the sub-matrices we are going to
    inner dot with the convolution filters. Furthermore, we want this view to
    be aligned with the resulting shape of the convolution. I.E we want to form a 4d array
    which has sub-matrix [i: i + k, j, j + k] at index [i, j].
    This means that moving from [i, 0, 0, 0] to [i + 1, 0, 0, 0] should be the same as
    moving from [i, 0] to [i + 1, 0] in the original matrix.
    Similarly, at the sub-matrix level, moving from [0, 0, i, 0] to [0, 0, i + 1, 0]
    is also the same. Thus the strides of our 4d array should
    be two copies of the original strides concatenated.

    Now we want to actually compute the convolution. This requires performing
    the Frobenius inner product between the filter and the sub-matrices.
    It can be achieved with the tricky but very handy, numpy.einsum function
    which extends einstein summation to a wide variety of linear operations.
    The summation can be represented by a string of the form
    "operand 1 axes, operand 2 axes -> output axes" where the axes which are
    stated for both operands and are absent from the output will be summed over.
    In the case of the Frobenius inner product we sum over the filter and the
    last two axes of our strided view. This can be written as "kl,ijkl->ij"
    or "kl,...kl->..." with ellipsis. To be perfectly precise it is "klm,...klm->..."
    when taking depth into account.

    N.B. The implementation expects a "batch" and "depth" dimensions.
    """
    img_shape = batch.shape
    if full:
        pad_width = (
            int(filtr.shape[0] / 2),
            int(filtr.shape[0] / 2) - filtr.shape[0] % 2,
        )
        batch = np.pad(batch, ((0, 0), pad_width, pad_width, (0, 0)))
        out_shape = img_shape[:3]
    else:
        out_shape = (img_shape[0], *(np.subtract(img_shape[1:3], filtr.shape[:2]) + 1))
    sub_images = np.lib.stride_tricks.as_strided(
        batch,
        shape=out_shape + filtr.shape,
        strides=(batch.strides[0],) + batch.strides[1:3] * 2 + (batch.strides[-1],),
        writeable=False,
    )
    return np.einsum("klm,...klm->...", filtr, sub_images)


def convolution(inpt, filters, full=True):
    if full:
        out_shape = inpt.shape[:3]
    else:
        out_shape = (inpt.shape[0], *(np.subtract(inpt.shape[1:3], filters.shape[1:3]) + 1))
    out = np.zeros((*out_shape, filters.shape[0]))
    for i, filtr in enumerate(filters):
        out[..., i] = convolve2d(inpt, filtr, full)
    return out


def grad_convolution(deltas, inpt, weights, full=True):
    grads = np.zeros((deltas.shape[0], *weights.shape))
    # if the forward convolution was "full" pad again
    if full:
        pad_width = (
            int(weights.shape[1] / 2),
            int(weights.shape[1] / 2) - (1 - weights.shape[1] % 2),
        )
        inpt = np.pad(inpt, ((0, 0), pad_width, pad_width, (0, 0)))
    # iterate in batch, because here the filters (deltas) are the "batched" entity
    for b in range(deltas.shape[0]):
        # iterate over each filter
        for f in range(deltas.shape[-1]):
            # iterate over channels
            for c in range(weights.shape[-1]):
                # extend dimensions to keep batch and depth dimensions in input
                grad = convolve2d(
                    inpt[b, ..., c][None, ..., None],
                    deltas[b, ..., f][..., None],
                    full=False,
                )
                grads[b, f, ..., c] = grad
    return grads


def deltas_convolution(deltas, weights):
    deltas_next = np.zeros((*deltas.shape[:-1], weights.shape[-1]))
    for c in range(weights.shape[-1]):
        for f in range(deltas.shape[-1]):
            deltas_next[..., c] += convolve2d(
                deltas[..., f][..., None],
                rot180(weights[f, ..., c][..., None]),
                full=True,
            )
    return deltas_next


def rot180(array, axes=(1, 2)):
    return np.rot90(np.rot90(array, axes=axes), axes=axes)
