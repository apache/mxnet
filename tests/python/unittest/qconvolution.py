import numpy as np


def STE(x):
    return np.ones_like(x)


def quantize_k(x, k): # x in [0,1]
    vmax = 2 ** k - 1
    return np.round(x * vmax) / vmax


def quantize_k_grad(x):
    return STE(x)


def bound_activation(x, method='clip'):
    return {
        'tanh': lambda x: (np.tanh(x) + 1) / 2,
        'min': lambda x: np.minimum(1, np.abs(x)),
        'clip': lambda x: np.clip(x, 0, 1)
    }[method](x)


def bound_activation_grad(x, method='clip'):
    return {
        'tanh': lambda x: (1 - np.tanh(x)**2) * 0.5,
        'min': lambda x: np.sign(x) * (np.clip(x, -1, 1) == x),
        'clip': lambda x: np.clip(x, 0, 1) == x
    }[method](x)


def dorefa_weight_act(w):
    return 0.5 * np.tanh(w) / np.max(np.abs(np.tanh(w))) + 0.5


def dorefa_weight_act_grad(w):
    '''
    1 - tanh^2(w)                               1
    -------------- (1 - ind_{max}[|tanh(w)|]) * -
    max(|tanh(w)|)                              2

    :param w: weights
    :return: dy/dw
    '''
    tanhw = np.tanh(w)
    abs_tanh_w = np.abs(tanhw)
    mw = np.max(abs_tanh_w)
    return 0.5 * (1 - tanhw**2) / mw * (1 - (abs_tanh_w == mw))


def get_weight_scaling(w):
    """
    w shape: FCHW
    """
    return np.abs(w).mean(axis=(1, 2, 3))#, keepdims=True)#.reshape(-1, 1, 1, 1)


# from chainer https://github.com/chainer/chainer/blob/master/chainer/utils/conv.py#L65
def get_conv_outsize(size, k, s, p, cover_all=False, d=1):
    """Calculates output size of convolution.
    This function takes the size of input feature map, kernel, stride, and
    pooling of one particular dimension, then calculates the output feature
    map size of that dimension.
    .. seealso:: :func:`~chainer.utils.get_deconv_outsize`
    Args:
        size (int): The size of input feature map. It usually is the length of
            a side of feature map.
        k (int): The size of convolution kernel.
        s (int): The size of stride.
        p (int): The size of padding.
        cover_all (bool): Use ``cover_all`` option or not.
        d (int): The size of dilation.
    Returns:
        int: The expected output size of the convolution operation.
    """
    dk = k + (k - 1) * (d - 1)
    if cover_all:
        return (size + p * 2 - dk + s - 1) // s + 1
    else:
        return (size + p * 2 - dk) // s + 1


def im2col_cpu(img, kh, kw, sy, sx, ph, pw, pval=0, cover_all=False, dy=1, dx=1, out_h=None, out_w=None):
    n, c, h, w = img.shape
    if out_h is None:
        out_h = get_conv_outsize(h, kh, sy, ph, cover_all, dy)
    assert out_h > 0, 'Height in the output should be positive.'
    if out_w is None:
        out_w = get_conv_outsize(w, kw, sx, pw, cover_all, dx)
    assert out_w > 0, 'Width in the output should be positive.'

    img = np.pad(img,
                 ((0, 0), (0, 0), (ph, ph + sy - 1), (pw, pw + sx - 1)),
                 mode='constant', constant_values=(pval,))
    col = np.ndarray((n, c, kh, kw, out_h, out_w), dtype=img.dtype)

    for j in range(kh):
        jdy = j * dy
        j_lim = jdy + sy * out_h
        for i in range(kw):
            idx = i * dx
            i_lim = idx + sx * out_w
            col[:, :, j, i, :, :] = img[:, :, jdy:j_lim:sy, idx:i_lim:sx]

    return col


def col2im_cpu(col, sy, sx, ph, pw, h, w, dy=1, dx=1):
    n, c, kh, kw, out_h, out_w = col.shape
    img = np.zeros((n, c, h + 2 * ph + sy - 1, w + 2 * pw + sx - 1),
                   dtype=col.dtype)
    for j in range(kh):
        jdy = j * dy
        j_lim = jdy + sy * out_h
        for i in range(kw):
            idx = i * dx
            i_lim = idx + sx * out_w
            img[:, :, jdy:j_lim:sy, idx:i_lim:sx] += col[:, :, j, i]
    return img[:, :, ph:h + ph, pw:w + pw]


def conv_fwd(d, w, sy=1, sx=1, ph=0, pw=0, pval=0, cover_all=False, dy=1, dx=1):
    """
    w shape: FCHW
    """
    nf, nc, kh, kw = w.shape
    col = im2col_cpu(d, kh, kw, sy, sx, ph, pw, pval=pval, cover_all=cover_all, dy=dy, dx=dx)
    return np.einsum('nchwab,ochw->noab', col, w)


def qconv_fwd(d, w, act_bit=32, weight_bit=2, sy=1, sx=1, ph=0, pw=0, pval=0, cover_all=False, dy=1, dx=1, scaling=False, pre_activated=False):
    if weight_bit < 32:
        if weight_bit == 1:
            qw = np.sign(w)
        else:
            qw = 2 * quantize_k(dorefa_weight_act(w), weight_bit) - 1
    if act_bit < 32:
        if act_bit == 1:
            qd = np.sign(d)
        else:
            if pre_activated:
                qd = quantize_k(d, act_bit)
            else:
                qd = quantize_k(bound_activation(d, method='clip'), act_bit)

    out = conv_fwd(qd, qw, sy=sy, sx=sx, ph=ph, pw=pw, pval=pval, cover_all=cover_all, dy=dy, dx=dx)

    if scaling:
        # out dims: NFHW
        out *= get_weight_scaling(w).reshape(1, -1, 1, 1)

    return out


def conv_bwd(x, w, gy, sy=1, sx=1, ph=0, pw=0, dy=1, dx=1, pval=0, cover_all=False, indexes=[0, 1]):
    # x, W = self.get_retained_inputs()
    # gy, = grad_outputs
    nf, nc, kh, kw = w.shape
    ret = []
    if 0 in indexes:
        xh, xw = x.shape[2:]
        gcol = np.tensordot(w, gy, (0, 1)).astype(x.dtype, copy=False)
        gcol = np.rollaxis(gcol, 3)
        gx = col2im_cpu(gcol, sy, sx, ph, pw, xh, xw)
        ret.append(gx)
    if 1 in indexes:
        col = im2col_cpu(x, kh, kw, sy, sx, ph, pw, pval=pval, cover_all=cover_all, dy=dy, dx=dx)
        # FCHW          NFHW  NCHWAB
        gW = np.tensordot(gy, col, ((0, 2, 3), (0, 4, 5))).astype(w.dtype, copy=False)
        ret.append(gW)
    if 2 in indexes:
        gb = gy.sum(axis=(0, 2, 3))
        ret.append(gb)
    return ret


def qconv_bwd(x, w, gy, act_bit=32, weight_bit=2, sy=1, sx=1, ph=0, pw=0, pval=0, cover_all=False, dy=1, dx=1, scaling=False, pre_activated=False):
    # QCNN(Q(A), Q(W))
    gy_, scaling_vec = None,  None
    if scaling:
        gy_ = np.copy(gy)
        scaling_vec = get_weight_scaling(w)

    qw = w
    if weight_bit < 32:
        if weight_bit == 1:
            qw = np.sign(w)
        else:
            qw = 2 * quantize_k(dorefa_weight_act(w), weight_bit) - 1

    qx = x
    if act_bit < 32:
        if act_bit == 1:
            qx = np.sign(x)
        else:
            if pre_activated:
                qx = quantize_k(x, act_bit)
            else:
                qx = quantize_k(bound_activation(x, method='clip'), act_bit)

    if scaling:
        # scaling_vec dims: FCHW -> F111, gy dims: NFHW -- bring scaling dimension from 0 to 1 --> 1F11
        gy *= scaling_vec.reshape(1, -1, 1, 1)
    gqx, gqw = conv_bwd(qx, qw, gy, sy=sy, sx=sx, ph=ph, pw=pw, pval=pval, cover_all=cover_all, dy=dy, dx=dx)

    # do not do activation gradient, because that is done in QActivation layer, not CNN in MXNET
    gx = gqx
    if act_bit < 32 and not pre_activated:
        gx = gqx * quantize_k_grad(bound_activation(x, method='clip')) * bound_activation_grad(x, method='clip')

    gw = gqw
    if weight_bit < 32:
        if weight_bit == 1:
            # clip or abs function??
            pass
        else:
            gw = gqw * 2 * quantize_k_grad(dorefa_weight_act(w)) * dorefa_weight_act_grad(w)
    if scaling:
        f = conv_fwd(qx, qw, sy=sy, sx=sx, ph=ph, pw=pw, pval=pval, cover_all=cover_all, dy=dy, dx=dx)
        F, C, H, W = w.shape
        N = C * H * W
        # gy_ * f dims: NFHW --> sum except F, broadcast and multiply with rest partials
        # (ds/dw = d/dw 1/N * abs(w) = 1/N sign(w)) dims: FCHW --> transpose sum dimension from 1 to 0
        # we may not omit 1/N by using mean() instead of sum(), because N depends on dimensions of W not f
        gw += (gy_ * f).sum(axis=(0, 2, 3)).reshape(-1, 1, 1, 1) * np.sign(w) / N

    return gx, gw
