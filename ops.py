import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim
from util import log
import functools

def weight_initializer(initializer="org", stddev=0.02):
    """Returns the initializer for the given name.
    Args:
    initializer: Name of the initalizer. Use one in consts.INITIALIZERS.
    stddev: Standard deviation passed to initalizer.
    Returns:
    Initializer from `tf.initializers`.
    """
    if initializer == "normal":
        return tf.initializers.random_normal(stddev=stddev)
    if initializer == "truncated":
        return tf.initializers.truncated_normal(stddev=stddev)
    if initializer == "org":
        return tf.initializers.orthogonal()
    raise ValueError("Unknown weight initializer {}.".format(initializer))

def lrelu(inputs, leak=0.2, name = "lrelu"):
    """Performs leaky-ReLU on the input."""   
    return tf.maximum(inputs, leak * inputs, name=name)
    
def spectral_norm(inputs, epsilon=1e-12, singular_value="auto"):
    """Performs Spectral Normalization on a weight tensor.
    Details of why this is helpful for GAN's can be found in "Spectral
    Normalization for Generative Adversarial Networks", Miyato T. et al., 2018.
    [https://arxiv.org/abs/1802.05957].
    Args:
    inputs: The weight tensor to normalize.
    epsilon: Epsilon for L2 normalization.
    singular_value: Which first singular value to store (left or right). Use
    "auto" to automatically choose the one that has fewer dimensions.
    Returns:
    The normalized weight tensor.
    """
    if len(inputs.shape) < 2:
        raise ValueError("Spectral norm can only be applied to multi-dimensional tensors")

    # The paper says to flatten convnet kernel weights from (C_out, C_in, KH, KW)
    # to (C_out, C_in * KH * KW). Our Conv2D kernel shape is (KH, KW, C_in, C_out)
    # so it should be reshaped to (KH * KW * C_in, C_out), and similarly for other
    # layers that put output channels as last dimension. This implies that w
    # here is equivalent to w.T in the paper.
    w = tf.reshape(inputs, (-1, inputs.shape[-1]))

    # Choose whether to persist the first left or first right singular vector.
    # As the underlying matrix is PSD, this should be equivalent, but in practice
    # the shape of the persisted vector is different. Here one can choose whether
    # to maintain the left or right one, or pick the one which has the smaller
    # dimension. We use the same variable for the singular vector if we switch
    # from normal weights to EMA weights.
    var_name = inputs.name.replace("/ExponentialMovingAverage", "").split("/")[-1]
    var_name = var_name.split(":")[0] + "/u_var"
    if singular_value == "auto":
        singular_value = "left" if w.shape[0] <= w.shape[1] else "right"
    u_shape = (w.shape[0], 1) if singular_value == "left" else (1, w.shape[-1])
    u_var = tf.get_variable(
        var_name,
        shape=u_shape,
        dtype=w.dtype,
        initializer=tf.random_normal_initializer(),
        trainable=False)
    u = u_var

    # Use power iteration method to approximate the spectral norm.
    # The authors suggest that one round of power iteration was sufficient in the
    # actual experiment to achieve satisfactory performance.
    power_iteration_rounds = 1
    for _ in range(power_iteration_rounds):
        if singular_value == "left":
            # `v` approximates the first right singular vector of matrix `w`.
            v = tf.math.l2_normalize(
                tf.matmul(tf.transpose(w), u), axis=None, epsilon=epsilon)
            u = tf.math.l2_normalize(tf.matmul(w, v), axis=None, epsilon=epsilon)
        else:
            v = tf.math.l2_normalize(tf.matmul(u, w, transpose_b=True),
                               epsilon=epsilon)
            u = tf.math.l2_normalize(tf.matmul(v, w), epsilon=epsilon)

    # Update the approximation.
    with tf.control_dependencies([tf.assign(u_var, u, name="update_u")]):
        u = tf.identity(u)

    # The authors of SN-GAN chose to stop gradient propagating through u and v
    # and we maintain that option.
    u = tf.stop_gradient(u)
    v = tf.stop_gradient(v)

    if singular_value == "left":
        norm_value = tf.matmul(tf.matmul(tf.transpose(u), w), v)
    else:
        norm_value = tf.matmul(tf.matmul(v, w), u, transpose_b=True)
    norm_value.shape.assert_is_fully_defined()
    norm_value.shape.assert_is_compatible_with([1, 1])

    w_normalized = w / norm_value

    # Deflate normalized weights to match the unnormalized tensor.
    w_tensor_normalized = tf.reshape(w_normalized, inputs.shape)
    return w_tensor_normalized

def conv2d(inputs, output_dim, k_h, k_w, d_h, d_w, stddev=0.02, 
           name="conv2d", use_sn=False, use_bias=True):
    with tf.variable_scope(name):
        w = tf.get_variable(
            "kernel", [k_h, k_w, inputs.shape[-1].value, output_dim],
            initializer=weight_initializer(stddev=stddev))
        if use_sn:
            w = spectral_norm(w)
        outputs = tf.nn.conv2d(inputs, w, strides=[1, d_h, d_w, 1], padding="SAME")
        if use_bias:
            bias = tf.get_variable(
                "bias", [output_dim], initializer=tf.constant_initializer(0.0))
            outputs += bias
        return outputs

def deconv2d(inputs, output_shape, k_h, k_w, d_h, d_w, stddev=0.02, 
             name='deconv2d', use_sn=False):
    with tf.variable_scope(name):
        w = tf.get_variable(
            "kernel", [k_h, k_w, output_shape[-1], inputs.get_shape()[-1]],
            initializer=weight_initializer(stddev=stddev))
        if use_sn:
            w = spectral_norm(w)
        deconv = tf.nn.conv2d_transpose(
            inputs, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        bias = tf.get_variable(
            "bias", [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        return tf.reshape(tf.nn.bias_add(deconv, bias), tf.shape(deconv))

def linear(inputs, output_size, scope=None, stddev=0.02, bias_start=0.0, use_sn=False, use_bias=True):
    shape = inputs.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        kernel = tf.get_variable(
            "kernel",
            [shape[1], output_size],
            initializer=weight_initializer(stddev=stddev))
        if use_sn:
            kernel = spectral_norm(kernel)
        outputs = tf.matmul(inputs, kernel)
        if use_bias:
            bias = tf.get_variable(
                  "bias",
                  [output_size],
                  initializer=tf.constant_initializer(bias_start))
            outputs += bias
        return outputs

conv1x1 = functools.partial(conv2d, k_h=1, k_w=1, d_h=1, d_w=1)

def non_local_block(x, name, use_sn):
    """Self-attention (non-local) block.
    This method is used to exactly reproduce SAGAN and ignores Gin settings on
    weight initialization and spectral normalization.
    Args:
    x: Input tensor of shape [batch, h, w, c].
    name: Name of the variable scope.
    use_sn: Apply spectral norm to the weights.
    Returns:
    A tensor of the same shape after self-attention was applied.
    """
    def _spatial_flatten(inputs):     
        shape = inputs.shape     
        return tf.reshape(inputs, (-1, shape[1] * shape[2], shape[3]))
        
    with tf.variable_scope(name):     
        h, w, num_channels = x.get_shape().as_list()[1:]     
        num_channels_attn = num_channels // 8     
        num_channels_g = num_channels // 2

        # Theta path
        theta = conv1x1(x, num_channels_attn, name="conv2d_theta", use_sn=use_sn,
                        use_bias=False)
        theta = _spatial_flatten(theta)
        
        # Phi path     
        phi = conv1x1(x, num_channels_attn, name="conv2d_phi", use_sn=use_sn, use_bias=False)
        phi = tf.layers.max_pooling2d(inputs=phi, pool_size=[2, 2], strides=2)     
        phi = _spatial_flatten(phi)
        
        attn = tf.matmul(theta, phi, transpose_b=True)
        attn = tf.nn.softmax(attn)
        
        # G path
        g = conv1x1(x, num_channels_g, name="conv2d_g", use_sn=use_sn,
                    use_bias=False)
        g = tf.layers.max_pooling2d(inputs=g, pool_size=[2, 2], strides=2)
        g = _spatial_flatten(g)
        
        attn_g = tf.matmul(attn, g)
        attn_g = tf.reshape(attn_g, [-1, h, w, num_channels_g])
        sigma = tf.get_variable("sigma", [], initializer=tf.zeros_initializer())
        attn_g = conv1x1(attn_g, num_channels, name="conv2d_attn_g", use_sn=use_sn,
                         use_bias=False)

        return x + sigma * attn_g
        
def conditional_batch_norm(inputs, y, is_training, use_sn, center=True,
                           scale=True, name="batch_norm", use_bias=False):
  """Conditional batch normalization."""
  if y is None:
    raise ValueError("You must provide y for conditional batch normalization.")
  if y.shape.ndims != 2:
    raise ValueError("Conditioning must have rank 2.")
  with tf.variable_scope(name, values=[inputs]):
    outputs = tf.contrib.layers.batch_norm(inputs, center=False, scale=False, decay=0.9, epsilon=1e-5, updates_collections=None, is_training=is_training)
    num_channels = inputs.shape[-1].value
    with tf.variable_scope("condition", values=[inputs, y]):
      if scale:
        gamma = linear(y, num_channels, scope="gamma", use_sn=use_sn,
                       use_bias=use_bias)
        gamma = tf.reshape(gamma, [-1, 1, 1, num_channels])
        outputs *= gamma
      if center:
        beta = linear(y, num_channels, scope="beta", use_sn=use_sn,
                      use_bias=use_bias)
        beta = tf.reshape(beta, [-1, 1, 1, num_channels])
        outputs += beta
      return outputs