from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl import logging

import arch_ops as ops
import resnet_ops

import utils

from six.moves import range
import tensorflow as tf
import time

class BigGanDeepResNetBlock(object):
  """ResNet block with bottleneck and identity preserving skip connections."""

  def __init__(self,
               name,
               in_channels,
               out_channels,
               scale,
               spectral_norm=False,
               batch_norm=None):
    """Constructs a new ResNet block with bottleneck.
    Args:
      name: Scope name for the resent block.
      in_channels: Integer, the input channel size.
      out_channels: Integer, the output channel size.
      scale: Whether or not to scale up or down, choose from "up", "down" or
        "none".
      spectral_norm: Use spectral normalization for all weights.
      batch_norm: Function for batch normalization.
    """
    assert scale in ["up", "down", "none"]
    self._name = name
    self._in_channels = in_channels
    self._out_channels = out_channels
    self._scale = scale
    self._spectral_norm = spectral_norm
    self.batch_norm = batch_norm

  def __call__(self, inputs, z, y, is_training):
    return self.apply(inputs=inputs, z=z, y=y, is_training=is_training)

  def _shortcut(self, inputs):
    """Constructs a skip connection from inputs."""
    with tf.variable_scope("shortcut", values=[inputs]):
      shortcut = inputs
      num_channels = inputs.shape[-1].value
      if num_channels > self._out_channels:
        assert self._scale == "up"
        # Drop redundant channels.
        logging.info("[Shortcut] Dropping %d channels in shortcut.",
                     num_channels - self._out_channels)
        shortcut = shortcut[:, :, :, :self._out_channels]
      if self._scale == "up":
        shortcut = resnet_ops.unpool(shortcut)
      if self._scale == "down":
        shortcut = tf.nn.pool(shortcut, [2, 2], "AVG", "SAME",
                              strides=[2, 2], name="pool")
      if num_channels < self._out_channels:
        assert self._scale == "down"
        # Increase number of channels if necessary.
        num_missing = self._out_channels - num_channels
        logging.info("[Shortcut] Adding %d channels in shortcut.", num_missing)
        added = ops.conv1x1(shortcut, num_missing, name="add_channels",
                            use_sn=self._spectral_norm)
        shortcut = tf.concat([shortcut, added], axis=-1)
      return shortcut

  def apply(self, inputs, z, y, is_training):
    """"ResNet block containing possible down/up sampling, shared for G / D.
    Args:
      inputs: a 3d input tensor of feature map.
      z: the latent vector for potential self-modulation. Can be None if use_sbn
        is set to False.
      y: `Tensor` of shape [batch_size, num_classes] with one hot encoded
        labels.
      is_training: boolean, whether or notthis is called during the training.
    Returns:
      output: a 3d output tensor of feature map.
    """
    if inputs.shape[-1].value != self._in_channels:
      raise ValueError(
          "Unexpected number of input channels (expected {}, got {}).".format(
              self._in_channels, inputs.shape[-1].value))

    bottleneck_channels = max(self._in_channels, self._out_channels) // 4
    bn = functools.partial(self.batch_norm, z=z, y=y, is_training=is_training)
    conv1x1 = functools.partial(ops.conv1x1, use_sn=self._spectral_norm)
    conv3x3 = functools.partial(ops.conv2d, k_h=3, k_w=3, d_h=1, d_w=1,
                                use_sn=self._spectral_norm)

    with tf.variable_scope(self._name, values=[inputs]):
      outputs = inputs

      with tf.variable_scope("conv1", values=[outputs]):
        outputs = bn(outputs, name="bn")
        outputs = tf.nn.relu(outputs)
        outputs = conv1x1(outputs, bottleneck_channels, name="1x1_conv")

      with tf.variable_scope("conv2", values=[outputs]):
        outputs = bn(outputs, name="bn")
        outputs = tf.nn.relu(outputs)
        if self._scale == "up":
          outputs = resnet_ops.unpool(outputs)
        outputs = conv3x3(outputs, bottleneck_channels, name="3x3_conv")

      with tf.variable_scope("conv3", values=[outputs]):
        outputs = bn(outputs, name="bn")
        outputs = tf.nn.relu(outputs)
        outputs = conv3x3(outputs, bottleneck_channels, name="3x3_conv")

      with tf.variable_scope("conv4", values=[outputs]):
        outputs = bn(outputs, name="bn")
        outputs = tf.nn.relu(outputs)
        if self._scale == "down":
          outputs = tf.nn.pool(outputs, [2, 2], "AVG", "SAME", strides=[2, 2],
                               name="avg_pool")
        outputs = conv1x1(outputs, self._out_channels, name="1x1_conv")

      # Add skip-connection.
      outputs += self._shortcut(inputs)

      logging.info("[Block] %s (z=%s, y=%s) -> %s", inputs.shape,
                   None if z is None else z.shape,
                   None if y is None else y.shape, outputs.shape)
      return outputs


class Generator(object):
  """ResNet-based generator supporting resolutions 32, 64, 128, 256, 512."""

  def __init__(self,
               name, h, w, c, is_train, use_sn,
               ch=128,
               embed_y=True,
               embed_y_dim=128,
               experimental_fast_conv_to_rgb=False,
               **kwargs):
    """Constructor for BigGAN generator.
    Args:
      ch: Channel multiplier.
      embed_y: If True use a learnable embedding of y that is used instead.
      embed_y_dim: Size of the embedding of y.
      experimental_fast_conv_to_rgb: If True optimize the last convolution to
        sacrifize memory for better speed.
      **kwargs: additional arguments past on to ResNetGenerator.
    """
    self.name = name
    self.s_h, self.s_w, self.colors = [h,w,c]
    self._image_shape = [h,w,c]
    self._is_train = is_train
    self._batch_norm_fn = ops.conditional_batch_norm
    self._spectral_norm = False
    
    self._ch = ch
    self._embed_y = embed_y
    self._embed_y_dim = embed_y_dim
    self._experimental_fast_conv_to_rgb = experimental_fast_conv_to_rgb

  def _resnet_block(self, name, in_channels, out_channels, scale):
    """ResNet block for the generator."""
    if scale not in ["up", "none"]:
      raise ValueError(
          "Unknown generator ResNet block scaling: {}.".format(scale))
    return BigGanDeepResNetBlock(
        name=name,
        in_channels=in_channels,
        out_channels=out_channels,
        scale=scale,
        spectral_norm=self._spectral_norm,
        batch_norm=self.batch_norm)

  def _get_in_out_channels(self):
    # See Table 7-9.
    resolution = self._image_shape[0]
    if resolution == 512:
      channel_multipliers = 4 * [16] + 4 * [8] + [4, 4, 2, 2, 1, 1, 1]
    elif resolution == 256:
      channel_multipliers = 4 * [16] + 4 * [8] + [4, 4, 2, 2, 1]
    elif resolution == 128:
      channel_multipliers = 4 * [16] + 2 * [8] + [4, 4, 2, 2, 1]
    elif resolution == 64:
      channel_multipliers = 4 * [16] + 2 * [8] + [4, 4, 2]
    elif resolution == 32:
      channel_multipliers = 8 * [4]
    else:
      raise ValueError("Unsupported resolution: {}".format(resolution))
    in_channels = [self._ch * c for c in channel_multipliers[:-1]]
    out_channels = [self._ch * c for c in channel_multipliers[1:]]
    return in_channels, out_channels
    
  def batch_norm(self, inputs, **kwargs):
    if self._batch_norm_fn is None:
      return inputs
    args = kwargs.copy()
    args["inputs"] = inputs
    if "use_sn" not in args:
      args["use_sn"] = self._spectral_norm
    return utils.call_with_accepted_args(self._batch_norm_fn, **args)

  def __call__(self, z, y):
      with tf.variable_scope(self.name, values=[z, y], reuse=tf.AUTO_REUSE):
        """Build the generator network for the given inputs.
        Args:
          z: `Tensor` of shape [batch_size, z_dim] with latent code.
          y: `Tensor` of shape [batch_size, num_classes] with one hot encoded
            labels.
          is_training: boolean, are we in train or eval model.
        Returns:
          A tensor of size [batch_size] + self._image_shape with values in [0, 1].
        """
        shape_or_none = lambda t: None if t is None else t.shape
        logging.info("[Generator] inputs are z=%s, y=%s", z.shape, shape_or_none(y))
        seed_size = 4

        if self._embed_y:
          y = ops.linear(y, self._embed_y_dim, scope="embed_y", use_sn=False,
                         use_bias=False)
        if y is not None:
          y = tf.concat([z, y], axis=1)
          z = y

        in_channels, out_channels = self._get_in_out_channels()
        num_blocks = len(in_channels)

        # Map noise to the actual seed.
        net = ops.linear(
            z,
            in_channels[0] * seed_size * seed_size,
            scope="fc_noise",
            use_sn=self._spectral_norm)
        # Reshape the seed to be a rank-4 Tensor.
        net = tf.reshape(
            net,
            [-1, seed_size, seed_size, in_channels[0]],
            name="fc_reshaped")

        for block_idx in range(num_blocks):
          scale = "none" if block_idx % 2 == 0 else "up"
          block = self._resnet_block(
              name="B{}".format(block_idx + 1),
              in_channels=in_channels[block_idx],
              out_channels=out_channels[block_idx],
              scale=scale)
          net = block(net, z=z, y=y, is_training=(self._is_train))
          # At resolution 64x64 there is a self-attention block.
          if scale == "up" and net.shape[1].value == 64:
            logging.info("[Generator] Applying non-local block to %s", net.shape)
            net = ops.non_local_block(net, "non_local_block",
                                      use_sn=self._spectral_norm)
        # Final processing of the net.
        # Use unconditional batch norm.
        logging.info("[Generator] before final processing: %s", net.shape)
        net = ops.batch_norm(net, is_training=(self._is_train), name="final_norm")
        net = tf.nn.relu(net)
        colors = self._image_shape[2]
        if self._experimental_fast_conv_to_rgb:

          net = ops.conv2d(net, output_dim=128, k_h=3, k_w=3,
                           d_h=1, d_w=1, name="final_conv",
                           use_sn=self._spectral_norm)
          net = net[:, :, :, :colors]
        else:
          net = ops.conv2d(net, output_dim=colors, k_h=3, k_w=3,
                           d_h=1, d_w=1, name="final_conv",
                           use_sn=self._spectral_norm)
        logging.info("[Generator] after final processing: %s", net.shape)
        net = (tf.nn.tanh(net) + 1.0) / 2.0
        return net