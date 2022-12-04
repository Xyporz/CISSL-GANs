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
      

class Classifier_proD(object):
  """ResNet-based discriminator supporting resolutions 32, 64, 128, 256, 512."""

  def __init__(self,
               name, num_class, use_sn,
               ch=128,
               blocks_with_attention="B1",
               project_y=True,
               **kwargs):
    """Constructor for BigGAN discriminator.
    Args:
      ch: Channel multiplier.
      blocks_with_attention: Comma-separated list of blocks that are followed by
        a non-local block.
      project_y: Add an embedding of y in the output layer.
      **kwargs: additional arguments past on to ResNetDiscriminator.
    """
    self.name = name
    self.num_class = num_class
    self._spectral_norm = True
    self._batch_norm_fn = None
    
    self._ch = ch
    self._blocks_with_attention = set(blocks_with_attention.split(","))
    self._project_y = project_y

  def _resnet_block(self, name, in_channels, out_channels, scale):
    """ResNet block for the generator."""
    if scale not in ["down", "none"]:
      raise ValueError(
          "Unknown discriminator ResNet block scaling: {}.".format(scale))
    return BigGanDeepResNetBlock(
        name=name,
        in_channels=in_channels,
        out_channels=out_channels,
        scale=scale,
        spectral_norm=self._spectral_norm,
        batch_norm=self.batch_norm)

  def _get_in_out_channels(self, colors, resolution):
    # See Table 7-9.
    if colors not in [1, 3]:
      raise ValueError("Unsupported color channels: {}".format(colors))
    if resolution == 512:
      channel_multipliers = [1, 1, 1, 2, 2, 4, 4] + 4 * [8] + 4 * [16]
    elif resolution == 256:
      channel_multipliers = [1, 2, 2, 4, 4] + 4 * [8] + 4 * [16]
    elif resolution == 128:
      channel_multipliers = [1, 2, 2, 4, 4] + 2 * [8] + 4 * [16]
    elif resolution == 64:
      channel_multipliers = [2, 4, 4] + 2 * [8] + 4 * [16]
    elif resolution == 32:
      channel_multipliers = 8 * [2]
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

  def __call__(self, x, y, _, __):
      with tf.variable_scope(self.name, values=[x, y], reuse=tf.AUTO_REUSE):
        """Apply the discriminator on a input.
        Args:
          x: `Tensor` of shape [batch_size, ?, ?, ?] with real or fake images.
          y: `Tensor` of shape [batch_size, num_classes] with one hot encoded
            labels.
          is_training: Boolean, whether the architecture should be constructed for
            training or inference.
        Returns:
          Tuple of 3 Tensors, the final prediction of the discriminator, the logits
          before the final output activation function and logits form the second
          last layer.
        """
        logging.info("[Discriminator] inputs are x=%s, y=%s", x.shape,
                     None if y is None else y.shape)
        resnet_ops.validate_image_inputs(x)

        in_channels, out_channels = self._get_in_out_channels(
            colors=x.shape[-1].value, resolution=x.shape[1].value)
        num_blocks = len(in_channels)

        net = ops.conv2d(x, output_dim=in_channels[0], k_h=3, k_w=3,
                         d_h=1, d_w=1, name="initial_conv",
                         use_sn=self._spectral_norm)

        for block_idx in range(num_blocks):
          scale = "down" if block_idx % 2 == 0 else "none"
          block = self._resnet_block(
              name="B{}".format(block_idx + 1),
              in_channels=in_channels[block_idx],
              out_channels=out_channels[block_idx],
              scale=scale)
          net = block(net, z=None, y=y, is_training=None)
          # At resolution 64x64 there is a self-attention block.
          if scale == "none" and net.shape[1].value == 64:
            logging.info("[Discriminator] Applying non-local block to %s",
                         net.shape)
            net = ops.non_local_block(net, "non_local_block",
                                      use_sn=self._spectral_norm)

        # Final part
        logging.info("[Discriminator] before final processing: %s", net.shape)
        net_conv = tf.nn.relu(net)
        h = tf.math.reduce_sum(net_conv, axis=[1, 2])
        out_logit_tf = ops.linear(h, 1, scope="final_fc", use_sn=self._spectral_norm)
        logging.info("[Discriminator] after final processing: %s", net.shape)
        if self._project_y:
          if y is None:
            raise ValueError("You must provide class information y to project.")
          with tf.variable_scope("embedding_fc"):
            y_embedding_dim = out_channels[-1]
            # We do not use ops.linear() below since it does not have an option to
            # override the initializer.
            kernel = tf.get_variable(
                "kernel", [y.shape[1], y_embedding_dim], tf.float32,
                initializer=tf.initializers.glorot_normal())
            if self._spectral_norm:
              kernel = ops.spectral_norm(kernel)
            embedded_y = tf.matmul(y, kernel)
            logging.info("[Discriminator] embedded_y for projection: %s",
                         embedded_y.shape)
            out_logit_tf += tf.reduce_sum(embedded_y * h, axis=1, keepdims=True)
        
        feature_matching = h
        t_SNE = h
        out_logit = ops.linear(h, self.num_class, scope="final_fc_cla", use_sn=self._spectral_norm)
        
        # grad cam
        cls = tf.argmax(y,axis=1)[0]
        out_logit_cam = tf.identity(out_logit)
        y_c = out_logit_cam[0, cls]
        grads = tf.gradients(y_c, net_conv)[0]
        output_conv = net_conv[0]
        grads_val = grads[0]
        
        # grad cam ++
        first = tf.exp(y_c)*grads
        second = tf.exp(y_c)*grads*grads
        third = tf.exp(y_c)*grads*grads*grads
        conv_first_grad, conv_second_grad, conv_third_grad = first[0], second[0], third[0]
        grad_val_plusplus = [conv_first_grad, conv_second_grad, conv_third_grad]
        
        # saliency
        signal = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out_logit[0, :], labels=y[0]))
        guide_grad = tf.gradients(signal, x)[0]
        
        return tf.nn.softmax(out_logit), out_logit, tf.nn.sigmoid(out_logit_tf), out_logit_tf, feature_matching, t_SNE, output_conv, grads_val, guide_grad, grad_val_plusplus, tf.argmax(y,axis=1)[0], tf.argmax(out_logit_cam,axis=1)[0]