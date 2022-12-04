from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import arch_ops as ops
import resnet_ops

import gin
from six.moves import range
import tensorflow as tf


class BigGanResNetBlock(resnet_ops.ResNetBlock):
  """ResNet block with options for various normalizations.
  This block uses a 1x1 convolution for the (optional) shortcut connection.
  """

  def __init__(self,
               add_shortcut=True,
               **kwargs):
    """Constructs a new ResNet block for BigGAN.
    Args:
      add_shortcut: Whether to add a shortcut connection.
      **kwargs: Additional arguments for ResNetBlock.
    """
    super(BigGanResNetBlock, self).__init__(**kwargs)
    self._add_shortcut = add_shortcut

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

    with tf.variable_scope(self._name, values=[inputs]):
      outputs = inputs

      outputs = self.batch_norm(
          outputs, z=z, y=y, is_training=is_training, name="bn1")
      if self._layer_norm:
        outputs = ops.layer_norm(outputs, is_training=is_training, scope="ln1")

      outputs = tf.nn.relu(outputs)
      outputs = self._get_conv(
          outputs, self._in_channels, self._out_channels, self._scale1,
          suffix="conv1")

      outputs = self.batch_norm(
          outputs, z=z, y=y, is_training=is_training, name="bn2")
      if self._layer_norm:
        outputs = ops.layer_norm(outputs, is_training=is_training, scope="ln2")

      outputs = tf.nn.relu(outputs)
      outputs = self._get_conv(
          outputs, self._out_channels, self._out_channels, self._scale2,
          suffix="conv2")

      # Combine skip-connection with the convolved part.
      if self._add_shortcut:
        shortcut = self._get_conv(
            inputs, self._in_channels, self._out_channels, self._scale,
            kernel_size=(1, 1),
            suffix="conv_shortcut")
        outputs += shortcut
      logging.info("[Block] %s (z=%s, y=%s) -> %s", inputs.shape,
                   None if z is None else z.shape,
                   None if y is None else y.shape, outputs.shape)
      return outputs

class Classifier_proD(object):
  """ResNet-based discriminator supporting resolutions 32, 64, 128, 256, 512."""

  def __init__(self,
               name, num_class, use_sn,
               ch=96,
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
    self._layer_norm = False
    
    self._ch = ch
    self._blocks_with_attention = set(blocks_with_attention.split(","))
    self._project_y = project_y

  def _resnet_block(self, name, in_channels, out_channels, scale):
    """ResNet block for the generator."""
    if scale not in ["down", "none"]:
      raise ValueError(
          "Unknown discriminator ResNet block scaling: {}.".format(scale))
    return BigGanResNetBlock(
        name=name,
        in_channels=in_channels,
        out_channels=out_channels,
        scale=scale,
        is_gen_block=False,
        add_shortcut=in_channels != out_channels,
        layer_norm=self._layer_norm,
        spectral_norm=self._spectral_norm,
        batch_norm=self.batch_norm)
        
  def batch_norm(self, inputs, **kwargs):
    if self._batch_norm_fn is None:
      return inputs
    args = kwargs.copy()
    args["inputs"] = inputs
    if "use_sn" not in args:
      args["use_sn"] = self._spectral_norm
    return utils.call_with_accepted_args(self._batch_norm_fn, **args)

  def _get_in_out_channels(self, colors, resolution):
    if colors not in [1, 3]:
      raise ValueError("Unsupported color channels: {}".format(colors))
    if resolution == 512:
      channel_multipliers = [1, 1, 2, 4, 8, 8, 16, 16]
    elif resolution == 256:
      channel_multipliers = [1, 2, 4, 8, 8, 16, 16]
    elif resolution == 128:
      channel_multipliers = [1, 2, 4, 8, 16, 16]
    elif resolution == 64:
      channel_multipliers = [2, 4, 8, 16, 16]
    elif resolution == 32:
      channel_multipliers = [2, 2, 2, 2]
    else:
      raise ValueError("Unsupported resolution: {}".format(resolution))
    out_channels = [self._ch * c for c in channel_multipliers]
    in_channels = [colors] + out_channels[:-1]
    return in_channels, out_channels

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

        net = x
        for block_idx in range(num_blocks):
          name = "B{}".format(block_idx + 1)
          is_last_block = block_idx == num_blocks - 1
          block = self._resnet_block(
              name=name,
              in_channels=in_channels[block_idx],
              out_channels=out_channels[block_idx],
              scale="none" if is_last_block else "down")
          net = block(net, z=None, y=y, is_training=None)
          if name in self._blocks_with_attention:
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
        
        
        