"""
This contains the custom layers implemented using tfga infrastructure.
"""
from typing import List, Union

import tensorflow as tf
from tensorflow.keras import (activations, constraints, initializers, layers,
                              regularizers)
#from keras.utils import register_keras_serializable

from tfga.blades import BladeKind
from tfga.layers import GeometricAlgebraLayer
from tfga.tfga import GeometricAlgebra
from tensorflow.keras.layers import Conv3D



class RotorConv2D(GeometricAlgebraLayer):
    """
    This is a  2D convolution layer as described in "Geometric Clifford Algebra
    Networks" (Ruhe et al.). It uses a weighted sandwich product with rotors in the kernel.


     Args:
        algebra: GeometricAlgebra instance to use for the parameters
        filters: How many channels the output will have
        kernel_size: Size for the convolution kernel
        stride: Stride to use for the convolution
        padding: "SAME" (zero-pad input length so output
            length == input length / stride) or "VALID" (no padding)
        blade_indices_kernel: Blade indices to use for the kernel parameter
        blade_indices_bias: Blade indices to use for the bias parameter (if used)
    """

    def __init__(
            self,
            algebra: GeometricAlgebra,
            filters: int,
            kernel_size: int,
            stride_horizontal: int,
            stride_vertical: int,
            padding: str,
            blade_indices_kernel: List[int] = None,
            blade_indices_bias: Union[None, List[int]] = None,
            dilations: Union[None, int] = None,
            activation=None,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=tf.keras.constraints.UnitNorm(axis=-1),
            bias_constraint=None,
            **kwargs
    ):
        super().__init__(
            algebra=algebra, activity_regularizer=activity_regularizer, **kwargs
        )

        self.filters = filters
        self.kernel_size = kernel_size
        self.stride_horizontal = stride_horizontal
        self.stride_vertical = stride_vertical
        self.padding = padding
        self.dilations = dilations

        # if no blade index specified, default to rotors (only even indices)
        if blade_indices_kernel is None:
            blade_indices_kernel = self.algebra.get_kind_blade_indices('even')

        self.blade_indices_kernel = tf.convert_to_tensor(
            blade_indices_kernel, dtype_hint=tf.int64
        )
        if use_bias:
            self.blade_indices_bias = tf.convert_to_tensor(
                blade_indices_bias, dtype_hint=tf.int64
            )

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def get_config(self):
        config = super().get_config().copy()
        
        return config
    
    def get_config(self):
        config = {'filters': self.filters,
                'kernel_size': self.kernel_size,
                'stride_horizontal': self.stride_horizontal,
                'stride_vertical': self.stride_vertical,
                'padding': self.padding}
        base_config = super(RotorConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    

    def build(self, input_shape: tf.TensorShape):
        # I: [..., S, S, C, B]
        self.num_input_filters = input_shape[-2]

        # K: [K, K, IC, OC, B]
        shape_kernel = [
            self.kernel_size,
            self.kernel_size,
            self.num_input_filters,
            self.filters,
            self.blade_indices_kernel.shape[0],
        ]
        self.kernel = self.add_weight(
            "kernel",
            shape=shape_kernel,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )
        self.kernel_weights = self.add_weight(
            "kernel_weights",
            shape=shape_kernel[:-1],
            regularizer=self.kernel_regularizer,
            dtype=self.dtype,
            trainable=True
        )
        if self.use_bias:
            shape_bias = [self.filters, self.blade_indices_bias.shape[0]]
            self.bias = self.add_weight(
                "bias",
                shape=shape_bias,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.bias = None
        self.built = True

    def rotor_conv2d(
            self,
            a_blade_values: tf.Tensor,
            k_blade_values: tf.Tensor,
            weights,
            stride_horizontal: int,
            stride_vertical: int,
            padding: str,
            dilations: Union[int, None] = None,
    ) -> tf.Tensor:
        # A: [..., S, S, CI, BI]
        # K: [K, K, CI, CO, BK]
        # C: [BI, BK, BO]

        kernel_size = k_blade_values.shape[0]

        a_batch_shape = tf.shape(a_blade_values)[:-4]


        # Reshape a_blade_values to a 2d image (since that's what the tf op expects)
        # [*, S, S, CI*BI]
        a_image_shape = tf.concat(
            [
                a_batch_shape,
                tf.shape(a_blade_values)[-4:-3],
                [tf.shape(a_blade_values)[-3], tf.reduce_prod(tf.shape(a_blade_values)[-2:])],
            ],
            axis=0,
        )
        a_image = tf.reshape(a_blade_values, a_image_shape)

        #print("a_image_shape", a_image_shape)

        sizes = [1, kernel_size, kernel_size, 1]
        strides = [1, stride_vertical, stride_horizontal, 1]

        # [*, P1, P2, K*K*CI*BI] where eg. number of patches P = S * K for
        # stride=1 and "SAME", (S-K+1) * K for "VALID", ...
        
        a_slices = tf.image.extract_patches(
            a_image, sizes=sizes, strides=strides, rates=[1, 1, 1, 1], padding=padding
        )

        #print("a_slices_shape", a_slices.shape)

        # [..., P1, P2, K, K, CI, BI]
        out_shape = tf.concat(
            [
                a_batch_shape,
                tf.shape(a_slices)[-3:-1],
                tf.shape(k_blade_values)[:2],
                tf.shape(a_blade_values)[-2:],
            ],
            axis=0,
        )

        a_slices = tf.reshape(a_slices, out_shape)

        #print("out_shape", out_shape)

        # no-sandwich product convolution:
        # a_...p,p,k,k,ci,bi; k,k,ci,co,bk; c_bi,bk,bo -> y_...p,p,co,bo
        #   ...a n b m c  d , b m c  f  g ,   d  g  h  ->   ...a n f  h

        # sandwich product adds additional cayley matrix, otherwise dimensions correspond; thus just need to add extra
        # dimension from 1d case to all kernel elements to maintain correspondence
        x = tf.einsum("...bmcf,...bmcfi,hij,...anbmcd,bmcfg,gdh->...anfj", weights,
                      self.algebra.reversion(k_blade_values), self.algebra._cayley, a_slices, k_blade_values,
                      self.algebra._cayley)

        return x

    def call(self, inputs):
        k_geom = self.algebra.from_tensor(self.kernel, self.blade_indices_kernel)

        inputs = tf.convert_to_tensor(inputs, dtype_hint=tf.float32)
        k_geom = tf.convert_to_tensor(k_geom, dtype_hint=tf.float32)
        weights = tf.convert_to_tensor(self.kernel_weights, dtype_hint=tf.float32)

        result = self.rotor_conv2d(
            inputs,
            k_geom,
            weights,
            stride_horizontal=self.stride_horizontal,
            stride_vertical=self.stride_vertical,
            padding=self.padding,
            dilations=self.dilations,
        )

        if self.bias is not None:
            b_geom = self.algebra.from_tensor(self.bias, self.blade_indices_bias)
            result += b_geom

        return self.activation(result)


class RotorConv3D(GeometricAlgebraLayer):
    """
    This is a  2D convolution layer as described in "Geometric Clifford Algebra
    Networks" (Ruhe et al.). It uses a weighted sandwich product with rotors in the kernel.


     Args:
        algebra: GeometricAlgebra instance to use for the parameters
        filters: How many channels the output will have
        kernel_size: Size for the convolution kernel
        stride: Stride to use for the convolution
        padding: "SAME" (zero-pad input length so output
            length == input length / stride) or "VALID" (no padding)
        blade_indices_kernel: Blade indices to use for the kernel parameter
        blade_indices_bias: Blade indices to use for the bias parameter (if used)
    """

    def __init__(
            self,
            algebra: GeometricAlgebra,
            filters: int,
            kernel_size: int,
            stride_horizontal: int,
            stride_vertical: int,
            stride_depth: int,
            padding: str,
            blade_indices_kernel: List[int] = None,
            blade_indices_bias: Union[None, List[int]] = None,
            dilations: Union[None, int] = None,
            activation=None,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=tf.keras.constraints.UnitNorm(axis=-1),
            bias_constraint=None,
            **kwargs
    ):
        super().__init__(
            algebra=algebra, activity_regularizer=activity_regularizer, **kwargs
        )

        self.filters = filters
        self.kernel_size = kernel_size
        self.stride_horizontal = stride_horizontal
        self.stride_vertical = stride_vertical
        self.stride_depth = stride_depth
        self.padding = padding
        self.dilations = dilations

        # if no blade index specified, default to rotors (only even indices)
        if blade_indices_kernel is None:
            blade_indices_kernel = self.algebra.get_kind_blade_indices('even')

        self.blade_indices_kernel = tf.convert_to_tensor(
            blade_indices_kernel, dtype_hint=tf.int64
        )
        if use_bias:
            self.blade_indices_bias = tf.convert_to_tensor(
                blade_indices_bias, dtype_hint=tf.int64
            )
        
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def get_config(self):
        config = super().get_config().copy()
        
        return config
    
    def get_config(self):
        config = {'filters': self.filters,
                'kernel_size': self.kernel_size,
                'stride_horizontal': self.stride_horizontal,
                'stride_vertical': self.stride_vertical,
                'stride_depth': self.stride_depth,
                'padding': self.padding}
        base_config = super(RotorConv3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    

    def build(self, input_shape: tf.TensorShape):
        # I: [..., S, S, S, C, B]
        self.num_input_filters = input_shape[-2]

        # K: [K, K, K, IC, OC, B]
        shape_kernel = [
            self.kernel_size,
            self.kernel_size,
            self.kernel_size,
            self.num_input_filters,
            self.filters,
            self.blade_indices_kernel.shape[0],
        ]
        self.kernel = self.add_weight(
            "kernel",
            shape=shape_kernel,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )
        self.kernel_weights = self.add_weight(
            "kernel_weights",
            shape=shape_kernel[:-1],
            regularizer=self.kernel_regularizer,
            dtype=self.dtype,
            trainable=True
        )
        if self.use_bias:
            shape_bias = [self.filters, self.blade_indices_bias.shape[0]]
            self.bias = self.add_weight(
                "bias",
                shape=shape_bias,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.bias = None
        self.built = True

    def rotor_conv3d(
            self,
            a_blade_values: tf.Tensor,
            k_blade_values: tf.Tensor,
            weights,
            stride_horizontal: int,
            stride_vertical: int,
            stride_depth: int,
            padding: str,
            dilations: Union[int, None] = None,
    ) -> tf.Tensor:
        # A: [..., S, S, S, CI, BI]
        # K: [K, K, K, CI, CO, BK]
        # C: [BI, BK, BO]

        kernel_size = k_blade_values.shape[0]

        #print("a_blade_values:", a_blade_values.shape)
        a_batch_shape = tf.shape(a_blade_values)[:-5]

        #print(tf.shape(a_blade_values))

        # Reshape a_blade_values to a 3d image (since that's what the tf op expects)
        # [*, S, S, S, CI*BI]

        #print("a_batch_shape:", a_batch_shape.shape)
        a_image_shape = tf.concat(
            [
                a_batch_shape,
                tf.shape(a_blade_values)[-5:-2],
                [tf.math.reduce_prod(tf.shape(a_blade_values)[-2:])],
            ],
            axis=0,
        )

   
        #print(tf.math.reduce_prod(tf.shape(a_blade_values)[-2:]))
        #print("a_image_shape:", a_image_shape)

        a_image = tf.reshape(a_blade_values, a_image_shape)
        #print("a_image:", a_image.shape)


        sizes = [1, kernel_size, kernel_size, kernel_size, 1]
        strides = [1, stride_vertical, stride_horizontal, stride_depth, 1]

        # [*, P1, P2, P3, K*K*K*CI*BI] where eg. number of patches P = S * K for
        # stride=1 and "SAME", (S-K+1) * K for "VALID", ...

    
        a_slices = tf.extract_volume_patches(
            a_image, ksizes=sizes, strides=strides, padding=padding
        )        

        # [..., P1, P2, P3, K, K, K, CI, BI]
        out_shape = tf.concat(
            [
                a_batch_shape,
                tf.shape(a_slices)[-3:-2],
                tf.shape(a_slices)[-3:-2],
                tf.shape(a_slices)[-3:-2],
                tf.shape(k_blade_values)[:3],
                tf.shape(a_blade_values)[-2:],
            ],
            axis=0,
        )

        a_slices = tf.reshape(a_slices, out_shape)

        #print(a_slices.shape)

        #print("output_shape", a_slices.shape)

        # no-sandwich product convolution:
        # a_...p,p,k,k,ci,bi; k,k,ci,co,bk; c_bi,bk,bo -> y_...p,p,co,bo
        #   ...a n b m c  d , b m c  f  g ,   d  g  h  ->   ...a n f  h

        # a_...p,p,p,k,k,k,ci,bi; k k,k,ci,co,bk; c_bi,bk,bo -> y_...p,p,p,co,bo
        #   ...a n l b m q c  d , q b m c  f  g ,   d  g  h  ->   ...a n l f  h

        # sandwich product adds additional cayley matrix, otherwise dimensions correspond; thus just need to add extra
        # dimension from 1d case to all kernel elements to maintain correspondence
        
        x = tf.einsum("...bmqcf,...bmqcfi,hij,...anlbmqcd,qbmcfg,gdh->...anlfj", weights,
                      self.algebra.reversion(k_blade_values), self.algebra._cayley, a_slices, k_blade_values,
                      self.algebra._cayley)
        '''
        term1 = tf.matmul(weights[..., tf.newaxis], self.algebra.reversion(k_blade_values)[..., tf.newaxis], transpose_b=True)
        term2 = tf.matmul(self.algebra._cayley[..., tf.newaxis], a_slices[..., tf.newaxis])
        term3 = tf.matmul(k_blade_values[..., tf.newaxis], self.algebra._cayley[..., tf.newaxis])

        # Perform element-wise multiplication
        elementwise_mult = term1 * term2 * term3

        # Sum along appropriate axes
        x = tf.reduce_sum(elementwise_mult, axis=[2, 3, 4, 5])
        '''
        return x

    def call(self, inputs):
        k_geom = self.algebra.from_tensor(self.kernel, self.blade_indices_kernel)

        inputs = tf.convert_to_tensor(inputs, dtype_hint=tf.float32)
        k_geom = tf.convert_to_tensor(k_geom, dtype_hint=tf.float32)
        weights = tf.convert_to_tensor(self.kernel_weights, dtype_hint=tf.float32)

        result = self.rotor_conv3d(
            inputs,
            k_geom,
            weights,
            stride_horizontal=self.stride_horizontal,
            stride_vertical=self.stride_vertical,
            stride_depth = self.stride_depth,
            padding=self.padding,
            dilations=self.dilations,
        )

        if self.bias is not None:
            b_geom = self.algebra.from_tensor(self.bias, self.blade_indices_bias)
            result += b_geom

        return self.activation(result)
