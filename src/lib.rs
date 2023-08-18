#![feature(generic_const_exprs)]

pub mod avg_pool_global;
pub mod batch_norm2d;
pub mod bias1d;
pub mod bias2d;
pub mod conv2d;
pub mod flatten2d;
pub mod generalized_add;
pub mod layer_norm1d;
pub mod linear;
pub mod matmul;
pub mod max_pool_2d;
pub mod multi_head_attention;
pub mod relu;
pub mod reshape;
pub mod residual_add;
pub mod sgd;
pub mod transformer;

pub use nn_core::*;
pub use nn_derives::*;
