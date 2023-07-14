use crate::{
    bias1d::{ConstBias1D, DynBias1D},
    matmul::{ConstMatMul, DynMatMul},
};

use derives::Sequential;

#[derive(Default, Debug, Clone, Copy, Sequential)]
pub struct ConstLinear<const I: usize, const O: usize> {
    pub matmul: ConstMatMul<I, O>,
    pub bias: ConstBias1D<O>,
}

#[derive(Debug, Clone, Copy, Sequential)]
pub struct DynLinear {
    pub matmul: DynMatMul,
    pub bias: DynBias1D,
}
