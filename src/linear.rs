use crate::{bias1d::Bias1D, matmul::MatMul};

use derives::Sequential;
use dfdx::shapes::{Const, Dim};

#[derive(Default, Debug, Clone, Copy, Sequential)]
pub struct LinearUnbiased<I: Dim, O: Dim> {
    pub matmul: MatMul<I, O>,
}

impl<I: Dim, O: Dim> LinearUnbiased<I, O> {
    pub fn new(inp: I, out: O) -> Self {
        Self {
            matmul: MatMul { inp, out },
        }
    }
}

#[derive(Default, Debug, Clone, Copy, Sequential)]
pub struct Linear<I: Dim, O: Dim> {
    pub matmul: MatMul<I, O>,
    pub bias: Bias1D<O>,
}

pub type ConstLinear<const I: usize, const O: usize> = Linear<Const<I>, Const<O>>;

impl<I: Dim, O: Dim> Linear<I, O> {
    pub fn new(inp: I, out: O) -> Self {
        Self {
            matmul: MatMul { inp, out },
            bias: Bias1D(out),
        }
    }
}
