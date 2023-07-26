use basenn::*;

use derives::*;
use dfdx::{
    prelude::{Const, Device, Dim, Dtype, Tape, Tensor},
    tensor_ops::{BroadcastTo, TryAdd},
};

#[derive(Default, Clone, Copy, Debug)]
pub struct ConstBias1D<const I: usize>;

impl<const I: usize, E: Dtype, D: Device<E>> BuildOnDevice<E, D> for ConstBias1D<I> {
    type Built = Bias1D<Const<I>, E, D>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, D::Err> {
        Ok(Bias1D {
            bias: device.try_zeros()?,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DynBias1D(pub usize);

impl<E: Dtype, D: Device<E>> BuildOnDevice<E, D> for DynBias1D {
    type Built = Bias1D<usize, E, D>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, D::Err> {
        Ok(Bias1D {
            bias: device.try_zeros_like(&(self.0,))?,
        })
    }
}

#[derive(Clone, Debug, UpdateParams, ZeroGrads, ToDtype, ToDevice)]
pub struct Bias1D<I: Dim, E: Dtype, D: Device<E>> {
    pub bias: Tensor<(I,), E, D>,
}

impl<I: Dim, E: Dtype, D: Device<E>> ResetParams for Bias1D<I, E, D> {
    type Error = D::Err;
    fn try_reset_params(&mut self) -> Result<(), Self::Error> {
        self.bias.try_fill_with_zeros()
    }
}

impl<I: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<(I,), E, D, T>>
    for Bias1D<I, E, D>
{
    type Output = Tensor<(I,), E, D, T>;
    type Error = D::Err;
    fn try_forward(&self, x: Tensor<(I,), E, D, T>) -> Result<Self::Output, Self::Error> {
        x.try_add(self.bias.clone())
    }
}

impl<Batch: Dim, I: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<(Batch, I), E, D, T>>
    for Bias1D<I, E, D>
{
    type Output = Tensor<(Batch, I), E, D, T>;
    type Error = D::Err;
    fn try_forward(&self, x: Tensor<(Batch, I), E, D, T>) -> Result<Self::Output, Self::Error> {
        self.bias.retaped::<T>().broadcast_like(&x).try_add(x)
    }
}

impl<Batch: Dim, Seq: Dim, I: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    Module<Tensor<(Batch, Seq, I), E, D, T>> for Bias1D<I, E, D>
{
    type Output = Tensor<(Batch, Seq, I), E, D, T>;
    type Error = D::Err;
    fn try_forward(
        &self,
        x: Tensor<(Batch, Seq, I), E, D, T>,
    ) -> Result<Self::Output, Self::Error> {
        self.bias.retaped::<T>().broadcast_like(&x).try_add(x)
    }
}
