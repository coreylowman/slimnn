use basenn::*;

use derives::*;
use dfdx::{
    prelude::{Device, Dim, Dtype, Tape, Tensor},
    tensor_ops::{BroadcastTo, TryAdd},
};

#[derive(Default, Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Bias1D<I: Dim>(pub I);

impl<I: Dim, E: Dtype, D: Device<E>> BuildOnDevice<E, D> for Bias1D<I> {
    type Built = DeviceBias1D<I, E, D>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, D::Err> {
        Ok(DeviceBias1D {
            bias: device.try_zeros_like(&(self.0,))?,
        })
    }
}

#[derive(Clone, Debug, UpdateParams, ZeroGrads, ToDtype, ToDevice)]
pub struct DeviceBias1D<I: Dim, Elem: Dtype, Dev: Device<Elem>> {
    #[param]
    pub bias: Tensor<(I,), Elem, Dev>,
}

impl<I: Dim, E: Dtype, D: Device<E>> ResetParams<E, D> for DeviceBias1D<I, E, D> {
    fn try_reset_params(&mut self) -> Result<(), D::Err> {
        self.bias.try_fill_with_zeros()
    }
}

impl<I: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<(I,), E, D, T>>
    for DeviceBias1D<I, E, D>
{
    type Output = Tensor<(I,), E, D, T>;
    type Error = D::Err;
    fn try_forward(&self, x: Tensor<(I,), E, D, T>) -> Result<Self::Output, Self::Error> {
        x.try_add(self.bias.clone())
    }
}

impl<Batch: Dim, I: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<(Batch, I), E, D, T>>
    for DeviceBias1D<I, E, D>
{
    type Output = Tensor<(Batch, I), E, D, T>;
    type Error = D::Err;
    fn try_forward(&self, x: Tensor<(Batch, I), E, D, T>) -> Result<Self::Output, Self::Error> {
        self.bias.retaped::<T>().broadcast_like(&x).try_add(x)
    }
}

impl<Batch: Dim, Seq: Dim, I: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    Module<Tensor<(Batch, Seq, I), E, D, T>> for DeviceBias1D<I, E, D>
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
