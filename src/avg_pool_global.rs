use derives::*;
use dfdx::{
    prelude::{Device, Dim, Dtype, Tape, Tensor},
    tensor_ops::MeanTo,
};

#[derive(Default, Debug, Clone, Copy, ResetParams, ZeroGrads, UpdateParams, ToDevice, ToDtype)]
pub struct AvgPoolGlobal;

// TODO derive this
impl<E: Dtype, D: Device<E>> crate::BuildOnDevice<E, D> for AvgPoolGlobal {
    type Built = Self;
    fn try_build_on_device(&self, _: &D) -> Result<Self::Built, <D>::Err> {
        Ok(self.clone())
    }
}

impl<C: Dim, H: Dim, W: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    crate::Module<Tensor<(C, H, W), E, D, T>> for AvgPoolGlobal
{
    type Output = Tensor<(C,), E, D, T>;
    type Error = D::Err;

    fn try_forward(&self, input: Tensor<(C, H, W), E, D, T>) -> Result<Self::Output, D::Err> {
        input.try_mean()
    }
}

impl<B: Dim, C: Dim, H: Dim, W: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    crate::Module<Tensor<(B, C, H, W), E, D, T>> for AvgPoolGlobal
{
    type Output = Tensor<(B, C), E, D, T>;
    type Error = D::Err;

    fn try_forward(&self, input: Tensor<(B, C, H, W), E, D, T>) -> Result<Self::Output, D::Err> {
        input.try_mean()
    }
}
