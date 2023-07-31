use derives::*;
use dfdx::prelude::*;

#[derive(Default, Clone, Copy, Debug)]
#[repr(transparent)]
pub struct LayerNorm1D<M: Dim>(pub M);

impl<M: Dim, E: Dtype, D: Device<E>> crate::BuildOnDevice<E, D> for LayerNorm1D<M> {
    type Built = DeviceLayerNorm1D<M, E, D>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, D::Err> {
        Ok(DeviceLayerNorm1D {
            gamma: device.try_ones_like(&(self.0,))?,
            beta: device.try_zeros_like(&(self.0,))?,
            epsilon: 1e-5,
        })
    }
}

#[derive(Clone, Debug, UpdateParams, ZeroGrads, ToDtype, ToDevice)]
pub struct DeviceLayerNorm1D<M: Dim, Elem: Dtype, Dev: Device<Elem>> {
    pub gamma: Tensor<(M,), Elem, Dev>,
    pub beta: Tensor<(M,), Elem, Dev>,
    pub epsilon: f64,
}

impl<M: Dim, E: Dtype, D: Device<E>> crate::ResetParams<E, D> for DeviceLayerNorm1D<M, E, D> {
    fn try_reset_params(&mut self) -> Result<(), D::Err> {
        self.gamma.try_fill_with_ones()?;
        self.beta.try_fill_with_zeros()
    }
}

impl<M: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>> crate::Module<Tensor<(M,), E, D, T>>
    for DeviceLayerNorm1D<M, E, D>
{
    type Output = Tensor<(M,), E, D, T>;
    type Error = D::Err;
    fn try_forward(&self, x: Tensor<(M,), E, D, T>) -> Result<Self::Output, Self::Error> {
        x.try_normalize(self.epsilon)?
            .try_mul(self.gamma.clone())?
            .try_add(self.beta.clone())
    }
}

impl<Batch: Dim, M: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    crate::Module<Tensor<(Batch, M), E, D, T>> for DeviceLayerNorm1D<M, E, D>
{
    type Output = Tensor<(Batch, M), E, D, T>;
    type Error = D::Err;
    fn try_forward(&self, x: Tensor<(Batch, M), E, D, T>) -> Result<Self::Output, Self::Error> {
        let x = x.try_normalize::<Axis<1>>(self.epsilon)?;
        let x = self.gamma.retaped::<T>().broadcast_like(&x).try_mul(x)?;
        self.beta.retaped::<T>().broadcast_like(&x).try_add(x)
    }
}

impl<Batch: Dim, Seq: Dim, M: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    crate::Module<Tensor<(Batch, Seq, M), E, D, T>> for DeviceLayerNorm1D<M, E, D>
{
    type Output = Tensor<(Batch, Seq, M), E, D, T>;
    type Error = D::Err;
    fn try_forward(
        &self,
        x: Tensor<(Batch, Seq, M), E, D, T>,
    ) -> Result<Self::Output, Self::Error> {
        let x = x.try_normalize::<Axis<2>>(self.epsilon)?;
        let x = self.gamma.retaped::<T>().broadcast_like(&x).try_mul(x)?;
        self.beta.retaped::<T>().broadcast_like(&x).try_add(x)
    }
}
