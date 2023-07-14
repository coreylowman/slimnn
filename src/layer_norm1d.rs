use derives::*;
use dfdx::prelude::*;

#[derive(Default, Clone, Copy, Debug)]
pub struct ConstLayerNorm1D<const M: usize>;

impl<const M: usize, E: Dtype, D: Device<E>> crate::BuildOnDevice<E, D> for ConstLayerNorm1D<M> {
    type Built = LayerNorm1D<Const<M>, E, D>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, D::Err> {
        Ok(LayerNorm1D {
            gamma: device.try_ones()?,
            beta: device.try_zeros()?,
            epsilon: 1e-5,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DynLayerNorm1D(pub usize);

impl<E: Dtype, D: Device<E>> crate::BuildOnDevice<E, D> for DynLayerNorm1D {
    type Built = LayerNorm1D<usize, E, D>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, D::Err> {
        Ok(LayerNorm1D {
            gamma: device.try_ones_like(&(self.0,))?,
            beta: device.try_zeros_like(&(self.0,))?,
            epsilon: 1e-5,
        })
    }
}

#[derive(Clone, Debug, UpdateParams, ZeroGrads, ToDtype, ToDevice)]
pub struct LayerNorm1D<M: Dim, E: Dtype, D: Device<E>> {
    pub gamma: Tensor<(M,), E, D>,
    pub beta: Tensor<(M,), E, D>,
    pub epsilon: f64,
}

impl<M: Dim, E: Dtype, D: Device<E>> crate::ResetParams for LayerNorm1D<M, E, D> {
    type Error = D::Err;
    fn try_reset_params(&mut self) -> Result<(), Self::Error> {
        self.gamma.try_fill_with_ones()?;
        self.beta.try_fill_with_zeros()
    }
}

impl<M: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>> crate::Module<Tensor<(M,), E, D, T>>
    for LayerNorm1D<M, E, D>
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
    crate::Module<Tensor<(Batch, M), E, D, T>> for LayerNorm1D<M, E, D>
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
    crate::Module<Tensor<(Batch, Seq, M), E, D, T>> for LayerNorm1D<M, E, D>
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
