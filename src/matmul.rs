use basenn::*;

use derives::*;
use dfdx::{
    prelude::{Const, Device, Dim, Dtype, Shape, Tape, Tensor},
    shapes::HasShape,
    tensor::HasErr,
    tensor_ops::TryMatMul,
};
use rand_distr::Uniform;

#[derive(Default, Clone, Copy, Debug)]
pub struct ConstMatMul<const I: usize, const O: usize>;

impl<const I: usize, const O: usize, E: Dtype, D: Device<E>> BuildOnDevice<E, D>
    for ConstMatMul<I, O>
{
    type Built = MatMul<Const<I>, Const<O>, E, D>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, D::Err> {
        Ok(MatMul {
            weight: device.try_zeros()?,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DynMatMul {
    pub inp: usize,
    pub out: usize,
}

impl<E: Dtype, D: Device<E>> BuildOnDevice<E, D> for DynMatMul {
    type Built = MatMul<usize, usize, E, D>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, D::Err> {
        Ok(MatMul {
            weight: device.try_zeros_like(&(self.inp, self.out))?,
        })
    }
}

#[derive(Clone, Debug, UpdateParams, ZeroGrads, ToDtype, ToDevice)]
pub struct MatMul<I: Dim, O: Dim, E: Dtype, D: Device<E>> {
    pub weight: Tensor<(I, O), E, D>,
}

// NOTE: others can simply #[derive(ResetParams)]
impl<I: Dim, O: Dim, E, D: Device<E>> ResetParams for MatMul<I, O, E, D>
where
    E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
{
    type Error = D::Err;
    fn try_reset_params(&mut self) -> Result<(), Self::Error> {
        let (i, _) = self.weight.shape();
        let scale = E::from_f64(1.0 / (i.size() as f64).sqrt()).unwrap();
        self.weight.try_fill_with_distr(Uniform::new(-scale, scale))
    }
}

impl<S: Shape, I: Dim, O: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>>
    for MatMul<I, O, E, D>
where
    Tensor<S, E, D, T>: TryMatMul<Tensor<(I, O), E, D>>,
{
    type Output = <Tensor<S, E, D, T> as TryMatMul<Tensor<(I, O), E, D>>>::Output;
    type Error = <Tensor<S, E, D, T> as HasErr>::Err;
    fn try_forward(&self, x: Tensor<S, E, D, T>) -> Result<Self::Output, Self::Error> {
        x.try_matmul(self.weight.clone())
    }
}
