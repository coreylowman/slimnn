use basenn::*;

use derives::*;
use dfdx::{
    prelude::{Device, Dim, Dtype, HasErr, HasShape, Shape, Tape, Tensor},
    tensor_ops::TryMatMul,
};
use rand_distr::Uniform;

#[derive(Clone, Copy, Debug, Default)]
pub struct MatMul<I: Dim, O: Dim> {
    pub inp: I,
    pub out: O,
}

impl<I: Dim, O: Dim, E: Dtype, D: Device<E>> BuildOnDevice<E, D> for MatMul<I, O> {
    type Built = DeviceMatMul<I, O, E, D>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, D::Err> {
        Ok(DeviceMatMul {
            weight: device.try_zeros_like(&(self.inp, self.out))?,
        })
    }
}

#[derive(Clone, Debug, UpdateParams, ZeroGrads, ToDtype, ToDevice)]
pub struct DeviceMatMul<I: Dim, O: Dim, Elem: Dtype, Dev: Device<Elem>> {
    pub weight: Tensor<(I, O), Elem, Dev>,
}

// NOTE: others can simply #[derive(ResetParams)]
impl<I: Dim, O: Dim, E, D: Device<E>> ResetParams for DeviceMatMul<I, O, E, D>
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
    for DeviceMatMul<I, O, E, D>
where
    Tensor<S, E, D, T>: TryMatMul<Tensor<(I, O), E, D>>,
{
    type Output = <Tensor<S, E, D, T> as TryMatMul<Tensor<(I, O), E, D>>>::Output;
    type Error = <Tensor<S, E, D, T> as HasErr>::Err;
    fn try_forward(&self, x: Tensor<S, E, D, T>) -> Result<Self::Output, Self::Error> {
        x.try_matmul(self.weight.clone())
    }
}
