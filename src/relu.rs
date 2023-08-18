use dfdx::prelude::{Device, Dtype, Shape, Tape, Tensor};
use nn_core::Module;
use nn_derives::CustomModule;

#[derive(Default, Debug, Clone, Copy, CustomModule)]
pub struct ReLU;

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>> for ReLU {
    type Output = Tensor<S, E, D, T>;
    type Error = D::Err;
    fn try_forward(&self, x: Tensor<S, E, D, T>) -> Result<Self::Output, Self::Error> {
        x.try_relu()
    }
}
