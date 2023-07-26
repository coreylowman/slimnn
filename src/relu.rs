use derives::Functional;
use dfdx::prelude::{Device, Dtype, Shape, Tape, Tensor};

fn try_relu<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>>(
    x: Tensor<S, E, D, T>,
) -> Result<Tensor<S, E, D, T>, D::Err> {
    x.try_relu()
}

#[derive(Default, Debug, Clone, Copy, Functional)]
#[calls_fn(try_relu)]
pub struct ReLU;
