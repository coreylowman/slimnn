use std::marker::PhantomData;

use dfdx::{
    shapes::{Dtype, Shape},
    tensor::{Gradients, Storage, Tensor, Tensorlike, UniqueId},
    tensor_ops::{Device, SgdConfig},
};

#[derive(Debug, Clone)]
pub struct Sgd<M, E: Dtype, D: Storage<E>> {
    /// Hyperparameter configuration
    pub cfg: SgdConfig,

    velocity: Gradients<E, D>,

    marker: PhantomData<*const M>,
}

impl<M, E: Dtype, D: Storage<E>> Sgd<M, E, D> {
    /// Constructs using hyperparameters from `cfg`
    pub fn new(_model: &M, cfg: SgdConfig) -> Self {
        Self {
            cfg,
            velocity: Gradients::leaky(),
            marker: PhantomData,
        }
    }
}

impl<M, E: Dtype, D: Device<E>> basenn::Optimizer<M, E, D> for Sgd<M, E, D> {
    fn update_tensor<S: Shape>(
        &mut self,
        t: &mut Tensor<S, E, D>,
        gradients: &Gradients<E, D>,
        missing_params: &mut Vec<UniqueId>,
    ) -> Result<(), D::Err> {
        let g = gradients.get_ref_checked(t);
        match g {
            None => missing_params.push(t.id()),
            Some(g) => {
                let v = self.velocity.get_or_alloc_mut(t)?;
                self.cfg.try_update(t, v, g)?;
            }
        }
        Ok(())
    }
}
