mod tuples;
mod vecs;

use dfdx::prelude::{Device, Dtype, Gradients, Shape, Tensor, UniqueId};

pub trait Module<X> {
    type Output;
    type Error: std::fmt::Debug;

    fn try_forward(&self, x: X) -> Result<Self::Output, Self::Error>;

    fn try_forward_mut(&mut self, x: X) -> Result<Self::Output, Self::Error> {
        self.try_forward(x)
    }

    fn forward(&self, x: X) -> Self::Output {
        self.try_forward(x).unwrap()
    }

    fn forward_mut(&mut self, x: X) -> Self::Output {
        self.try_forward_mut(x).unwrap()
    }
}

/// An error indicating that a parameter was not used in gradient
/// computation, and was therefore not present in [Gradients]
/// during an update.
#[derive(Debug)]
pub enum OptimizerUpdateError<Err> {
    UnusedTensors(Vec<UniqueId>),
    DeviceError(Err),
}

impl<Err: std::fmt::Display> std::fmt::Display for OptimizerUpdateError<Err> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnusedTensors(unused) => write!(f, "Unused tensors: {unused:?}"),
            Self::DeviceError(err) => write!(f, "{err}"),
        }
    }
}

#[cfg(feature = "std")]
impl<Err: std::fmt::Debug + std::fmt::Display> std::error::Error for OptimizerUpdateError<Err> {}

pub trait Optimizer<M, E: Dtype, D: Device<E>>: Sized {
    fn update_tensor<S: Shape>(
        &mut self,
        t: &mut Tensor<S, E, D>,
        gradients: &Gradients<E, D>,
        missing_tensors: &mut Vec<UniqueId>,
    ) -> Result<(), D::Err>;

    fn update(
        &mut self,
        module: &mut M,
        gradients: &Gradients<E, D>,
    ) -> Result<(), OptimizerUpdateError<D::Err>>
    where
        M: UpdateParams<E, D>,
    {
        let mut missing_tensors = Vec::new();
        module
            .try_update_params(self, gradients, &mut missing_tensors)
            .map_err(OptimizerUpdateError::DeviceError)?;
        if missing_tensors.is_empty() {
            Ok(())
        } else {
            Err(OptimizerUpdateError::UnusedTensors(missing_tensors))
        }
    }
}

pub trait BuildOnDevice<E: Dtype, D: Device<E>>: Clone {
    type Built: Clone + std::fmt::Debug;
    fn build_on_device(&self, device: &D) -> Self::Built {
        self.try_build_on_device(device).unwrap()
    }
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, D::Err>;
}

pub trait ResetParams<E: Dtype, D: Device<E>> {
    fn reset_params(&mut self) {
        self.try_reset_params().unwrap()
    }
    fn try_reset_params(&mut self) -> Result<(), D::Err>;
}

pub trait UpdateParams<E: Dtype, D: Device<E>> {
    fn update_params<M, Optim: Optimizer<M, E, D>>(
        &mut self,
        optimizer: &mut Optim,
        gradients: &Gradients<E, D>,
        missing_tensors: &mut Vec<UniqueId>,
    ) {
        self.try_update_params(optimizer, gradients, missing_tensors)
            .unwrap()
    }
    fn try_update_params<M, Optim: Optimizer<M, E, D>>(
        &mut self,
        optimizer: &mut Optim,
        gradients: &Gradients<E, D>,
        missing_tensors: &mut Vec<UniqueId>,
    ) -> Result<(), D::Err>;
}

pub trait ZeroGrads<E: Dtype, D: Device<E>> {
    fn zero_grads(&self, grads: &mut Gradients<E, D>) {
        self.try_zero_grads(grads).unwrap()
    }
    fn try_zero_grads(&self, grads: &mut Gradients<E, D>) -> Result<(), D::Err>;

    fn alloc_grads(&self) -> Gradients<E, D> {
        self.try_alloc_grads().unwrap()
    }
    fn try_alloc_grads(&self) -> Result<Gradients<E, D>, D::Err> {
        let mut grads = Gradients::leaky();
        self.try_zero_grads(&mut grads)?;
        grads.retain_current_grads_as_leafs();
        Ok(grads)
    }
}

pub trait BuildModuleExt<M>: Sized {
    fn build_module_ext<E: Dtype>(&self, m: M) -> M::Built
    where
        M: BuildOnDevice<E, Self>,
        M::Built: ResetParams<E, Self>,
        Self: Device<E>,
    {
        let mut module = m.build_on_device(self);
        module.reset_params();
        module
    }
}
impl<D, M> BuildModuleExt<M> for D {}
