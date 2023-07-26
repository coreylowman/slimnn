use dfdx::prelude::{Device, Dtype, Gradients, Shape, Tensor};

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

pub trait Optimizer<E: Dtype, D: Device<E>> {
    fn update_tensor<S: Shape>(
        &mut self,
        t: &mut Tensor<S, E, D>,
        gradients: &Gradients<E, D>,
    ) -> Result<(), D::Err>;
}

pub trait BuildOnDevice<E: Dtype, D: Device<E>>: Clone {
    type Built: Clone;
    fn build_on_device(&self, device: &D) -> Self::Built {
        self.try_build_on_device(device).unwrap()
    }
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, D::Err>;
}

pub trait ResetParams {
    type Error: std::fmt::Debug;
    fn reset_params(&mut self) {
        self.try_reset_params().unwrap()
    }
    fn try_reset_params(&mut self) -> Result<(), Self::Error>;
}

pub trait UpdateParams<E: Dtype, D: Device<E>> {
    fn update_params<Optim: Optimizer<E, D>>(
        &mut self,
        optimizer: &mut Optim,
        gradients: &Gradients<E, D>,
    ) {
        self.try_update_params(optimizer, gradients).unwrap()
    }
    fn try_update_params<Optim: Optimizer<E, D>>(
        &mut self,
        optimizer: &mut Optim,
        gradients: &Gradients<E, D>,
    ) -> Result<(), D::Err>;
}

pub trait BuildModuleExt<M>: Sized {
    fn build_module_ext<E: Dtype>(&self, m: M) -> M::Built
    where
        M: BuildOnDevice<E, Self>,
        M::Built: ResetParams,
        Self: Device<E>,
    {
        let mut module = m.build_on_device(self);
        module.reset_params();
        module
    }
}
impl<D, M> BuildModuleExt<M> for D {}

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
        Ok(grads)
    }
}

pub trait ToDtype<E> {
    type AsDtype;
    fn to_dtype(self) -> Self::AsDtype;
}

pub trait ToDevice<D> {
    type OnDevice;
    fn to_device(self, device: &D) -> Self::OnDevice;
}
