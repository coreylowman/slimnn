mod tuples;
mod vecs;

use dfdx::{
    prelude::{Device, Dtype, Gradients, Shape, Tensor, UniqueId},
    shapes::HasShape,
};

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

pub trait SaveSafeTensors {
    fn save_safetensors<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), safetensors::SafeTensorError> {
        let mut tensors = Vec::new();
        self.write_safetensors("", &mut tensors);
        let data = tensors
            .iter()
            .map(|(k, dtype, shape, data)| {
                (
                    k.clone(),
                    safetensors::tensor::TensorView::new(dtype.clone(), shape.clone(), data)
                        .unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let data = data.iter().map(|i| (i.0.clone(), &i.1)).collect::<Vec<_>>();

        safetensors::serialize_to_file(data, &None, path.as_ref())
    }
    fn write_safetensors(
        &self,
        location: &str,
        tensors: &mut Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)>,
    );
}

impl<S: Shape, E: Dtype, D: Device<E>, T> SaveSafeTensors for Tensor<S, E, D, T> {
    fn write_safetensors(
        &self,
        location: &str,
        tensors: &mut Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)>,
    ) {
        tensors.push((
            location.to_string(),
            <E as dfdx::dtypes::SafeTensorsDtype>::DTYPE,
            self.shape().concrete().into(),
            self.as_vec().iter().flat_map(|e| e.to_le_bytes()).collect(),
        ));
    }
}

impl SaveSafeTensors for bool {
    fn write_safetensors(
        &self,
        location: &str,
        tensors: &mut Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)>,
    ) {
        tensors.push((
            location.to_string(),
            safetensors::Dtype::BOOL,
            Vec::new(),
            vec![*self as u8],
        ));
    }
}

impl SaveSafeTensors for f32 {
    fn write_safetensors(
        &self,
        location: &str,
        tensors: &mut Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)>,
    ) {
        tensors.push((
            location.to_string(),
            safetensors::Dtype::F32,
            Vec::new(),
            self.to_le_bytes().to_vec(),
        ));
    }
}

impl SaveSafeTensors for f64 {
    fn write_safetensors(
        &self,
        location: &str,
        tensors: &mut Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)>,
    ) {
        tensors.push((
            location.to_string(),
            safetensors::Dtype::F64,
            Vec::new(),
            self.to_le_bytes().to_vec(),
        ));
    }
}

pub trait LoadSafeTensors {
    fn load_safetensors<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
    ) -> Result<(), safetensors::SafeTensorError> {
        let f = std::fs::File::open(path)?;
        let buffer = unsafe { memmap2::MmapOptions::new().map(&f)? };
        let tensors = safetensors::SafeTensors::deserialize(&buffer)?;
        self.read_safetensors("", &tensors)
    }

    fn read_safetensors<'a>(
        &mut self,
        location: &str,
        tensors: &safetensors::SafeTensors<'a>,
    ) -> Result<(), safetensors::SafeTensorError>;
}

impl<S: Shape, E: Dtype, D: Device<E>, T> LoadSafeTensors for Tensor<S, E, D, T> {
    fn read_safetensors<'a>(
        &mut self,
        location: &str,
        tensors: &safetensors::SafeTensors<'a>,
    ) -> Result<(), safetensors::SafeTensorError> {
        self.load_safetensor(tensors, location)
    }
}

impl LoadSafeTensors for bool {
    fn read_safetensors<'a>(
        &mut self,
        location: &str,
        tensors: &safetensors::SafeTensors<'a>,
    ) -> Result<(), safetensors::SafeTensorError> {
        let view = tensors.tensor(location)?;
        *self = view.data()[0] != 0;
        Ok(())
    }
}

impl LoadSafeTensors for f32 {
    fn read_safetensors<'a>(
        &mut self,
        location: &str,
        tensors: &safetensors::SafeTensors<'a>,
    ) -> Result<(), safetensors::SafeTensorError> {
        let view = tensors.tensor(location)?;
        *self = Self::from_le_bytes(view.data().try_into().unwrap());
        Ok(())
    }
}

impl LoadSafeTensors for f64 {
    fn read_safetensors<'a>(
        &mut self,
        location: &str,
        tensors: &safetensors::SafeTensors<'a>,
    ) -> Result<(), safetensors::SafeTensorError> {
        let view = tensors.tensor(location)?;
        *self = Self::from_le_bytes(view.data().try_into().unwrap());
        Ok(())
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
