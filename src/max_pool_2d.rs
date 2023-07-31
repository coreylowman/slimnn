use derives::*;
use dfdx::{
    shapes::{Const, Dim, Dtype},
    tensor_ops::{Device, TryPool2D},
};

#[derive(Debug, Default, Clone, UpdateParams, ResetParams, ZeroGrads, ToDtype, ToDevice)]
pub struct MaxPool2D<
    KernelSize: Dim,
    Stride: Dim = Const<1>,
    Padding: Dim = Const<0>,
    Dilation: Dim = Const<1>,
> {
    pub kernel_size: KernelSize,
    pub stride: Stride,
    pub padding: Padding,
    pub dilation: Dilation,
}

impl<K: Dim, S: Dim, P: Dim, L: Dim, Elem: Dtype, Dev: Device<Elem>> crate::BuildOnDevice<Elem, Dev>
    for MaxPool2D<K, S, P, L>
{
    type Built = Self;
    fn try_build_on_device(&self, _: &Dev) -> Result<Self::Built, <Dev>::Err> {
        Ok(self.clone())
    }
}

impl<K: Dim, S: Dim, P: Dim, L: Dim, Img: TryPool2D<K, S, P, L>> crate::Module<Img>
    for MaxPool2D<K, S, P, L>
{
    type Output = Img::Pooled;
    type Error = Img::Error;

    fn try_forward(&self, x: Img) -> Result<Self::Output, Self::Error> {
        x.try_pool2d(
            dfdx::tensor_ops::Pool2DKind::Max,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )
    }
}
