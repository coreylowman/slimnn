use derives::*;
use dfdx::{
    shapes::{Const, Dtype},
    tensor_ops::{Device, TryPool2D},
};

#[derive(Debug, Default, Clone, UpdateParams, ZeroGrads, ToDtype, ToDevice)]
pub struct ConstMaxPool2D<
    const KERNEL_SIZE: usize,
    const STRIDE: usize = 1,
    const PADDING: usize = 0,
    const DILATION: usize = 1,
>;

impl<
        const K: usize,
        const S: usize,
        const P: usize,
        const L: usize,
        Elem: Dtype,
        Dev: Device<Elem>,
    > crate::BuildOnDevice<Elem, Dev> for ConstMaxPool2D<K, S, P, L>
{
    type Built = Self;
    fn try_build_on_device(&self, _: &Dev) -> Result<Self::Built, <Dev>::Err> {
        Ok(self.clone())
    }
}

impl<
        const K: usize,
        const S: usize,
        const P: usize,
        const L: usize,
        Img: TryPool2D<Const<K>, Const<S>, Const<P>, Const<L>>,
    > crate::Module<Img> for ConstMaxPool2D<K, S, P, L>
{
    type Output = Img::Pooled;
    type Error = Img::Error;

    fn try_forward(&self, x: Img) -> Result<Self::Output, Self::Error> {
        x.try_pool2d(
            dfdx::tensor_ops::Pool2DKind::Max,
            Const,
            Const,
            Const,
            Const,
        )
    }
}

#[derive(Debug, Default, Clone, UpdateParams, ZeroGrads, ToDtype, ToDevice)]
pub struct DynMaxPool2D {
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
}

impl<Elem: Dtype, Dev: Device<Elem>> crate::BuildOnDevice<Elem, Dev> for DynMaxPool2D {
    type Built = Self;
    fn try_build_on_device(&self, _: &Dev) -> Result<Self::Built, <Dev>::Err> {
        Ok(self.clone())
    }
}

impl<Img: TryPool2D<usize, usize, usize, usize>> crate::Module<Img> for DynMaxPool2D {
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
