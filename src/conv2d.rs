use derives::*;
use dfdx::{
    shapes::{Const, Dim, Dtype, HasShape},
    tensor::Tensor,
    tensor_ops::{Device, TryConv2D},
};

#[derive(Debug, Default, Clone, Copy)]
pub struct Conv2D<
    InChan: Dim,
    OutChan: Dim,
    KernelSize: Dim,
    Stride: Dim = Const<1>,
    Padding: Dim = Const<0>,
    Dilation: Dim = Const<1>,
    Groups: Dim = Const<1>,
> {
    pub in_chan: InChan,
    pub out_chan: OutChan,
    pub kernel_size: KernelSize,
    pub stride: Stride,
    pub padding: Padding,
    pub dilation: Dilation,
    pub groups: Groups,
}

impl<I: Dim, O: Dim, K: Dim, S: Dim, P: Dim, L: Dim, G: Dim, E: Dtype, D: Device<E>>
    crate::BuildOnDevice<E, D> for Conv2D<I, O, K, S, P, L, G>
where
    I: std::ops::Div<G>,
    <I as std::ops::Div<G>>::Output: Dim,
{
    type Built = DeviceConv2D<I, O, K, S, P, L, G, E, D>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, <D>::Err> {
        assert_eq!(self.in_chan.size() % self.groups.size(), 0);
        assert_eq!(self.out_chan.size() % self.groups.size(), 0);
        let i_over_g = self.in_chan / self.groups;
        let weight = device.try_zeros_like(&(
            self.out_chan,
            i_over_g,
            self.kernel_size,
            self.kernel_size,
        ))?;
        Ok(DeviceConv2D {
            weight,
            stride: self.stride,
            padding: self.padding,
            dilation: self.dilation,
            groups: self.groups,
        })
    }
}

#[derive(Debug, Clone, UpdateParams, ToDtype, ToDevice)]
pub struct DeviceConv2D<InChan, OutChan, KernelSize, Stride, Padding, Dilation, Groups, Elem, Dev>
where
    InChan: std::ops::Div<Groups>,
    <InChan as std::ops::Div<Groups>>::Output: Dim,
    InChan: Dim,
    OutChan: Dim,
    KernelSize: Dim,
    Stride: Dim,
    Padding: Dim,
    Dilation: Dim,
    Groups: Dim,
    Elem: Dtype,
    Dev: Device<Elem>,
{
    #[param]
    pub weight: Tensor<
        (
            OutChan,
            <InChan as std::ops::Div<Groups>>::Output,
            KernelSize,
            KernelSize,
        ),
        Elem,
        Dev,
    >,
    pub stride: Stride,
    pub padding: Padding,
    pub dilation: Dilation,
    pub groups: Groups,
}

impl<I: Dim, O: Dim, K: Dim, S: Dim, P: Dim, L: Dim, G: Dim, E, D> crate::ResetParams<E, D>
    for DeviceConv2D<I, O, K, S, P, L, G, E, D>
where
    I: std::ops::Div<G>,
    <I as std::ops::Div<G>>::Output: Dim,
    E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
    D: Device<E>,
{
    fn try_reset_params(&mut self) -> Result<(), D::Err> {
        let (_, i_over_g, k, _) = self.weight.shape();
        let scale = E::from_f64(1.0 / (k.size() * k.size() * i_over_g.size()) as f64).unwrap();
        let b = scale.sqrt();
        self.weight
            .try_fill_with_distr(rand_distr::Uniform::new(-b, b))
    }
}

impl<I: Dim, O: Dim, K: Dim, S: Dim, P: Dim, L: Dim, G: Dim, E, D, Img> crate::Module<Img>
    for DeviceConv2D<I, O, K, S, P, L, G, E, D>
where
    I: std::ops::Div<G>,
    <I as std::ops::Div<G>>::Output: Dim,
    E: Dtype,
    D: Device<E>,
    (
        Img,
        Tensor<(O, <I as std::ops::Div<G>>::Output, K, K), E, D>,
    ): TryConv2D<S, P, L, G>,
{
    type Output = <(
        Img,
        Tensor<(O, <I as std::ops::Div<G>>::Output, K, K), E, D>,
    ) as TryConv2D<S, P, L, G>>::Convolved;
    type Error = <(
        Img,
        Tensor<(O, <I as std::ops::Div<G>>::Output, K, K), E, D>,
    ) as TryConv2D<S, P, L, G>>::Error;

    fn try_forward(&self, x: Img) -> Result<Self::Output, Self::Error> {
        (x, self.weight.clone()).try_conv2d(self.stride, self.padding, self.dilation, self.groups)
    }
}
