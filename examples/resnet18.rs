#![feature(generic_const_exprs)]

use slimnn::{
    avg_pool_global::AvgPoolGlobal, batch_norm2d::ConstBatchNorm2DConfig,
    conv2d::ConstConv2DConfig, generalized_add::GeneralizedAdd, linear::ConstLinearConfig,
    max_pool_2d::ConstMaxPool2D, relu::ReLU, residual_add::ResidualAdd, *,
};

fn main() {
    #[derive(Default, Clone, Sequential)]
    pub struct BasicBlockInternal<const C: usize> {
        conv1: ConstConv2DConfig<C, C, 3, 1, 1>,
        bn1: ConstBatchNorm2DConfig<C>,
        relu: ReLU,
        conv2: ConstConv2DConfig<C, C, 3, 1, 1>,
        bn2: ConstBatchNorm2DConfig<C>,
    }

    #[derive(Default, Clone, Sequential)]
    pub struct DownsampleA<const C: usize, const D: usize> {
        conv1: ConstConv2DConfig<C, D, 3, 2, 1>,
        bn1: ConstBatchNorm2DConfig<D>,
        relu: ReLU,
        conv2: ConstConv2DConfig<D, D, 3, 1, 1>,
        bn2: ConstBatchNorm2DConfig<D>,
    }

    #[derive(Default, Clone, Sequential)]
    pub struct DownsampleB<const C: usize, const D: usize> {
        conv1: ConstConv2DConfig<C, D, 1, 2, 0>,
        bn1: ConstBatchNorm2DConfig<D>,
    }

    pub type BasicBlock<const C: usize> = ResidualAdd<BasicBlockInternal<C>>;

    pub type Downsample<const C: usize, const D: usize> =
        GeneralizedAdd<DownsampleA<C, D>, DownsampleB<C, D>>;

    #[derive(Default, Clone, Sequential)]
    pub struct Head {
        conv: ConstConv2DConfig<3, 64, 7, 2, 3>,
        bn: ConstBatchNorm2DConfig<64>,
        relu: ReLU,
        pool: ConstMaxPool2D<3, 2, 1>,
    }

    #[derive(Default, Clone, Sequential)]
    pub struct Resnet18<const NUM_CLASSES: usize> {
        head: Head,
        l1: (BasicBlock<64>, ReLU, BasicBlock<64>, ReLU),
        l2: (Downsample<64, 128>, ReLU, BasicBlock<128>, ReLU),
        l3: (Downsample<128, 256>, ReLU, BasicBlock<256>, ReLU),
        l4: (Downsample<256, 512>, ReLU, BasicBlock<512>, ReLU),
        l5: (AvgPoolGlobal, ConstLinearConfig<512, NUM_CLASSES>),
    }

    {
        use dfdx::prelude::*;

        let dev = AutoDevice::default();
        let arch = Resnet18::<1000>::default();
        let m = dev.build_module_ext::<f32>(arch);

        let x: Tensor<Rank3<3, 224, 224>, f32, _> = dev.sample_normal();
        let _: Tensor<Rank1<1000>, f32, _> = m.forward(x.clone());
    }
}
