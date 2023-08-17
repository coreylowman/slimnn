use basenn::{BuildModuleExt, Module};
use dfdx::shapes::Const;
use slimnn::{linear::LinearConfig, relu::ReLU, *};

mod arch {
    use super::*;

    #[derive(Default, Clone, Sequential)]
    pub struct MixedMlp {
        pub l1: LinearConfig<Const<3>, usize>,
        pub act1: ReLU,
        pub l2: LinearConfig<usize, Const<10>>,
        pub act2: ReLU,
    }
}

fn main() {
    use dfdx::prelude::*;
    let dev: Cpu = Default::default();

    let structure = arch::MixedMlp {
        l1: slimnn::linear::LinearConfig::new(Const, 5),
        act1: Default::default(),
        l2: slimnn::linear::LinearConfig::new(5, Const),
        act2: Default::default(),
    };
    let module = dev.build_module_ext::<f32>(structure);
    let x: Tensor<(Const<10>, Const<3>), f32, _> = dev.sample_normal();
    let _: Tensor<(Const<10>, Const<10>), f32, _> = module.forward(x);
}
