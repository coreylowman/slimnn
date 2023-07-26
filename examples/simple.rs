use basenn::{BuildModuleExt, Module};

mod arch {
    use slimnn::{linear::ConstLinear, relu::ReLU, *};

    #[derive(Default, Clone, Sequential)]
    pub struct MyMlp {
        pub l1: ConstLinear<3, 5>,
        pub act1: ReLU,
        pub l2: ConstLinear<5, 10>,
        pub act2: ReLU,
    }
}

fn main() {
    use dfdx::prelude::*;
    let dev: Cpu = Default::default();
    let module = dev.build_module_ext::<f32>(arch::MyMlp::default());
    let x: Tensor<Rank2<10, 3>, f32, _> = dev.sample_normal();
    let _: Tensor<Rank2<10, 10>, f32, _> = module.forward(x);
}
