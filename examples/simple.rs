use basenn::{BuildModuleExt, Module};

mod arch {
    use slimnn::{linear::ConstLinear, relu::ReLU, *};

    #[derive(Default, Sequential)]
    pub struct MyMlp {
        l1: ConstLinear<3, 5>,
        act1: ReLU,
        l2: ConstLinear<5, 10>,
        act2: ReLU,
    }
}

fn main() {
    use dfdx::prelude::*;
    let dev: Cpu = Default::default();
    let module = dev.build_module_ext::<f32>(arch::MyMlp::default());
    let x: Tensor<Rank2<10, 3>, f32, _> = dev.sample_normal();
    let y: Tensor<Rank2<10, 10>, f32, _> = module.forward(x);
}
