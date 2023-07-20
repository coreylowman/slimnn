use dfdx::prelude::*;
use slimnn::{Module, *};

fn main() {
    let dev: Cpu = Default::default();
    let arch: slimnn::linear::ConstLinear<3, 5> = Default::default();
    let module = dev.build_module_ext::<f32>(arch);
    let x: Tensor<Rank2<10, 3>, f32, _> = dev.sample_normal();
    let y = module.forward(x);
}
