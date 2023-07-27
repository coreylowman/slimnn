use basenn::{BuildModuleExt, Module};
use slimnn::{
    linear::{ConstLinear, DynLinear},
    relu::ReLU,
    *,
};

mod arch {
    use super::*;

    #[derive(Default, Clone, Sequential)]
    pub struct ConstMlp {
        pub l1: ConstLinear<3, 5>,
        pub act1: ReLU,
        pub l2: ConstLinear<5, 10>,
        pub act2: ReLU,
    }

    #[derive(Clone, Sequential)]
    pub struct DynMlp {
        pub l1: DynLinear,
        pub act1: ReLU,
        pub l2: DynLinear,
        pub act2: ReLU,
    }
}

fn main() {
    use dfdx::prelude::*;
    let dev: Cpu = Default::default();

    {
        let const_structure = arch::ConstMlp::default();
        let module = dev.build_module_ext::<f32>(const_structure);
        let x: Tensor<Rank2<10, 3>, f32, _> = dev.sample_normal();
        let _: Tensor<Rank2<10, 10>, f32, _> = module.forward(x);
    }

    {
        let const_structure = arch::DynMlp {
            l1: DynLinear::new(3, 5),
            act1: Default::default(),
            l2: DynLinear::new(5, 10),
            act2: Default::default(),
        };
        let module = dev.build_module_ext::<f32>(const_structure);
        let x: Tensor<(Const<10>, usize), f32, _> = dev.sample_normal_like(&(Const, 3));
        let y: Tensor<(Const<10>, usize), f32, _> = module.forward(x);
        assert_eq!(y.shape(), &(Const, 10));
    }
}
