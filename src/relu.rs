use derives::Functional;

#[derive(Default, Debug, Clone, Copy, Functional)]
#[calls_fn(try_relu)]
pub struct ReLU;
