use derives::*;
use dfdx::tensor_ops::TryAdd;

use crate::Module;

#[derive(Default, Clone, Debug, ResetParams, ZeroGrads, UpdateParams, ToDtype, ToDevice)]
pub struct ResidualAdd<T>(pub T);

impl<X: Clone, T: Module<X>> Module<X> for ResidualAdd<T>
where
    X: TryAdd<T::Output, Err = T::Error>,
{
    type Output = X;
    type Error = T::Error;
    fn try_forward(&self, x: X) -> Result<Self::Output, Self::Error> {
        let y = self.0.try_forward(x.clone())?;
        x.try_add(y)
    }

    fn try_forward_mut(&mut self, x: X) -> Result<Self::Output, Self::Error> {
        let y = self.0.try_forward_mut(x.clone())?;
        x.try_add(y)
    }
}
