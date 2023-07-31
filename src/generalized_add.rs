use derives::*;
use dfdx::tensor_ops::TryAdd;

use crate::Module;

#[derive(
    Default, Clone, Debug, BuildOnDevice, ResetParams, ZeroGrads, UpdateParams, ToDevice, ToDtype,
)]
pub struct GeneralizedAdd<T, U>(pub T, pub U);

impl<X: Clone, T: Module<X>, U: Module<X, Error = T::Error>> Module<X> for GeneralizedAdd<T, U>
where
    T::Output: TryAdd<U::Output, Err = T::Error>,
{
    type Output = T::Output;
    type Error = T::Error;
    fn try_forward(&self, x: X) -> Result<Self::Output, Self::Error> {
        let t = self.0.try_forward(x.clone())?;
        let u = self.1.try_forward(x)?;
        t.try_add(u)
    }

    fn try_forward_mut(&mut self, x: X) -> Result<Self::Output, Self::Error> {
        let t = self.0.try_forward_mut(x.clone())?;
        let u = self.1.try_forward_mut(x)?;
        t.try_add(u)
    }
}
