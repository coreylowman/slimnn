use dfdx::{dtypes::Dtype, tensor_ops::Device};

macro_rules! tuple_impls {
    ([$($name:ident),+] [$($idx:tt),+], $last:ident, [$($rev_tail:ident),*]) => {
        impl<Dev: Device<Elem>, Elem: Dtype, $($name: crate::BuildOnDevice<Elem, Dev>),+> crate::BuildOnDevice<Elem, Dev> for ($($name,)+) {
            type Built = ($($name::Built, )+);
            fn try_build_on_device(&self, device: &Dev) -> Result<Self::Built, Dev::Err> {
                Ok(($(
                    self.$idx.try_build_on_device(device)?,
                )+))
            }
        }

        /*This macro expands like this for a 4-tuple:

        impl<
            Input: Tensor,

            // `$last:`
            D:

            // `$(Module::<$rev_tail ::Output>, $rev_tail: )+`
            Module<C ::Output>, C:
            Module<B ::Output>, B:
            Module<A ::Output>, A:

            Module<Input>
        > Module<Input> for (A, B, C, D) {
            type Output = D::Output;
            fn forward(&self, x: Input) -> Self::Output {
                let x = self.0.forward(x);
                let x = self.1.forward(x);
                let x = self.2.forward(x);
                let x = self.3.forward(x);
                x
            }
        }
        */
        impl<
            Input,
            $last:
            $(crate::Module::<$rev_tail ::Output, Error=$rev_tail::Error>, $rev_tail: )*
            crate::Module<Input>
        > crate::Module<Input> for ($($name,)+) {
            type Output = $last ::Output;
            type Error = $last ::Error;

            /// Calls forward sequentially on each module in the tuple.
            fn try_forward(&self, x: Input) -> Result<Self::Output, Self::Error> {
                $(let x = self.$idx.try_forward(x)?;)+
                Ok(x)
            }

            /// Calls forward sequentially on each module in the tuple.
            fn try_forward_mut(&mut self, x: Input) -> Result<Self::Output, Self::Error> {
                $(let x = self.$idx.try_forward_mut(x)?;)+
                Ok(x)
            }
        }
    };
}

tuple_impls!([M1][0], M1, []);
tuple_impls!([M1, M2] [0, 1], M2, [M1]);
tuple_impls!([M1, M2, M3] [0, 1, 2], M3, [M2, M1]);
tuple_impls!([M1, M2, M3, M4] [0, 1, 2, 3], M4, [M3, M2, M1]);
tuple_impls!([M1, M2, M3, M4, M5] [0, 1, 2, 3, 4], M5, [M4, M3, M2, M1]);
tuple_impls!([M1, M2, M3, M4, M5, M6] [0, 1, 2, 3, 4, 5], M6, [M5, M4, M3, M2, M1]);
