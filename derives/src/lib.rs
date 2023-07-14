use quote::{quote, quote_spanned};
use syn::spanned::Spanned;
use syn::{parse_macro_input, parse_quote, Data, DeriveInput, Fields, Index};

macro_rules! match_type {
    ($F:ident, $Where:ident, $Ty:ident, tensor=$TensorStmt:tt, module=$ModuleStmt:tt, bound=$Bound:path) => {
        match $Ty {
            syn::Type::Path(path)=> {
                match path.path.segments[0].ident.to_string().as_str() {
                    "bool" => quote_spanned!($F.span() => ();),
                    "i8" | "i16" | "i32" | "i64" |"isize" => quote_spanned!($F.span() => ();),
                    "u8" | "u16" | "u32" | "u64" |"usize" => quote_spanned!($F.span() => ();),
                    "f32" | "f64" => quote_spanned!($F.span() => ();),
                    "Tensor" => quote_spanned!($F.span() => $TensorStmt;),
                    _ => {
                        $Where
                            .predicates
                            .push(parse_quote!(#$Ty: $Bound));
                        quote_spanned!($F.span() => $ModuleStmt;)
                    }
                }
            }
            _ => {
                $Where
                    .predicates
                    .push(parse_quote!(#$Ty: $Bound));
                quote_spanned!($F.span() => $ModuleStmt;)
            },
        }
    };
}

#[proc_macro_derive(Sequential)]
pub fn sequential(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    proc_macro::TokenStream::new()
}

#[proc_macro_derive(Functional, attributes(calls_fn))]
pub fn functional(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    proc_macro::TokenStream::new()
}

#[proc_macro_derive(ToDtype)]
pub fn to_dtype(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    proc_macro::TokenStream::new()
}

#[proc_macro_derive(ToDevice)]
pub fn to_device(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    proc_macro::TokenStream::new()
}

#[proc_macro_derive(ResetParams)]
pub fn reset_params(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let mut input = parse_macro_input!(input as DeriveInput);

    let name = input.ident;

    let where_clause = input.generics.make_where_clause();
    let resets = match &input.data {
        Data::Struct(ref obj) => match obj.fields {
            Fields::Named(ref fields) => {
                let resets = fields.named.iter().map(|f| {
                    let name = &f.ident;
                    let ty = &f.ty;
                    match_type!(
                        f, where_clause, ty,
                        tensor={self.#name.try_reset_params().unwrap();},
                        module={self.#name.try_reset_params().unwrap();},
                        bound=crate::ResetParams
                    )
                });
                quote! { #(#resets)* }
            }
            Fields::Unnamed(ref fields) => {
                let resets = fields.unnamed.iter().enumerate().map(|(i, f)| {
                    let index = Index::from(i);
                    let ty = &f.ty;
                    match_type!(
                        f, where_clause, ty,
                        tensor={self.#index.try_reset_params().unwrap();},
                        module={self.#index.try_reset_params().unwrap();},
                        bound=crate::ResetParams
                    )
                });
                quote! { #(#resets)* }
            }
            Fields::Unit => Default::default(),
        },
        Data::Enum(_) => unimplemented!("ResetParams not implemented for enums."),
        Data::Union(_) => unimplemented!("ResetParams not implemented for unions."),
    };

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    proc_macro::TokenStream::from(quote! {
        impl #impl_generics crate::ResetParams for #name #ty_generics #where_clause {
            type Error = std::convert::Infallible;
            fn try_reset_params(&mut self) -> Result<(), Self::Error> {
                #resets
                Ok(())
            }
        }
    })
}

#[proc_macro_derive(UpdateParams)]
pub fn update_params(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let mut input = parse_macro_input!(input as DeriveInput);

    let struct_name = input.ident;

    let mut custom_generics = input.generics.clone();
    if custom_generics
        .params
        .iter()
        .position(|param| match param {
            syn::GenericParam::Type(type_param) if type_param.ident == "E" => true,
            _ => false,
        })
        .is_none()
    {
        custom_generics
            .params
            .push(parse_quote!(E: dfdx::prelude::Dtype));
    }

    if custom_generics
        .params
        .iter()
        .position(|param| match param {
            syn::GenericParam::Type(type_param) if type_param.ident == "D" => true,
            _ => false,
        })
        .is_none()
    {
        custom_generics
            .params
            .push(parse_quote!(D: dfdx::prelude::Device<E>));
    }

    let where_clause = input.generics.make_where_clause();
    let updates = match &input.data {
        Data::Struct(ref obj) => match obj.fields {
            Fields::Named(ref fields) => {
                let updates = fields.named.iter().map(|f| {
                    let name = &f.ident;
                    let ty = &f.ty;
                    match_type!(
                        f, where_clause, ty,
                        tensor={optimizer.update_tensor(&mut self.#name, gradients)?;},
                        module={self.#name.try_update_params(optimizer, gradients)?;},
                        bound=crate::UpdateParams<E, D>
                    )
                });
                quote! { #(#updates)* }
            }
            Fields::Unnamed(ref fields) => {
                let updates = fields.unnamed.iter().enumerate().map(|(i, f)| {
                    let index = Index::from(i);
                    let ty = &f.ty;
                    match_type!(
                        f, where_clause, ty,
                        tensor={optimizer.update_tensor(&mut self.#index, gradients)?;},
                        module={self.#index.try_update_params(optimizer, gradients)?;},
                        bound=crate::UpdateParams<E, D>
                    )
                });
                quote! { #(#updates)* }
            }
            Fields::Unit => Default::default(),
        },
        Data::Enum(_) => unimplemented!("UpdateParams not implemented for enums."),
        Data::Union(_) => unimplemented!("UpdateParams not implemented for unions."),
    };

    let (impl_generics, _, _) = custom_generics.split_for_impl();
    let (_, ty_generics, where_clause) = input.generics.split_for_impl();

    proc_macro::TokenStream::from(quote! {
        impl #impl_generics crate::UpdateParams<E, D> for #struct_name #ty_generics #where_clause {
            fn try_update_params<Optim: crate::Optimizer<E, D>>(
                &mut self,
                optimizer: &mut Optim,
                gradients: &dfdx::tensor::Gradients<E, D>,
            ) -> Result<(), D::Err> {
                #updates
                Ok(())
            }
        }
    })
}

#[proc_macro_derive(ZeroGrads)]
pub fn zero_grads(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let mut input = parse_macro_input!(input as DeriveInput);

    let name = input.ident;

    let mut custom_generics = input.generics.clone();
    if custom_generics
        .params
        .iter()
        .position(|param| match param {
            syn::GenericParam::Type(type_param) if type_param.ident == "E" => true,
            _ => false,
        })
        .is_none()
    {
        custom_generics
            .params
            .push(parse_quote!(E: dfdx::prelude::Dtype));
    }

    if custom_generics
        .params
        .iter()
        .position(|param| match param {
            syn::GenericParam::Type(type_param) if type_param.ident == "D" => true,
            _ => false,
        })
        .is_none()
    {
        custom_generics
            .params
            .push(parse_quote!(D: dfdx::prelude::Device<E>));
    }

    let where_clause = input.generics.make_where_clause();
    let zero_grads = match &input.data {
        Data::Struct(ref obj) => match obj.fields {
            Fields::Named(ref fields) => {
                let zero_grads = fields.named.iter().map(|f| {
                    let name = &f.ident;
                    let ty = &f.ty;
                    match_type!(
                        f, where_clause, ty,
                        tensor={
                            let grad = grads.get_or_alloc_mut(&self.#name)?;
                            self.#name.device().try_fill_with_zeros(grad)?;
                        },
                        module={self.#name.try_zero_grads(grads)?;},
                        bound=crate::ZeroGrads<E, D>
                    )
                });
                quote! { #(#zero_grads)* }
            }
            Fields::Unnamed(ref fields) => {
                let zero_grads = fields.unnamed.iter().enumerate().map(|(i, f)| {
                    let index = Index::from(i);
                    let ty = &f.ty;
                    match_type!(
                        f, where_clause, ty,
                        tensor={
                            let grad = grads.get_or_alloc_mut(&self.#index)?;
                            self.#index.device().try_fill_with_zeros(grad)?;
                        },
                        module={self.#index.try_zero_grads(grads)?;},
                        bound=crate::ZeroGrads<E, D>
                    )
                });
                quote! { #(#zero_grads)* }
            }
            Fields::Unit => Default::default(),
        },
        Data::Enum(_) => unimplemented!("ZeroGrads not implemented for enums."),
        Data::Union(_) => unimplemented!("ZeroGrads not implemented for unions."),
    };

    let (impl_generics, _, _) = custom_generics.split_for_impl();
    let (_, ty_generics, where_clause) = input.generics.split_for_impl();

    proc_macro::TokenStream::from(quote! {
        impl #impl_generics crate::ZeroGrads<E, D> for #name #ty_generics #where_clause {
            fn try_zero_grads(&self, grads: &mut dfdx::prelude::Gradients<E, D>) -> Result<(), D::Err> {
                #zero_grads
                Ok(())
            }
        }
    })
}
