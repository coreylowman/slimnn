use quote::{quote, quote_spanned};
use syn::spanned::Spanned;
use syn::{parse_macro_input, parse_quote, Data, DeriveInput, Fields, Index};

macro_rules! match_type {
    ($F:ident, $Where:ident, $Ty:ident, tensor=$TensorStmt:tt, module=$ModuleStmt:tt, bound=$Bound:path) => {
        match $Ty {
            syn::Type::Path(path)=> {
                dbg!(&path.path.segments[0].ident);
                match path.path.segments[0].ident.to_string().as_str() {
                    "bool" => quote_spanned!($F.span() => ();),
                    "i8" | "i16" | "i32" | "i64" |"isize" => quote_spanned!($F.span() => ();),
                    "u8" | "u16" | "u32" | "u64" |"usize" => quote_spanned!($F.span() => ();),
                    "f32" | "f64" => quote_spanned!($F.span() => ();),
                    "Const" => quote_spanned!($F.span() => ();),
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

#[proc_macro_derive(Functional, attributes(calls_fn))]
pub fn functional(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let name = input.ident;
    let fn_name = input
        .attrs
        .iter()
        .find(|attr| attr.path().is_ident("calls_fn"))
        .map(|attr| attr.parse_args::<syn::Ident>().unwrap())
        .expect("Need to specify #[calls_fn(<fn name>)] attribute");

    let mut built_generics = input.generics.clone();
    built_generics
        .params
        .push(parse_quote!(Elem: dfdx::prelude::Dtype));
    built_generics
        .params
        .push(parse_quote!(Dev: dfdx::prelude::Device<Elem>));

    // get the generics for the impl. `Input` must be added only to the impl_generics.
    // NOTE: without cloning, `Input` will appear in both impl & ty generics.
    let mut module_generics = built_generics.clone();
    module_generics
        .params
        .push(parse_quote!(S: dfdx::prelude::Shape));
    module_generics
        .params
        .push(parse_quote!(T: dfdx::prelude::Tape<Elem, Dev>));

    let (_, builder_ty, builder_where) = input.generics.split_for_impl();
    let (built_impl, _, built_where) = built_generics.split_for_impl();
    let (module_impl, _, _) = module_generics.split_for_impl();

    proc_macro::TokenStream::from(quote! {
        impl #built_impl basenn::BuildOnDevice<Elem, Dev> for #name #builder_ty #built_where {
            type Built = Self;
            fn try_build_on_device(&self, device: &Dev) -> Result<Self::Built, Dev::Err> {
                Ok(*self)
            }
        }

        impl #built_impl basenn::ResetParams<Elem, Dev> for #name #builder_ty #builder_where {
            fn try_reset_params(&mut self) -> Result<(), Dev::Err> { Ok(()) }
        }

        impl #built_impl basenn::UpdateParams<Elem, Dev> for #name #builder_ty #built_where {
            fn try_update_params<Optim: basenn::Optimizer<Elem, Dev>>(
                &mut self,
                optimizer: &mut Optim,
                gradients: &dfdx::prelude::Gradients<Elem, Dev>,
            ) -> Result<(), Dev::Err> {
                Ok(())
            }
        }

        impl #module_impl basenn::Module<dfdx::prelude::Tensor<S, Elem, Dev, T>> for #name #builder_ty #built_where {
            type Output = dfdx::prelude::Tensor<S, Elem, Dev, T>;
            type Error = Dev::Err;
            fn try_forward(&self, x: dfdx::prelude::Tensor<S, Elem, Dev, T>) -> Result<Self::Output, Self::Error> {
                #fn_name(x)
            }
        }
    })
}

#[proc_macro_derive(BuildOnDevice)]
pub fn build_on_device(_: proc_macro::TokenStream) -> proc_macro::TokenStream {
    proc_macro::TokenStream::new()
}

#[proc_macro_derive(ToDtype)]
pub fn to_dtype(_: proc_macro::TokenStream) -> proc_macro::TokenStream {
    proc_macro::TokenStream::new()
}

#[proc_macro_derive(ToDevice)]
pub fn to_device(_: proc_macro::TokenStream) -> proc_macro::TokenStream {
    proc_macro::TokenStream::new()
}

#[proc_macro_derive(Sequential)]
pub fn sequential(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let builder_name = input.ident.clone();

    let built_name = input
        .attrs
        .iter()
        .find(|attr| attr.path().is_ident("built"))
        .map(|attr| attr.parse_args::<syn::Ident>().unwrap())
        .unwrap_or_else(|| syn::Ident::new(&format!("{}OnDevice", builder_name), input.span()));
    let mut built_generics = input.generics.clone();
    built_generics
        .params
        .push(parse_quote!(Elem: dfdx::prelude::Dtype));
    built_generics
        .params
        .push(parse_quote!(Dev: dfdx::prelude::Device<Elem>));

    // get the generics for the impl. `Input` must be added only to the impl_generics.
    // NOTE: without cloning, `Input` will appear in both impl & ty generics.
    let mut module_generics = built_generics.clone();
    module_generics.params.push(parse_quote!(Input));

    let struct_def = {
        let where_clause = built_generics.make_where_clause();
        let fields = {
            match &input.data {
                Data::Struct(ref obj) => match obj.fields {
                    Fields::Named(ref fields) => {
                        let fields = fields.named.iter().map(|f| {
                            let name = &f.ident;
                            let ty = &f.ty;
                            let vis = &f.vis;
                            where_clause
                                .predicates
                                .push(parse_quote!(#ty: basenn::BuildOnDevice<Elem, Dev>));
                            quote_spanned!(f.span()=> #vis #name: <#ty as basenn::BuildOnDevice<Elem, Dev>>::Built,)
                        });
                        quote! { #(#fields)* }
                    }
                    Fields::Unnamed(ref fields) => {
                        let fields = fields.unnamed.iter().map(|f| {
                            let ty = &f.ty;
                            let vis = &f.vis;
                            where_clause
                                .predicates
                                .push(parse_quote!(#ty: basenn::BuildOnDevice<Elem, Dev>));
                            quote_spanned!(f.span()=> #vis <#ty as basenn::BuildOnDevice<Elem, Dev>>::Built,)
                        });
                        quote! { #(#fields)* }
                    }
                    Fields::Unit => Default::default(),
                },
                Data::Enum(_) => unimplemented!("Sequential cannot be derived for enums."),
                Data::Union(_) => unimplemented!("Sequential cannot be derived for unions."),
            }
        };

        let (built_impl, _, built_where) = built_generics.split_for_impl();

        quote! {
            #[derive(Clone, derives::ResetParams, derives::UpdateParams, derives::ZeroGrads, derives::ToDevice, derives::ToDtype)]
            pub struct #built_name #built_impl #built_where {
                #fields
            }
        }
    };

    let impl_build_on_device = {
        let (_, builder_ty, _) = input.generics.split_for_impl();
        let (built_impl, built_ty, built_where) = built_generics.split_for_impl();

        match input.data {
            Data::Struct(ref data) => match data.fields {
                Fields::Named(ref fields) => {
                    let recurse = fields.named.iter().map(|f| {
                        let name = &f.ident;
                        quote_spanned! {f.span()=> #name: self.#name.try_build_on_device(device)?, }
                    });
                    quote! {
                        impl #built_impl basenn::BuildOnDevice<Elem, Dev> for #builder_name #builder_ty #built_where {
                            type Built = #built_name #built_ty;
                            fn try_build_on_device(&self, device: &Dev) -> Result<Self::Built, Dev::Err> {
                                let built = #built_name {
                                    #(#recurse)*
                                };
                                Ok(built)
                            }
                        }
                    }
                }
                Fields::Unnamed(ref fields) => {
                    let recurse = fields.unnamed.iter().enumerate().map(|(i, f)| {
                        let index = Index::from(i);
                        quote_spanned! {f.span()=> self.#index.try_build_on_device(device)?, }
                    });
                    quote! {
                        impl #built_impl basenn::BuildOnDevice<Elem, Dev> for #builder_name #builder_ty #built_where {
                            type Built = #built_name #built_ty;
                            fn try_build_on_device(&self, device: &Dev) -> Result<Self::Built, Dev::Err> {
                                #built_name(
                                    #(#recurse)*
                                )
                            }
                        }
                    }
                }
                Fields::Unit => proc_macro2::TokenStream::new(),
            },
            _ => unreachable!(),
        }
    };

    // Get's the output type of the sequential. Also adds Module bounds to the where clause.
    let mut last_ty = quote!(Input);
    let err = quote!(<Input as dfdx::prelude::HasErr>::Err);
    let output_ty = {
        let where_clause = module_generics.make_where_clause();
        where_clause
            .predicates
            .push(parse_quote!(Input: dfdx::prelude::HasErr));
        match &input.data {
            Data::Struct(ref obj) => match obj.fields {
                Fields::Named(ref fields) => {
                    fields.named.iter().for_each(|f| {
                        let ty = &f.ty;
                        where_clause
                            .predicates
                            .push(parse_quote!(#ty: basenn::BuildOnDevice<Elem, Dev>));
                        where_clause
                            .predicates
                            .push(parse_quote!(<#ty as basenn::BuildOnDevice<Elem, Dev>>::Built: basenn::Module<#last_ty, Error = #err>));
                        last_ty = parse_quote!(<<#ty as basenn::BuildOnDevice<Elem, Dev>>::Built as basenn::Module<#last_ty>>::Output);
                    });
                }
                Fields::Unnamed(ref fields) => {
                    fields.unnamed.iter().for_each(|f| {
                        let ty = &f.ty;
                        where_clause
                            .predicates
                            .push(parse_quote!(#ty: basenn::BuildOnDevice<Elem, Dev>));
                        where_clause
                            .predicates
                            .push(parse_quote!(<#ty as basenn::BuildOnDevice<Elem, Dev>>::Built: basenn::Module<#last_ty, Error = #err>));
                        last_ty = parse_quote!(<<#ty as basenn::BuildOnDevice<Elem, Dev>>::Built as basenn::Module<#last_ty>>::Output);
                    });
                }
                Fields::Unit => {}
            },
            Data::Enum(_) => unimplemented!("Sequential cannot be derived for enums."),
            Data::Union(_) => unimplemented!("Sequential cannot be derived for unions."),
        };
        last_ty
    };

    let impl_module = {
        let src = match input.data {
            Data::Struct(ref data) => match data.fields {
                Fields::Named(ref fields) => {
                    let recurse = fields.named.iter().map(|f| {
                        let name = &f.ident;
                        quote_spanned! {f.span()=> self.#name.try_forward(x)? }
                    });
                    quote! { #(let x = #recurse;)* }
                }
                Fields::Unnamed(ref fields) => {
                    let recurse = fields.unnamed.iter().enumerate().map(|(i, f)| {
                        let index = Index::from(i);
                        quote_spanned! {f.span()=> self.#index.try_forward(x)? }
                    });
                    quote! { #(let x = #recurse;)* }
                }
                Fields::Unit => quote! { let x = x; },
            },
            _ => unreachable!(),
        };

        let (_, built_ty, _) = built_generics.split_for_impl();
        let (module_impl, _, module_where) = module_generics.split_for_impl();

        quote! {
            impl #module_impl basenn::Module<Input> for #built_name #built_ty #module_where {
                type Output = #output_ty;
                type Error = #err;
                fn try_forward(&self, x: Input) -> Result<Self::Output, Self::Error> {
                    #src
                    Ok(x)
                }
            }
        }
    };

    proc_macro::TokenStream::from(quote! {
        #struct_def
        #impl_build_on_device
        #impl_module
    })
}

#[proc_macro_derive(ResetParams)]
pub fn reset_params(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let mut input = parse_macro_input!(input as DeriveInput);

    let name = input.ident;

    let mut custom_generics = input.generics.clone();
    if custom_generics
        .params
        .iter()
        .position(|param| match param {
            syn::GenericParam::Type(type_param) if type_param.ident == "Elem" => true,
            _ => false,
        })
        .is_none()
    {
        custom_generics
            .params
            .push(parse_quote!(Elem: dfdx::prelude::Dtype));
    }

    if custom_generics
        .params
        .iter()
        .position(|param| match param {
            syn::GenericParam::Type(type_param) if type_param.ident == "Dev" => true,
            _ => false,
        })
        .is_none()
    {
        custom_generics
            .params
            .push(parse_quote!(Dev: dfdx::prelude::Device<Elem>));
    }

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
                        bound=basenn::ResetParams<Elem, Dev>
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
                        bound=basenn::ResetParams<Elem, Dev>
                    )
                });
                quote! { #(#resets)* }
            }
            Fields::Unit => Default::default(),
        },
        Data::Enum(_) => unimplemented!("ResetParams not implemented for enums."),
        Data::Union(_) => unimplemented!("ResetParams not implemented for unions."),
    };

    let (impl_generics, _, _) = custom_generics.split_for_impl();
    let (_, ty_generics, where_clause) = input.generics.split_for_impl();

    proc_macro::TokenStream::from(quote! {
        impl #impl_generics basenn::ResetParams<Elem, Dev> for #name #ty_generics #where_clause {
            fn try_reset_params(&mut self) -> Result<(), Dev::Err> {
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
            syn::GenericParam::Type(type_param) if type_param.ident == "Elem" => true,
            _ => false,
        })
        .is_none()
    {
        custom_generics
            .params
            .push(parse_quote!(Elem: dfdx::prelude::Dtype));
    }

    if custom_generics
        .params
        .iter()
        .position(|param| match param {
            syn::GenericParam::Type(type_param) if type_param.ident == "Dev" => true,
            _ => false,
        })
        .is_none()
    {
        custom_generics
            .params
            .push(parse_quote!(Dev: dfdx::prelude::Device<Elem>));
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
                        bound=basenn::UpdateParams<Elem, Dev>
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
                        bound=basenn::UpdateParams<Elem, Dev>
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
        impl #impl_generics basenn::UpdateParams<Elem, Dev> for #struct_name #ty_generics #where_clause {
            fn try_update_params<Optim: basenn::Optimizer<Elem, Dev>>(
                &mut self,
                optimizer: &mut Optim,
                gradients: &dfdx::tensor::Gradients<Elem, Dev>,
            ) -> Result<(), Dev::Err> {
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
            syn::GenericParam::Type(type_param) if type_param.ident == "Elem" => true,
            _ => false,
        })
        .is_none()
    {
        custom_generics
            .params
            .push(parse_quote!(Elem: dfdx::prelude::Dtype));
    }

    if custom_generics
        .params
        .iter()
        .position(|param| match param {
            syn::GenericParam::Type(type_param) if type_param.ident == "Dev" => true,
            _ => false,
        })
        .is_none()
    {
        custom_generics
            .params
            .push(parse_quote!(Dev: dfdx::prelude::Device<Elem>));
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
                        bound=basenn::ZeroGrads<Elem, Dev>
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
                        bound=basenn::ZeroGrads<Elem, Dev>
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
        impl #impl_generics basenn::ZeroGrads<Elem, Dev> for #name #ty_generics #where_clause {
            fn try_zero_grads(&self, grads: &mut dfdx::prelude::Gradients<Elem, Dev>) -> Result<(), Dev::Err> {
                #zero_grads
                Ok(())
            }
        }
    })
}
