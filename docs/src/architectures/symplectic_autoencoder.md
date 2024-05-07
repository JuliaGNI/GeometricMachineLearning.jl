# Symplectic Autoencoder 

A visualization of an instance of [SymplecticAutoencoder](@ref) is shown below: 

```@example 
Main.include_graphics("../tikz/symplectic_autoencoder_architecture") # hide
```

The *intermediate dimension* ``M`` is calculated via `n : (N - n) ÷ (n_blocks - 1) : N`. Further we have the following choices:
- `n_encoder_layers::Integer = 4`
- `n_encoder_blocks::Integer = 2` 
- `n_decoder_layers::Integer = 2` 
- `n_decoder_blocks::Integer = 3`
- `encoder_init_q::Bool = true`
- `decoder_init_q::Bool = true`

Note that all of these are keyword arguments that can be supplied to [SymplecticAutoencoder](@ref).