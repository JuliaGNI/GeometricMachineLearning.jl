# Symplectic Autoencoder 

A visualization of an instance of [SymplecticAutoencoder](@ref) is shown below: 

```@example
import Images, Plots # hide
if Main.output_type == :html # hide
  HTML("""<object type="image/svg+xml" class="display-light-only" data=$(joinpath(Main.buildpath, "../tikz/symplectic_autoencoder_architecture.png"))></object>""") # hide
else # hide
  Plots.plot(Images.load("../tikz/symplectic_autoencoder_architecture.png"), axis=([], false)) # hide
end # hide
```

```@example
if Main.output_type == :html # hide
  HTML("""<object type="image/svg+xml" class="display-dark-only" data=$(joinpath(Main.buildpath, "../tikz/symplectic_autoencoder_architecture_dark.png"))></object>""") # hide
end # hide
```

The *intermediate dimension* ``M`` is calculated via `n : (N - n) ÷ (n_blocks - 1) : N`. Further we have the following choices:
- `n_encoder_layers::Integer = 4`
- `n_encoder_blocks::Integer = 2` 
- `n_decoder_layers::Integer = 2` 
- `n_decoder_blocks::Integer = 3`
- `encoder_init_q::Bool = true`
- `decoder_init_q::Bool = true`

Note that all of these are keyword arguments that can be supplied to [SymplecticAutoencoder](@ref).