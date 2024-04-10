# `SymmetricMatrix` and `SkewSymMatrix`

There are special implementations of symmetric and skew-symmetric matrices in `GeometricMachineLearning.jl`. They are implemented to work on GPU and for multiplication with tensors. The following image demonstrates how the data necessary for an instance of `SkewSymMatrix` are stored[^1]:

[^1]: It works similarly for `SymmetricMatrix`. 

```@example 
import Images, Plots # hide
if Main.output_type == :html # hide
    HTML("""<object type="image/svg+xml" class="display-light-only" data=$(joinpath(Main.buildpath, "../tikz/skew_sym_visualization.png"))></object>""") # hide
else # hide
    Plots.plot(Images.load("../tikz/skew_sym_visualization.png"), axis=([], false)) # hide
end # hide
```

```@example
if Main.output_type == :html # hide
    HTML("""<object type="image/svg+xml" class="display-dark-only" data=$(joinpath(Main.buildpath, "../tikz/skew_sym_visualization_dark.png"))></object>""") # hide
end # 
```

So what is stored internally is a vector of size ``n(n-1)/2`` for the skew-symmetric matrix and a vector of size ``n(n+1)/2`` for the symmetric matrix. We can sample a random skew-symmetric matrix: 

```@example skew_sym
using GeometricMachineLearning # hide 

A = rand(SkewSymMatrix, 5)
```

and then access the vector:

```@example skew_sym
A.S 
```