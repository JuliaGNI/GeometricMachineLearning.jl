
abstract type AbstractArchitecture end

"""
The function `chain` returns a chain of a specific neural network architecture
and a specific backend and is called by:
```
chain(::AbstractArchitecture, ::AbstractBackend)
```
"""
function chain end


dim(arch::AbstractArchitecture) = @error "You forgot to implement dim for "*string(typeof(arch))*"."