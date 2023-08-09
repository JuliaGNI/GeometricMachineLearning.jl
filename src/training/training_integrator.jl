#Define common strucutre Method
struct TrainingMethod{TIT<:AbstractTrainingMethod, TSymbol<:AbstractDataSymbol, TShape<:AbstractDataShape, TD} <:AbstractTrainingMethod
    sqdist::TD
end

@inline type(::TrainingMethod{T}) where T<: AbstractTrainingMethod = T
@inline symbols(::TrainingMethod{T,Symbols}) where {T<: AbstractTrainingMethod, Symbols<:AbstractDataSymbol} = Symbols
@inline shape(::TrainingMethod{T,Symbols, Shape}) where {T<: AbstractTrainingMethod, Symbols<:AbstractDataSymbol, Shape<:AbstractDataShape} = Shape


min_length_batch(ti::TrainingMethod) = min_length_batch(type(ti)())