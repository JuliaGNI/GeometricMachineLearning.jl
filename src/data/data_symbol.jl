
abstract type AbstractDataSymbol end

abstract type PositionSymbol <: AbstractDataSymbol end

abstract type PhaseSpaceSymbol <: PositionSymbol end
abstract type DerivativePhaseSpaceSymbol <: PhaseSpaceSymbol end

abstract type PosVeloSymbol <: PositionSymbol end
abstract type PosVeloAccSymbol <: PosVeloSymbol end


struct DataSymbol{T <: AbstractDataSymbol} end

type(::DataSymbol{T}) where T<: AbstractDataSymbol = T


data_symbol(::DataSymbol{AbstractDataSymbol}) = nothing
data_symbol(::DataSymbol{PositionSymbol}) = (:q,)
data_symbol(::DataSymbol{PhaseSpaceSymbol}) = (:q,:p)
data_symbol(::DataSymbol{DerivativePhaseSpaceSymbol}) = (:q,:p,:q̇,:ṗ)
data_symbol(::DataSymbol{PosVeloSymbol}) = (:q,:q̇)
data_symbol(::DataSymbol{PosVeloAccSymbol}) = (:q,:q̇,:q̈)


function can_reduce(::DataSymbol{T}, s₂::DataSymbol{M}) where {T<:AbstractDataSymbol, M  <:AbstractDataSymbol}
    T <: M ? true : false
end

function reduce(s₁::DataSymbol{T}, s₂::DataSymbol{M}) where {T<:AbstractDataSymbol, M  <:AbstractDataSymbol}
    T <: M ? _setdiff(s₁,s₂) : @error "Impossible to reduce "*string(T)*" in "*string(M)*" !"
end

transform(::DataSymbol{T}, ::DataSymbol{U}) where {T<:AbstractDataSymbol,U<:AbstractDataSymbol} = @error "No method to convert "*string(T)*" in "*string(M)*" !"

can_transform(::DataSymbol{T}, ::DataSymbol{U}) where {T<:AbstractDataSymbol,U<:AbstractDataSymbol} = false


#=
   DataSymbol(keys; kwargs...) give a DataSymbol{T} where T is the smaler (in sens of <:) type of AbstractDataSymbol
   the symbols of which are includes in keys.
=#

function DataSymbol(keys::Tuple; symbol = DataSymbol{AbstractDataSymbol}())
    for subtype in subtypes(type(symbol))
        if data_symbol(DataSymbol{subtype}()) == keys
            return DataSymbol{subtype}
        elseif data_symbol(DataSymbol{subtype}()) ⊆ keys
            return DataSymbol(keys; symbol = DataSymbol{subtype}())
        end
    end
    return symbol
end


