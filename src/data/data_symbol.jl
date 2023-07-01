
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


function reduce(::DataSymbol{T}, s₂::DataSymbol{M}) where {T<:AbstractDataSymbol, M  <:AbstractDataSymbol}
    T <: M ? s₂ : @error "Impossible to reduce "*string(T)*" in "*string(M)*" !"
end


function DataSymbol(keys::NamedTuple; symbol = DataSymbol{AbstractDataSymbol}())
    for subtype in subtypes(type(symbol))
        if data_symbol(DataSymbol{subtype}()) == keys
            return DataSymbol{subtype}
        elseif data_symbol(DataSymbol{subtype}()) ⊆ keys
            return DataSymbol(keys; symbol = DataSymbol{subtype}())
        end
    end
    return symbol
end


