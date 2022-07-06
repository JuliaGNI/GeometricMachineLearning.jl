
abstract type AbstractNeuralNetwork{DT <: Number} end

function apply end
function train end
function jacobian end
