
abstract type AbstractNeuralNetwork end
abstract type AbstractBackend end

function apply! end
function train! end
function jacobian! end
