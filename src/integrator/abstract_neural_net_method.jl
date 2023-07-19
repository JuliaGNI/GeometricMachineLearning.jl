abstract type NeuralNetMethod <: GeometricMethod end

method(nns::NeuralNetSolution) = throw(ArgumentError("No intgertator method associated to "*string(typeof(nn(nns)))))
