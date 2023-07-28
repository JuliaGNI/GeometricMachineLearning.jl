function (sinput, M_1, M_2, M_3, M_4, M_5, M_6)
    (*)([0.0 1.0; -1.0 0.0], (Differential(sinput))((broadcast)(AbstractNeuralNetworks.IdentityActivation(), (*)(M_5, (broadcast)(tanh, (broadcast)(+, (*)(M_3, (broadcast)(tanh, (broadcast)(+, (*)(M_1, sinput), M_2))), M_4))))))
end