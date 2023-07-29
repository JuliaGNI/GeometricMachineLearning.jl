function (sinput, M_1, M_2, M_3, M_4, M_5, M_6)
    (broadcast)(AbstractNeuralNetworks.IdentityActivation(), (*)(M_5, (broadcast)(tanh, (broadcast)(+, (*)(M_3, (broadcast)(tanh, (broadcast)(+, (*)(M_1, sinput), M_2))), M_4))))
end