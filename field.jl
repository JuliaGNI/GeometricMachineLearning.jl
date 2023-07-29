function (ˍ₋out, sinput, M_1, M_2, M_3, M_4)
    begin
         @inbounds begin
                ˍ₋out[1] = (+)((*)((*)((+)(1, (*)(-1, (^)((tanh)((+)((+)((*)((getindex)(sinput, 1), (getindex)(M_1, 1, 1)), (*)((getindex)(sinput, 2), (getindex)(M_1, 1, 2))), (getindex)(M_2, 1))), 2))), (getindex)(M_1, 1, 1)), (getindex)(M_3, 1, 1)), (*)((*)((+)(1, (*)(-1, (^)((tanh)((+)((+)((*)((getindex)(sinput, 1), (getindex)(M_1, 2, 1)), (*)((getindex)(sinput, 2), (getindex)(M_1, 2, 2))), (getindex)(M_2, 2))), 2))), (getindex)(M_1, 2, 1)), (getindex)(M_3, 1, 2)))
                ˍ₋out[2] = (+)((*)((*)((+)(1, (*)(-1, (^)((tanh)((+)((+)((*)((getindex)(sinput, 1), (getindex)(M_1, 1, 1)), (*)((getindex)(sinput, 2), (getindex)(M_1, 1, 2))), (getindex)(M_2, 1))), 2))), (getindex)(M_1, 1, 2)), (getindex)(M_3, 1, 1)), (*)((*)((+)(1, (*)(-1, (^)((tanh)((+)((+)((*)((getindex)(sinput, 1), (getindex)(M_1, 2, 1)), (*)((getindex)(sinput, 2), (getindex)(M_1, 2, 2))), (getindex)(M_2, 2))), 2))), (getindex)(M_1, 2, 2)), (getindex)(M_3, 1, 2)))
                nothing
            end
    end
end