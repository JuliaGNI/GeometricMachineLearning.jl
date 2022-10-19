#build J matrices
function make_J(n)
	zer_n = zeros(n,n); id_n = I(n)
	J_n = hcat(vcat(zer_n,-id_n),vcat(id_n,zer_n))
	return J_n
end


