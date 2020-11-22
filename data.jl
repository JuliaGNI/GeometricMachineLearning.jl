using Zygote

####get data set

function get_data_set(num=10)
	#define Hamiltonian
	H(y) = .5 * sum(y[1,:].^2 + y[2,:].^2)
	
	#compute vector field  
	dH(y) = [0 1;-1 0] * gradient(χ -> H(χ),y)[1]
	
	
	#range in which the data should be in
	rang = range(-1.2,stop=1.2,length=num)
	xy = [[x,y] for x in rang, y in rang]
	x = reshape([point[1] for point in xy],num^2)
	y = reshape([point[2] for point in xy],num^2)
	dat = hcat(x,y)'
	#compute the value of the vector field 
	target = dH(dat)
	return((dat,target))
end
