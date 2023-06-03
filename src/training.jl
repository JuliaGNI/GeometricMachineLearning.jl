
# select a number of points at random (one batch)
function get_batchs(data, target, batch_size=10)
	index = rand(eachindex(data, target), batch_size)
	return (data[index], target[index])
end
