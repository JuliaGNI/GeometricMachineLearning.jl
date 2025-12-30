file = open("build/GeometricMachineLearning.jl.tex")

collection_of_lines = Tuple(line for line in eachline(file));
close(file)

new_contents = ""

for (i, line) in zip(1:length(collection_of_lines), collection_of_lines)
    global new_contents *= replace(line, raw"``{\textbackslash}mathcal\{M\}``" => raw"$\mathcal{M}$")
    global new_contents *= "\n"
end

new_file = open("build/GeometricMachineLearning.jl-copy.tex", "a")
write(new_file, new_contents)
close(new_file)