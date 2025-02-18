file = open("build/GeometricMachineLearning.jl.tex")

collection_of_lines = Tuple(line for line in eachline(file))
close(file)

new_contents = ""

for (i, line) in zip(1:length(collection_of_lines), collection_of_lines)
    if line == raw" \begin{figure}"
        global new_contents *= raw"\iffalse"
    elseif line == raw"\end{figure}"
            if collection_of_lines[i + 1] == ""
                global new_contents *= raw"\fi"
            else
                global new_contents *= raw"\end{figure}"
            end
    else
        global new_contents *= line
    end
    global new_contents *= "\n"
end

new_file = open("build/GeometricMachineLearning.jl-copy.tex", "a")
write(new_file, new_contents)
close(new_file)