# Call `f` on each leaf of the parameters `g` and on the leaf of the same name in each of `refs`.
function foreach_leaf(f, g, refs...)
    for name in keys(g)
        for k in keys(g[name])
            f(g[name][k], (r[name][k] for r in refs)...)
        end
    end
end
