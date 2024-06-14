# taken from https://docs.makie.org/stable/how-to/save-figure-with-transparency
function calculate_rgba(rgb1, rgb2, rgba_bg)::RGBAf
    rgb1 == rgb2 && return RGBAf(GLMakie.red(rgb1), GLMakie.green(rgb1), GLMakie.blue(rgb1), 1)
    c1 = Float64.((GLMakie.red(rgb1), GLMakie.green(rgb1), GLMakie.blue(rgb1)))
    c2 = Float64.((GLMakie.red(rgb2), GLMakie.green(rgb2), GLMakie.blue(rgb2)))
    alphas_fg = 1 .+ c1 .- c2
    alpha_fg = clamp(sum(alphas_fg) / 3, 0, 1)
    alpha_fg == 0 && return rgba_bg
    rgb_fg = clamp.((c1 ./ alpha_fg), 0, 1)
    rgb_bg = Float64.((rgba_bg.r, rgba_bg.g, rgba_bg.b))
    alpha_final = alpha_fg + (1 - alpha_fg) * rgba_bg.alpha
    rgb_final = @. 1 / alpha_final * (alpha_fg * rgb_fg + (1 - alpha_fg) * rgba_bg.alpha * rgb_bg)
    return RGBAf(rgb_final..., alpha_final)
end

function alpha_colorbuffer(figure)
    scene = figure.scene
    bg = scene.backgroundcolor[]
    scene.backgroundcolor[] = RGBAf(0, 0, 0, 1)
    b1 = copy(colorbuffer(scene))
    scene.backgroundcolor[] = RGBAf(1, 1, 1, 1)
    b2 = colorbuffer(scene)
    scene.backgroundcolor[] = bg
    return map(b1, b2) do b1, b2
        calculate_rgba(b1, b2, bg)
    end
end
