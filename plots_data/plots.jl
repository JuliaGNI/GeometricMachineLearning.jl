using Plots

#plot vector field
field_val = field(dat)
field_x = [field_val[i][1] for i in axes(dat,2)]
field_y = [field_val[i][2] for i in axes(dat,2)]
plt = quiver(dat[1,:],dat[2,:],quiver=(field_x,field_y))


#make on rk4 step
function rk4(y0, dt)
        dt2 = dt / 2.0
        k1 = field(y0)[1]
        k2 = field(y0 + dt2*k1)[1]
        k3 = field(y0 + dt2*k2)[1]
        k4 = field(y0 + dt   *k3)[1]
  dy = dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
  return dy
end


#initial value(s) -> plot a few trajectories to check e-conservation
y0 = [[1,0],[0,1.2],[.5,0],[0,.25],[0,1.5]]
#number of steps for rk4
steps = 2000
#one period is π long - so make 4 turns
dt = 4*π/steps

for k in y0
        global p = zeros(steps); global q = zeros(steps)
        p[1] = k[1]; q[1] = k[2]
        #make rk4 steps
        for i=2:steps
                p[i],q[i] = [p[i-1],q[i-1]] + rk4([p[i-1],q[i-1]],dt)
        end
        #closed curve indicates a conservative field
        display(plot!(p,q))
end


