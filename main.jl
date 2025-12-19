using LinearAlgebra, Random
using JuMP, SCS
using OSQP, SparseArrays
using Plots
using DelimitedFiles

function hat(v::AbstractVector{<:Real})
    return [ 0.0   -v[3]   v[2];
             v[3]   0.0   -v[1];
            -v[2]   v[1]   0.0 ]
end

function vee(X::AbstractMatrix{<:Real})
    return [X[3,2], X[1,3], X[2,1]]
end

function expSO3(phi::AbstractVector{<:Real})
    phi = Vector{Float64}(phi)
    θ = norm(phi)
    Φ = hat(phi)
    I3 = Matrix{Float64}(I, 3, 3)
    if θ < 1e-10
        return I3 + Φ
    end
    A = sin(θ)/θ
    B = (1 - cos(θ)) / (θ^2)
    return I3 + A*Φ + B*(Φ*Φ)
end

function logSO3(R::AbstractMatrix{<:Real})
    R = Matrix{Float64}(R)
    c = (tr(R) - 1) / 2
    c = clamp(c, -1.0, 1.0)
    θ = acos(c)

    if θ < 1e-10
        return 0.5 * vee(R - R')
    end

    if abs(pi - θ) < 1e-6
        M = R - Matrix{Float64}(I, 3, 3)
        U, S, Vt = svd(M)
        axis = Vt'[:, end]
        na = norm(axis)
        if na < 1e-12
            s = vee(R - R')
            ns = norm(s)
            axis = (ns > 1e-12) ? (s / ns) : [1.0, 0.0, 0.0]
        else
            axis ./= na
        end

        s = vee(R - R')
        if dot(axis, s) < 0
            axis .*= -1
        end
        return θ * axis
    end

    Φ = (θ / (2*sin(θ))) * (R - R')
    return vee(Φ)
end

function so3_angle(R::AbstractMatrix{<:Real})
    R = Matrix{Float64}(R)
    c = (tr(R) - 1)/2
    c = clamp(c, -1.0, 1.0)
    return acos(c)
end

function random_rotation()
    M = randn(3,3)
    U, _, Vt = svd(M)
    R = U*Vt
    if det(R) < 0
        U[:,3] .*= -1
        R = U*Vt
    end
    return R
end

function project_to_SO3(M::AbstractMatrix{<:Real})
    U, _, Vt = svd(Matrix{Float64}(M))
    R = U * Vt
    if det(R) < 0
        U[:,3] .*= -1
        R = U * Vt
    end
    return R
end

function wahba_residual(Q::AbstractMatrix{<:Real},
                        X::AbstractMatrix{<:Real},
                        Y::AbstractMatrix{<:Real})
    Q = Matrix{Float64}(Q)
    X = Matrix{Float64}(X)
    Y = Matrix{Float64}(Y)
    N = size(X,2)
    s = 0.0
    for k in 1:N
        r = Y[:,k] - Q' * X[:,k]
        s += dot(r, r)
    end
    return s
end

function Q_of_Z(Z::AbstractMatrix{<:Real})
    Z = Matrix{Float64}(Z)
    Z11, Z22, Z33, Z44 = Z[1,1], Z[2,2], Z[3,3], Z[4,4]
    Z12, Z13, Z14      = Z[1,2], Z[1,3], Z[1,4]
    Z23, Z24, Z34      = Z[2,3], Z[2,4], Z[3,4]

    Q11 =  Z11 - Z22 - Z33 + Z44
    Q12 = -2*(Z13 + Z24)
    Q13 = -2*Z12 + 2*Z34
    Q21 =  2*(Z13 - Z24)
    Q22 =  Z11 + Z22 - Z33 - Z44
    Q23 = -2*(Z14 + Z23)
    Q31 =  2*(Z12 + Z34)
    Q32 =  2*(Z14 - Z23)
    Q33 =  Z11 - Z22 + Z33 - Z44

    return [Q11 Q12 Q13;
            Q21 Q22 Q23;
            Q31 Q32 Q33]
end

function K_from_B(B::Matrix{Float64})
    b11,b12,b13 = B[1,1], B[1,2], B[1,3]
    b21,b22,b23 = B[2,1], B[2,2], B[2,3]
    b31,b32,b33 = B[3,1], B[3,2], B[3,3]

    K = zeros(4,4)
    K[1,1] =  b11 + b22 + b33
    K[2,2] = -b11 + b22 - b33
    K[3,3] = -b11 - b22 + b33
    K[4,4] =  b11 - b22 - b33

    K[1,2] =  (b31 - b13);  K[2,1] = K[1,2]
    K[1,3] =  (b21 - b12);  K[3,1] = K[1,3]
    K[1,4] =  (b32 - b23);  K[4,1] = K[1,4]
    K[2,3] = -(b23 + b32);  K[3,2] = K[2,3]
    K[2,4] = -(b12 + b21);  K[4,2] = K[2,4]
    K[3,4] =  (b13 + b31);  K[4,3] = K[3,4]

    return Symmetric(K)
end

function wahba_sdp_Z(B::Matrix{Float64}; verbose::Bool=false)
    model = Model(optimizer_with_attributes(
        SCS.Optimizer,
        "verbose"   => (verbose ? 1 : 0),
        "eps_abs"   => 1e-8,
        "eps_rel"   => 1e-8,
        "max_iters" => 200_000
    ))
    @variable(model, Z[1:4,1:4], PSD)
    @constraint(model, tr(Z) == 1)

    K = K_from_B(B)
    @objective(model, Max, sum(K[i,j] * Z[i,j] for i in 1:4, j in 1:4))

    optimize!(model)
    status = termination_status(model)

    Zhat = value.(Z)
    Zhat = 0.5*(Zhat + Zhat')
    return Zhat, status
end

function recover_Q_via_K(B::Matrix{Float64},
                         X::Matrix{Float64},
                         Y::Matrix{Float64})
    K = K_from_B(B)
    evals, evecs = eigen(K)
    q = evecs[:, argmax(evals)]
    q ./= norm(q)

    Z1 = q*q'
    Qraw = Q_of_Z(Z1)
    Qcand = project_to_SO3(Qraw)

    J1 = wahba_residual(Qcand, X, Y)
    J2 = wahba_residual(Qcand', X, Y)
    return (J1 <= J2) ? (Qcand, J1) : (Qcand', J2)
end

function blockdiag(mats::AbstractMatrix...)
    matsM = [Matrix{Float64}(M) for M in mats]
    n = sum(size(M,1) for M in matsM)
    m = sum(size(M,2) for M in matsM)
    out = zeros(n,m)
    i = 1; j = 1
    for M in matsM
        r,c = size(M)
        out[i:i+r-1, j:j+c-1] .= M
        i += r; j += c
    end
    return out
end

function mpc_qp_osqp(A::AbstractMatrix, Bd::AbstractMatrix,
                     Qx::AbstractMatrix, Ru::AbstractMatrix, Qf::AbstractMatrix,
                     x0::AbstractVector,
                     H::Int, umax::Real)

    A  = Matrix{Float64}(A)
    Bd = Matrix{Float64}(Bd)
    Qx = Matrix{Float64}(Qx)
    Ru = Matrix{Float64}(Ru)
    Qf = Matrix{Float64}(Qf)
    x0 = Vector{Float64}(x0)
    umax = Float64(umax)

    nx = size(A,1)
    nu = size(Bd,2)

    Sx = zeros(nx*(H+1), nx)
    Su = zeros(nx*(H+1), nu*H)
    Sx[1:nx, :] .= Matrix{Float64}(I, nx, nx)

    for k in 1:H
        Sx[k*nx+1:(k+1)*nx, :] .= A^k
        for j in 0:k-1
            Su[k*nx+1:(k+1)*nx, j*nu+1:(j+1)*nu] .= (A^(k-1-j)) * Bd
        end
    end

    Qbar = blockdiag((Qx for _ in 1:H)..., Qf)
    Rbar = blockdiag((Ru for _ in 1:H)...)

    xTbar = zeros(nx*(H+1))

    P = 2*(Su' * Qbar * Su + Rbar)
    P = 0.5*(P + P')
    q = 2*(Su' * Qbar * (Sx*x0 - xTbar))

    nU = nu*H
    l = -umax * ones(nU)
    u =  umax * ones(nU)

    model = OSQP.Model()
    OSQP.setup!(model; P=sparse(P), q=q,
                A=sparse(1:nU, 1:nU, ones(nU), nU, nU),
                l=l, u=u, verbose=false)
    res = OSQP.solve!(model)

    status_str = lowercase(String(res.info.status))
    Uopt = occursin("solved", status_str) ? res.x : zeros(nU)

    return Uopt[1:nu], Uopt
end

function write_csv_with_header(fname::String, header::String, data::AbstractMatrix)
    open(fname, "w") do io
        write(io, header * "\n")
        writedlm(io, data, ',')
    end
end

function main()
    Random.seed!(2)

    Nvec  = 12
    sigma = 0.02

    Qtrue = random_rotation()

    X = randn(3, Nvec)
    X = X ./ reshape([norm(c) for c in eachcol(X)], 1, :)

    Y = zeros(3, Nvec)
    for k in 1:Nvec
        yk = Qtrue' * X[:,k] + sigma*randn(3)
        yk /= norm(yk)
        Y[:,k] = yk
    end

    Bdata = X * Y'

    println("Sanity: residual at Qtrue = ", wahba_residual(Qtrue, X, Y))

    Zhat, status = wahba_sdp_Z(Bdata; verbose=false)
    println("SDP status: ", status)
    evZ = sort(eigvals(Symmetric(Zhat)); rev=true)
    println("Top eigenvalues of Zhat: ", evZ[1:4])

    Qhat, Jhat = recover_Q_via_K(Bdata, X, Y)
    println("Wahba residual (K-eig recover) = ", Jhat)
    println("Estimation angle error (rad): ", so3_angle(Qhat * Qtrue'))
    println("det(Qhat) = ", det(Qhat), "   ||Qhat'Qhat - I|| = ",
            norm(Qhat' * Qhat - Matrix{Float64}(I,3,3)))

    Ibody = Diagonal([0.08, 0.10, 0.12])
    Iinv  = inv(Matrix(Ibody))

    dt   = 0.02
    Tf   = 4.0
    Tsim = Int(round(Tf/dt))
    H    = 25
    umax = 0.15

    Qx = Matrix(Diagonal([10,10,10, 0.5,0.5,0.5]))
    Ru = 0.05 * Matrix{Float64}(I, 3, 3)
    Qf = 20 * Matrix(Diagonal([10,10,10, 1,1,1]))

    theta0 = [0.25, -0.15, 0.18]
    omega0 = [0.0, 0.0, 0.0]

    Q = expSO3(theta0)
    ω = copy(omega0)

    Qtgt = Qhat

    theta_err_hist = zeros(Tsim+1)
    geo_err_hist   = zeros(Tsim+1)
    omega_mag      = zeros(Tsim+1)
    u_mag          = zeros(Tsim)

    I3 = Matrix{Float64}(I, 3, 3)
    Z3 = zeros(3,3)
    A  = [I3  dt*I3;
          Z3  I3]
    Bd = [Z3;
          dt*Iinv]

    Qe0 = Qtgt' * Q
    theta_err_hist[1] = norm(logSO3(Qe0))
    geo_err_hist[1]   = so3_angle(Qe0)
    omega_mag[1]      = norm(ω)

    for t in 1:Tsim
        Qe = Qtgt' * Q
        θ  = logSO3(Qe)
        x0 = vcat(θ, ω)

        u0, _ = mpc_qp_osqp(A, Bd, Qx, Ru, Qf, x0, H, umax)
        u = u0
        u_mag[t] = norm(u)

        ω = ω + dt * (Iinv * u)
        Q = Q * expSO3(dt * ω)

        Qe_next = Qtgt' * Q
        theta_err_hist[t+1] = norm(logSO3(Qe_next))
        geo_err_hist[t+1]   = so3_angle(Qe_next)
        omega_mag[t+1]      = norm(ω)
    end

    time   = (0:Tsim) .* dt
    time_u = (0:Tsim-1) .* dt

    p1 = plot(time, theta_err_hist, lw=2, grid=true,
              xlabel="Time (s)", ylabel="‖log(QtgtᵀQ)‖ (rad)",
              title="Lie-algebra intrinsic error magnitude")
    p2 = plot(time, omega_mag, lw=2, grid=true,
              xlabel="Time (s)", ylabel="‖ω‖ (rad/s)",
              title="Angular velocity magnitude")
    p3 = plot(time_u, u_mag, lw=2, grid=true,
              xlabel="Time (s)", ylabel="‖u‖ (N·m)",
              title="Control effort magnitude")
    p4 = plot(time, geo_err_hist, lw=2, grid=true,
              xlabel="Time (s)", ylabel="Angle error (rad)",
              title="Geodesic error angle angle(QtgtᵀQ)")
    p5 = scatter(1:length(evZ), evZ, grid=true,
                 xlabel="Index", ylabel="Eigenvalue",
                 title="Eigenvalues of SDP lift Z")

    savefig(p1, "theta_err_mag.png")
    savefig(p2, "omega_mag.png")
    savefig(p3, "u_mag.png")
    savefig(p4, "attitude_error.png")
    savefig(p5, "Z_eigs.png")
    display(p1); display(p2); display(p3); display(p4); display(p5)

    traj = [time  theta_err_hist  geo_err_hist  omega_mag]
    write_csv_with_header("traj.csv",
        "time,theta_err_norm,geo_angle_err,omega_norm",
        traj)

    ulog = [time_u  u_mag]
    write_csv_with_header("u.csv",
        "time,u_norm",
        ulog)

    Ze = [collect(1:length(evZ))  evZ]
    write_csv_with_header("Z_eigs.csv",
        "idx,eig",
        Ze)

    println("Saved: traj.csv, u.csv, Z_eigs.csv, and PNG plots.")
end

main()
