module project

using LinearAlgebra
using Distributions
using Mocking
using Random
using Plots
using LsqFit
using DelimitedFiles

export calculate_source_term,
    create_tridiagonal_matrix,
    create_V,
    initial_conditions_numerical,
    extract_mid_price,
    calculate_left_jump_probability,
    calculate_right_jump_probability,
    calculate_self_jump_probability,
    calculate_jump_probabilities,
    get_sub_period_time,
    simulate_intra_time_period,
    get_time_steps,
    sample_mid_price_path,
    dtrw_solver,
    run_sequential_limit_order_book_mid_price_model,
    plot_mid_price

function calculate_source_term(x::Float64,p::Float64, λ::Float64 = 1.0, μ::Float64  = 1.0)::Float64
    return λ*tanh(μ*(p-x))

    # return λ*(p - x)exp(-μ*(p-x)^2)
end

function create_tridiagonal_matrix(; Δx::Float64, diffusion_constant::Float64, M::Int64, cancellation_rate::Float64, V::Float64)
    upper_diagonal = (-V/(2.0*Δx) + diffusion_constant/(Δx^2)) * ones(Float64, M)
    middle_diagonal = ((-2.0*diffusion_constant)/(Δx^2) - cancellation_rate) * ones(Float64, M+1)
    lower_diagonal = (V/(2.0*Δx) + diffusion_constant/(Δx^2)) * ones(Float64, M)

    A = Tridiagonal(lower_diagonal, middle_diagonal, upper_diagonal)

    A[1,2] = 2*diffusion_constant/(Δx^2)
    A[end, end-1] = 2*diffusion_constant/(Δx^2)

    return A
end

function create_V(; σ::Float64, Δx::Float64, Δt::Float64)::Float64
    ϵ = rand(Normal(0.0, 1.0))
    V₀ = sign(ϵ) * min(abs(σ * ϵ), Δx / Δt)
    return V₀
end

function initial_conditions_numerical(;
    Δx::Float64,
    diffusion_constant::Float64,
    M::Int64,
    cancellation_rate::Float64,
    p₀::Float64,
    x::Vector{Float64},
    σ::Float64,
    Δt::Float64,
    use_V::Bool,
)
    if use_V
        V₀ = @mock create_V(σ=σ, Δx=Δx, Δt=Δt)
    else
        V₀ = 0.0
    end

    A = create_tridiagonal_matrix(Δx=Δx, diffusion_constant=diffusion_constant, M=M, cancellation_rate=cancellation_rate, V=V₀)
    source_term_matrix = .-[calculate_source_term(xᵢ, p₀) for xᵢ in x]

    initial_φ = A \ (source_term_matrix .+ 0.1)

    return initial_φ
end

function extract_mid_price(; x::Vector{Float64}, Δx::Float64, latent_order_book_density::Vector{Float64})::Float64
    mid_price_index = 2
    while (latent_order_book_density[mid_price_index] > 0) || isapprox(latent_order_book_density[mid_price_index],latent_order_book_density[mid_price_index-1], atol=0.001)
        mid_price_index += 1
    end
    φ₁ = latent_order_book_density[mid_price_index - 1]
    φ₂ = latent_order_book_density[mid_price_index]
    x₁ = x[mid_price_index - 1]

    mid_price = (-φ₁ * Δx)/(φ₂ - φ₁) + x₁
    return mid_price
end

function calculate_left_jump_probability(Z::Float64)::Float64
    return (exp(-Z))/(exp(Z) + exp(-Z) + 1.0)
end

function calculate_right_jump_probability(Z::Float64)::Float64
    return (exp(Z))/(exp(Z) + exp(-Z) + 1.0)
end

function calculate_self_jump_probability(Z::Float64)::Float64
    return (1.0)/(exp(Z) + exp(-Z) + 1.0)
end

function calculate_jump_probabilities(; Δx::Float64, diffusion_constant::Float64, V::Float64)::Tuple{Float64,Float64,Float64}
    Z = (3/4 * V * Δx) / diffusion_constant
    p⁻ = calculate_left_jump_probability(Z)
    p⁺ = calculate_right_jump_probability(Z)
    p = calculate_self_jump_probability(Z)

    return p⁺, p⁻, p
end

function get_sub_period_time(; α::Float64, Δt::Float64, t::Int64, time_steps::Int64)::Float64
    remaining_time = time_steps - t + 1

    if α <= 0.0
        return remaining_time
    end

    τ = rand(Exponential(α))
    τ_periods = min(floor(Int, τ/Δt), remaining_time)
    return τ_periods
end

function simulate_intra_time_period(;
    x::Vector{Float64},
    diffusion_constant::Float64,
    σ::Float64,
    Δx::Float64,
    Δt::Float64,
    cancellation_rate::Float64,
    φ::Array{Float64,1},
    p::Float64
)::Vector{Float64}
    Vₜ = @mock create_V(σ=σ, Δx=Δx, Δt=Δt)
    P⁺, P⁻, P = calculate_jump_probabilities(Δx=Δx, diffusion_constant=diffusion_constant, V=Vₜ)

    φ₋₁ = φ[1]
    φ₊₁ = φ[end]
    φ_next = zeros(Float64, size(φ,1))
    φ_next[1] = P⁺ * φ₋₁ + P⁻ * φ[2] + P * φ[1] - cancellation_rate * Δt * φ[1] + Δt * calculate_source_term(x[1], p)
    φ_next[end] = P⁻ * φ₊₁ + P⁺ * φ[end-1] + P * φ[end] - cancellation_rate * Δt * φ[end] + Δt * calculate_source_term(x[end], p)
    φ_next[2:end-1] = P⁺ * φ[1:end-2] + P⁻ * φ[3:end] + P * φ[2:end-1] - cancellation_rate * Δt * φ[2:end-1] + [Δt * calculate_source_term(xᵢ, p) for xᵢ in x[2:end-1]]

    return φ_next
end

function get_time_steps(T::Int64, Δt::Float64)::Int64
    time_steps =  T / Δt

    return floor(Int, time_steps + eps(time_steps))
end

function sample_mid_price_path(; T::Int64, Δt::Float64, price_path::Vector{Float64})::Vector{Float64}
    mid_price_indices = get_time_steps.(0:T, Δt) .+ 1
    mid_prices = price_path[mid_price_indices]

    return mid_prices
end

function dtrw_solver(;
    x::Vector{Float64},
    diffusion_constant::Float64,
    Δx::Float64,
    M::Int64,
    T::Int64,
    Δt::Float64,
    p₀::Float64,
    α::Float64,
    σ::Float64,
    cancellation_rate::Float64
)::Vector{Float64}
    time_steps = get_time_steps(T, Δt)
    p = fill(1.0, time_steps + 1)
    p[1] = p₀
    t = 1
    φ = initial_conditions_numerical(Δx=Δx, diffusion_constant=diffusion_constant, M=M, cancellation_rate=cancellation_rate, x=x, σ=σ, Δt=Δt, p₀=p[t], use_V=false)

    while t <= time_steps
        τ_periods = get_sub_period_time(α=α, Δt=Δt, t=t, time_steps=time_steps)

        for τₖ in 1:τ_periods
            t += 1
            φ = simulate_intra_time_period(x=x, diffusion_constant=diffusion_constant, σ=σ, Δx=Δx, Δt=Δt, cancellation_rate=cancellation_rate, φ=φ, p=p[t-1])
            p[t] = extract_mid_price(x=x, Δx=Δx, latent_order_book_density=φ)
        end

        if t > time_steps
            return sample_mid_price_path(T=T, Δt=Δt, price_path=p)
        end

        t += 1
        if α > 0.0
            φ = initial_conditions_numerical(Δx=Δx, diffusion_constant=diffusion_constant, M=M, cancellation_rate=cancellation_rate, p₀=p[t-1], x=x, σ=σ, Δt=Δt, use_V=true)
        end

        p[t] = extract_mid_price(x=x, Δx=Δx, latent_order_book_density=φ)
    end

    return sample_mid_price_path(T=T, Δt=Δt, price_path=p)
end

function run_sequential_limit_order_book_mid_price_model(;
    num_paths::Int64,
    T::Int64,
    p₀::Float64,
    M::Int64,
    L::Real,
    diffusion_constant::Float64,
    σ::Float64,
    cancellation_rate::Float64,
    α::Float64,
    # source_term::SourceTerm
    seed::Int = -1
)
    x₀ = p₀ - 0.5*L
    x_m = p₀ + 0.5*L
    @assert x₀ >= 0
    x = collect(Float64, range(x₀, stop=x_m, length=M+1))
    Δx = L/M
    Δt = (Δx^2) / (2.0*diffusion_constant)

    if seed == -1
        seeds = Int.(rand(MersenneTwister(), UInt32, num_paths))
    else
        seeds = Int.(rand(MersenneTwister(seed), UInt32, num_paths))
    end

    mid_price_paths = ones(Float64, T + 1, num_paths)
    for path in 1:num_paths
        Random.seed!(seeds[path])
        mid_price_paths[:, path] = dtrw_solver(x=x,diffusion_constant=diffusion_constant,Δx=Δx,M=M,T=T,Δt=Δt,p₀=p₀,α=α,σ=σ,cancellation_rate=cancellation_rate)
    end

    return mid_price_paths
end

function plot_mid_price()
    price_path = run_sequential_limit_order_book_mid_price_model(num_paths=1 ,T=2300, p₀=238.745, M=200, L=100.0, diffusion_constant=1.0, σ=0.1, cancellation_rate=0.1, α=10.0, seed=1)
    plot(1:size(price_path, 1), price_path, legend=false, xlab="Time", ylab="Mid-Price");
end

end # module project
