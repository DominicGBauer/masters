using project
using Mocking
using Test

@testset "calculate_source_term" begin
    @test calculate_source_term(1.0,1.0) == 0
    @test calculate_source_term(1.0,0.0) ≈ -0.761 atol=0.001
    @test calculate_source_term(0.0,1.0) ≈ 0.761 atol=0.001
    @test calculate_source_term(0.5,1.0) ≈ 0.462 atol=0.001
end

@testset "create_tridiagonal_matrix" begin
    @test create_tridiagonal_matrix(Δx=1.0, diffusion_constant=1.0, M=1, cancellation_rate=1.0, V=1.0) == [-3.0 2.0; 2.0 -3.0]
    @test create_tridiagonal_matrix(Δx=1.0, diffusion_constant=1.0, M=2, cancellation_rate=1.0, V=1.0) == [-3.0 2.0 0.0; 1.5 -3.0 0.5; 0.0 2.0 -3.0]
    @test create_tridiagonal_matrix(Δx=1.0, diffusion_constant=1.0, M=3, cancellation_rate=1.0, V=1.0) == [-3.0 2.0 0.0 0.0; 1.5 -3.0 0.5 0.0; 0.0 1.5 -3.0 0.5; 0.0 0.0 2.0 -3.0]
end


@testset "create_V" begin
    @test create_V(σ=1.0, Δx=1.0, Δt=1.0) <= 1
    @test create_V(σ=1.0, Δx=1.0, Δt=1.0) >= -1
end

@testset "extract_mid_price" begin
    @test extract_mid_price(x=[2.0, 3.0, 3.0, 3.0, 3.0], Δx=1.0, latent_order_book_density=[2.0, 3.0, 1.0, 2.0, 1.0, 0.0, 0.0]) === 4.0
    @test extract_mid_price(x=[2.0, 3.0, 1.0, 1.0, 5.0], Δx=1.0, latent_order_book_density=[2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0]) === 6.0
    @test extract_mid_price(x=[2.0, 3.0, 1.0, 1.0, 2.0], Δx=1.0, latent_order_book_density=[4.0, 3.0, 2.0, 3.0, 2.0, 0.0, 0.0]) === 3.0
end

@testset "calculate_left_jump_probability" begin
    @test calculate_left_jump_probability(1.0) ≈ 0.090 atol=0.0001
end

@testset "calculate_right_jump_probability" begin
    @test calculate_right_jump_probability(1.0) ≈ 0.6652 atol=0.0001
end

@testset "calculate_self_jump_probability" begin
    @test calculate_self_jump_probability(1.0) ≈ 0.2447 atol=0.0001
end

@testset "calculate_jump_probabilities" begin
    @test calculate_jump_probabilities(Δx=0.5, diffusion_constant=1.0, V=2.0) === (0.5897976636568126, 0.13160164714691433, 0.278600689196273)
    @test sum(calculate_jump_probabilities(Δx=0.5, diffusion_constant=1.0, V=2.0)) <= 1
end

@testset "get_sub_period_time" begin
    @test get_sub_period_time(α=1.0, Δt=0.3, t=4, time_steps=3) === 0.0
    @test get_sub_period_time(α=0.0, Δt=0.3, t=4, time_steps=3) === 0.0
    @test get_sub_period_time(α=-1.0, Δt=0.3, t=4, time_steps=3) === 0.0
    @test get_sub_period_time(α=-1.0, Δt=0.3, t=10, time_steps=3) === -6.0
    @test get_sub_period_time(α=-1.0, Δt=0.3, t=2, time_steps=13) === 12.0
end

Mocking.activate()
mock_create_V = @patch create_V(; σ=1.0, Δx=0.2, Δt=1.0) = 0.5
@testset "initial_conditions_numerical" begin
    @test initial_conditions_numerical(Δx=1.0, diffusion_constant=1.0, M=1, cancellation_rate=1.0, p₀=1.0, x=[1.0, 1.0], σ=1.0, Δt=1.0, use_V=false) == [0.0, 0.0]
    @test initial_conditions_numerical(Δx=0.2, diffusion_constant=1.0, M=2, cancellation_rate=1.0, p₀=121.10, x=[1.0, 2.0, 3.0], σ=1.0, Δt=1.0, use_V=false) == [0.9999999999999984, 0.9999999999999984, 0.9999999999999984]
    @test initial_conditions_numerical(Δx=0.2, diffusion_constant=1.0, M=3, cancellation_rate=1.0, p₀=1.10, x=[1.0, 2.0, 3.0, 4.0], σ=1.0, Δt=1.0, use_V=false) == [-0.685037784594889, -0.7007319001792859, -0.7158033769628933, -0.7212574904999068]
    apply(mock_create_V) do
        @test initial_conditions_numerical(Δx=1.0, diffusion_constant=1.0, M=1, cancellation_rate=1.0, p₀=1.0, x=[1.0, 1.0], σ=1.0, Δt=1.0, use_V=true) == [0.0, 0.0]
        @test initial_conditions_numerical(Δx=0.2, diffusion_constant=1.0, M=2, cancellation_rate=1.0, p₀=121.10, x=[1.0, 2.0, 3.0], σ=1.0, Δt=1.0, use_V=true) == [0.9999999999999984, 0.9999999999999984, 0.9999999999999984]
        @test initial_conditions_numerical(Δx=0.2, diffusion_constant=1.0, M=3, cancellation_rate=1.0, p₀=1.10, x=[1.0, 2.0, 3.0, 4.0], σ=1.0, Δt=1.0, use_V=true) == [-0.6636503235684693, -0.6789166899323379, -0.6942160977969638, -0.700093491317623]
    end
end

@testset "simulate_intra_time_period" begin
    apply(mock_create_V) do
        @test simulate_intra_time_period(x=[1.0, 1.0, 1.0, 1.0], diffusion_constant=1.0, σ=1.0, Δx=0.5, Δt=0.3, cancellation_rate=1.0, φ=[1.0, 1.0, 1.0, 1.0], p=1.0) == [0.7, 0.7, 0.7, 0.7]
        @test simulate_intra_time_period(x=[1.0, 1.0, 1.0, 1.0], diffusion_constant=0.2, σ=1.0, Δx=0.2, Δt=0.4, cancellation_rate=121.10, φ=[1.0, 2.0, 3.0, 4.0], p=1.0) == [-47.22127693040518, -95.12431367237097, -142.564313672371, -190.22303674196579]
    end
end

@testset "get_time_steps" begin
    @test get_time_steps(10, 0.2) === 50
    @test get_time_steps(1, 0.2) === 5
    @test get_time_steps(20, 0.2) === 100
end

@testset "sample_mid_price_path" begin
    @test sample_mid_price_path(T=1, Δt=1.0, price_path=[122.0, 122.5, 124.7, 125.0]) == [122.0, 122.5]
    @test sample_mid_price_path(T=1, Δt=0.5, price_path=[122.0, 122.5, 124.7, 125.0]) == [122.0, 124.7]
    @test sample_mid_price_path(T=1, Δt=0.2, price_path=[122.0, 122.5, 124.7, 125.0, 122.0, 122.5, 124.7, 125.0]) == [122.0, 122.5]
end

@testset "dtrw_solver" begin
    apply(mock_create_V) do
        @test isapprox(
            dtrw_solver(x=[122.0, 122.5, 124.7, 125.0],diffusion_constant=1.0,Δx=0.5,M=3,T=10,Δt=0.2,p₀=120.0,α=0.0,σ=1.0,cancellation_rate=1.0),
            [120.0, -1201.9466453107539, -17925.54354040754, -241196.33950090283, -3.226685821355823e6, -4.313399552847608e7, -5.765444876238925e8, -7.706050044289734e9, -1.0299569731540987e11, -1.3772466141962102e12, -1.845737122639521e13],
            atol=0.01,
        )
    end
end
