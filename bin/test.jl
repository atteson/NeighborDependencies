

function single_transitions!( neighbor_dependencies, single_transition, live, stationary_distribution )
    alphabet_size = size( neighbor_dependencies, 1 )
    alphabet = 1:alphabet_size
    
    for a in alphabet
        for b in alphabet
            single_transition[a, b] = 0.0
            denominator = 0.0
            for c in alphabet
                single_transition[a, b] += neighbor_dependencies[c, a, b] * stationary_distribution[live, c, a]
                denominator += stationary_distribution[live, c, a]
            end
            single_transition[a, b] /= denominator
        end
    end
    return single_transition
end

function calculate_single_transition( neighbor_dependencies, stationary_distribution )
    alphabet_size = size( neighbor_dependencies, 1 )
    single_transition = zeros( alphabet_size, alphabet_size )
    stationary_distribution = reshape( stationary_distribution, ( 1, alphabet_size, alphabet_size ) )
    single_transitions!( neighbor_dependencies, single_transition, 1, stationary_distribution )
    return single_transition
end

function stationary_distributions!( neighbor_dependencies, single_transition, live, stationary_distribution )
    alphabet_size = size( neighbor_dependencies, 1 )
    alphabet = 1:alphabet_size
    
    for a in alphabet
        for b in alphabet
            stationary_distribution[3-live, a, b] = 0.0
            for c in alphabet
                for d in alphabet
                    stationary_distribution[3-live, a, b] +=
                        stationary_distribution[live, c, d] * single_transition[c, a] * neighbor_dependencies[c, d, b]
                end
            end
        end
    end
    return stationary_distribution[3-live, :, :]
end

const alphabet_size = 2
const alphabet = 1:alphabet_size

neighbor_dependencies = zeros( alphabet_size, alphabet_size, alphabet_size )

neighbor_dependencies[1, 1, 2] = 0.01
neighbor_dependencies[2, 2, 1] = 0.02
neighbor_dependencies[1, 2, 1] = 0.1
neighbor_dependencies[2, 1, 2] = 0.15
for a in alphabet
    for b in alphabet
        neighbor_dependencies[a, b, b] = 1 - sum(neighbor_dependencies[a, b, :])
    end
end

stationary_distribution = zeros( alphabet_size, alphabet_size )
stationary_distribution[1, 1] = 0.35
stationary_distribution[1, 2] = 0.05
stationary_distribution[2, 1] = 0.1
stationary_distribution[2, 2] = 0.5

single_transition = zeros( alphabet_size, alphabet_size )

function contraction( neighbor_dependencies, single_transition, stationary_distribution )
    alphabet_size = size( neighbor_dependencies, 1 )
    scratch = zeros( 2, alphabet_size, alphabet_size )
    scratch[1, :, :] = stationary_distribution[:, :]
    live = 1
    diff = true
    while diff
        single_transitions!( neighbor_dependencies, single_transition, live, scratch )
        stationary_distributions!( neighbor_dependencies, single_transition, live, scratch )
        live = 3 - live
        diff = maximum(abs.(scratch[live, :, :] - scratch[3-live, :, :])) > 0
    end
    return (single_transition, scratch[live, :, :])
end

result = contraction( neighbor_dependencies, single_transition, stationary_distribution )
@time contraction( neighbor_dependencies, single_transition, stationary_distribution );
sum(result[1], dims=2)
sum(result[2])

# order of variables:
# LHS: p_0->0, p_0->1, p_1->0, p_1->1
# RHS: p00, p01, p10, p11
# I should really change this to use higher order tensors
index( alphabet_size, a, b ) = alphabet_size * (a - 1) + b

function f( neighbor_dependencies )
    alphabet_size = size( neighbor_dependencies, 1 )
    alphabet = 1:alphabet_size
    a2 = alphabet_size^2

    return (x; y = zeros( 2 * alphabet_size^2 )) -> begin
        for a in alphabet
            for b in alphabet
                numerator = 0.0
                denominator = 0.0
                for c in alphabet
                    j = a2 + index( alphabet_size, c, a )
                    numerator -= neighbor_dependencies[c, a, b] * x[j]
                    denominator += x[j]
                end
                i = index( alphabet_size, a, b )
                y[i] = x[i] + numerator/denominator
            end
        end
    
        for a in alphabet
            for b in alphabet
                i = a2 + index( alphabet_size, a, b )
                last = i == 2 * a2
                y[i] = last ? 1.0 : x[i]
                for c in alphabet
                    j = index( alphabet_size, c, a )
                    for d in alphabet
                        k = a2 + index( alphabet_size, c, d )
                        # if this is the last equation, replace it with unity sum
                        y[i] -= last ? x[k] : x[k] * x[j] * neighbor_dependencies[c, d, b]
                    end
                end
            end
        end

            
        return y
    end
end

function Df( neighbor_dependencies )
    alphabet_size = size( neighbor_dependencies, 1 )
    alphabet = 1:alphabet_size
    a2 = alphabet_size^2

    return (x; dy = zeros( 2 * a2, 2 * a2 )) -> begin
        for a in alphabet
            for b in alphabet
                numerator = 0.0
                denominator = 0.0
                for c in alphabet
                    j = a2 + index( alphabet_size, c, a )
                    numerator -= neighbor_dependencies[c, a, b] * x[j]
                    denominator += x[j]
                end
                i = index( alphabet_size, a, b )
                for c in alphabet
                    j = a2 + index( alphabet_size, c, a )
                    dy[i,j] = (-denominator * neighbor_dependencies[c, a, b] - numerator)/denominator^2
                end
                dy[i,i] += 1
            end
        end
    
        for a in alphabet
            for b in alphabet
                i = a2 + index( alphabet_size, a, b )
                last = i == 2 * a2
                dy[i,i] = last ? 0 : 1
                for c in alphabet
                    j = index( alphabet_size, c, a )
                    for d in alphabet
                        k = a2 + index( alphabet_size, c, d )
                        if last
                            dy[i,k] -= 1.0
                        else
                            dy[i,k] -= x[j] * neighbor_dependencies[c, d, b]
                            dy[i,j] -= x[k] * neighbor_dependencies[c, d, b]
                        end
                    end
                end
            end
        end
        return dy
    end
end

function Newton( neighbor_dependencies, x; epsilon = 1e-15 )
    g = f( neighbor_dependencies )
    dg = Df( neighbor_dependencies )

    delta = inv(dg(x))*g(x)
    while norm(delta) >= epsilon
        x = x - delta
        delta = inv(dg(x))*g(x)
    end
    return x
end

vst = vec(result[1]')
vsd = vec(result[2]')

single_transition = [0.9 0.1; 0.05 0.95]
x = [vec(single_transition'); vec(stationary_distribution')]
Newton( neighbor_dependencies, x )
@time Newton( neighbor_dependencies, x );

f( neighbor_dependencies )( [vst; vsd] )

g = f( neighbor_dependencies )
g( x )
dy = Df( neighbor_dependencies )( x )

using LinearAlgebra

a2 = alphabet_size^2
fidi = hcat((g.([x] .+ getindex.([1e-8 * I[1:2*a2,1:2*a2]], 1:2*a2, [:] )) .- [g(x)])./1e-8...)
@assert( maximum(abs.(dy - fidi)) .< 1e-6 )

using Utilities
using Distributions
using Random

Random.seed!( 1 )
s = rand( [1,2], 1_000_000 );
for i = 1:100
    s = rand.( Categorical.(deaccumulate( (x,y) -> getindex( neighbor_dependencies, x, y, : ), s )) );
end
mean(s .== 1)
sum(result[2], dims=1)
mean((s[1:end-1] .== 1) .& (s[2:end] .== 1))
mean((s[1:end-1] .== 2) .& (s[2:end] .== 2))
