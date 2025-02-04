

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

function tensors( neighbor_dependencies )
    alphabet_size = size( neighbor_dependencies, 1 )
    alphabet = 1:alphabet_size

    as2 = alphabet_size^2

    A = zeros( as2, as2, 2*as2 )
    b = zeros( 2*as2, 2*as2 )
    
    single_transition_A = view( A, :, :, 1:as2 )
    single_transition_b = view( b, :, : )
    
    for a in alphabet
        for b in alphabet
            # p_a->b equation
            i = index( alphabet_size, a, b )
            for c in alphabet
                # p_ca
                j = index( alphabet_size, c, a )

                # coefficient for p_a->b * p_ca
                single_transition_A[i, j, i] = 1.0
                # coefficient for p_ca
               single_transition_b[i, as2 + j] += - neighbor_dependencies[c, a, b]
            end
        end
    end

    stationary_distribution_indices = as2 .+ (1:as2)
    stationary_distribution_A = view( A, :, :, stationary_distribution_indices )
    stationary_distribution_b = view( b, stationary_distribution_indices, stationary_distribution_indices )
    for a in alphabet
        for b in alphabet
            # p_ab equation
            i = index( alphabet_size, a, b )
            stationary_distribution_b[i, i] = 1.0
            for c in alphabet
                # p_c->a coefficients
                j = index( alphabet_size, c, a )
                for d in alphabet
                    # p_cd
                    k = index( alphabet_size, c, d )
                    # coefficient for p_c->A * p_cd
                    stationary_distribution_A[j, k, i] = - neighbor_dependencies[c, d, b]
                end
            end
        end
    end

    return (A, b)
end

(A, b) = tensors( neighbor_dependencies )

function f( A, b )
    a2 = size( A, 1 )
    return x -> vec(sum(A .* x[1:a2] .* reshape( x[a2+1:2*a2], (1, a2) ), dims=(1,2))) + b * x
end

vst = vec(result[1]')
vsd = vec(result[2]')

vec(sum(A .* vst .* reshape( vsd, (1, alphabet_size.^2) ), dims=(1,2) )) + b * [vst; vsd]
f( A, b )( [vst;vsd] )

vst' * A[:,:,1] * vsd

dot( b[1,:], [vst; vsd] )

single_transition = [0.9 0.1; 0.05 0.95]

a2 = size( A, 1 )
g = f( A, b )
x = [vec(single_transition'); vec(stationary_distribution')] 
g( x )

maximum(abs.(hcat((g.([x] .+ getindex.([1e-8 * I[1:2*a2,1:2*a2]], 1:2*a2, [:] )) .- [g(x)])./1e-8...) - D))

function Newton( A, b, x; epsilon=1e-15 )
    i = 0
    delta = fill( Inf, length(x) )
    epsilon = 1e-15
    while maximum(abs.(delta)) >= epsilon
        hi = reshape( sum(A .* reshape( x[a2.+(1:a2)], (1, a2 ) ), dims=2 ), (a2, 2*a2) )
        lo = reshape( sum(A .* x[1:a2], dims=1 ), (a2, 2*a2) )
        D = [hi; lo]' + b
        delta = pinv(D)*g(x)
        x = x - delta
        i += 1
    end
    return (x, i)
end

@time Newton( A, b, x )
@time Newton( A, b, x );

y = vcat(vec.(transpose.(result))...)
hcat((g.([y] .+ getindex.([1e-8 * I[1:2*a2,1:2*a2]], 1:2*a2, [:] )) .- [g(y)])./1e-8...)
hcat((g.([x] .+ getindex.([1e-8 * I[1:2*a2,1:2*a2]], 1:2*a2, [:] )) .- [g(x)])./1e-8...)

g(x)
g(y)
