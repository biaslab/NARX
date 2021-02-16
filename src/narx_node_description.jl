using ForneyLab
using LinearAlgebra
import ForneyLab: SoftFactor, @ensureVariables, generateId, addNode!, associate!,
                  averageEnergy, Interface, Variable, slug, ProbabilityDistribution,
                  differentialEntropy, unsafeLogMean, unsafeMean, unsafeCov, unsafePrecision, unsafeMeanCov,
                  collectAverageEnergyInbounds, localEdgeToRegion, ultimatePartner, region, Region, isClamped, currentInferenceAlgorithm
import SpecialFunctions: digamma
export NAutoregressiveX, NARX

"""
Description:

    A Nonlinear Autoregressive model with eXogeneous input (NARX)

    y_t = f(y_t-1, …, y_t-M, u_t, u_t-1, …, u_t-N) + e_t

    where M is the auto-regression order of observations and N of inputs.
    Concatenate previous observations: x_t-1 = [y_t-1, …, y_t-M]' and
    previous inputs z_t-1 = [u_t-1, …, u_t-N]'.

    Assume y_t, x_t-1, z_t-1 and u_t are observed and e_t ~ N(0, τ^-1).

    f is assumed to be a linear product of coefficients θ and a basis expansion
    of inputs and outputs ϕ: 
    
        f(...) = θ'*ϕ(y_t-1, …, y_t-n_y, u_t, …, u_t-n_u)

Interfaces:

    1. y (output)
    2. θ (function coefficients)
    3. x (previous observations vector)
    4. u (input)
    5. z (previous inputs vector)
    6. τ (precision)

Construction:

    NAutoregressiveX(y, θ, x, u, z, τ, g=ϕ, id=:some_id)
"""

mutable struct NAutoregressiveX <: SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    g::Function # Scalar function between autoregression coefficients and state variable

    function NAutoregressiveX(y, θ, x, u, z, τ; g::Function, id=generateId(NAutoregressiveX))
        @ensureVariables(y, θ, x, u, z, τ)
        self = new(id, Array{Interface}(undef, 6), Dict{Symbol,Interface}(), g)
        addNode!(currentGraph(), self)
        self.i[:y] = self.interfaces[1] = associate!(Interface(self), y)
        self.i[:θ] = self.interfaces[2] = associate!(Interface(self), θ)
        self.i[:x] = self.interfaces[3] = associate!(Interface(self), x)
        self.i[:u] = self.interfaces[4] = associate!(Interface(self), u)
        self.i[:z] = self.interfaces[5] = associate!(Interface(self), z)
        self.i[:τ] = self.interfaces[6] = associate!(Interface(self), τ)
        return self
    end
end

slug(::Type{NAutoregressiveX}) = "NARX"

function averageEnergy(::Type{NAutoregressiveX},
                       g::Function,
                       marg_y::ProbabilityDistribution{Univariate},
                       marg_θ::ProbabilityDistribution{Multivariate},
                       marg_x::ProbabilityDistribution{Multivariate},
                       marg_u::ProbabilityDistribution{Univariate},
                       marg_z::ProbabilityDistribution{Multivariate},
                       marg_τ::ProbabilityDistribution{Univariate})

    
    # Extract moments of beliefs
    mθ,Vθ = unsafeMeanCov(marg_θ)
    my = unsafeMean(marg_y)
    mx = unsafeMean(marg_x)
    mu = unsafeMean(marg_u)
    mz = unsafeMean(marg_z)
    mτ = unsafeMean(marg_τ)  
    aτ = marg_τ.params[:a]
    bτ = marg_τ.params[:b]

    # Gradient of f evaluated at mθ
	# Jθ = Zygote.gradient(g, mθ, mx, mu, mz)[1]
	Jθ = g([mx; mu; mz])

	# Evaluate f at mθ
	fθ = mθ'*Jθ
    
    return 1/2*log(2*pi) - 1/2*(digamma(aτ)-log(bτ)) + 1/2*mτ*(my^2 - 2*my*fθ + fθ^2 + Jθ'*Vθ*Jθ)
end

function collectAverageEnergyInbounds(node::NAutoregressiveX)
    inbounds = Any[]
    
    # Push function to calling signature (g needs to be defined in user scope)
	push!(inbounds, Dict{Symbol, Any}(:g => node.g, :keyword => false))
    
    local_edge_to_region = localEdgeToRegion(node)

    encountered_regions = Region[] # Keep track of encountered regions
    for node_interface in node.interfaces
        inbound_interface = ultimatePartner(node_interface)
        current_region = region(node_interface.node, node_interface.edge)

        if isClamped(inbound_interface)
            # Hard-code marginal of constant node in schedule
            push!(inbounds, assembleClamp!(copy(inbound_interface.node), ProbabilityDistribution)) # Copy Clamp before assembly to prevent overwriting dist_or_msg field
        elseif !(current_region in encountered_regions)
            # Collect marginal entry from marginal dictionary (if marginal entry is not already accepted)
            target = local_edge_to_region[node_interface.edge]
            current_inference_algorithm = currentInferenceAlgorithm()
            push!(inbounds, current_inference_algorithm.target_to_marginal_entry[target])
        end

        push!(encountered_regions, current_region)
    end

    return inbounds
end

