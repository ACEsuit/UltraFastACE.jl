
function ACEbase.evaluate(ace::UFACE, Rs, Zs, zi) 
   i_zi = _z2i(ace, zi)
   ace_inner = ace.ace_inner[i_zi]
   Ei = ( evaluate(ace_inner, Rs, Zs) + 
          evaluate(ace.pairpot, Rs, Zs, zi) + 
          ace.E0s[zi] )
end

# --------------------------------------------------------
# UF_ACE evaluation code. 

function ACEbase.evaluate(ace::UFACE_inner, Rs, Zs)
   TF = eltype(eltype(Rs))
   rbasis = ace.rbasis 
   NZ = length(rbasis._i2z)
  
   @no_escape begin 
   
   # embeddings    
   # element embedding 
   Ez = @withalloc embed_z!(ace, Rs, Zs)
   # radial embedding 
   Rn = @withalloc evaluate!(rbasis, Rs, Zs)
   # angular embedding 
   Zlm = @withalloc evaluate_ylm!(ace, Rs)
   
   # pooling 
   A = ace.abasis((Ez, Rn, Zlm))

   # n correlations
   φ = ace.aadot(A)

   # release the borrowed arrays 
   release!(A)

   end

   return φ
end

function evaluate_ed(ace::UFACE, Rs, Zs, z0)
   ∇φ = acquire!(ace.pool, :out_dEs, (length(Rs),), eltype(Rs))
   return evaluate_ed!(∇φ, ace, Rs, Zs, z0)
end

function evaluate_ed!(∇φ, ace::UFACE, Rs, Zs, z0)
   i_z0 = _z2i(ace, z0)
   ace_inner = ace.ace_inner[i_z0]
   φ, _ = evaluate_ed!(∇φ, ace_inner, Rs, Zs)
   add_evaluate_d!(∇φ, ace.pairpot, Rs, Zs, z0)
   φ += ace.E0s[z0] + evaluate(ace.pairpot, Rs, Zs, z0)
   return φ, ∇φ
end


function ACEbase.evaluate_ed!(∇φ, ace::UFACE_inner, Rs, Zs)
   TF = eltype(eltype(Rs))
   rbasis = ace.rbasis 
   NZ = length(rbasis._i2z)

   # embeddings    
   # element embedding  (there is no gradient)
   Ez = embed_z(ace, Rs, Zs)

   # radial embedding 
   Rn, dRn = evaluate_ed(ace, rbasis, Rs, Zs)

   # angular embedding 
   Zlm, dZlm = evaluate_ylm_ed(ace, Rs)

   # pooling 
   A = ace.abasis((Ez, Rn, Zlm))

   # n correlations - compute with gradient, do it in-place 
   ∂φ_∂A = acquire!(ace.pool, :∂A, size(A), TF)
   φ = evaluate_and_gradient!(∂φ_∂A, ace.aadot, A)
   
   # backprop through A  =>  this part could be done more nicely I think
   ∂φ_∂Ez = BlackHole(TF) 
   # ∂φ_∂Ez = zeros(TF, size(Ez))
   ∂φ_∂Rn = acquire!(ace.pool, :∂Rn, size(Rn), TF)
   ∂φ_∂Zlm = acquire!(ace.pool, :∂Zlm, size(Zlm), TF)
   fill!(∂φ_∂Rn, zero(TF))
   fill!(∂φ_∂Zlm, zero(TF))
   P4ML._pullback_evaluate!((∂φ_∂Ez, unwrap(∂φ_∂Rn), unwrap(∂φ_∂Zlm)), 
                             unwrap(∂φ_∂A), 
                             ace.abasis, 
                             (unwrap(Ez), unwrap(Rn), unwrap(Zlm)); 
                             sizecheck=false)

   # backprop through the embeddings 
   # depending on whether there is a bottleneck here, this can be 
   # potentially implemented more efficiently without needing writing/reading 
   # (to be investigated where the bottleneck is)
   
   # we just ignore Ez (hence the black hole)

   # backprop through Rn 
   # We already computed the gradients in the forward pass
   fill!(∇φ, zero(SVector{3, TF}))
   @inbounds for n = 1:size(Rn, 2)
      @simd ivdep for j = 1:length(Rs)
         ∇φ[j] += ∂φ_∂Rn[j, n] * dRn[j, n]
      end
   end

   # ... and Ylm 
   @inbounds for i_lm = 1:size(Zlm, 2)
      @simd ivdep for j = 1:length(Rs)
         ∇φ[j] += ∂φ_∂Zlm[j, i_lm] * dZlm[j, i_lm]
      end
   end

   # release the borrowed arrays 
   release!(Zlm); release!(dZlm)
   release!(Rn); release!(dRn)
   release!(Ez)
   release!(A)
   release!(∂φ_∂Rn); release!(∂φ_∂Zlm); release!(∂φ_∂A)

   return φ, ∇φ 
end
