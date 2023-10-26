

struct UFACE_inner{TR, TY, TA, TAA}
   rbasis::TR
   ybasis::TY
   abasis::TA
   aadot::TAA
end

struct UFACE{N, TR, TY, TA, TAA}
   _i2z::NTuple{N, Int}
   ace_inner::NTuple{N, UFACE_inner{TR, TY, TA, TAA}}
end


function evaluate(ace::UFACE, Rs, Zs, zi) 
   i_zi = _z2i(ace, zi)
   ace_inner = ace.ace_inner[i_zi]
   return evaluate(ace_inner, Rs, Zs)
end


function evaluate(ace::UFACE_inner, Rs, Zs)
   

end