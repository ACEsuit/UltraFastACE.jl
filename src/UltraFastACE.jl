module UltraFastACE

_i2z(obj, i::Integer) = obj._i2z[i] 

function _z2i(basis, Z)
   for i_Z = 1:length(obj._i2z)
      if basis._i2z[i_Z] == Z
         return i_Z
      end
   end
   error("_z2i : Z = $Z not found in obj._i2z")
   return -1 # never reached
end


include("zlms.jl")
include("ncorr.jl")
include("splines.jl")

include("convert_c2r.jl")

end
