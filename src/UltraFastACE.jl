module UltraFastACE

import ACEbase 
import ACEbase: evaluate, evaluate!, 
                evaluate_ed, evaluate_ed!

_i2z(obj, i::Integer) = obj._i2z[i] 

function _z2i(obj, Z)
   for i_Z = 1:length(obj._i2z)
      if obj._i2z[i_Z] == Z
         return i_Z
      end
   end
   error("_z2i : Z = $Z not found in obj._i2z")
   return -1 # never reached
end


# include("zlms.jl")
include("ncorr.jl")
include("splines.jl")

include("convert_c2r.jl")

include("auxiliary.jl")

include("uface.jl")

end
