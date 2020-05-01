print("gail-driver installer")
if VERSION != v"0.6.4"
  print("version is wrong. it must be 0.6.4")
  return
end

gail_root = ENV["GAIL_DRIVER_GITHUB"]

Pkg.clone(gail_root * "/Vec.jl")
Pkg.build("Vec.jl")
Pkg.clone("https://github.com/JuliaIO/VideoIO.jl")
Pkg.clone("https://github.com/JuliaGraphics/Cairo.jl")
Pkg.checkout("VideoIO.jl", "mindeps")
Pkg.checkout("Cairo.jl", "app")
Pkg.build("VideoIO.jl")
Pkg.build("Cairo.jl")
Pkg.clone(gail_root * "/AutomotiveDrivingModels.jl")
Pkg.build("AutomotiveDrivingModels.jl")
Pkg.clone(gail_root * "/AutoViz.jl")
Pkg.build("AutoViz.jl")
Pkg.clone(gail_root * "/ForwardNets.jl")
Pkg.build("ForwardNets.jl")
Pkg.clone(gail_root * "/NGSIM.jl")
Pkg.build("NGSIM.jl")
Pkg.clone(gail_root * "/Records.jl")
Pkg.build("Records.jl")
using Vec
using AutomotiveDrivingModels
using NGSIM
using AutoViz
using Records
using ForwardNets
