using UltraFastACE
using Documenter

DocMeta.setdocmeta!(UltraFastACE, :DocTestSetup, :(using UltraFastACE); recursive=true)

makedocs(;
    modules=[UltraFastACE],
    authors="Christoph Ortner <christohortner@gmail.com> and contributors",
    repo="https://github.com/ACEsuit/UltraFastACE.jl/blob/{commit}{path}#{line}",
    sitename="UltraFastACE.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ACEsuit.github.io/UltraFastACE.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ACEsuit/UltraFastACE.jl",
    devbranch="main",
)
