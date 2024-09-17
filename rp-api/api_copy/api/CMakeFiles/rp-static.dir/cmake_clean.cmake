file(REMOVE_RECURSE
  "output/librp.a"
  "output/librp.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang C)
  include(CMakeFiles/rp-static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
