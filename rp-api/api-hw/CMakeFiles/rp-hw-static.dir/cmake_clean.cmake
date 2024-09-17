file(REMOVE_RECURSE
  "output/librp-hw.a"
  "output/librp-hw.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang C)
  include(CMakeFiles/rp-hw-static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
