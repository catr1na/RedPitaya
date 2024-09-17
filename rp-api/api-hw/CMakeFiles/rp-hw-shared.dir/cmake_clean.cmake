file(REMOVE_RECURSE
  "output/librp-hw.pdb"
  "output/librp-hw.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang C)
  include(CMakeFiles/rp-hw-shared.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
