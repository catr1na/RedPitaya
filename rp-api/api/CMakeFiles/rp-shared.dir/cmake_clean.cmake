file(REMOVE_RECURSE
  "output/librp.pdb"
  "output/librp.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang C)
  include(CMakeFiles/rp-shared.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
