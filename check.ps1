$m = [ordered]@{
  'src\hnsw_index.cpp'           = 'std::sort, not stable_sort'
  'include\hnsw_index.h'         = 'select_neighbors(NodeId base,'
  'src\upkeep.cpp'               = 'select_neighbors(node, options_l'
  'tests\test_comprehensive.cpp' = 'uk12_identical_vectors_are_linked_at_the_source'
  'README.md'                    = '793 tests'
  'docs\user-guide.md'           = 'after building an index from heavily duplicated data'
  'docs\testing.md'              = 'Found while linking identical vectors at the source'
  'docs\architecture.md'         = 'select_neighbors(base, candidates, max)'
  'docs\getting-started.md'      = '12 of 793 tests selected'
  'docs\project-notes.md'        = 'keep some of them unreachable until upkeep'
}
$h = [ordered]@{
  'tests\data\golden_hnsw_v3.bin'        = 'EF90E45D371AE5F5247E28F0E41335EF4A510AB1EDA9D624C4C686A3772754F1'
  'tests\data\golden_hnsw_sq8_v3.bin'    = '7C36A701DBE2F1B143DCBA69D9BF1312FFB2C1AF084FE10BDF1D21E85A38539E'
  'tests\data\golden_hnsw_binary_v3.bin' = 'D1A6848317568576A40262E72CDDFC887CB699038E4113289D399546461821F0'
  'tests\data\golden_hnsw_pq_v3.bin'     = 'DCD2DC74DDD96E5CA6A065B2723637E86422FE3A0682172C3F0957941F8CC8F3'
}
$bad = 0
foreach ($f in $m.Keys) { $s = if (-not (Test-Path $f)) {'MISSING'} elseif (Select-String -Path $f -Pattern $m[$f] -SimpleMatch -Quiet) {'OK'} else {'OLD'}; if ($s -ne 'OK') {$bad++}; '{0,-9} {1}' -f $s, $f }
foreach ($f in $h.Keys) { $s = if (-not (Test-Path $f)) {'MISSING'} elseif ((Get-FileHash $f -Algorithm SHA256).Hash -eq $h[$f]) {'OK'} else {'DIFFERENT'}; if ($s -ne 'OK') {$bad++}; '{0,-9} {1}' -f $s, $f }
"$bad problem(s) in 14 files"