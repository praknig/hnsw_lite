$markers = [ordered]@{
  '.github\workflows\ci.yml'      = '|health|upkeep|rebuild)'
  'CMakeLists.txt'                = 'src/rebuild.cpp'
  'README.md'                     = '`rebuild()` retraining compression'
  'docs\api-reference.md'         = '| `RebuildOptions`, `RebuildStats` |'
  'docs\architecture.md'          = '**Rebuilding** (`rebuild.cpp`)'
  'docs\feature-catalog.md'       = 'better removal repair, incremental `upkeep()`'
  'docs\getting-started.md'       = 'that entry''s share, not tests left out'
  'docs\project-notes.md'         = '- [x] Rebuilds while in use'
  'docs\testing.md'               = '**Found while building rebuilds:**'
  'docs\user-guide.md'            = '**Rebuilding** (`rebuild()`'
  'include\flat_index.h'          = 'RebuildStats rebuild('
  'include\health.h'              = 'struct RebuildOptions'
  'include\hnsw_index.h'          = 'UpkeepStats upkeep_unlocked('
  'src\flat_index.cpp'            = 'note_change(id);'
  'src\hnsw_index.cpp'            = 'void HnswIndex::maybe_auto_compact() noexcept'
  'src\rebuild.cpp'               = 'RebuildStats FlatIndex::rebuild('
  'src\upkeep.cpp'                = 'UpkeepStats HnswIndex::upkeep_unlocked('
  'tests\test_comprehensive.cpp'  = 'rb11_flat_cancellation_and_out_of_memory'
}
$hashes = [ordered]@{
  'tests\data\golden_hnsw_v3.bin'        = '0ABBBAB3029002F44093F5FB28D5555B6CC47B5FCF656D0245CBD4A7312BC41D'
  'tests\data\golden_hnsw_sq8_v3.bin'    = 'B5E406A1735710F8DE7307B2DE6CC805F781BBFCD44A296CAB5945BA1ED934DB'
  'tests\data\golden_hnsw_binary_v3.bin' = '28CEFAF062748CCDD30373646A93037E780D338F905EE3B3FE8E14B4C06410C7'
  'tests\data\golden_hnsw_pq_v3.bin'     = '50F70E5F357CCC71CFD44CED09CDE76F54DA15A4634BCA2C162D057B1ABA04D9'
}
$bad = 0
foreach ($f in $markers.Keys) {
  $s = if (-not (Test-Path $f)) { 'MISSING' }
       elseif (Select-String -Path $f -Pattern $markers[$f] -SimpleMatch -Quiet) { 'OK' }
       else { 'OLD' }
  if ($s -ne 'OK') { $bad++ }
  '{0,-8} {1}' -f $s, $f
}
foreach ($f in $hashes.Keys) {
  $s = if (-not (Test-Path $f)) { 'MISSING' }
       elseif ((Get-FileHash $f -Algorithm SHA256).Hash -eq $hashes[$f]) { 'OK' }
       else { 'DIFFERENT' }
  if ($s -ne 'OK') { $bad++ }
  '{0,-9} {1}' -f $s, $f
}
"$bad problem(s) in 22 files"