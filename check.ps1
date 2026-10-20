$new = 'include\wal.h','include\wal_ops.h','include\durable.h','src\wal.cpp','src\wal_ops.cpp','src\wal_replay.cpp','src\durable.cpp','bench\bench_durability.cpp'
$m = [ordered]@{
  '.github\workflows\ci.yml'='|wal|durable)'; 'CMakeLists.txt'='src/durable.cpp'; 'README.md'='**Durability:** a write-ahead log'
  'docs\api-reference.md'='| `RecoveryStats` |'; 'docs\architecture.md'='**Durability** (`wal.cpp`'; 'docs\feature-catalog.md'='three modes'
  'docs\getting-started.md'='825 tests'; 'docs\performance.md'='### Durability'; 'docs\project-notes.md'='- [x] Durability: a write-ahead log'
  'docs\testing.md'='Found while building durability'; 'docs\user-guide.md'='## Durability'
  'include\flat_index.h'='open_durable('; 'include\hnsw_index.h'='struct RngGuard {'; 'src\flat_index.cpp'='note_change(id);  // under the lock'
  'src\hnsw_index.cpp'='const RngGuard rng_guard(rng_);'; 'src\persistence.cpp'='HnswIndex::save_unlocked('; 'src\rebuild.cpp'='walop::kRebuild'
  'src\upkeep.cpp'='walop::kUpkeep'; 'tests\test_comprehensive.cpp'='du14_real_process_kills'
}
$bad = 0
foreach ($f in $new) { $s = if (Test-Path $f) {'OK'} else {'MISSING'}; if ($s -ne 'OK') {$bad++}; '{0,-8} {1}' -f $s, $f }
foreach ($f in $m.Keys) { $s = if (-not (Test-Path $f)) {'MISSING'} elseif (Select-String -Path $f -Pattern $m[$f] -SimpleMatch -Quiet) {'OK'} else {'OLD'}; if ($s -ne 'OK') {$bad++}; '{0,-8} {1}' -f $s, $f }
"$bad problem(s) in 27 files"