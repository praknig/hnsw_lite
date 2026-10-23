#!/usr/bin/env bash
# Fuzzes every file loader with libFuzzer, AddressSanitizer and UBSan.
#   bash .github/scripts/fuzz.sh [seconds per loader, default 60]
# Needs clang. Checksums are skipped in this build (FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION)
# so that mutated inputs reach the validation behind them. A crashing input is
# kept in fuzz-out/artifacts/<loader>/; reproduce it with: fuzz-out/fuzz<N> <file>
set -euo pipefail
seconds="${1:-60}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
out="${FUZZ_OUT:-$root/fuzz-out}"  # FUZZ_OUT: elsewhere, e.g. to keep the corpus between runs
flags="-fsanitize=address,undefined -fno-sanitize-recover=undefined -O1 -g -DFUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION"
names=(x hnsw flat sparse collection wal disk)

CXX=clang++ cmake -S "$root" -B "$out/build" -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  "-DCMAKE_CXX_FLAGS=$flags -fsanitize=fuzzer-no-link" \
  -DHNSW_LITE_BUILD_TESTS=OFF -DHNSW_LITE_BUILD_BENCHMARKS=OFF -DHNSW_LITE_BUILD_EXAMPLES=OFF > /dev/null
cmake --build "$out/build" -j "$(nproc)"
inc="-I$root/include -I$root/src"
clang++ -std=c++20 $flags $inc "$root/fuzz/make_seeds.cpp" -L"$out/build" -lhnsw_lite -pthread -o "$out/make_seeds"
rm -rf "$out/seeds" && "$out/make_seeds" "$out/seeds"
for t in 1 2 3 4 5 6; do
  clang++ -std=c++20 $flags -fsanitize=fuzzer -DFUZZ_TARGET=$t $inc "$root/fuzz/fuzz_load.cpp" -L"$out/build" -lhnsw_lite -pthread -o "$out/fuzz$t" &
done
wait

status=0
for t in 1 2 3 4 5 6; do
  mkdir -p "$out/corpus/${names[$t]}" "$out/artifacts/${names[$t]}"
  cp -n "$out/seeds/$t/"* "$out/corpus/${names[$t]}/" 2>/dev/null || true
  "$out/fuzz$t" "$out/corpus/${names[$t]}" -max_total_time="$seconds" -rss_limit_mb=2048 -malloc_limit_mb=1024 -max_len=20000 \
    -artifact_prefix="$out/artifacts/${names[$t]}/" -print_final_stats=1 > "$out/${names[$t]}.log" 2>&1 &
done
for job in $(jobs -p); do wait "$job" || status=1; done
for t in 1 2 3 4 5 6; do
  runs=$(grep -oE "number_of_executed_units: *[0-9]+" "$out/${names[$t]}.log" | grep -oE "[0-9]+$" || echo "?")
  found=$(ls "$out/artifacts/${names[$t]}" | wc -l)
  printf "%-11s %10s inputs, %s findings\n" "${names[$t]}" "$runs" "$found"
  if [ "$found" -gt 0 ]; then status=1; grep -E "ERROR|FUZZ FINDING|runtime error" "$out/${names[$t]}.log" | head -5; fi
done
exit $status