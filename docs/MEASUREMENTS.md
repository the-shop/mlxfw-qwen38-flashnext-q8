title: GPU/CPU expert tiering on Apple Silicon (M5 Max, 128 GiB)
model: Qwen3.8-Flash-Next Q8_0, 5-shard split, 188.5 GB, arch qwen4exp
method: 12 varied prompts, 24 warm queries discarded per config, token-weighted
        tok/s, thinking disabled, content verified nonempty, span-gated preflight
date: 2026-10-04

headline:
  untiered_tok_s: 9.7
  best_efficient: "-ngl 16 -> 20.45 tok/s, wired 45.1 GiB, 39.2 GiB free"
  best_speed: "-ngl 24 -> 23.35 tok/s, wired 65.6 GiB, 27.6 GiB free"
  gain: 2.1x to 2.4x

curve[5]{config,tok_s,wired_GiB,free_GiB,tok_s_per_GiB}:
  untiered (ngl 6),9.7,19.4,~60,0.50
  -ngl 16,20.45,45.1,39.2,0.450
  -ngl 20,20.43,55.8,32.4,0.366
  -ngl 24,23.35,65.6,27.6,0.356
  -ngl 28,22.36,76.0,21.8,0.294

mechanism_1_span:
  rule: "Metal cost per FILE = max_offset - min_offset over tensors assigned to that
         device, NOT bytes. A GPU-assigned tensor near offset 0 stretches the span
         across the whole file."
  evidence: "dense weights for layers 0-31 are 2.8 GiB of data but 79 GiB of span,
             because they are scattered across two 41 GiB shards."
  limit: "total span vs recommendedMaxWorkingSetSize (120 GiB). There is NO
          per-span limit: ggml splits spans > max_buffer_size into overlapping
          views (ggml-metal-device.m:1753-1780, up to GGML_METAL_MAX_BUFFERS=64),
          so maxBufferLength 80.64 GiB is the view stride, not a cliff."
  tool: tier/span.py

mechanism_2_crossings:
  rule: "Each CPU<->GPU transition costs a full GPU drain plus host-side tensor
         copies. Contiguous layer blocks pay 2 crossings per TOKEN; scattered
         per-layer expert placement pays 2 per LAYER."
  measured: "-ngl 16 (contiguous) 20.80 vs experts 32-47 via -ot (scattered) 16.18
             at equal GPU memory (45.0 vs 46.8 GiB) = +28.6% for contiguity"
  source: "ggml-backend.cpp:1604,1743 sequential split loop; CPU cpy_tensor_async
           == NULL at :1726-1736 forces ggml_backend_synchronize = waitUntilCompleted"

mechanism_3_budget:
  formula: "wired + held + ~36 GiB working cache <= 122 GiB"
  consequence: "60 GiB free and a large hot tier are mutually exclusive on 128 GiB"
  counterintuitive: "Under memory pressure a BIGGER GPU tier protects throughput,
                     because GPU weights are wired and unevictable:
                     ngl24+hold15 = 10.29 t/s beats ngl6+hold45 = 5.75 t/s"

geometry:
  experts_per_layer_MiB: 2550.0
  per_expert_MiB: 4.98
  dense_per_layer_MiB: 88.2
  dense_all_48_layers_GiB: 4.10
  PLE_MiB: 51880
  shard_to_layer: "s3: 0-16, s4: 16-32, s5: 32-47"
  per_token_expert_reads_GB: 1.93

closed_paths[7]{path,verdict,evidence}:
  per-expert GPU/CPU placement,IMPOSSIBLE as shipped,"mul_mat_id does stride arithmetic off ONE base pointer (ggml-cpu.c:1654, ggml-metal.metal:11235); a tensor has exactly one buffer (ggml.h:676); routing is decided IN-GRAPH via argsort_top_k (llama-graph.cpp:2057) so residency cannot follow the router. Viable path is offline expert permutation + group split, 2-4 weeks."
  CPU/GPU compute overlap,PATCHABLE BUT UNSAFE,"ggml_gallocr reuses scratch across splits assuming serial execution; concurrent splits alias buffers -> silently WRONG NUMBERS, no crash. Upside only 3.5-5.3%."
  MTLIO / sparse heaps,NO GAIN,"measured 11.99-13.53 GB/s = matches the pread ceiling, does not beat it; pollutes page cache with no F_NOCACHE equivalent; placement-sparse is Private-only so no CPU fallback."
  GGML_METAL_NO_RESIDENCY,NO-OP,"wired 45.1 (on) vs 45.2 (off) GiB. Wiring comes from newBufferWithBytesNoCopy, not residency sets."
  speculative decoding (PR #27739 branch),NO GAIN THERE,"20.73 / 20.84 / 20.68 at draft depth 4 / 8 / 16 vs 20.62-20.80 baseline. Later superseded: MTP drafting on llama.cpp v0.6.0 gave +44% with experts cached (docs/2026-10-06-v060-copy-mode-mtp.md)."
  internal-disk relocation,~9% AT BEST (unreplicated),"CORRECTED: the 22.06 GB/s @ QD8 external figure was a page-cache artifact -- F_NOCACHE does not evict already-resident pages, so the benchmark timed RAM. With offsets verified non-resident via mincore: external saturates at 7.08 GB/s, internal NVMe at 13.52 (1.91x). Bandwidth arithmetic then predicted +24-53% under memory pressure; the weights were copied to internal and measured: 20.78 vs 20.45 tok/s unheld (no change, as predicted) and 5.98 vs 5.47 at 30 GiB held (+9%, n=1). Three replication attempts VOID (hog ABORT_GUARD_HOLD, swapfile exhausted at 8.1/9.2 GB after a day of testing). Why the projection failed: llama.cpp faults weights in via mmap at ~1.5 GB/s per thread, so the FAULT PATH is the ceiling, not the device; a 1.91x faster device leaves it untouched. Same mechanism as the prefetch no-op."
  LLAMA_MMAP_RANDOM (PLE readahead),NO-OP; PREMISE FALSE,"patch built and measured: 20.78 stock vs 20.72 patched. mincore measures the PLE at 0.03-0.47 GiB of page cache, not the predicted 3-7 GiB, so there is nothing to reclaim. Expert shard holds ~32 GiB resident -- that is the real consumer."

ceilings[5]{limit,measured,tok_s_cap}:
  CPU RAM read (12 threads),235 GB/s,122
  SSD random 3 MiB saturated (external TBT5),7.08 GB/s,3.67
  SSD random 3 MiB saturated (internal NVMe),13.52 GB/s,7.0
  SSD random 3 MiB QD1,6.70 GB/s,3.47
  mmap page-fault path,~1.47 GB/s per stream,~0.8 per thread

threads:
  finding: "6 and 12 threads are equivalent; 18 threads (every core) is much slower. The M5 Max has 18 CPU cores: 6 at its top performance level and 12 at the next (sysctl hw.perflevel0.physicalcpu / hw.perflevel1.physicalcpu)"
  measured: "threads 6 -> 18.88, threads 12 -> 19.14 (+1.4%), threads 18 -> 11.45 (-39%)"
  cause: "even work split + barrier per layer; with every core busy, the slowest threads become stragglers the others wait on"
  regime_dependent: "under memory pressure MORE threads DO help (2.80 -> 4.22) because they buy I/O fault concurrency, not compute"

corrections_made[4]:
  - "maxBufferLength 80.64 GiB is NOT a failure threshold (ggml splits into views). An earlier span.py rejected valid configs, and a false warning nearly went into a public model card; an independent run falsified it empirically."
  - "All early tiering scripts omitted --chat-template-kwargs enable_thinking:false, so content was empty in 6 of 8 responses. The empty counter was wrongly dismissed as a parser artifact. Relative curve survives (identical deterministic workload, ntok=512 every row); correctness claim did not."
  - "A local tree used to inspect PR #27742 was called 'master' when no master checkout existed. Actual master has a full qwen4exp MTP implementation. The correct report was 'cannot verify'."
  - "A 3.86 tok/s storage ceiling was reported from a QD1 benchmark, although a measured 4.22 tok/s had already exceeded it. The QD8 figure first used to correct it (22.06 GB/s) was itself a page-cache artifact; see internal-disk relocation above."
