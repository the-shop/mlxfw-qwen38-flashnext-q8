import sys, time
t=int(sys.argv[1])
print(f"[{time.time():.3f}] HOLDING={t} GiB nonzero_probe={t}/{t} swap=100 MiB", flush=True)
print(f"[{time.time():.3f}] hold swap=100 (+0) swapouts=+0", flush=True)
time.sleep(2); raise SystemExit(9)          # dies mid-decode, logging nothing further
