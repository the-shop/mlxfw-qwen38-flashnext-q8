import sys, time
t=int(sys.argv[1])
print(f"[{time.time():.3f}] HOLDING={t} GiB nonzero_probe={t}/{t} swap=100 MiB", flush=True)
time.sleep(2)
print(f"[{time.time():.3f}] RELEASED (swapouts +412 while holding) swap=900 MiB", flush=True)
print(f"[{time.time():.3f}] ABORT_GUARD_HOLD", flush=True)
raise SystemExit(5)
