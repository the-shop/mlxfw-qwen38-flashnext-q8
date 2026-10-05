import sys, os, time
t=int(sys.argv[1]); stop=sys.argv[2]
print(f"[{time.time():.3f}] HOLDING={t-20} GiB nonzero_probe=1/1 swap=100 MiB", flush=True)
while not os.path.exists(stop): time.sleep(1)
print(f"[{time.time():.3f}] RELEASED (stopfile) swap=100 MiB", flush=True)
print(f"[{time.time():.3f}] DONE held={t-20} GiB swapouts=+0 swap=+0 MiB", flush=True)
