import sys, os, time
t=int(sys.argv[1]); stop=sys.argv[2]
print(f"[{time.time():.3f}] HOLDING={t} GiB nonzero_probe={t}/{t} swap=100 MiB", flush=True)
while not os.path.exists(stop):
    print(f"[{time.time():.3f}] hold swap=900 (+800) swapouts=+777", flush=True); time.sleep(1)
print(f"[{time.time():.3f}] RELEASED (stopfile) swap=900 MiB", flush=True)
print(f"[{time.time():.3f}] DONE held={t} GiB swapouts=+777 swap=+800 MiB", flush=True)
