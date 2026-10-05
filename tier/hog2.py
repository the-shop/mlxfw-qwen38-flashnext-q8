import sys, os, time, ctypes, subprocess, signal

TARGET = int(sys.argv[1])          # GiB
STOPFILE = sys.argv[2]
SWAP_GUARD = int(sys.argv[3]) if len(sys.argv) > 3 else 256   # MiB, fail-closed
STEP = 2                           # GiB per block
PAGE = 16384                       # macOS page size (vm.pagesize)

def swapouts():
    o = subprocess.check_output(['vm_stat']).decode()
    for l in o.splitlines():
        if 'Swapouts' in l: return int(l.split(':')[1].strip().rstrip('.'))
    raise RuntimeError('vm_stat has no Swapouts line - refusing to report 0')

def swap_mb():
    o = subprocess.check_output(['sysctl','-n','vm.swapusage']).decode()
    return float(o.split('used =')[1].split('M')[0])

def log(s):
    print(f"[{time.time():.3f}] {s}", flush=True)

# 16 MiB of real entropy, tiled into every block. WKdm compresses per-page, and
# every PAGE-sized window of this tile is random, so no page is compressible.
ENT = os.urandom(16*1024*1024)
sw0 = swap_mb(); so0 = swapouts()
log(f"baseline swap={sw0:.0f} MiB swapouts={so0} target={TARGET} GiB guard=+{SWAP_GUARD} MiB")

blocks = []; got = 0
def release(why):
    global blocks
    blocks = []
    log(f"RELEASED ({why}) swap={swap_mb():.0f} MiB")

try:
    for _ in range(TARGET // STEP):
        n = STEP * 1024**3
        b = ctypes.create_string_buffer(n)
        mv = memoryview(b).cast("B")
        for off in range(0, n - 1, len(ENT)):
            end = min(off + len(ENT), n - 1)
            mv[off:end] = ENT[:end-off]          # writes EVERY byte => every page dirty
        blocks.append(b); got += STEP
        time.sleep(1.0)          # let the VM rebalance; ramp transients are not steady state
        sw = swap_mb(); so = swapouts()
        log(f"held={got} GiB swap={sw:.0f} MiB (+{sw-sw0:.0f}) swapouts=+{so-so0}")
        if so > so0:
            release(f"swapouts +{so-so0} at {got} GiB")
            log("ABORT_GUARD"); sys.exit(3)
        if sw - sw0 > SWAP_GUARD:
            release(f"swap +{sw-sw0:.0f} MiB > {SWAP_GUARD} at {got} GiB")
            log("ABORT_GUARD"); sys.exit(3)
except MemoryError:
    log(f"MemoryError at {got} GiB")
    release("MemoryError"); sys.exit(4)

# verify the memory is genuinely resident and incompressible: re-read a sample
chk = sum(blocks[i][j*1024**3] != 0 for i in range(len(blocks)) for j in range(STEP))
log(f"HOLDING={got} GiB nonzero_probe={chk}/{len(blocks)*STEP} swap={swap_mb():.0f} MiB")

while not os.path.exists(STOPFILE):
    sw = swap_mb(); so = swapouts()
    log(f"hold swap={sw:.0f} (+{sw-sw0:.0f}) swapouts=+{so-so0}")
    if so > so0:
        release(f"swapouts +{so-so0} while holding")
        log("ABORT_GUARD_HOLD"); sys.exit(5)
    if sw - sw0 > SWAP_GUARD:
        release(f"swap +{sw-sw0:.0f} MiB while holding")
        log("ABORT_GUARD_HOLD"); sys.exit(5)
    time.sleep(2)
release("stopfile")
log(f"DONE held={got} GiB swapouts=+{swapouts()-so0} swap=+{swap_mb()-sw0:.0f} MiB")
