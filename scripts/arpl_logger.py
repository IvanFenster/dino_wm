# [ARPL-WM] Session logger. Runs in the background next to training and saves to Drive every minute:
#   logs/<run>/sessions.csv              one row per session: start, GPU, settings, epoch it resumed from
#   logs/<run>/epoch_timing.csv          one row per finished epoch: train/val minutes, speed, save time
#   logs/<run>/gpu_<session>.csv         GPU use, memory, temperature, power every minute
#   logs/<run>/session_<session>_stdout.log   full training output
# Usage: python3 arpl_logger.py <run_name> <stdout_log> [session_id]. Starting it again replaces the old one.
import os, re, sys, time, signal, shutil, subprocess, datetime as dt

run, src = sys.argv[1], sys.argv[2]
sid = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("ARPL_SESSION") or time.strftime("%Y%m%d_%H%M", time.gmtime())
WM = os.environ.get("WM", "/content/drive/MyDrive/ARPL-WM")
out = f"{WM}/logs/{run}"
ckdir = f"{WM}/outputs/{run}/checkpoints"
os.makedirs(out, exist_ok=True)

pidfile = "/tmp/arpl_logger.pid"
try:
    old = int(open(pidfile).read())
    if old != os.getpid():
        os.kill(old, signal.SIGTERM)
except Exception:
    pass
open(pidfile, "w").write(str(os.getpid()))

def utc(t=None):
    return dt.datetime.fromtimestamp(t if t is not None else time.time(), dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        return f"error {e}"

def append(path, header, row):
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write(header + "\n")
        f.write(row + "\n")
        f.flush(); os.fsync(f.fileno())

def rows(path):
    return open(path).read().splitlines()[1:] if os.path.exists(path) else []

def saved_epochs():
    try:
        return sorted(int(m.group(1)) for f in os.listdir(ckdir) if (m := re.fullmatch(r"model_(\d+)\.pth", f)))
    except Exception:
        return []

def minutes(s):  # tqdm "47:46" or "1:02:03"
    p = [int(x) for x in s.split(":")]
    sec = p[-1] + 60 * p[-2] + (3600 * p[-3] if len(p) == 3 else 0)
    return sec / 60

gpu = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
if not any(r.split(",")[0] == sid for r in rows(f"{out}/sessions.csv")):
    # the epoch training resumed from: from train.py's own message if the logger starts late, else the last checkpoint
    said = re.findall(r"Resuming from epoch (\d+)", open(src, errors="ignore").read()) if os.path.exists(src) else []
    done = [int(said[-1])] if said else saved_epochs()
    append(f"{out}/sessions.csv",
           "session,logger_start_utc,gpu,resumed_from_epoch,epochs_this_session,batch,amp_fp16,sdpa,resume_fix_in_train_py,host",
           ",".join([sid, utc(), gpu.replace(",", " /"), str(max(done, default=0)), os.environ.get("ARPL_EPOCHS", "?"),
                     os.environ.get("ARPL_BATCH", "?"), os.environ.get("ARPL_AMP", "?"), os.environ.get("ARPL_SDPA", "?"),
                     str("[ARPL-WM patch] resume" in open(f"{WM}/dino_wm/train.py").read()), sh("hostname")]))

TS = r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+\]"
while True:
    try:
        if os.path.exists(src):  # 1. full output
            tmp = f"{out}/.session_{sid}_stdout.tmp"
            shutil.copyfile(src, tmp)
            os.replace(tmp, f"{out}/session_{sid}_stdout.log")
        smi = sh("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits")
        append(f"{out}/gpu_{sid}.csv", "time_utc,gpu_util_pct,mem_used_mib,mem_total_mib,temp_c,power_w",  # 2. GPU
               utc() + "," + smi.replace(" ", ""))
        text = open(src, errors="ignore").read().replace("\r", "\n") if os.path.exists(src) else ""  # 3. epoch timing
        starts = re.findall(TS + r".*Resuming from epoch \d+", text) or re.findall(TS + r".*dataloader batch size", text)
        start = starts[-1] if starts else ""
        have = {tuple(r.split(",")[:2]) for r in rows(f"{out}/epoch_timing.csv")}
        for m in re.finditer(TS + r".*Epoch (\d+)\s+Training loss: ([\d.]+)\s+Validation loss: ([\d.]+)", text):
            ts, ep, tr, va = m.groups()
            if (ep, sid) in have:
                continue
            t = re.findall(rf"Epoch {ep} Train: 100%.*?\|\s*(\d+)/\d+ \[([\d:]+)<[^,]*,\s*([\d.]+)(it/s|s/it)", text)
            v = re.findall(rf"Epoch {ep} Valid: 100%.*?\|\s*\d+/\d+ \[([\d:]+)<", text)
            saved = re.findall(TS + r".*Saved model to", text[m.end():])
            ck = f"{ckdir}/model_{ep}.pth"
            if not saved and not os.path.exists(ck):
                continue  # checkpoint not written yet; try again next minute
            prev = re.findall(TS + r".*Saved model to", text[:m.start()])
            began = prev[-1] if prev and prev[-1] > start else start
            end = saved[0] if saved else utc(os.path.getmtime(ck))
            total = (dt.datetime.fromisoformat(end) - dt.datetime.fromisoformat(began)).total_seconds() / 60 if began else float("nan")
            steps, tmin, speed, unit = t[-1] if t else ("", "", "", "")
            ips = (float(speed) if unit == "it/s" else 1 / float(speed)) if speed else float("nan")
            append(f"{out}/epoch_timing.csv",
                   "epoch,session,started_utc,saved_utc,total_min,train_min,val_min,train_steps,train_it_per_s,train_loss,val_loss,gpu",
                   ",".join([ep, sid, began, end, f"{total:.1f}", f"{minutes(tmin):.1f}" if tmin else "",
                             f"{minutes(v[-1]):.1f}" if v else "", steps, f"{ips:.2f}", tr, va, gpu.split(",")[0]]))
    except Exception as e:
        append(f"{out}/logger_errors.txt", "time_utc,error", f"{utc()},{e!r}")
    time.sleep(60)
