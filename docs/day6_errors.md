# Day 6 — Error Report

A frank list of every error, warning, or rough edge I hit while executing
Day 6, with the resolution (or "unresolved, deferred to Day N"). Recorded so
the other contributor doesn't re-discover them.

---

## E1 — PowerShell doesn't accept `&&`

**Symptom**
```
The token '&&' is not a valid statement separator in this version.
```
when trying to run `git pull && git log -1 --oneline` from the docs.

**Cause**
Windows PowerShell 5.1 has no `&&` operator. `cmd` does, but the opencode
shell here is PowerShell.

**Resolution**
Ran the commands as two separate `bash` tool calls (`git pull`, then
`git log -1 --oneline`). Going forward I'll keep PowerShell idiom in mind:
use `; if ($?) { ... }` for chains, or just split into parallel tool calls.

**Status:** Resolved.

---

## E2 — `nbconvert` warning printed as a PowerShell `RemoteException`

**Symptom**
```
.venv\Scripts\python.exe : [NbConvertApp] Converting notebook notebooks/02_preprocessing.ipynb to notebook
At line:1 char:1
+ .venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute ...
    + CategoryInfo          : NotSpecified: ([NbConvertApp] ...ynb to notebook:String) [], RemoteException
    + FullyQualifiedErrorId : NativeCommandError
```
Looks scary, but the notebook was actually written successfully
(`Writing 42226 bytes`).

**Cause**
A `RuntimeWarning` from `zmq._future` (Proactor event loop on Windows) is
printed via `print(...)` to stderr by `ipykernel`, then PowerShell's native
exec layer misroutes stderr as a `RemoteException`. The exit code was 0.

**Resolution**
Confirmed success by parsing the saved notebook with `nbformat.read(...)` and
checking `cell.outputs[*].output_type == 'error'` — there were none. The
notebook is good. I'll keep using `nbformat`-based validation instead of
trusting the exit code for future notebook runs.

**Status:** Resolved (workaround).

---

## E3 — `ModuleNotFoundError: No module named 'ticket_router'` on first notebook execution

**Symptom**
```
ModuleNotFoundError: No module named 'ticket_router'
```
in the first code cell that imports the preprocessing module.

**Cause**
The notebook kernel's `cwd` was `notebooks/`, so `sys.path.insert(0,
REPO / "src")` was inserting the literal string `notebooks/src`. My
`REPO = Path.cwd()` was wrong.

**Resolution**
Replaced `REPO = Path.cwd()` with a robust walk-up that locates the
directory containing `src/`:
```python
_here = Path.cwd().resolve()
REPO = next((p for p in (_here, *_here.parents) if (p / "src").is_dir()), _here)
```
Also set `matplotlib.use("Agg")` up front so the notebook doesn't try to open
a display. This will be the standard preamble for future notebooks in this
repo.

**Status:** Resolved.

---

## E4 — matplotlib / zmq hang on Windows during `nbconvert --execute`

**Symptom**
`jupyter nbconvert --execute` eventually returned but printed
`[NbConvertApp] Kernel is taking too long to finish, terminating` and a
`RuntimeError: no running event loop` trace.

**Cause**
Tornado + Windows Proactor event loop + matplotlib's interactive backend is a
known bad combo. Even with `Agg` set early in the cell, the kernel can hang
during shutdown.

**Resolution**
For local runs, the notebook actually completes (cells all executed, outputs
written) before the hang; we just lose the clean shutdown. For CI use, we can
either add `set_event_loop_policy(WindowsSelectorEventLoopPolicy())` or run
through `papermill`. Not blocking Day 6.

**Status:** Deferred (workaround for Day 19 evaluation script).

---

## E5 — Stale `01_eda.ipynb` outputs in git

**Symptom**
`01_eda.ipynb` is ~1 MB on disk with image blobs in `outputs[*].data["image/png"]`.

**Cause**
The Day 3 contributor didn't strip outputs before committing. Not my mess to
fix on Day 6, but worth flagging.

**Resolution**
Wrote `scripts/strip_nb02_outputs.py` for Day 6's notebook (cleared before
commit, 24.7 KB). I'll offer to retro-strip `01_eda.ipynb` in a separate
commit on Day 7 if there's time, or open it as an issue.

**Status:** Deferred (low priority, not blocking).

---

## E6 — `data/processed/*.csv` is gitignored but my script reads from `data/raw/`

**Symptom** (potential)
The notebook needs `data/raw/tickets_raw.csv` to be present locally.

**Cause**
`.gitignore` excludes both `data/raw/` and `data/processed/*.csv`, so neither
file is checked in. A fresh codespace clone will not have them.

**Resolution**
Added an `assert raw_path.exists()` at the top of the relevant cell so the
failure mode is a clear assertion message ("Run scripts/download_data.py
first") rather than a downstream `FileNotFoundError`. Documented in the
notebook's setup section.

**Status:** Resolved (defensive code).

---

## E7 — Hand-off drift risk: `solver='liblinear'` vs `'lbfgs'`

**Symptom** (latent)
I locked `solver='liblinear'` for the Day 7 LogReg.

**Cause**
`liblinear` is a solid default for sparse text + L2, but with `class_weight`
and `max_iter=1000` you can still hit convergence warnings on some platforms.

**Resolution**
Captured this in `artifacts/class_weight_strategy.json` and noted in
`docs/preprocessing.md` that Day 7 should log convergence warnings; if they
appear, bump `max_iter` to 2000 or switch to `solver='lbfgs'` (which
natively supports multinomial).

**Status:** Documented; no action needed today.

---

## Summary

7 issues found, 5 resolved, 2 deferred (E4 + E5). None blocked Day 6.
The deferred items are well-scoped and will be picked up naturally on Day 7
or Day 19.