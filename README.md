# PrecedentIQ — Project Setup Guide

Follow these steps **in order**, on your own machine (not a sandbox), since Steps 4–5
need real internet access to AWS and Hugging Face.

---

## Step 0 — Prerequisites check

Open a terminal and confirm you have Python 3.10+ and at least ~15 GB free disk space
(the SC/HC judgment subset + validation sets + model downloads add up).

```bash
python3 --version        # should show 3.10 or higher
df -h .                  # check free disk space (Linux/Mac) — or just check your drive on Windows
```

If Python isn't installed, get it from python.org (Windows/Mac) or your package manager (Linux).

---

## Step 1 — Get the project folder onto your machine

Unzip the `precedentiq/` folder I've prepared (structure below) into wherever you keep your
projects, then `cd` into it:

```bash
cd precedentiq
```

```
precedentiq/
├── config/
│   └── config.yaml              <- controls how much data you pull
├── data/
│   ├── raw/                     <- SC/HC judgments land here
│   ├── processed/                <- your chunked output goes here later
│   └── validation/               <- IL-TUR / IL-PCSR land here
├── scripts/
│   ├── download_judgments.py
│   ├── download_validation_sets.py
│   └── verify_setup.py
├── src/                          <- your pipeline code goes here as you build it
└── requirements.txt
```

---

## Step 2 — Create a virtual environment and install dependencies

A virtual environment keeps this project's packages separate from everything else on
your machine — always use one.

```bash
python3 -m venv venv

# Activate it:
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows (Command Prompt)

pip install --upgrade pip
pip install -r requirements.txt
```

**Note on `torch`:** if you don't have an NVIDIA GPU, install the CPU-only build instead
(much smaller download, and you don't need a GPU for this phase of the project):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

This step will take a while the first time (torch and transformers are large). Get a coffee.

---

## Step 3 — Verify everything installed correctly

```bash
python scripts/verify_setup.py
```

You should see:
```
Packages:  PASS
Folders:   PASS
Network:   PASS
```

**If Packages shows FAIL:** the output tells you exactly which package is missing and the
exact `pip install` command to fix it — run that, then re-run this script.

**If Network shows FAIL:** you're on a restricted network (college/office wifi sometimes
blocks S3 or Hugging Face). Try a mobile hotspot and re-run.

Do not proceed to Step 4 until this shows all PASS.

---

## Step 4 — Configure how much data to pull

Open `config/config.yaml`. The defaults are intentionally small so your first run is fast:

```yaml
aws:
  sc_years: [2022, 2023]      # just 2 years to start
  hc_courts: ["27_1"]         # just 1 High Court to start
```

**Leave these as-is for your first run.** Once Step 5 succeeds, come back here and expand
towards your 50,000-judgment target by adding more years/courts.

To find more High Court bench codes, list what's actually in the bucket first:
```bash
pip install awscli
aws s3 ls s3://indian-high-court-judgments/data/tar/ --no-sign-request
```
This prints every available `court=XX_Y/` folder — pick the ones you want and add them to
`hc_courts` in the config.

---

## Step 5 — Download the datasets

Two separate scripts — run both:

```bash
# Supreme Court + High Court judgments (primary corpus)
python scripts/download_judgments.py

# IL-TUR + IL-PCSR (validation/evaluation sets)
python scripts/download_validation_sets.py
```

Each prints progress bars and clear `[skip]` / `[ERROR]` messages as it goes — don't just
let it run silently in the background on your first try, watch the output.

**A note on the validation sets script:** it also tries to explain what to do if LawSum or
ILDC specifically fail — those two don't have a confirmed one-line download, so the script
points you to a documented fallback (the LexSumm collection) instead of failing silently.

---

## Step 6 — Verify the data actually landed

```bash
python scripts/verify_setup.py
```

Check the "Downloaded data on disk" section — you should now see non-zero file counts for
Supreme Court, High Court, and validation sets.

Then spot-check by hand — open a couple of the downloaded files to make sure they look like
real judgment text, not empty or corrupted:

```bash
# Mac/Linux
find data/raw -name "*.tar" | head -3

# then extract one and look inside
tar -tvf data/raw/supreme_court/2023/english.tar | head -10
```

---

## Step 7 — Scale up

Once Steps 5–6 succeed with the small default config, edit `config/config.yaml` again to
add more years and High Courts, and re-run Step 5. The script skips anything already
downloaded, so it's safe to re-run repeatedly as you expand scope.

---

## If something goes wrong

Run `python scripts/verify_setup.py` first — it's designed to be the first thing you run
whenever anything seems broken. It will tell you specifically which of Packages / Folders /
Network / Data is the problem, rather than you having to guess from a stack trace.
