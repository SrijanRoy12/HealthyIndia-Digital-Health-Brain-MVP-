# Health Platform MVP — Streamlit (Anaconda Friendly)

This is a **hands-on, offline-friendly MVP** that shows:
- Digital Health Wallet (simulated NFT-style ownership with signatures + QR)
- Health record upload + hashing
- AI Triage Avatar (rule-based, multilingual UI labels)
- IoT vitals dashboard (with sample data)
- Emergency SOS (simulated)
- Govt Scheme matching (rule engine)
- Digital Twin (India health map demo)
- Token economy (earn for healthy actions)
- Simple Federated Learning demo (local-only, privacy-preserving concept)

## 🍼 Run It Step-by-Step (Windows + Anaconda Navigator)

1) **Open Anaconda Navigator** → click **Environments** → **Create** new env  
   - Name: `health_mvp`  
   - Python version: `3.10` or `3.11`

2) Select the `health_mvp` environment → open a **Terminal** (right side).  
   Run the following commands **exactly**:

```
pip install -r requirements.txt
```

3) In the same terminal, start the app:

```
streamlit run app.py
```

4) Your browser opens automatically. If it doesn’t, copy the displayed URL (usually http://localhost:8501) into Chrome.

5) **No data leaves your PC.** Everything is local: `data/` folder holds uploads, keys, DB, logs.

---

## Folders You Will See

```
health_platform_mvp/
  app.py
  requirements.txt
  data/
    iot_samples.csv
    records/           # your uploaded files go here
    city_coords.csv
```

If something breaks: close the terminal, reopen, and run `streamlit run app.py` again.

> Tip: Keep this folder path **short** (e.g., `C:\health_mvp`) to avoid Windows path issues.
