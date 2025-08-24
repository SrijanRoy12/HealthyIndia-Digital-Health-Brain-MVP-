# app.py (Polished, complete version)
import os, io, json, time, base64, hashlib, sqlite3, textwrap, datetime
from datetime import datetime as dt, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import qrcode
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from streamlit_folium import st_folium
import folium

# -------------------
# App config & paths
# -------------------
APP_TITLE = "HealthyIndia — Digital Health Brain (MVP)"
DATA_DIR = Path("data")
RECORDS_DIR = DATA_DIR / "records"
DB_PATH = DATA_DIR / "health.db"
IOT_CSV = DATA_DIR / "iot_samples.csv"
CITY_CSV = DATA_DIR / "city_coords.csv"

# -------------------
# Styling helpers
# -------------------
PRIMARY = "#0b76ff"
ACCENT = "#0b9df1"
CARD_BG = "#0f1720"

st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")

def local_css():
    st.markdown(f"""
    <style>
    .app-header {{
        background: linear-gradient(90deg, {PRIMARY}, {ACCENT});
        padding: 16px;
        border-radius: 8px;
        color: white;
    }}
    .card {{
        background: #0b1220;
        padding: 16px;
        border-radius: 10px;
        box-shadow: 0 4px 14px rgba(2,6,23,0.6);
    }}
    .muted {{ color: #9aa6b2; font-size:14px; }}
    .small {{ font-size:12px; color:#aabcbd; }}
    </style>
    """, unsafe_allow_html=True)

# -------------------
# Basic helpers & DB
# -------------------
def ensure_dirs():
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    RECORDS_DIR.mkdir(exist_ok=True, parents=True)

def get_db():
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS user(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        phone TEXT,
        language TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS wallet(
        user_id INTEGER,
        public_key BLOB,
        private_key BLOB,
        did TEXT,
        PRIMARY KEY (user_id)
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS record(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        filename TEXT,
        sha256 TEXT,
        signature BLOB,
        created_at TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS tokens(
        user_id INTEGER PRIMARY KEY,
        balance INTEGER
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS sos_log(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        message TEXT,
        created_at TEXT
    )""")
    conn.commit()
    return conn

def init_user(name, phone, language):
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO user(name, phone, language) VALUES(?,?,?)", (name, phone, language))
    uid = c.lastrowid
    conn.commit()
    conn.close()
    return uid

def get_user(uid):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id, name, phone, language FROM user WHERE id=?", (uid,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "name": row[1], "phone": row[2], "language": row[3]}
    return None

# -------------------
# Wallet & signatures
# -------------------
def get_or_create_wallet(uid):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT user_id, public_key, private_key, did FROM wallet WHERE user_id=?", (uid,))
    row = c.fetchone()
    if row:
        conn.close()
        return row[3]
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    pub_bytes = public_key.public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
    priv_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption()
    )
    did = "did:health:" + hashlib.sha256(pub_bytes).hexdigest()[:20]
    c.execute("INSERT INTO wallet(user_id, public_key, private_key, did) VALUES(?,?,?,?)", (uid, pub_bytes, priv_bytes, did))
    conn.commit()
    conn.close()
    return did

def load_private_key(uid):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT private_key FROM wallet WHERE user_id=?", (uid,))
    row = c.fetchone()
    conn.close()
    if not row: return None
    return Ed25519PrivateKey.from_private_bytes(row[0])

def sign_bytes(uid, message_bytes: bytes):
    pk = load_private_key(uid)
    if pk is None:
        return None
    return pk.sign(message_bytes)

def verify_signature(uid, message_bytes: bytes, signature: bytes) -> bool:
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT public_key FROM wallet WHERE user_id=?", (uid,))
    row = c.fetchone()
    conn.close()
    if not row: return False
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    pub = Ed25519PublicKey.from_public_bytes(row[0])
    try:
        pub.verify(signature, message_bytes)
        return True
    except InvalidSignature:
        return False

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def add_tokens(uid, amount):
    conn = get_db()
    c = conn.cursor()
    c.execute("""INSERT INTO tokens(user_id, balance) VALUES(?, ?)
                 ON CONFLICT(user_id) DO UPDATE SET balance = balance + excluded.balance""",
              (uid, amount))
    conn.commit()
    conn.close()

def get_balance(uid):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT balance FROM tokens WHERE user_id=?", (uid,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else 0

def redeem_tokens(uid, amount):
    conn = get_db()
    c = conn.cursor()
    c.execute("UPDATE tokens SET balance = MAX(balance - ?, 0) WHERE user_id=?", (amount, uid))
    conn.commit()
    conn.close()

# -------------------
# QR + image helpers
# -------------------
def qr_image(text: str) -> Image.Image:
    qr = qrcode.QRCode(version=1, box_size=8, border=2)
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    if hasattr(img, "get_image"):
        img = img.get_image()
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# -------------------
# Data loaders / demos
# -------------------
def load_iot_df() -> pd.DataFrame:
    if not IOT_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(IOT_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def localized_labels(lang: str):
    labels = {
        "en": {
            "welcome": "Welcome",
            "create_wallet": "Create/Show Health Wallet",
            "upload_record": "Upload Health Record",
            "ai_avatar": "AI Triage Avatar",
            "iot": "Vitals & IoT",
            "sos": "Emergency SOS",
            "schemes": "Govt Scheme Matching",
            "twin": "Digital Twin Map",
            "tokens": "Health Tokens",
            "federated": "Federated AI Demo",
            "disclaimer": "This MVP is not a medical device. For emergencies, call local services immediately."
        },
        "hi": { "welcome": "स्वागत है", "create_wallet": "हेल्थ वॉलेट", "upload_record": "हेल्थ रिकॉर्ड अपलोड",
                "ai_avatar": "एआई ट्रायाज अवतार", "iot": "वाइटल्स और IoT", "sos": "आपातकाल SOS",
                "schemes": "सरकारी योजनाएँ", "twin": "डिजिटल ट्विन मानचित्र", "tokens": "हेल्थ टोकन",
                "federated": "फेडरेटेड एआई डेमो", "disclaimer": "यह MVP मेडिकल डिवाइस नहीं है। आपातकाल में तुरंत सहायता लें।"
        },
        "kn": { "welcome": "ಸ್ವಾಗತ", "create_wallet": "ಹೆಲ್ತ್ ವಾಲೆಟ್", "upload_record": "ರೆಕಾರ್ಡ್ ಅಪ್‌ಲೋಡ್",
                "ai_avatar": "ಎಐ ಟ್ರೈಯೇಜ್ ಅವತಾರ", "iot": "ವೈಟಲ್ಸ್ ಮತ್ತು IoT", "sos": "ತುರ್ತು SOS",
                "schemes": "ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು", "twin": "ಡಿಜಿಟಲ್ ಟ್ವಿನ್ ನಕ್ಷೆ", "tokens": "ಹೆಲ್ತ್ ಟೋಕನ್",
                "federated": "ಫೆಡರೇಟೆಡ್ ಎಐ ಡೆಮೋ", "disclaimer": "ಇದು ವೈದ್ಯಕೀಯ ಸಾಧನ ಅಲ್ಲ. ತುರ್ತು ಪರಿಸ್ಥಿತಿಯಲ್ಲಿ ತಕ್ಷಣ ಸಹಾಯ ಕೋರಿರಿ."
        }
    }
    return labels.get(lang, labels["en"])

# -------------------
# UI sections
# -------------------
def section_wallet(uid, labels):
    st.subheader(labels["create_wallet"])
    did = get_or_create_wallet(uid)

    # Top row: DID + Quick actions
    col1, col2 = st.columns([2,1])
    with col1:
        st.success("Your DID (wallet id) has been created")
        st.code(did)
        st.caption("Copy the DID above — do not click (it's not a web link). Use QR to share.")
    with col2:
        img = qr_image(did)
        st.image(pil_to_png_bytes(img), width=160, caption="Scan to share DID (demo)")

    st.markdown("<div class='muted'>DID is derived from your public key — this simulates ownership via cryptographic signatures. No data leaves your PC.</div>", unsafe_allow_html=True)

def section_upload_record(uid, labels):
    st.subheader(labels["upload_record"])
    st.markdown("Upload medical reports (PDF, JPG, PNG). Records are hashed and signed to prove ownership by your DID.")
    uploaded = st.file_uploader("", type=["pdf","png","jpg","jpeg"])
    if uploaded is not None:
        # Save file
        content = uploaded.read()
        sha = sha256_bytes(content)
        sig = sign_bytes(uid, sha.encode())
        if sig is None:
            st.error("Create wallet first to sign records.")
            return
        safe_name = uploaded.name.replace(" ", "_")
        ts = dt.now().strftime("%Y%m%d_%H%M%S")
        path = RECORDS_DIR / f"{ts}_{safe_name}"
        with open(path, "wb") as f:
            f.write(content)

        # Insert DB
        conn = get_db()
        c = conn.cursor()
        c.execute("INSERT INTO record(user_id, filename, sha256, signature, created_at) VALUES(?,?,?,?,?)",
                  (uid, str(path), sha, sig, dt.now().isoformat(timespec="seconds")))
        conn.commit()
        conn.close()

        st.success(f"Record saved — SHA {sha[:12]}... (full hash stored).")
        img = qr_image(sha)
        st.image(pil_to_png_bytes(img), width=220, caption="QR: record hash (integrity token)")
        add_tokens(uid, 5)
        st.balloons()

    # List records & verify panel
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id, filename, sha256, created_at FROM record WHERE user_id=? ORDER BY id DESC", (uid,))
    rows = c.fetchall()
    conn.close()

    if rows:
        st.markdown("---")
        st.markdown("### Your recent records")
        for rid, fname, sha, created in rows[:6]:
            file_name = Path(fname).name
            rs1, rs2 = st.columns([6,1])
            with rs1:
                st.write(f"**{file_name}** — {created}")
                st.write(f"`SHA:` {sha}")
            with rs2:
                # small verify button
                if st.button(f"Verify {rid}", key=f"verify_{rid}"):
                    verified = verify_signature(uid, sha.encode(), _get_signature_for_record(rid))
                    if verified:
                        st.success("Signature: ✅ VALID")
                    else:
                        st.error("Signature: ❌ INVALID")

def _get_signature_for_record(rid):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT signature FROM record WHERE id=?", (rid,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

def rule_based_triage(symptoms: dict) -> dict:
    urgent = False
    notes = []
    chest_pain = symptoms.get("chest_pain", False)
    breathless = symptoms.get("breathless", False)
    sweating = symptoms.get("sweating", False)
    fever = symptoms.get("fever", False)
    rash = symptoms.get("rash", False)
    glucose = symptoms.get("glucose", None)
    spo2 = symptoms.get("spo2", None)
    temp_c = symptoms.get("temp_c", None)
    headache = symptoms.get("headache", False)
    city_temp = symptoms.get("city_temp", None)

    if chest_pain and (breathless or sweating):
        urgent = True
        notes.append("Possible cardiac emergency. Seek immediate care (golden hour).")
    if glucose is not None and glucose < 60:
        urgent = True
        notes.append("Severe hypoglycemia suspected. Take fast-acting sugar and seek care.")
    if spo2 is not None and spo2 < 92:
        urgent = True
        notes.append("Low oxygen saturation detected. Possible respiratory distress.")
    if fever and rash:
        notes.append("Dengue/chikungunya suspicion. Hydrate, check platelets, consult doctor.")
    if temp_c and temp_c >= 39.5:
        notes.append("High fever. Monitor and consider medical evaluation.")
    if city_temp and city_temp >= 42:
        notes.append("Severe heatwave risk. Avoid sun, hydrate, ORS recommended.")
    if headache and fever:
        notes.append("Fever + headache: monitor for dengue/malaria; seek test if persists.")
    if not notes:
        notes.append("Low immediate risk based on inputs. Monitor and practice hydration, rest, and follow local guidelines.")
    return {"urgent": urgent, "advice": notes}

def section_ai_avatar(labels):
    st.subheader(labels["ai_avatar"])
    st.caption("Offline rule-based triage. NOT a medical device.")
    col1, col2 = st.columns(2)
    with col1:
        chest_pain = st.checkbox("Chest pain")
        breathless = st.checkbox("Shortness of breath")
        sweating = st.checkbox("Cold sweats")
        fever = st.checkbox("Fever")
        rash = st.checkbox("Rash")
        headache = st.checkbox("Headache")
    with col2:
        glucose = st.number_input("Blood glucose (mg/dL)", min_value=0, max_value=600, value=110)
        spo2 = st.number_input("SpO₂ (%)", min_value=50, max_value=100, value=97)
        temp_c = st.number_input("Body temperature (°C)", min_value=30.0, max_value=45.0, value=36.9, step=0.1)
        city_temp = st.number_input("Outside temperature (°C)", min_value=10.0, max_value=50.0, value=34.0, step=0.5)

    if st.button("Analyze Symptoms"):
        res = rule_based_triage({
            "chest_pain": chest_pain, "breathless": breathless, "sweating": sweating,
            "fever": fever, "rash": rash, "headache": headache,
            "glucose": glucose, "spo2": spo2, "temp_c": temp_c, "city_temp": city_temp
        })
        if res["urgent"]:
            st.error("⚠️ URGENT FLAG")
        for n in res["advice"]:
            st.write("- " + n)
        if res["urgent"]:
            st.session_state["urgent_flag"] = True
        add_tokens(st.session_state["uid"], 2)

def section_iot(labels):
    st.subheader(labels["iot"])
    df = load_iot_df()
    if df.empty:
        st.info("No sample IoT data found. Use sample CSV in data/ to demo.")
        return
    st.line_chart(df.set_index("timestamp")[["heart_rate","systolic","diastolic","spo2","glucose"]])
    latest = df.iloc[-1]
    st.write(f"Latest vitals @ {latest['timestamp']}: HR {latest['heart_rate']} bpm, BP {latest['systolic']}/{latest['diastolic']}, SpO₂ {latest['spo2']}%, Glucose {latest['glucose']} mg/dL")
    if latest["spo2"] < 92 or latest["systolic"] > 180 or latest["glucose"] < 60:
        st.error("Threshold breach detected! Consider SOS.")
    add_tokens(st.session_state["uid"], 1)

def section_sos(labels):
    st.subheader(labels["sos"])
    if st.session_state.get("urgent_flag", False):
        st.error("AI flagged urgent condition previously.")
    msg = st.text_input("Message to record (simulated)", value="Emergency detected. Please help.")
    if st.button("Trigger SOS (Simulated)"):
        conn = get_db()
        c = conn.cursor()
        c.execute("INSERT INTO sos_log(user_id, message, created_at) VALUES(?,?,?)",
                  (st.session_state["uid"], msg, dt.now().isoformat(timespec="seconds")))
        conn.commit()
        conn.close()
        st.success("SOS recorded locally.")
        add_tokens(st.session_state["uid"], 1)

def section_schemes(labels):
    st.subheader(labels["schemes"])
    st.caption("Offline eligibility suggestions.")
    col1, col2 = st.columns(2)
    with col1:
        state = st.selectbox("State", ["UP","Bihar","Rajasthan","Karnataka","Delhi","Maharashtra","Tamil Nadu","Telangana","West Bengal","MP"])
        caste = st.selectbox("Category", ["General","OBC","SC","ST"])
        income = st.number_input("Household annual income (₹)", min_value=0, max_value=2000000, value=120000, step=1000)
    with col2:
        ration = st.selectbox("Ration card", ["None","BPL","APL","Antyodaya (AAY)","Priority"])
        disabled = st.checkbox("Disability (PwD)")
        female_head = st.checkbox("Female-headed household")
        urban = st.checkbox("Urban resident")

    suggestions = []
    if ration in ["BPL","Antyodaya (AAY)","Priority"] or income <= 150000:
        suggestions.append("Ayushman Bharat PM-JAY — Likely eligible.")
    if state == "Karnataka" and income <= 200000:
        suggestions.append("State health scheme (Karnataka).")
    if disabled:
        suggestions.append("PwD benefits: extra coverage/support.")
    if female_head:
        suggestions.append("Women-focused health subsidies.")
    if urban:
        suggestions.append("Urban Health Mission clinics (UPHC).")

    if suggestions:
        st.success("Suggested programs:")
        for s in suggestions:
            st.write("- " + s)
    else:
        st.info("No obvious matches from basic inputs.")

def section_twin(labels):
    st.subheader(labels["twin"])
    st.caption("Digital Twin demo (sample vitals aggregated by city).")
    if not (IOT_CSV.exists() and CITY_CSV.exists()):
        st.info("Missing iot_samples.csv or city_coords.csv in data/.")
        return
    df = pd.read_csv(IOT_CSV)
    cc = pd.read_csv(CITY_CSV)
    def risk_row(r):
        score = 0
        if r["spo2"] < 93: score += 2
        if r["systolic"] > 160: score += 2
        if r["glucose"] > 180 or r["glucose"] < 60: score += 2
        if r["temperature_c"] > 38.5: score += 1
        return score
    df["risk"] = df.apply(risk_row, axis=1)
    agg = df.groupby("city")["risk"].mean().reset_index()
    m = folium.Map(location=[22.97, 79.59], zoom_start=5)
    merged = pd.merge(agg, cc, on="city", how="left")
    for _, row in merged.iterrows():
        folium.CircleMarker(location=[row["lat"], row["lon"]],
                            radius=6 + float(row["risk"])*2,
                            popup=f"{row['city']}: risk {row['risk']:.2f}",
                            fill=True).add_to(m)
    st_folium(m, width=700, height=480)
    add_tokens(st.session_state["uid"], 1)

def section_tokens(uid, labels):
    st.subheader(labels["tokens"])
    bal = get_balance(uid)
    st.metric("Your Health Tokens", bal)
    if st.button("Redeem 10 tokens (demo)"):
        if bal >= 10:
            redeem_tokens(uid, 10)
            st.success("Redeemed 10 tokens (demo).")
        else:
            st.warning("Not enough tokens.")
    st.caption("Earn tokens by uploading records, triage checks, and IoT readings.")

def section_federated(labels):
    st.subheader(labels["federated"])
    st.caption("Basic local federated demo (toy).")
    rng = np.random.default_rng(42)
    n = 200
    X1 = np.column_stack([rng.normal(125, 15, n), rng.normal(80, 8, n), rng.normal(95, 1.5, n)])
    y1 = (X1[:,0] > 140).astype(int)
    X2 = np.column_stack([rng.normal(130, 20, n), rng.normal(85, 10, n), rng.normal(96, 1.2, n)])
    y2 = (X2[:,0] > 145).astype(int)
    m1 = LogisticRegression(max_iter=500).fit(X1, y1)
    m2 = LogisticRegression(max_iter=500).fit(X2, y2)
    coef = (m1.coef_ + m2.coef_) / 2.0
    intercept = (m1.intercept_ + m2.intercept_) / 2.0
    m_global = LogisticRegression()
    m_global.classes_ = np.array([0,1])
    m_global.coef_ = coef
    m_global.intercept_ = intercept
    m_global.n_features_in_ = X1.shape[1]
    X_test = np.vstack([X1[:50], X2[:50]])
    y_test = np.hstack([y1[:50], y2[:50]])
    y_pred = (m_global.predict_proba(X_test)[:,1] > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Global model accuracy: **{acc:.2f}**")
    st.info("This is a simple averaged-weights demo to show the federated concept.")

# -------------------
# Onboarding & Main
# -------------------
def onboarding():
    st.markdown("### Create Profile")
    name = st.text_input("Your name")
    phone = st.text_input("Phone (optional)")
    lang = st.selectbox("Language", ["en","hi","kn"])
    if st.button("Start"):
        uid = init_user(name or "User", phone or "", lang)
        st.session_state["uid"] = uid
        st.session_state["lang"] = lang
        st.rerun()

def main():
    local_css()
    ensure_dirs()
    st.markdown(f"<div class='app-header'><h2 style='margin:0'>{APP_TITLE}</h2></div>", unsafe_allow_html=True)
    if "uid" not in st.session_state:
        onboarding()
        st.stop()

    user = get_user(st.session_state["uid"])
    labels = localized_labels(user["language"] or "en")

    # Sidebar
    with st.sidebar:
        st.markdown(f"**{labels['welcome']}, {user['name']}**")
        page = st.radio("Navigate", [
            labels["create_wallet"],
            labels["upload_record"],
            labels["ai_avatar"],
            labels["iot"],
            labels["sos"],
            labels["schemes"],
            labels["twin"],
            labels["tokens"],
            labels["federated"]
        ])
        st.markdown("---")
        st.markdown("Demo info:")
        st.markdown("<div class='small'>Local demo. Not a medical device. No network calls by default.</div>", unsafe_allow_html=True)

    # Route pages
    if page == labels["create_wallet"]:
        section_wallet(user["id"], labels)
    elif page == labels["upload_record"]:
        section_upload_record(user["id"], labels)
    elif page == labels["ai_avatar"]:
        section_ai_avatar(labels)
    elif page == labels["iot"]:
        section_iot(labels)
    elif page == labels["sos"]:
        section_sos(labels)
    elif page == labels["schemes"]:
        section_schemes(labels)
    elif page == labels["twin"]:
        section_twin(labels)
    elif page == labels["tokens"]:
        section_tokens(user["id"], labels)
    elif page == labels["federated"]:
        section_federated(labels)

if __name__ == "__main__":
    main()
