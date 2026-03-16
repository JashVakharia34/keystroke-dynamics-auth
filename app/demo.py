import streamlit as st
import streamlit.components.v1 as components
import pickle
import numpy as np
import json
import os
import pandas as pd
import plotly.express as px
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Page config ───────────────────────────────────────
st.set_page_config(
    page_title="KeyDNA — Keystroke Authentication",
    page_icon="🧬",
    layout="centered"
)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, '..', 'data', 'raw')
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
TARGET_SENTENCE = "the quick brown fox jumps over the lazy dog near the river bank"

# ── Feature extraction ────────────────────────────────
def extract_features(key_log):
    keydowns, dwell_times, flight_times = {}, {}, []
    last_up_key = last_up_time = None
    for event in key_log:
        if event['event'] == 'down':
            keydowns[event['key']] = event['time']
            if last_up_time is not None:
                flight_times.append({
                    'from': last_up_key, 'to': event['key'],
                    'time': event['time'] - last_up_time
                })
        if event['event'] == 'up':
            if event['key'] in keydowns:
                dwell = event['time'] - keydowns[event['key']]
                dwell_times.setdefault(event['key'], []).append(dwell)
            last_up_key, last_up_time = event['key'], event['time']
    dwell_means = {k: np.mean(v) for k, v in dwell_times.items()}
    all_dwells  = [d for v in dwell_times.values() for d in v]
    all_flights = [f['time'] for f in flight_times]
    return {
        'dwell_per_key': dwell_means, 'flight_times': flight_times,
        'mean_dwell':    np.mean(all_dwells)  if all_dwells  else 0,
        'std_dwell':     np.std(all_dwells)   if all_dwells  else 0,
        'mean_flight':   np.mean(all_flights) if all_flights else 0,
        'std_flight':    np.std(all_flights)  if all_flights else 0,
        'total_keys':    len([e for e in key_log if e['event'] == 'down']),
        'backspace_count': len([e for e in key_log
                                if e['event'] == 'down' and e['key'] == 'Backspace']),
        'total_time_ms': key_log[-1]['time'] - key_log[0]['time']
                         if len(key_log) > 1 else 0
    }

def build_feature_vector(f):
    row = {
        'mean_dwell':      f['mean_dwell'],   'std_dwell':    f['std_dwell'],
        'mean_flight':     f['mean_flight'],  'std_flight':   f['std_flight'],
        'total_time_ms':   f['total_time_ms'],'total_keys':   f['total_keys'],
        'backspace_count': f['backspace_count'],
        'backspace_rate':  f['backspace_count'] / max(f['total_keys'], 1),
        'typing_speed':    f['total_keys'] / max(f['total_time_ms'], 1) * 1000,
    }
    for key in list(set('thequickbrownfoxjumpsoverthelazydog')):
        row[f'dwell_{key}'] = f['dwell_per_key'].get(key, np.nan)
    digraph_lookup = {}
    for ft in f['flight_times']:
        if ft['from'] and ft['to']:
            digraph_lookup.setdefault(ft['from']+ft['to'], []).append(ft['time'])
    for d in ['th','he','er','qu','ow','br','fo','ox',
              'ju','um','ov','la','az','zy','do','og']:
        row[f'digraph_{d}'] = np.nanmean(digraph_lookup.get(d, [np.nan]))
    return list(row.values())

# ── Retrain pipeline ──────────────────────────────────
def retrain_models():
    all_data, rows = [], []
    for f in os.listdir(DATA_DIR):
        if f.endswith('.json'):
            with open(os.path.join(DATA_DIR, f)) as fp:
                all_data.append(json.load(fp))

    for person in all_data:
        for attempt in person['attempts']:
            feat = extract_features(attempt['raw_log'])
            row  = build_feature_vector(feat)
            row.append(person['username'])
            rows.append(row)

    cols = [f'f{i}' for i in range(len(rows[0])-1)] + ['user']
    df   = pd.DataFrame(rows, columns=cols)

    X = df.drop('user', axis=1).values.astype(float)
    y = df['user'].values

    # Fill NaNs
    col_medians = np.nanmedian(X, axis=0)
    for i in range(X.shape[1]):
        X[np.isnan(X[:, i]), i] = col_medians[i]

    le      = LabelEncoder()
    y_enc   = le.fit_transform(y)
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)

    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    rf  = RandomForestClassifier(n_estimators=200, random_state=42)
    svm.fit(X_sc, y_enc)
    rf.fit(X_sc, y_enc)

    for name, obj in [('svm_model', svm), ('rf_model', rf),
                      ('scaler', scaler), ('label_encoder', le)]:
        with open(os.path.join(MODELS_DIR, f'{name}.pkl'), 'wb') as fp:
            pickle.dump(obj, fp)

    st.cache_resource.clear()
    return le.classes_.tolist()

# ── Load models ───────────────────────────────────────
@st.cache_resource
def load_models():
    with open(os.path.join(MODELS_DIR, 'svm_model.pkl'),    'rb') as f: svm    = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'rf_model.pkl'),     'rb') as f: rf     = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'scaler.pkl'),       'rb') as f: scaler = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'label_encoder.pkl'),'rb') as f: le     = pickle.load(f)
    return svm, rf, scaler, le

def get_enrolled_users():
    users = []
    for f in os.listdir(DATA_DIR):
        if f.endswith('.json'):
            with open(os.path.join(DATA_DIR, f)) as fp:
                d = json.load(fp)
                users.append({
                    'name':     d['username'],
                    'attempts': len(d['attempts']),
                    'file':     f
                })
    return users

# ── Styling ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
* { box-sizing: border-box; }
[data-testid="stAppViewContainer"] {
    background: #080810;
    background-image:
        radial-gradient(ellipse at 20% 20%, rgba(99,102,241,0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(16,185,129,0.06) 0%, transparent 50%);
}
[data-testid="stHeader"] { background: transparent; }
.main .block-container { padding-top: 2rem; max-width: 780px; }
h1,h2,h3 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.02em; }
p,div,span,label { font-family: 'Space Mono', monospace !important; font-size: 13px; }

.hero { text-align: center; padding: 40px 0 24px; }
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.4);
    color: #818cf8;
    font-family: 'Space Mono', monospace;
    font-size: 11px; letter-spacing: 0.15em;
    padding: 6px 16px; border-radius: 100px;
    margin-bottom: 20px; text-transform: uppercase;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 52px; font-weight: 800;
    color: #f8fafc; line-height: 1.1; margin: 0 0 8px;
}
.hero-title span {
    background: linear-gradient(135deg, #6366f1, #10b981);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; font-family: 'Syne', sans-serif;
}
.hero-sub { color: #64748b; font-size: 13px; font-family: 'Space Mono', monospace; margin: 0; }

.stats-bar {
    display: flex; gap: 1px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px; overflow: hidden; margin: 24px 0;
}
.stat-item { flex: 1; padding: 18px 12px; text-align: center; background: rgba(8,8,16,0.8); }
.stat-num  { font-family: 'Syne', sans-serif; font-size: 22px; font-weight: 800; color: #f8fafc; display: block; }
.stat-num.green  { color: #10b981; }
.stat-num.indigo { color: #818cf8; }
.stat-label { font-family: 'Space Mono', monospace; font-size: 10px; color: #475569; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 4px; display: block; }

.result-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 20px 0; }
.result-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px; padding: 24px; position: relative; overflow: hidden;
}
.result-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #6366f1, #10b981);
}
.card-model { font-family: 'Space Mono', monospace; font-size: 10px; color: #475569; text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 12px; }
.card-name  { font-family: 'Syne', sans-serif; font-size: 30px; font-weight: 800; color: #f8fafc; margin: 0 0 6px; }
.card-conf  { font-family: 'Space Mono', monospace; font-size: 12px; color: #10b981; }
.conf-bar   { height: 3px; background: rgba(255,255,255,0.06); border-radius: 100px; margin-top: 10px; overflow: hidden; }
.conf-fill  { height: 100%; background: linear-gradient(90deg, #6366f1, #10b981); border-radius: 100px; }

.agree-banner {
    background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.25);
    border-radius: 12px; padding: 14px 20px; color: #10b981;
    font-family: 'Space Mono', monospace; font-size: 12px; text-align: center; margin: 8px 0 20px;
}
.disagree-banner {
    background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.25);
    border-radius: 12px; padding: 14px 20px; color: #f59e0b;
    font-family: 'Space Mono', monospace; font-size: 12px; text-align: center; margin: 8px 0 20px;
}

.metrics-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin: 20px 0; }
.metric-card {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px; padding: 16px; text-align: center;
}
.metric-val { font-family: 'Syne', sans-serif; font-size: 20px; font-weight: 700; color: #f8fafc; display: block; }
.metric-lbl { font-family: 'Space Mono', monospace; font-size: 10px; color: #475569; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 4px; display: block; }

.section-header { font-family: 'Syne', sans-serif; font-size: 18px; font-weight: 700; color: #f8fafc; margin: 28px 0 14px; }
.divider { height: 1px; background: rgba(255,255,255,0.05); margin: 28px 0; }

.enroll-step {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px; padding: 24px; margin-bottom: 16px;
}
.enroll-step-num {
    font-family: 'Syne', sans-serif; font-size: 11px; font-weight: 700;
    color: #6366f1; text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 8px;
}
.enroll-step-title { font-family: 'Syne', sans-serif; font-size: 18px; font-weight: 700; color: #f8fafc; margin-bottom: 4px; }
.enroll-step-desc  { font-family: 'Space Mono', monospace; font-size: 11px; color: #475569; }

.success-box {
    background: rgba(16,185,129,0.06); border: 1px solid rgba(16,185,129,0.3);
    border-radius: 20px; padding: 40px; text-align: center; margin: 20px 0;
}
.success-icon  { font-size: 48px; margin-bottom: 16px; }
.success-title { font-family: 'Syne', sans-serif; font-size: 28px; font-weight: 800; color: #10b981; margin-bottom: 8px; }
.success-sub   { font-family: 'Space Mono', monospace; font-size: 12px; color: #475569; }

.user-card {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px; padding: 16px 20px;
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 10px;
}
.user-name   { font-family: 'Syne', sans-serif; font-size: 16px; font-weight: 700; color: #f8fafc; }
.user-meta   { font-family: 'Space Mono', monospace; font-size: 10px; color: #475569; margin-top: 2px; }
.user-badge  {
    background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.3);
    color: #818cf8; font-family: 'Space Mono', monospace;
    font-size: 10px; padding: 4px 10px; border-radius: 100px;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🧬 Behavioral Biometrics</div>
    <h1 class="hero-title">Key<span>DNA</span></h1>
    <p class="hero-sub">identifying humans from typing rhythm alone</p>
</div>
""", unsafe_allow_html=True)

# ── Dynamic stats bar ─────────────────────────────────
enrolled = get_enrolled_users()
n_users  = len(enrolled)
n_samples = n_users * 15

st.markdown(f"""
<div class="stats-bar">
    <div class="stat-item">
        <span class="stat-num indigo">{n_users}</span>
        <span class="stat-label">Enrolled Users</span>
    </div>
    <div class="stat-item">
        <span class="stat-num">51</span>
        <span class="stat-label">Features</span>
    </div>
    <div class="stat-item">
        <span class="stat-num">{n_samples}</span>
        <span class="stat-label">Training Samples</span>
    </div>
    <div class="stat-item">
        <span class="stat-num green">98.67%</span>
        <span class="stat-label">SVM Accuracy</span>
    </div>
    <div class="stat-item">
        <span class="stat-num green">100%</span>
        <span class="stat-label">RF Accuracy</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔐  Authenticate", "➕  Enroll", "👥  Users"])

# ══════════════════════════════════════════════════════
# TAB 1 — AUTHENTICATE
# ══════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <span style="font-family:'Space Mono',monospace;font-size:11px;color:#475569;">
    Upload your keystroke JSON to identify yourself.
    New user? Head to the <strong style="color:#818cf8;">➕ Enroll</strong> tab first.
    </span>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    uploaded = st.file_uploader("", type="json",
                                 label_visibility="collapsed",
                                 key="auth_upload")

    if uploaded:
        data = json.load(uploaded)
        with st.spinner("Analyzing typing pattern..."):
            all_vectors, all_mean_dwells = [], []
            all_mean_flights, all_speeds, all_backspaces = [], [], []

            for attempt in data['attempts']:
                feat = extract_features(attempt['raw_log'])
                vec  = build_feature_vector(feat)
                all_vectors.append(vec)
                all_mean_dwells.append(feat['mean_dwell'])
                all_mean_flights.append(feat['mean_flight'])
                all_speeds.append(feat['total_keys'] / max(feat['total_time_ms'],1)*1000)
                all_backspaces.append(feat['backspace_count'])

            fv = np.nanmean(all_vectors, axis=0).tolist()
            fv = [0 if (isinstance(v,float) and np.isnan(v)) else v for v in fv]

            svm, rf, scaler, le = load_models()
            X_sc = scaler.transform(np.array(fv).reshape(1,-1))

            svm_pred  = svm.predict(X_sc)[0]
            svm_proba = svm.predict_proba(X_sc)[0]
            rf_pred   = rf.predict(X_sc)[0]
            rf_proba  = rf.predict_proba(X_sc)[0]

            svm_name = le.inverse_transform([svm_pred])[0]
            rf_name  = le.inverse_transform([rf_pred])[0]
            svm_conf = svm_proba.max() * 100
            rf_conf  = rf_proba.max()  * 100

        st.markdown('<div class="section-header">🎯 Identification Result</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="result-grid">
            <div class="result-card">
                <div class="card-model">SVM · RBF Kernel</div>
                <div class="card-name">{svm_name}</div>
                <div class="card-conf">↑ {svm_conf:.1f}% confidence</div>
                <div class="conf-bar">
                    <div class="conf-fill" style="width:{svm_conf}%"></div>
                </div>
            </div>
            <div class="result-card">
                <div class="card-model">Random Forest · 200 trees</div>
                <div class="card-name">{rf_name}</div>
                <div class="card-conf">↑ {rf_conf:.1f}% confidence</div>
                <div class="conf-bar">
                    <div class="conf-fill" style="width:{rf_conf}%"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if svm_name == rf_name:
            st.markdown(f"""
            <div class="agree-banner">
                ✓ &nbsp; Both models agree — identity confirmed as
                <strong>{svm_name}</strong>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="disagree-banner">
                ⚡ Models disagree — SVM: <strong>{svm_name}</strong>
                · RF: <strong>{rf_name}</strong>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">⌨️ Typing Signature</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <span class="metric-val">{np.mean(all_mean_dwells):.0f}ms</span>
                <span class="metric-lbl">Mean Dwell</span>
            </div>
            <div class="metric-card">
                <span class="metric-val">{np.mean(all_mean_flights):.0f}ms</span>
                <span class="metric-lbl">Mean Flight</span>
            </div>
            <div class="metric-card">
                <span class="metric-val">{np.mean(all_speeds):.1f}</span>
                <span class="metric-lbl">Keys / sec</span>
            </div>
            <div class="metric-card">
                <span class="metric-val">{np.mean(all_backspaces):.1f}</span>
                <span class="metric-lbl">Avg Backspaces</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">📊 Model Confidence</div>',
                    unsafe_allow_html=True)
        proba_df = pd.DataFrame({
            'User': le.classes_,
            'SVM': svm_proba * 100,
            'Random Forest': rf_proba * 100
        })
        fig = px.bar(
            proba_df.melt(id_vars='User', var_name='Model', value_name='Confidence %'),
            x='User', y='Confidence %', color='Model', barmode='group',
            color_discrete_map={'SVM': '#6366f1', 'Random Forest': '#10b981'},
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Space Mono', color='#64748b', size=11),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            xaxis=dict(showgrid=False, title=None),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', title=None),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.markdown("""
        <div style="text-align:center;padding:70px 40px;
                    border:1.5px dashed rgba(99,102,241,0.2);
                    border-radius:24px;margin:20px 0;
                    background:rgba(99,102,241,0.02);">
            <div style="font-size:44px;margin-bottom:14px;">🔐</div>
            <div style="font-family:'Syne',sans-serif;font-size:18px;
                        font-weight:700;color:#334155;margin-bottom:6px;">
                Upload a keystroke file to authenticate
            </div>
            <div style="font-family:'Space Mono',monospace;font-size:11px;color:#1e293b;">
                collect · type 15 times · download JSON · upload here
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 2 — ENROLL
# ══════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Check if we just finished enrolling
    if st.session_state.get('enrolled_name'):
        name = st.session_state['enrolled_name']
        st.markdown(f"""
        <div class="success-box">
            <div class="success-icon">🧬</div>
            <div class="success-title">Welcome, {name}!</div>
            <div class="success-sub">
                Your typing DNA has been captured.<br>
                The model has been retrained with your pattern.<br><br>
                Head to the <strong>Authenticate</strong> tab to verify yourself.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("➕ Enroll Another Person", use_container_width=True):
            st.session_state['enrolled_name'] = None
            st.rerun()

    else:
        # Step 1 — Name
        st.markdown("""
        <div class="enroll-step">
            <div class="enroll-step-num">Step 01</div>
            <div class="enroll-step-title">Enter Your Name</div>
            <div class="enroll-step-desc">
                This will be your identity label in the model.
            </div>
        </div>
        """, unsafe_allow_html=True)

        enroll_name = st.text_input("", placeholder="e.g. Jash",
                                     label_visibility="collapsed",
                                     key="enroll_name_input")

        # Check duplicate
        existing_names = [u['name'].lower() for u in enrolled]
        name_taken = enroll_name.strip().lower() in existing_names if enroll_name.strip() else False

        if name_taken:
            st.markdown("""
            <div style="color:#f59e0b;font-family:'Space Mono',monospace;
                        font-size:11px;margin-top:6px;">
                ⚠️ This name is already enrolled.
                Choose a different name or delete the existing one first.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Step 2 — Type
        st.markdown(f"""
        <div class="enroll-step">
            <div class="enroll-step-num">Step 02</div>
            <div class="enroll-step-title">Type the Sentence 15 Times</div>
            <div class="enroll-step-desc">
                Type exactly as you normally would. Don't try to type differently.
                Press Submit after each attempt.
            </div>
        </div>
        <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);
                    border-radius:12px;padding:14px 18px;margin-bottom:16px;">
            <span style="font-family:'Space Mono',monospace;font-size:12px;color:#818cf8;">
                📝 {TARGET_SENTENCE}
            </span>
        </div>
        """, unsafe_allow_html=True)

        typing_component = f"""
        <style>
            body {{ background:transparent; font-family:'Space Mono',monospace; }}
            #attempt-counter {{
                color:#475569; font-size:12px; margin-bottom:10px;
                font-family:'Space Mono',monospace;
            }}
            #progress-bar {{
                height:3px; background:rgba(255,255,255,0.06);
                border-radius:100px; margin-bottom:14px; overflow:hidden;
            }}
            #progress-fill {{
                height:100%; width:0%;
                background:linear-gradient(90deg,#6366f1,#10b981);
                border-radius:100px; transition:width 0.4s ease;
            }}
            #typing-area {{
                width:100%; padding:12px;
                background:#1a1a1a; color:white;
                font-size:15px; border:2px solid #6366f1;
                border-radius:10px; font-family:'Space Mono',monospace;
                box-sizing:border-box; resize:none;
            }}
            #typing-area:focus {{ outline:none; border-color:#10b981; }}
            #submit-btn {{
                margin-top:10px; padding:10px 24px;
                background:linear-gradient(135deg,#6366f1,#10b981);
                color:#fff; border:none; border-radius:8px;
                font-size:14px; font-weight:700;
                cursor:pointer; font-family:'Space Mono',monospace;
            }}
            #submit-btn:disabled {{
                background:#1e293b; color:#334155; cursor:not-allowed;
            }}
            #status {{ margin-top:8px; font-size:11px; color:#10b981; }}
            #output {{ display:none; }}
        </style>

        <div id="progress-bar"><div id="progress-fill"></div></div>
        <div id="attempt-counter">Attempt 1 of 15</div>
        <textarea id="typing-area" rows="2"
            autocomplete="off" autocorrect="off"
            autocapitalize="off" spellcheck="false"
            placeholder="Type the sentence here..."></textarea>
        <br>
        <button id="submit-btn" onclick="submitAttempt()">
            Submit Attempt →
        </button>
        <div id="status"></div>
        <textarea id="output"></textarea>

        <script>
        const TOTAL = 15;
        let attempt = 1, allAttempts = [], keyLog = [], isRecording = true;
        const area = document.getElementById('typing-area');
        area.focus();

        area.addEventListener('keydown', e => {{
            if (!isRecording) return;
            keyLog.push({{ key: e.key, event: 'down', time: performance.now() }});
        }});
        area.addEventListener('keyup', e => {{
            if (!isRecording) return;
            keyLog.push({{ key: e.key, event: 'up', time: performance.now() }});
        }});

        function submitAttempt() {{
            if (area.value.trim().length < 10) {{
                document.getElementById('status').textContent =
                    '⚠ Type the full sentence first.';
                return;
            }}
            isRecording = false;
            allAttempts.push({{ attempt, raw_log: keyLog }});

            if (attempt < TOTAL) {{
                attempt++;
                keyLog = []; area.value = ''; isRecording = true; area.focus();
                document.getElementById('attempt-counter').textContent =
                    `Attempt ${{attempt}} of ${{TOTAL}}`;
                document.getElementById('progress-fill').style.width =
                    `${{((attempt-1)/TOTAL)*100}}%`;
                document.getElementById('status').textContent =
                    `✓ Attempt ${{attempt-1}} saved`;
            }} else {{
                document.getElementById('progress-fill').style.width = '100%';
                document.getElementById('attempt-counter').textContent =
                    '✓ All 15 attempts captured!';
                document.getElementById('submit-btn').disabled = true;
                document.getElementById('status').textContent = '';

                const payload = JSON.stringify({{
                    username: 'PLACEHOLDER',
                    attempts: allAttempts
                }});
                document.getElementById('output').value = payload;

                navigator.clipboard.writeText(payload).then(() => {{
                    document.getElementById('status').textContent =
                        '✅ Data copied to clipboard! Paste below and click Enroll.';
                }}).catch(() => {{
                    document.getElementById('status').textContent =
                        '✅ Done! Copy the data from the hidden box and paste below.';
                }});
            }}
        }}
        </script>
        """

        components.html(typing_component, height=220)

        st.markdown("<br>", unsafe_allow_html=True)

        # Step 3 — Paste + Enroll
        st.markdown("""
        <div class="enroll-step">
            <div class="enroll-step-num">Step 03</div>
            <div class="enroll-step-title">Paste Data & Enroll</div>
            <div class="enroll-step-desc">
                After 15 attempts, data is auto-copied.
                Paste it below and click Enroll.
            </div>
        </div>
        """, unsafe_allow_html=True)

        pasted = st.text_area("", placeholder="Paste keystroke data here (Ctrl+V)...",
                               height=80, label_visibility="collapsed",
                               key="enroll_paste")

        can_enroll = (
            bool(enroll_name.strip()) and
            not name_taken and
            bool(pasted.strip())
        )

        if st.button("🧬 Enroll & Retrain Model",
                     use_container_width=True,
                     disabled=not can_enroll,
                     type="primary"):
            try:
                raw_data = json.loads(pasted.strip())

                # Replace placeholder name with actual name
                raw_data['username'] = enroll_name.strip()

                # Save JSON to data/raw
                filename = f"{enroll_name.strip()}_keystrokes.json"
                filepath = os.path.join(DATA_DIR, filename)
                with open(filepath, 'w') as fp:
                    json.dump(raw_data, fp, indent=2)

                # Retrain
                with st.spinner(f"Training model with {enroll_name.strip()}'s data..."):
                    retrain_models()

                st.session_state['enrolled_name'] = enroll_name.strip()
                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}. Please make sure you pasted the data correctly.")

# ══════════════════════════════════════════════════════
# TAB 3 — USERS
# ══════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-header">👥 {n_users} Enrolled Users</div>',
                unsafe_allow_html=True)

    for user in enrolled:
        st.markdown(f"""
        <div class="user-card">
            <div>
                <div class="user-name">{user['name']}</div>
                <div class="user-meta">{user['attempts']} attempts · {user['file']}</div>
            </div>
            <div class="user-badge">enrolled</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:11px;color:#1e293b;
                text-align:center;padding:16px;">
        KeyDNA · built by
        <a href="https://github.com/JashVakharia34"
           style="color:#334155;">Jash Vakharia</a>
        · 51 features · SVM 98.67% · RF 100%
    </div>
    """, unsafe_allow_html=True)