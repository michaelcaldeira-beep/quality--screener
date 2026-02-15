import io
import yaml
import numpy as np
import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

from engine import compute_actions

st.set_page_config(page_title="Quality Screener", layout="wide")

st.title("Quality Stock Screener")
st.caption("Google Sheets → Perfil → BUY / SELL Engine")

# ----------------------------
# Google Sheets setup (via Streamlit Secrets)
# ----------------------------
SHEETS_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

def get_gspread_client():
    info = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(info, scopes=SHEETS_SCOPES)
    return gspread.authorize(creds)

def sheet_to_df(gc, spreadsheet_id: str, worksheet_name: str):
    sh = gc.open_by_key(spreadsheet_id)
    ws = sh.worksheet(worksheet_name)
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()
    headers = values[0]
    rows = values[1:]
    return pd.DataFrame(rows, columns=headers)

# ----------------------------
# Load profiles
# ----------------------------
@st.cache_data
def load_profiles():
    with open("profiles.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

profiles = load_profiles()

# ----------------------------
# Sidebar configuration
# ----------------------------
st.sidebar.header("Perfil do investidor")

profile_names = list(profiles.keys())
profile = st.sidebar.selectbox("Perfil base", profile_names)

base_cfg = dict(profiles[profile])

st.sidebar.markdown("---")
st.sidebar.subheader("Risco personalizado")

risk = st.sidebar.slider("Risco (0 = conservador, 100 = agressivo)", 0, 100, 50)

score_min = int(np.interp(risk, [0, 50, 100], [85, 70, 55]))
dd_buy = float(np.interp(risk, [0, 50, 100], [-0.35, -0.20, -0.10]))
dd_strong = float(np.interp(risk, [0, 50, 100], [-0.45, -0.30, -0.20]))

SCORE_BUY_MIN = st.sidebar.number_input("Score mínimo", 0, 100, score_min)
DD_BUY = st.sidebar.number_input("Drawdown BUY", value=dd_buy, format="%.2f")
DD_STRONG = st.sidebar.number_input("Drawdown STRONG BUY", value=dd_strong, format="%.2f")

cfg = {
    "SCORE_BUY_MIN": float(SCORE_BUY_MIN),
    "DD_BUY": float(DD_BUY),
    "DD_STRONG": float(DD_STRONG),
    "REQUIRE_PASS_DEBT": base_cfg.get("REQUIRE_PASS_DEBT", True),
    "REQUIRE_PASS_INTEREST": base_cfg.get("REQUIRE_PASS_INTEREST", True),
    "REQUIRE_PASS_FCF": base_cfg.get("REQUIRE_PASS_FCF", True),
    "REQUIRE_PASS_ROIC": base_cfg.get("REQUIRE_PASS_ROIC", False),
    "REQUIRE_PASS_PAYOUT": base_cfg.get("REQUIRE_PASS_PAYOUT", False),
    "ALLOW_SPECULATIVE": base_cfg.get("ALLOW_SPECULATIVE", True),
}

st.sidebar.markdown("---")
st.sidebar.code(cfg, language="json")

# ----------------------------
# Google Sheets input
# ----------------------------
st.subheader("Google Sheets Source")

spreadsheet_id = st.text_input("Spreadsheet ID")
worksheet_name = st.text_input("Worksheet name", value="Sheet1")

if not spreadsheet_id:
    st.info("Coloca o Spreadsheet ID para continuar.")
    st.stop()

try:
    gc = get_gspread_client()
    df = sheet_to_df(gc, spreadsheet_id, worksheet_name)
except Exception as e:
    st.error(f"Erro ao ligar ao Google Sheets: {e}")
    st.stop()

if df.empty:
    st.warning("Sheet vazia ou sem dados.")
    st.stop()

df.columns = [c.strip() for c in df.columns]

# ----------------------------
# Compute actions
# ----------------------------
with st.spinner("A calcular decisões..."):
    out = compute_actions(df, cfg)

st.success("Cálculo concluído.")

# ----------------------------
# Display
# ----------------------------
st.subheader("Resultados")

st.dataframe(out, use_container_width=True, height=500)

# ----------------------------
# Download Excel
# ----------------------------
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    out.to_excel(writer, index=False)

st.download_button(
    label="Download Excel",
    data=buffer.getvalue(),
    file_name="screener_output.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
  )
