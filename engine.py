import numpy as np
import pandas as pd

EXPECTED_COLUMNS = [
    "ticker","name","sector","price",
    "drawdown_from_52w_high","drawdown_trigger",
    "fcf_last_year","fcf_positive_last_n_years","fcf_trend_slope","pass_fcf",
    "roic_proxy","pass_roic",
    "payout_to_fcf","pass_payout",
    "net_debt_to_ebitda","interest_cover","pass_debt","pass_interest_cover",
    "QUALITY_PASS","ENTRY_PERMITTED","BUY_CANDIDATE","score"
]

BOOL_COLS = [
    "pass_fcf","pass_roic","pass_payout","pass_debt","pass_interest_cover",
    "QUALITY_PASS","ENTRY_PERMITTED","BUY_CANDIDATE"
]

NUM_COLS = [
    "price","drawdown_from_52w_high","fcf_last_year","fcf_positive_last_n_years",
    "fcf_trend_slope","roic_proxy","payout_to_fcf","net_debt_to_ebitda",
    "interest_cover","score"
]

def _to_bool(x):
    if pd.isna(x):
        return False
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    s = str(x).strip().lower()
    return s in {"true","1","yes","y","sim","s","ok"}

def _to_num(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().replace("%","").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan

def _normalize_drawdown(x):
    """
    Suporta:
      -0.25 (fração)  -> -0.25
      -25  (percent) -> -0.25
    """
    if pd.isna(x):
        return np.nan
    if x <= -1:
        return x / 100.0
    return x

def _safe_get(r, col, default=None):
    return r[col] if col in r.index else default

def _passes_required(r, cfg):
    """
    Regras ligadas/desligadas via config:
      REQUIRE_PASS_DEBT, REQUIRE_PASS_INTEREST, REQUIRE_PASS_FCF, REQUIRE_PASS_ROIC, REQUIRE_PASS_PAYOUT
    """
    failed = []

    if cfg.get("REQUIRE_PASS_DEBT", True):
        if not bool(_safe_get(r, "pass_debt", False)):
            failed.append("pass_debt")

    if cfg.get("REQUIRE_PASS_INTEREST", True):
        if not bool(_safe_get(r, "pass_interest_cover", False)):
            failed.append("pass_interest_cover")

    if cfg.get("REQUIRE_PASS_FCF", True):
        if not bool(_safe_get(r, "pass_fcf", False)):
            failed.append("pass_fcf")

    if cfg.get("REQUIRE_PASS_ROIC", False):
        if not bool(_safe_get(r, "pass_roic", False)):
            failed.append("pass_roic")

    if cfg.get("REQUIRE_PASS_PAYOUT", False):
        if not bool(_safe_get(r, "pass_payout", False)):
            failed.append("pass_payout")

    return failed

def compute_actions(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Entrada: df com as tuas colunas
    Saída: df com ACTION, BUY_SIGNAL, SELL_SIGNAL, REASON_BUY, REASON_SELL, FAILED_CHECKS
    """

    out = df.copy()
    out.columns = [c.strip() for c in out.columns]

    # Normaliza tickers/strings
    for c in ["ticker","name","sector","drawdown_trigger"]:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()

    # Converte tipos
    for c in NUM_COLS:
        if c in out.columns:
            out[c] = out[c].apply(_to_num)

    for c in BOOL_COLS:
        if c in out.columns:
            out[c] = out[c].apply(_to_bool)

    # Drawdown normalizado (fração negativa)
    if "drawdown_from_52w_high" in out.columns:
        out["dd_norm"] = out["drawdown_from_52w_high"].apply(_normalize_drawdown)
    else:
        out["dd_norm"] = np.nan

    # Thresholds
    SCORE_BUY_MIN = float(cfg.get("SCORE_BUY_MIN", 70))
    DD_BUY = float(cfg.get("DD_BUY", -0.20))
    DD_STRONG = float(cfg.get("DD_STRONG", -0.30))
    ALLOW_SPEC = bool(cfg.get("ALLOW_SPECULATIVE", True))

    def sell_signal(r):
        # Breaks estruturais
        if not bool(_safe_get(r, "QUALITY_PASS", True)):
            return True, "QUALITY_PASS=FALSE"
        required_failed = _passes_required(r, cfg)
        if required_failed:
            return True, " / ".join(required_failed)
        return False, ""

    def buy_gate(r):
        if not bool(_safe_get(r, "ENTRY_PERMITTED", False)):
            return False, "ENTRY_PERMITTED=FALSE"
        if not bool(_safe_get(r, "QUALITY_PASS", False)):
            return False, "QUALITY_PASS=FALSE"

        buy_candidate = bool(_safe_get(r, "BUY_CANDIDATE", False))
        score = _safe_get(r, "score", np.nan)
        score_ok = (pd.notna(score) and float(score) >= SCORE_BUY_MIN)

        if not (buy_candidate or score_ok):
            return False, "NO_BUY_CANDIDATE_OR_SCORE"

        dd = _safe_get(r, "dd_norm", np.nan)
        if pd.notna(dd) and float(dd) > DD_BUY:
            return False, f"INSUFFICIENT_DRAWDOWN({float(dd):.0%})"

        required_failed = _passes_required(r, cfg)
        if required_failed:
            if ALLOW_SPEC:
                return True, "ENTRY_OK_WITH_FLAGS:" + ",".join(required_failed)
            return False, "FAILED_REQUIRED:" + ",".join(required_failed)

        return True, "ENTRY_OK"

    def decide(r):
        s, rs = sell_signal(r)
        if s:
            return "REVIEW SELL", "", rs, rs

        b, rb = buy_gate(r)
        if b:
            dd = _safe_get(r, "dd_norm", np.nan)
            flags = rb.replace("ENTRY_OK_WITH_FLAGS:", "") if rb.startswith("ENTRY_OK_WITH_FLAGS:") else ""
            failed = flags.replace(",", " / ") if flags else ""

            if pd.notna(dd) and float(dd) <= DD_STRONG and not failed:
                return "STRONG BUY", rb, "", ""
            if failed and ALLOW_SPEC:
                return "SPECULATIVE / WATCH", rb, "", failed
            return "BUY", rb, "", ""

        watch_flags = []
        if bool(_safe_get(r, "ENTRY_PERMITTED", False)):
            for col in ["pass_debt","pass_interest_cover","pass_fcf","pass_payout","pass_roic"]:
                if col in r.index and (bool(r[col]) is False):
                    watch_flags.append(col)

        if watch_flags:
            return "WATCH", "", " / ".join(watch_flags), " / ".join(watch_flags)

        return "HOLD", "", "", ""

    decided = out.apply(decide, axis=1, result_type="expand")
    decided.columns = ["ACTION","REASON_BUY","REASON_SELL","FAILED_CHECKS"]
    out = pd.concat([out, decided], axis=1)

    out["BUY_SIGNAL"] = out["ACTION"].isin(["BUY","STRONG BUY"])
    out["SELL_SIGNAL"] = out["ACTION"].eq("REVIEW SELL")

    if "score" in out.columns:
        out = out.sort_values(by=["BUY_SIGNAL","SELL_SIGNAL","score"], ascending=[False, False, False])
    else:
        out = out.sort_values(by=["BUY_SIGNAL","SELL_SIGNAL"], ascending=[False, False])

    return out
