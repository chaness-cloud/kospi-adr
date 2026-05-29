# -*- coding: utf-8 -*-
"""
코스피 / 코스닥 ADR (등락비율) 웹 대시보드
실행: streamlit run adr_app.py
"""

import os
import pickle
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import FinanceDataReader as fdr

warnings.filterwarnings("ignore")

# ── 페이지 설정 ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KOSPI / KOSDAQ ADR 분석",
    page_icon="📈",
    layout="wide",
)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".adr_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
YEARS = 5


# ── 날짜 ────────────────────────────────────────────────────────────────────
def _today() -> str:
    return datetime.today().strftime("%Y-%m-%d")


def _start_date() -> str:
    return (datetime.today() - timedelta(days=365 * YEARS + 90)).strftime("%Y-%m-%d")


# ── 데이터 로드 (캐시) ────────────────────────────────────────────────────────
def get_tickers(market: str) -> list[str]:
    cache_file = os.path.join(CACHE_DIR, f"tickers_{market}.pkl")
    today = _today()
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            saved = pickle.load(f)
        if saved.get("date") == today:
            return saved["tickers"]

    # ① 번들된 CSV 파일 (가장 안정적 — KRX 접근 불필요)
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"{market.lower()}_tickers.csv")
    if os.path.exists(csv_path):
        try:
            tickers = pd.read_csv(csv_path)["Code"].astype(str).str.zfill(6).tolist()
            if tickers:
                with open(cache_file, "wb") as f:
                    pickle.dump({"date": today, "tickers": tickers}, f)
                return tickers
        except Exception:
            pass

    # ② KRX-DESC (로컬 환경용 fallback)
    try:
        desc = fdr.StockListing('KRX-DESC')
        listing = desc[desc['Market'] == market]
        tickers = listing["Code"].tolist()
        if tickers:
            with open(cache_file, "wb") as f:
                pickle.dump({"date": today, "tickers": tickers}, f)
            return tickers
    except Exception:
        pass

    # ③ fdr.StockListing(market) 최후 fallback
    try:
        listing = fdr.StockListing(market)
        tickers = listing["Code"].tolist()
        with open(cache_file, "wb") as f:
            pickle.dump({"date": today, "tickers": tickers}, f)
        return tickers
    except Exception:
        return []


def _download_closes(market: str, start: str, end: str, progress_bar=None) -> pd.DataFrame:
    tickers = get_tickers(market)
    frames = {}
    total = len(tickers)
    for i, ticker in enumerate(tickers, 1):
        if progress_bar is not None:
            progress_bar.progress(i / total, text=f"[{market}] {i}/{total} 종목 다운로드 중...")
        try:
            df = fdr.DataReader(ticker, start, end)
            if not df.empty and "Close" in df.columns:
                frames[ticker] = df["Close"].rename(ticker)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    result = pd.DataFrame(frames)
    result.index = pd.to_datetime(result.index, errors="coerce")
    result = result[result.index.notna()].sort_index()
    return result


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """인덱스가 DatetimeIndex가 아니면 강제 변환."""
    if df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[df.index.notna()].sort_index()
    return df


def _load_close_prices_inner(market: str, force_refresh: bool = False,
                              progress_bar=None) -> pd.DataFrame:
    """실제 데이터 로드 로직 — st.* 호출 없음 (캐시 함수 안에서 호출 불가)."""
    cache_file = os.path.join(CACHE_DIR, f"close_{market}.pkl")
    start = _start_date()
    end = _today()

    existing = None
    if os.path.exists(cache_file) and not force_refresh:
        try:
            with open(cache_file, "rb") as f:
                existing = pickle.load(f)
            existing = _ensure_datetime_index(existing)
        except Exception:
            existing = None

    if existing is not None and not existing.empty:
        cached_end = existing.index.max().strftime("%Y-%m-%d")
        if cached_end >= end:
            mask = existing.index >= pd.to_datetime(start)
            return existing.loc[mask]

        fetch_start = (existing.index.max() + timedelta(days=1)).strftime("%Y-%m-%d")
        new_df = _download_closes(market, fetch_start, end, progress_bar)
        new_df = _ensure_datetime_index(new_df)
        if not new_df.empty:
            combined = pd.concat([existing, new_df])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        else:
            combined = existing
    else:
        combined = _download_closes(market, start, end, progress_bar)
        combined = _ensure_datetime_index(combined)

    if not combined.empty:
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(combined, f)
        except Exception:
            pass

    if combined.empty:
        return combined
    mask = combined.index >= pd.to_datetime(start)
    return combined.loc[mask]


@st.cache_data(show_spinner=False, ttl=3600)
def _load_close_prices_cached(market: str, today_key: str) -> pd.DataFrame:
    """오늘 날짜를 키로 써서 당일 RAM 캐시 유지. st.* 호출 없음."""
    return _load_close_prices_inner(market)


def load_close_prices(market: str, force_refresh: bool = False,
                      progress_bar=None) -> pd.DataFrame:
    """RAM 캐시(당일) → pickle(디스크) → 다운로드 순으로 데이터 반환."""
    if force_refresh:
        st.cache_data.clear()
    return _load_close_prices_cached(market, _today())


@st.cache_data(show_spinner=False)
def load_index(market: str) -> pd.Series | None:
    sym = {"KOSPI": "KS11", "KOSDAQ": "KQ11"}.get(market)
    if not sym:
        return None
    try:
        df = fdr.DataReader(sym, _start_date(), _today())
        if not df.empty and "Close" in df.columns:
            return df["Close"].rename(market)
    except Exception:
        pass
    return None


# ── ADR 계산 ─────────────────────────────────────────────────────────────────
def calc_adr(close: pd.DataFrame, period: int) -> pd.Series:
    ma = close.rolling(period, min_periods=period).mean()
    above = (close > ma).sum(axis=1)
    below = (close < ma).sum(axis=1)
    adr = above / below.replace(0, np.nan)
    valid_start = ma.dropna(how="all").index.min()
    return adr.loc[adr.index >= valid_start].rename(f"ADR {period}일")


# ── 차트 ─────────────────────────────────────────────────────────────────────
COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0", "#00BCD4"]


def build_chart(close: pd.DataFrame, periods: list[int], market: str,
                index_s: pd.Series | None, date_from: datetime, date_to: datetime) -> go.Figure:
    mask = (close.index >= pd.to_datetime(date_from)) & (close.index <= pd.to_datetime(date_to))
    close_sliced = close.loc[mask]

    rows = len(periods) + (1 if index_s is not None else 0)
    subplot_titles = []
    if index_s is not None:
        subplot_titles.append(f"{market} 지수")
    for p in periods:
        subplot_titles.append(f"ADR {p}일")

    row_heights = [1.5] + [1.0] * len(periods) if index_s is not None else [1.0] * len(periods)
    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
        vertical_spacing=0.03,
    )

    row = 1

    # 지수 패널
    if index_s is not None:
        idx_mask = (index_s.index >= pd.to_datetime(date_from)) & (index_s.index <= pd.to_datetime(date_to))
        s = index_s.loc[idx_mask].dropna()
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            fill="tozeroy", fillcolor="rgba(96,125,139,0.15)",
            line=dict(color="#607D8B", width=1.2),
            name=market, hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.2f}<extra></extra>",
        ), row=row, col=1)
        fig.update_yaxes(tickformat=",.0f", row=row, col=1)
        row += 1

    # ADR 패널
    for p, color in zip(periods, COLORS):
        adr = calc_adr(close_sliced, p).dropna()
        if adr.empty:
            row += 1
            continue

        latest = adr.iloc[-1]

        # 배경 색 (1 기준 위/아래)
        above_mask = adr >= 1
        below_mask = adr < 1

        for mask_b, fill_color in [(above_mask, color), (below_mask, "#F44336")]:
            seg = adr.where(mask_b)
            fig.add_trace(go.Scatter(
                x=seg.index, y=seg.values,
                fill="tonexty" if False else "tozeroy",
                fillcolor=f"rgba({_hex2rgb(fill_color)},0.15)",
                line=dict(color="rgba(0,0,0,0)", width=0),
                showlegend=False, hoverinfo="skip",
            ), row=row, col=1)

        fig.add_trace(go.Scatter(
            x=adr.index, y=adr.values,
            line=dict(color=color, width=1.3),
            name=f"ADR {p}일",
            hovertemplate="%{x|%Y-%m-%d}<br>ADR: %{y:.3f}<extra></extra>",
        ), row=row, col=1)

        # 기준선 1.0
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                      line_width=0.8, opacity=0.6, row=row, col=1)

        # 최신값 표시
        fig.add_annotation(
            x=adr.index[-1], y=latest,
            text=f"  {latest:.2f}",
            showarrow=False, font=dict(color=color, size=11),
            xanchor="left", row=row, col=1,
        )

        row += 1

    fig.update_layout(
        height=280 * rows,
        title=dict(text=f"{market} ADR (등락비율) — 이동평균선 대비", font=dict(size=16)),
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=60, r=60, t=80, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")

    return fig


def _hex2rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return ",".join(str(int(h[i:i+2], 16)) for i in (0, 2, 4))


# ── 통계 분포 분석 ────────────────────────────────────────────────────────────
def adr_stats(adr: pd.Series) -> dict:
    """현재 ADR의 역사적 분포상 위치 계산."""
    s = adr.dropna()
    current = s.iloc[-1]
    mean = s.mean()
    std = s.std()
    pct = float((s <= current).mean() * 100)          # 백분위수
    z = (current - mean) / std if std > 0 else 0.0    # Z-스코어

    # 극단 임계값 (상/하위 10%, 5%, 1%)
    q1, q5, q10 = s.quantile(0.01), s.quantile(0.05), s.quantile(0.10)
    q90, q95, q99 = s.quantile(0.90), s.quantile(0.95), s.quantile(0.99)

    if current <= q1:
        signal, signal_color = "극단 과매도 ⚡", "#D32F2F"
    elif current <= q5:
        signal, signal_color = "강한 과매도 🔴", "#F44336"
    elif current <= q10:
        signal, signal_color = "과매도 🟠", "#FF9800"
    elif current >= q99:
        signal, signal_color = "극단 과매수 ⚡", "#1565C0"
    elif current >= q95:
        signal, signal_color = "강한 과매수 🔵", "#2196F3"
    elif current >= q90:
        signal, signal_color = "과매수 🔷", "#64B5F6"
    else:
        signal, signal_color = "중립 ⚪", "#9E9E9E"

    return dict(
        current=current, mean=mean, std=std,
        pct=pct, z=z, signal=signal, signal_color=signal_color,
        q1=q1, q5=q5, q10=q10, q90=q90, q95=q95, q99=q99,
        history=s,
    )


def build_dist_chart(adr: pd.Series, period: int, color: str) -> go.Figure:
    """히스토그램 + KDE + 현재값 마커 + 분위수 경계선."""
    st_data = adr_stats(adr)
    s = st_data["history"]
    current = st_data["current"]

    # KDE (scipy 없이 numpy로 근사)
    hist_vals, bin_edges = np.histogram(s, bins=60, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Gaussian KDE 근사
    x_kde = np.linspace(s.min() * 0.8, s.max() * 1.2, 300)
    bw = 1.06 * s.std() * len(s) ** (-0.2)
    kde_y = np.array([
        np.mean(np.exp(-0.5 * ((x - s.values) / bw) ** 2) / (bw * np.sqrt(2 * np.pi)))
        for x in x_kde
    ])

    fig = go.Figure()

    # 히스토그램
    fig.add_trace(go.Bar(
        x=bin_centers, y=hist_vals,
        marker_color=f"rgba({_hex2rgb(color)},0.25)",
        marker_line_color=f"rgba({_hex2rgb(color)},0.5)",
        marker_line_width=0.5,
        name="빈도",
        hovertemplate="ADR: %{x:.3f}<br>밀도: %{y:.4f}<extra></extra>",
    ))

    # KDE 곡선
    fig.add_trace(go.Scatter(
        x=x_kde, y=kde_y,
        line=dict(color=color, width=2),
        name="분포 곡선",
        hoverinfo="skip",
    ))

    # 분위수 영역 색칠 (하위 10%, 상위 10%)
    for lo, hi, fc in [
        (s.min(), st_data["q10"], "rgba(255,152,0,0.12)"),
        (st_data["q90"], s.max(), "rgba(33,150,243,0.12)"),
        (s.min(), st_data["q5"], "rgba(244,67,54,0.15)"),
        (st_data["q95"], s.max(), "rgba(33,150,243,0.2)"),
    ]:
        fig.add_vrect(x0=lo, x1=hi, fillcolor=fc, layer="below", line_width=0)

    # 분위수 경계 수직선
    for qv, label, lc in [
        (st_data["q5"],  "하위 5%",  "#F44336"),
        (st_data["q10"], "하위 10%", "#FF9800"),
        (st_data["q90"], "상위 10%", "#64B5F6"),
        (st_data["q95"], "상위 5%",  "#2196F3"),
    ]:
        fig.add_vline(x=qv, line_dash="dot", line_color=lc, line_width=1.2,
                      annotation_text=label, annotation_position="top",
                      annotation_font=dict(size=9, color=lc))

    # 평균선
    fig.add_vline(x=st_data["mean"], line_dash="dash", line_color="#555", line_width=1.5,
                  annotation_text=f"평균 {st_data['mean']:.2f}",
                  annotation_position="top right",
                  annotation_font=dict(size=9))

    # 현재값 마커
    fig.add_vline(x=current, line_color=st_data["signal_color"], line_width=2.5,
                  annotation_text=f"현재 {current:.3f} ({st_data['pct']:.1f}%ile)",
                  annotation_position="top left",
                  annotation_font=dict(size=11, color=st_data["signal_color"]))

    fig.update_layout(
        title=f"ADR {period}일 — 역사적 확률분포  ({len(s):,}일 기준)",
        xaxis_title="ADR 값",
        yaxis_title="확률 밀도",
        height=340,
        showlegend=False,
        margin=dict(l=50, r=50, t=60, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig


def build_percentile_chart(adr: pd.Series, period: int, color: str,
                            date_from: datetime, date_to: datetime) -> go.Figure:
    """시간에 따른 백분위수 추이 차트."""
    s = adr.dropna()
    # 각 시점에서 전체 역사 기준 백분위수 계산 (expanding window)
    pct_series = s.expanding(min_periods=20).apply(
        lambda x: float((x <= x.iloc[-1]).mean() * 100), raw=False
    )
    mask = (pct_series.index >= pd.to_datetime(date_from)) & \
           (pct_series.index <= pd.to_datetime(date_to))
    pct_sliced = pct_series.loc[mask]

    fig = go.Figure()

    # 구간별 배경
    for lo, hi, fc in [(0, 10, "rgba(244,67,54,0.10)"), (90, 100, "rgba(33,150,243,0.10)")]:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=fc, layer="below", line_width=0)

    for yv, lc, label in [(10, "#FF9800", "하위 10%"), (90, "#64B5F6", "상위 10%"),
                           (5, "#F44336", "하위 5%"), (95, "#2196F3", "상위 5%")]:
        fig.add_hline(y=yv, line_dash="dot", line_color=lc, line_width=0.8,
                      annotation_text=label, annotation_position="right",
                      annotation_font=dict(size=8, color=lc))
    fig.add_hline(y=50, line_dash="dash", line_color="#888", line_width=1)

    fig.add_trace(go.Scatter(
        x=pct_sliced.index, y=pct_sliced.values,
        line=dict(color=color, width=1.2),
        fill="tozeroy",
        fillcolor=f"rgba({_hex2rgb(color)},0.10)",
        name=f"ADR {period}일 백분위수",
        hovertemplate="%{x|%Y-%m-%d}<br>백분위: %{y:.1f}%<extra></extra>",
    ))

    latest_pct = pct_sliced.iloc[-1] if not pct_sliced.empty else float("nan")
    fig.update_layout(
        title=f"ADR {period}일 — 역사적 백분위수 추이  (현재: {latest_pct:.1f}%ile)",
        yaxis=dict(title="백분위수 (%)", range=[0, 100]),
        height=260,
        showlegend=False,
        margin=dict(l=50, r=80, t=50, b=30),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig


# ── KSC 업종 → KOSPI 11업종 매핑 ────────────────────────────────────────────
_SECTOR_KEYWORDS = [
    ("전기전자",   ["반도체", "전자부품", "통신장비", "디스플레이", "이차전지", "일차전지",
                    "전자", "컴퓨터", "영상", "음향", "광학"]),
    ("화학",       ["화학", "플라스틱", "고무", "도료", "비료", "농약", "세제", "의약품원료",
                    "석유화학", "정밀화학"]),
    ("바이오·의약",["의약", "의료", "바이오", "제약", "의료기기", "생명과학", "임상"]),
    ("철강·금속",  ["철강", "비철금속", "금속", "주물", "제련", "알루미늄", "동"]),
    ("기계·장비",  ["기계", "산업용 기계", "공작기계", "냉난방", "펌프", "밸브",
                    "로봇", "자동화", "항공기", "방위산업"]),
    ("자동차",     ["자동차", "자동차 부품", "차량", "전기차"]),
    ("건설·건재",  ["건설", "건물", "토목", "시멘트", "유리", "도자기", "요업"]),
    ("금융·보험",  ["금융", "은행", "보험", "증권", "투자", "신탁", "저축"]),
    ("에너지",     ["석유", "가스", "발전", "전력", "에너지", "연료", "원자력"]),
    ("유통·소비",  ["소매", "도매", "유통", "백화점", "마트", "편의점", "의류",
                    "패션", "식품", "음료", "주류", "담배", "농업"]),
    ("미디어·통신",["통신", "방송", "미디어", "광고", "콘텐츠", "게임", "소프트웨어",
                    "인터넷", "플랫폼", "영화", "출판"]),
    ("운수·물류",  ["운수", "운송", "항공", "해운", "항만", "물류", "창고", "택배"]),
    ("서비스·기타",["서비스", "호텔", "숙박", "레저", "여행", "교육", "부동산",
                    "임대", "사업지원", "전문직"]),
]

def _map_industry(industry: str) -> str:
    if not isinstance(industry, str):
        return "기타"
    for sector_name, keywords in _SECTOR_KEYWORDS:
        if any(kw in industry for kw in keywords):
            return sector_name
    return "기타"


@st.cache_data(show_spinner=False)
def get_sector_map(market: str) -> dict[str, str]:
    """ticker -> 업종명 매핑. KRX-DESC의 Industry 컬럼 기반."""
    try:
        desc = fdr.StockListing('KRX-DESC')
        target_market = market  # 'KOSPI' or 'KOSDAQ'
        desc_filtered = desc[desc['Market'] == target_market].copy()
        desc_filtered['_sector'] = desc_filtered['Industry'].apply(_map_industry)
        return desc_filtered.set_index('Code')['_sector'].to_dict()
    except Exception:
        pass
    return {}


# ── 섹터별 ADR ────────────────────────────────────────────────────────────────
def calc_sector_adr(close: pd.DataFrame, sector_map: dict, period: int) -> pd.DataFrame:
    latest = close.index.max()
    row = close.loc[latest]
    ma_row = close.rolling(period, min_periods=period).mean().loc[latest]

    sectors = sorted(set(sector_map.get(t, "기타") for t in close.columns))
    rows = []
    for sec in sectors:
        tickers = [t for t in close.columns if sector_map.get(t, "기타") == sec]
        if not tickers:
            continue
        r = row[tickers].dropna()
        m = ma_row[tickers].dropna()
        common = r.index.intersection(m.index)
        if len(common) < 3:
            continue
        above = int((r[common] > m[common]).sum())
        below = int((r[common] < m[common]).sum())
        total = above + below
        adr_val = above / below if below > 0 else np.nan
        rows.append({
            "업종": sec,
            "종목수": total,
            "MA위": above,
            "MA아래": below,
            "ADR": round(adr_val, 3) if not np.isnan(adr_val) else None,
            "위비중": round(above / total * 100, 1) if total > 0 else 0,
        })
    return pd.DataFrame(rows).sort_values("위비중", ascending=False)


def build_sector_chart(df_sec: pd.DataFrame, market: str, period: int) -> go.Figure:
    df = df_sec.dropna(subset=["ADR"]).copy()
    colors = ["#F44336" if v < 30 else "#FF9800" if v < 50 else "#4CAF50" if v < 70 else "#2196F3"
              for v in df["위비중"]]
    fig = go.Figure(go.Bar(
        x=df["위비중"], y=df["업종"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.0f}%" for v in df["위비중"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>MA위 비중: %{x:.1f}%<br>ADR: %{customdata:.3f}<extra></extra>",
        customdata=df["ADR"],
    ))
    fig.add_vline(x=50, line_dash="dash", line_color="gray", line_width=1)
    fig.update_layout(
        title=f"{market} 업종별 ADR {period}일 — MA 위 종목 비중",
        xaxis=dict(title="MA 위 종목 비중 (%)", range=[0, 110]),
        height=max(300, len(df) * 26 + 80),
        margin=dict(l=160, r=60, t=50, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


# ── 백테스트 ─────────────────────────────────────────────────────────────────
def run_backtest(adr: pd.Series, index_s: pd.Series,
                 threshold_pct: float, direction: str) -> dict:
    """
    ADR이 threshold_pct 이하(과매도) 또는 이상(과매수)인 날의 이후 수익률 분석.
    direction: 'low' | 'high'
    """
    s = adr.dropna()
    # 전체 기간 기준 분위수
    q = s.quantile(threshold_pct / 100)
    if direction == "low":
        signal_dates = s[s <= q].index
        label = f"하위 {threshold_pct:.0f}% (과매도)"
    else:
        q = s.quantile(1 - threshold_pct / 100)
        signal_dates = s[s >= q].index
        label = f"상위 {threshold_pct:.0f}% (과매수)"

    # 연속 신호 중 첫날만 사용 (에피소드 분리)
    episodes = []
    prev = None
    for d in sorted(signal_dates):
        if prev is None or (d - prev).days > 5:
            episodes.append(d)
        prev = d

    idx_aligned = index_s.reindex(s.index).ffill()
    fwd_days = [5, 10, 20, 40, 60]
    results = {fd: [] for fd in fwd_days}

    for d in episodes:
        pos = idx_aligned.index.get_loc(d) if d in idx_aligned.index else None
        if pos is None:
            continue
        for fd in fwd_days:
            target_pos = pos + fd
            if target_pos < len(idx_aligned):
                ret = (idx_aligned.iloc[target_pos] / idx_aligned.iloc[pos] - 1) * 100
                results[fd].append(ret)

    summary = []
    for fd in fwd_days:
        arr = np.array(results[fd])
        if len(arr) == 0:
            continue
        summary.append({
            "기간": f"{fd}일 후",
            "에피소드수": len(arr),
            "평균수익률": round(arr.mean(), 2),
            "승률(%)": round((arr > 0).mean() * 100, 1),
            "중앙값": round(np.median(arr), 2),
            "최대": round(arr.max(), 2),
            "최소": round(arr.min(), 2),
        })
    return {"label": label, "episodes": episodes, "summary": pd.DataFrame(summary),
            "q_val": q, "raw": results}


def build_backtest_dist_chart(raw: dict, fwd_days: list[int]) -> go.Figure:
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0"]
    fig = go.Figure()
    for fd, color in zip(fwd_days, colors):
        arr = np.array(raw.get(fd, []))
        if len(arr) == 0:
            continue
        fig.add_trace(go.Box(
            y=arr, name=f"{fd}일 후",
            marker_color=color,
            boxpoints="outliers",
            hovertemplate=f"{fd}일 후: %{{y:.2f}}%<extra></extra>",
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig.update_layout(
        title="신호 발생 후 수익률 분포 (지수 기준)",
        yaxis_title="수익률 (%)",
        height=340, showlegend=False,
        margin=dict(l=50, r=30, t=50, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def build_episode_chart(episodes: list, adr: pd.Series, index_s: pd.Series,
                        q_val: float, market: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=[f"{market} 지수", "ADR"],
                        row_heights=[0.6, 0.4], vertical_spacing=0.05)

    # 지수
    idx = index_s.dropna()
    fig.add_trace(go.Scatter(x=idx.index, y=idx.values,
                             line=dict(color="#607D8B", width=1),
                             name="지수"), row=1, col=1)
    # ADR
    s = adr.dropna()
    fig.add_trace(go.Scatter(x=s.index, y=s.values,
                             line=dict(color="#2196F3", width=1),
                             name="ADR"), row=2, col=1)
    fig.add_hline(y=q_val, line_dash="dot", line_color="#F44336",
                  line_width=1, row=2, col=1)

    # 에피소드 마킹
    for d in episodes:
        fig.add_vline(x=d, line_color="rgba(244,67,54,0.4)",
                      line_width=1, line_dash="dot")

    fig.update_layout(
        height=380, showlegend=False,
        margin=dict(l=50, r=30, t=50, b=30),
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x unified",
    )
    fig.update_yaxes(tickformat=",.0f", row=1, col=1)
    return fig


# ── 다이버전스 분석 ───────────────────────────────────────────────────────────
def calc_divergence(adr: pd.Series, index_s: pd.Series, window: int = 20) -> pd.DataFrame:
    """
    롤링 window 기간 동안 지수 방향 vs ADR 방향 비교.
    - 강세 다이버전스: 지수 하락 + ADR 상승 (바닥 신호)
    - 약세 다이버전스: 지수 상승 + ADR 하락 (고점 경고)
    """
    common = adr.dropna().index.intersection(index_s.dropna().index)
    s = adr.loc[common]
    idx = index_s.loc[common]

    idx_chg = idx.pct_change(window)
    adr_chg = s.pct_change(window)

    bear_div = (idx_chg > 0) & (adr_chg < -0.10)   # 지수↑ ADR↓10%+ : 약세다이버전스
    bull_div = (idx_chg < 0) & (adr_chg > 0.10)    # 지수↓ ADR↑10%+ : 강세다이버전스

    df = pd.DataFrame({
        "index": idx,
        "adr": s,
        "idx_chg": idx_chg * 100,
        "adr_chg": adr_chg * 100,
        "bear_div": bear_div,
        "bull_div": bull_div,
    })
    return df


def build_divergence_chart(df: pd.DataFrame, market: str, period: int,
                           date_from: datetime, date_to: datetime) -> go.Figure:
    mask = (df.index >= pd.to_datetime(date_from)) & (df.index <= pd.to_datetime(date_to))
    d = df.loc[mask]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=[f"{market} 지수", f"ADR {period}일",
                                        "다이버전스 강도 (지수변화율 - ADR변화율)"],
                        row_heights=[0.4, 0.3, 0.3], vertical_spacing=0.05)

    fig.add_trace(go.Scatter(x=d.index, y=d["index"],
                             line=dict(color="#607D8B", width=1.2), name="지수",
                             hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}<extra></extra>"),
                  row=1, col=1)

    # 약세 다이버전스 구간 마킹 (지수)
    bear_dates = d.index[d["bear_div"]]
    for bd in bear_dates:
        fig.add_vline(x=bd, line_color="rgba(244,67,54,0.15)",
                      line_width=1, row=1, col=1)

    fig.add_trace(go.Scatter(x=d.index, y=d["adr"],
                             line=dict(color="#2196F3", width=1.2), name=f"ADR {period}일",
                             hovertemplate="%{x|%Y-%m-%d}<br>ADR: %{y:.3f}<extra></extra>"),
                  row=2, col=1)
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                  line_width=0.8, row=2, col=1)

    # 다이버전스 강도: idx_chg - adr_chg (양수=약세다이버전스)
    div_strength = d["idx_chg"] - d["adr_chg"]
    pos = div_strength.clip(lower=0)
    neg = div_strength.clip(upper=0)

    fig.add_trace(go.Bar(x=d.index, y=pos.values, name="약세 다이버전스",
                         marker_color="rgba(244,67,54,0.6)",
                         hovertemplate="%{x|%Y-%m-%d}<br>강도: %{y:.1f}%p<extra></extra>"),
                  row=3, col=1)
    fig.add_trace(go.Bar(x=d.index, y=neg.values, name="강세 다이버전스",
                         marker_color="rgba(33,150,243,0.6)",
                         hovertemplate="%{x|%Y-%m-%d}<br>강도: %{y:.1f}%p<extra></extra>"),
                  row=3, col=1)

    # 현재 상황 강조
    latest = d.index[-1]
    latest_strength = div_strength.iloc[-1]
    fig.add_annotation(
        x=latest, y=d["adr"].iloc[-1],
        text=f"  현재: ADR {d['adr'].iloc[-1]:.3f}",
        showarrow=False, font=dict(color="#F44336", size=10),
        xanchor="left", row=2, col=1,
    )

    fig.update_layout(
        height=520,
        title=f"{market} ADR {period}일 — 지수 vs ADR 다이버전스 분석",
        showlegend=True,
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=50, r=50, t=70, b=30),
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x unified", barmode="overlay",
    )
    fig.update_yaxes(tickformat=",.0f", row=1, col=1)
    return fig


# ── 회복 시점 예측 ────────────────────────────────────────────────────────────
def recovery_analysis(adr: pd.Series, threshold_pct: float = 10.0,
                       recovery_pct: float = 50.0) -> dict:
    """
    과거 극단 과매도(하위 threshold_pct%) 에피소드에서
    recovery_pct 백분위수까지 회복하는 데 걸린 일수 분석.
    """
    s = adr.dropna()
    q_enter = s.quantile(threshold_pct / 100)    # 진입 기준
    q_exit = s.quantile(recovery_pct / 100)      # 회복 기준

    # 에피소드 탐색
    in_episode = False
    enter_date = None
    durations = []
    episode_records = []

    for date, val in s.items():
        if not in_episode and val <= q_enter:
            in_episode = True
            enter_date = date
            enter_val = val
        elif in_episode and val >= q_exit:
            dur = (date - enter_date).days
            durations.append(dur)
            episode_records.append({
                "진입일": enter_date.strftime("%Y-%m-%d"),
                "회복일": date.strftime("%Y-%m-%d"),
                "소요일(달력)": dur,
                "진입ADR": round(enter_val, 3),
                "회복ADR": round(val, 3),
            })
            in_episode = False

    # 현재 에피소드
    current_in_episode = in_episode
    current_enter = enter_date

    if not durations:
        return {"durations": [], "episode_records": pd.DataFrame(),
                "current_in_episode": current_in_episode,
                "current_enter": current_enter, "q_enter": q_enter, "q_exit": q_exit}

    arr = np.array(durations)
    return {
        "durations": durations,
        "episode_records": pd.DataFrame(episode_records),
        "current_in_episode": current_in_episode,
        "current_enter": current_enter,
        "q_enter": q_enter,
        "q_exit": q_exit,
        "median_days": int(np.median(arr)),
        "mean_days": int(arr.mean()),
        "p25_days": int(np.percentile(arr, 25)),
        "p75_days": int(np.percentile(arr, 75)),
        "max_days": int(arr.max()),
    }


def build_recovery_chart(rec: dict, adr: pd.Series, period: int) -> go.Figure:
    if not rec["durations"]:
        return go.Figure()

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["회복 소요 기간 분포 (달력일)", "ADR 추이 & 에피소드 구간"],
                        column_widths=[0.4, 0.6])

    # 히스토그램
    arr = np.array(rec["durations"])
    fig.add_trace(go.Histogram(
        x=arr, nbinsx=20,
        marker_color="rgba(33,150,243,0.5)",
        marker_line_color="#2196F3", marker_line_width=0.5,
        name="회복 소요일",
        hovertemplate="소요일: %{x}일<br>빈도: %{y}회<extra></extra>",
    ), row=1, col=1)

    for val, label, color in [
        (rec["median_days"], f"중앙값 {rec['median_days']}일", "#FF9800"),
        (rec["mean_days"], f"평균 {rec['mean_days']}일", "#F44336"),
    ]:
        fig.add_vline(x=val, line_dash="dash", line_color=color, line_width=1.5,
                      annotation_text=label, annotation_position="top",
                      annotation_font=dict(color=color, size=9), row=1, col=1)

    # ADR 추이
    s = adr.dropna()
    fig.add_trace(go.Scatter(
        x=s.index, y=s.values,
        line=dict(color="#2196F3", width=1),
        name=f"ADR {period}일",
        hovertemplate="%{x|%Y-%m-%d}<br>ADR: %{y:.3f}<extra></extra>",
    ), row=1, col=2)

    fig.add_hline(y=rec["q_enter"], line_dash="dot", line_color="#F44336",
                  line_width=1, annotation_text="진입선", row=1, col=2)
    fig.add_hline(y=rec["q_exit"], line_dash="dot", line_color="#4CAF50",
                  line_width=1, annotation_text="회복선", row=1, col=2)

    # 에피소드 구간 표시
    for _, ep in rec["episode_records"].iterrows():
        fig.add_vrect(
            x0=ep["진입일"], x1=ep["회복일"],
            fillcolor="rgba(244,67,54,0.07)", layer="below", line_width=0,
            row=1, col=2,
        )

    # 현재 에피소드
    if rec["current_in_episode"] and rec["current_enter"]:
        fig.add_vrect(
            x0=rec["current_enter"], x1=s.index[-1],
            fillcolor="rgba(244,67,54,0.18)", layer="below", line_width=0,
            row=1, col=2,
        )

    fig.update_layout(
        height=360, showlegend=False,
        margin=dict(l=50, r=50, t=60, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


# ── 종목 스크리너 ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_stock_names(market: str) -> dict[str, str]:
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"{market.lower()}_tickers.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df["Code"] = df["Code"].astype(str).str.zfill(6)
            return df.set_index("Code")["Name"].to_dict()
        except Exception:
            pass
    try:
        desc = fdr.StockListing('KRX-DESC')
        listing = desc[desc['Market'] == market]
        return listing.set_index("Code")["Name"].to_dict()
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def get_marcap_map(market: str) -> dict[str, float]:
    try:
        listing = fdr.StockListing(market)
        if "Marcap" in listing.columns:
            return listing.set_index("Code")["Marcap"].to_dict()
    except Exception:
        pass
    return {}


def screen_stocks(close: pd.DataFrame, period: int, direction: str,
                  days_back: int = 3, min_days_below: int = 5) -> pd.DataFrame:
    """
    direction='cross_up':  MA 아래 → 위로 전환 (days_back일 이내)
    direction='cross_down': MA 위 → 아래로 전환
    direction='far_below': MA 대비 많이 하락한 종목 (반등 후보)
    direction='far_above': MA 대비 많이 상승한 종목
    """
    ma = close.rolling(period, min_periods=period).mean()
    above = close > ma  # True/False

    rows = []
    latest = close.index[-1]

    for ticker in close.columns:
        s = close[ticker].dropna()
        m = ma[ticker].dropna()
        ab = above[ticker].dropna()
        if len(s) < period + days_back + min_days_below:
            continue

        cur_price = s.iloc[-1]
        cur_ma = m.iloc[-1]
        if np.isnan(cur_price) or np.isnan(cur_ma):
            continue
        gap_pct = (cur_price / cur_ma - 1) * 100

        if direction == "cross_up":
            # 최근 days_back일 이내에 False→True 전환
            recent = ab.iloc[-(days_back + min_days_below):]
            was_below = not any(recent.iloc[:min_days_below])
            now_above = ab.iloc[-1]
            crossed = was_below and now_above
            if not crossed:
                continue
        elif direction == "cross_down":
            recent = ab.iloc[-(days_back + min_days_below):]
            was_above = all(recent.iloc[:min_days_below])
            now_below = not ab.iloc[-1]
            crossed = was_above and now_below
            if not crossed:
                continue
        elif direction == "far_below":
            if ab.iloc[-1] or gap_pct > -5:
                continue
        elif direction == "far_above":
            if not ab.iloc[-1] or gap_pct < 5:
                continue

        rows.append({
            "종목코드": ticker,
            "현재가": int(cur_price),
            f"MA{period}일": int(cur_ma),
            "MA대비(%)": round(gap_pct, 1),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("MA대비(%)", ascending=(direction in ("far_below", "cross_up")))
    return df.reset_index(drop=True)


# ── 쏠림 완화 반등 잠재력 스코어 ─────────────────────────────────────────────
def calc_rebound_score(
    close: pd.DataFrame,
    sector_map: dict,
    marcap_map: dict,
    periods: list[int],
    top_n: int = 50,
) -> pd.DataFrame:
    """
    쏠림장 해소 시 반등 잠재력 스코어 (0~100).
    구성:
      - MA 이격 점수  (40pt): 이평선 대비 낙폭이 클수록 반등 여력 큼
      - RSI 과매도    (25pt): 단기 RSI 낮을수록 기술적 반등 여력
      - 업종 약세도   (20pt): 업종 ADR이 낮을수록 해당 업종 자체가 과매도 → 회복 시 업종 전체 반등
      - 시총 가중     (15pt): 중소형주일수록 ADR 회복 시 베타가 높음
    """
    latest = close.index[-1]
    row = close.loc[latest]

    # 업종별 MA위 비중 (낮을수록 업종 자체가 과매도)
    ref_p = periods[1] if len(periods) > 1 else periods[0]
    ma = close.rolling(ref_p, min_periods=ref_p).mean()
    ma_row = ma.loc[latest]

    sector_above_ratio = {}
    for sec in set(sector_map.values()):
        tickers = [t for t in close.columns if sector_map.get(t) == sec]
        if not tickers:
            continue
        r = row[tickers].dropna()
        m = ma_row[tickers].dropna()
        common = r.index.intersection(m.index)
        if len(common) < 3:
            continue
        sector_above_ratio[sec] = (r[common] > m[common]).mean()  # 0~1, 낮을수록 과매도

    rows = []
    for ticker in close.columns:
        s = close[ticker].dropna()
        if len(s) < max(periods) + 10:
            continue
        cur = s.iloc[-1]
        if np.isnan(cur):
            continue

        # ① MA 이격 점수 (40pt): 여러 이평 대비 평균 이격
        gaps = []
        for p in periods:
            m_val = s.rolling(p, min_periods=p).mean().iloc[-1]
            if not np.isnan(m_val) and m_val > 0:
                gaps.append((cur / m_val - 1) * 100)
        if not gaps:
            continue
        avg_gap = np.mean(gaps)  # 음수일수록 MA 아래

        # ② RSI (25pt): 14일 RSI
        delta = s.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi_val = float(100 - 100 / (1 + gain.iloc[-1] / loss.iloc[-1])) if loss.iloc[-1] > 0 else 50

        # ③ 업종 약세도 (20pt)
        sec = sector_map.get(ticker, "기타")
        sec_ratio = sector_above_ratio.get(sec, 0.5)  # 낮을수록 과매도

        # ④ 시총 (15pt): 로그 시총, 낮을수록 소형주
        marcap = marcap_map.get(ticker, 0)
        log_cap = np.log1p(marcap) if marcap > 0 else 0

        rows.append({
            "종목코드": ticker,
            "업종": sec,
            "현재가": int(cur),
            "MA이격(%)": round(avg_gap, 1),
            "RSI14": round(rsi_val, 1),
            "업종MA위비중(%)": round(sec_ratio * 100, 1),
            "_log_cap": log_cap,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # 정규화 (0~1)
    def norm(series, invert=False):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(0.5, index=series.index)
        s = (series - mn) / (mx - mn)
        return 1 - s if invert else s

    # MA 이격: 낙폭 클수록(음수) 높은 점수
    df["s_gap"] = norm(df["MA이격(%)"], invert=True)  # 낮은값(더 많이 빠진 것)→높은점수
    # RSI: 낮을수록 과매도 → 높은점수
    df["s_rsi"] = norm(df["RSI14"], invert=True)
    # 업종 약세도: 낮을수록 → 높은점수
    df["s_sec"] = norm(df["업종MA위비중(%)"], invert=True)
    # 시총: 작을수록 → 높은점수 (소형주 베타)
    df["s_cap"] = norm(df["_log_cap"], invert=True)

    df["반등잠재력"] = (
        df["s_gap"] * 40 +
        df["s_rsi"] * 25 +
        df["s_sec"] * 20 +
        df["s_cap"] * 15
    ).round(1)

    df = df.drop(columns=["s_gap","s_rsi","s_sec","s_cap","_log_cap"])
    df = df.sort_values("반등잠재력", ascending=False).head(top_n).reset_index(drop=True)
    df.index = df.index + 1  # 1위부터
    return df


def build_rebound_chart(df: pd.DataFrame) -> go.Figure:
    top = df.head(20).copy()
    label = top.apply(lambda r: f"{r['종목코드']} ({r['업종']})", axis=1)
    colors = top["반등잠재력"].apply(
        lambda v: "#D32F2F" if v >= 70 else "#F44336" if v >= 60 else "#FF9800" if v >= 50 else "#FFC107"
    )
    fig = go.Figure(go.Bar(
        x=top["반등잠재력"], y=label,
        orientation="h",
        marker_color=colors,
        text=top["반등잠재력"].apply(lambda v: f"{v:.0f}점"),
        textposition="outside",
        customdata=top[["MA이격(%)","RSI14","업종MA위비중(%)"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "반등잠재력: %{x:.1f}점<br>"
            "MA이격: %{customdata[0]:.1f}%<br>"
            "RSI14: %{customdata[1]:.1f}<br>"
            "업종MA위비중: %{customdata[2]:.1f}%<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="쏠림 완화 시 반등 잠재력 Top 20",
        xaxis=dict(title="반등 잠재력 점수 (0~100)", range=[0, 115]),
        height=max(400, len(top) * 28 + 80),
        margin=dict(l=200, r=60, t=50, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


# ── 스냅샷 테이블 ─────────────────────────────────────────────────────────────
def snapshot_table(close: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    latest_date = close.index.max()
    row = close.loc[latest_date]
    total = int(row.notna().sum())
    rows = []
    for p in periods:
        ma = close.rolling(p, min_periods=p).mean()
        if latest_date not in ma.index:
            continue
        last_ma = ma.loc[latest_date]
        above = int((row > last_ma).sum())
        below = int((row < last_ma).sum())
        adr_val = above / below if below > 0 else float("nan")
        ratio = above / total * 100 if total > 0 else 0
        rows.append({
            "기간": f"{p}일",
            "MA 위 종목": above,
            "MA 아래 종목": below,
            "ADR": round(adr_val, 3),
            "위 비중(%)": round(ratio, 1),
            "강도": "강세 🟢" if adr_val > 1 else ("약세 🔴" if adr_val < 1 else "중립 ⚪"),
        })
    return pd.DataFrame(rows)


# ── UI ───────────────────────────────────────────────────────────────────────
st.title("📈 KOSPI / KOSDAQ ADR 분석")
st.caption(f"ADR = 이동평균선 위 종목 수 ÷ 이동평균선 아래 종목 수  |  기준: 최근 {YEARS}년")

with st.sidebar:
    st.header("⚙️ 설정")

    markets = st.multiselect(
        "마켓", ["KOSPI", "KOSDAQ"],
        default=["KOSPI", "KOSDAQ"],
    )

    st.markdown("**이동평균 기간 (일)**")
    col1, col2 = st.columns(2)
    preset = col1.selectbox("프리셋", ["5 / 10 / 20", "10 / 20 / 60", "20 / 60 / 120", "직접 입력"])
    if preset == "직접 입력":
        custom = col2.text_input("기간 (쉼표 구분)", "5,10,20")
        try:
            periods = sorted(set(int(x) for x in custom.split(",") if x.strip().isdigit()))
        except Exception:
            periods = [5, 10, 20]
    else:
        periods = [int(x) for x in preset.split(" / ")]

    st.markdown(f"선택된 기간: **{periods}**")

    st.divider()
    date_from = st.date_input("시작일", value=datetime.today() - timedelta(days=365 * 3))
    date_to = st.date_input("종료일", value=datetime.today())

    show_index = st.toggle("지수 차트 표시", value=True)

    st.divider()
    if st.button("🔄 데이터 새로고침", use_container_width=True):
        st.cache_data.clear()
        for f in os.listdir(CACHE_DIR):
            os.remove(os.path.join(CACHE_DIR, f))
        st.rerun()

if not markets:
    st.warning("마켓을 하나 이상 선택해주세요.")
    st.stop()

# ── 마켓별 탭 ─────────────────────────────────────────────────────────────────
tabs = st.tabs(markets)

for tab, market in zip(tabs, markets):
    with tab:
        _msg = st.empty()
        _pb = st.progress(0)
        _msg.info(f"[{market}] 데이터 로드 중...")
        close = load_close_prices(market, progress_bar=_pb)
        _pb.empty(); _msg.empty()
        index_s = load_index(market) if show_index else None

        if close.empty:
            st.error(f"{market} 데이터를 불러오지 못했습니다.")
            continue

        latest = close.index.max()
        st.markdown(f"**기준일:** {latest.strftime('%Y-%m-%d')}  |  **전체 종목:** {close.shape[1]:,}개  |  **거래일:** {close.shape[0]:,}일")

        # 스냅샷 테이블
        df_snap = snapshot_table(close, periods)
        st.dataframe(
            df_snap,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ADR": st.column_config.NumberColumn(format="%.3f"),
                "위 비중(%)": st.column_config.NumberColumn(format="%.1f%%"),
                "MA 위 종목": st.column_config.NumberColumn(format="%d"),
                "MA 아래 종목": st.column_config.NumberColumn(format="%d"),
            },
        )

        adr_all = {p: calc_adr(close, p) for p in periods}

        # ── 현재 쏠림장 경보 배너 ─────────────────────────────────────────────
        ref_period = periods[1] if len(periods) > 1 else periods[0]
        st_ref = adr_stats(adr_all[ref_period])
        if index_s is not None and st_ref["pct"] < 10:
            idx_now = index_s.dropna().iloc[-1]
            idx_1y_ago = index_s.dropna().iloc[max(0, len(index_s.dropna()) - 252)]
            idx_chg_1y = (idx_now / idx_1y_ago - 1) * 100
            if idx_chg_1y > 5 and st_ref["pct"] < 10:
                st.error(
                    f"⚡ **극단 쏠림장 감지** — 지수 1년 수익률 **+{idx_chg_1y:.1f}%** (신고가 근접) "
                    f"vs ADR {ref_period}일 백분위 **{st_ref['pct']:.1f}%** (역대급 약세 내부). "
                    f"소수 대형주가 지수를 주도하는 집중 장세입니다. "
                    f"'🔮 회복 시점 예측' 탭에서 평균 회귀 시나리오를 확인하세요."
                )

        # ── 서브탭 ────────────────────────────────────────────────────────────
        tab_chart, tab_stat, tab_bt, tab_sector, tab_div, tab_screen, tab_recover, tab_rebound = st.tabs([
            "📊 ADR 차트",
            "🔔 통계 분포",
            "📈 백테스트",
            "🏭 섹터별 ADR",
            "📡 다이버전스",
            "🔍 종목 스크리너",
            "🔮 회복 시점 예측",
            "🚀 반등 잠재력",
        ])

        # ── ① ADR 차트 ───────────────────────────────────────────────────────
        with tab_chart:
            fig = build_chart(close, periods, market, index_s, date_from, date_to)
            st.plotly_chart(fig, use_container_width=True)

        # ── ② 통계 분포 ──────────────────────────────────────────────────────
        with tab_stat:
            st.markdown("#### 현재 ADR의 역사적 확률분포상 위치")
            st.caption("전체 관측 기간(최대 5년) 기준. 극단 구간(상/하위 5~10%)은 평균 회귀 가능성이 높습니다.")

            cols_stat = st.columns(len(periods))
            for col_s, p in zip(cols_stat, periods):
                st_d = adr_stats(adr_all[p])
                with col_s:
                    st.metric(label=f"ADR {p}일  |  {st_d['signal']}",
                              value=f"{st_d['current']:.3f}",
                              delta=f"{st_d['current'] - st_d['mean']:+.3f} (평균 대비)")
                    st.markdown(f"**백분위: {st_d['pct']:.1f}%** | Z: `{st_d['z']:+.2f}`",
                                unsafe_allow_html=True)
                    st.markdown(f"평균 `{st_d['mean']:.3f}` ± σ `{st_d['std']:.3f}`")

            st.divider()
            for p, color in zip(periods, COLORS):
                st.markdown(f"##### ADR {p}일")
                c1, c2 = st.columns([1.1, 1])
                with c1:
                    st.plotly_chart(build_dist_chart(adr_all[p], p, color), use_container_width=True)
                with c2:
                    st.plotly_chart(build_percentile_chart(adr_all[p], p, color, date_from, date_to),
                                    use_container_width=True)
                st_d = adr_stats(adr_all[p])
                st.dataframe(pd.DataFrame([{
                    "최솟값": round(st_d["history"].min(), 3),
                    "하위1%": round(st_d["q1"], 3), "하위5%": round(st_d["q5"], 3),
                    "하위10%": round(st_d["q10"], 3), "평균": round(st_d["mean"], 3),
                    "중앙값": round(st_d["history"].median(), 3),
                    "상위10%": round(st_d["q90"], 3), "상위5%": round(st_d["q95"], 3),
                    "상위1%": round(st_d["q99"], 3), "최댓값": round(st_d["history"].max(), 3),
                    "현재값": round(st_d["current"], 3),
                }]), use_container_width=True, hide_index=True)
                st.divider()

        # ── ③ 백테스트 ───────────────────────────────────────────────────────
        with tab_bt:
            st.markdown("#### ADR 극단 신호 → 이후 지수 수익률 백테스트")
            st.caption("과거 ADR이 특정 극단에 진입했을 때, 실제 지수 수익률이 어땠는지 역사적으로 검증합니다.")

            if index_s is None:
                st.warning("지수 데이터가 없어 백테스트를 실행할 수 없습니다.")
            else:
                bc1, bc2, bc3 = st.columns(3)
                bt_period = bc1.selectbox("ADR 기간", periods, key=f"bt_p_{market}")
                bt_threshold = bc2.slider("극단 기준 (%ile)", 1, 20, 10, key=f"bt_th_{market}")
                bt_dir = bc3.radio("방향", ["과매도 (하위)", "과매수 (상위)"], key=f"bt_dir_{market}")
                direction = "low" if "하위" in bt_dir else "high"

                res = run_backtest(adr_all[bt_period], index_s, bt_threshold, direction)

                if res["summary"].empty:
                    st.info("해당 기준으로 발생한 에피소드가 없습니다.")
                else:
                    st.markdown(f"**{res['label']}** — 에피소드 {len(res['episodes'])}회 감지")
                    # 컬러 포맷: 수익률 양수=초록, 음수=빨강
                    df_s = res["summary"]

                    def color_ret(val):
                        if isinstance(val, float):
                            color = "green" if val > 0 else "red"
                            return f"color: {color}"
                        return ""

                    st.dataframe(
                        df_s.style.map(color_ret, subset=["평균수익률", "중앙값", "최대", "최소"]),
                        use_container_width=True, hide_index=True,
                    )
                    rc1, rc2 = st.columns([1, 1.2])
                    with rc1:
                        st.plotly_chart(build_backtest_dist_chart(res["raw"], [5, 10, 20, 40, 60]),
                                        use_container_width=True)
                    with rc2:
                        st.plotly_chart(build_episode_chart(res["episodes"], adr_all[bt_period],
                                                            index_s, res["q_val"], market),
                                        use_container_width=True)

        # ── ④ 섹터별 ADR ─────────────────────────────────────────────────────
        with tab_sector:
            st.markdown("#### 업종별 ADR — 섹터 로테이션 맵")
            st.caption("MA 위 종목 비중 50% 이상이면 해당 업종 강세. 어느 업종이 주도하고 있는지 파악합니다.")

            sc1, sc2 = st.columns(2)
            sec_period = sc1.selectbox("ADR 기간", periods, key=f"sec_p_{market}")

            with st.spinner("업종 데이터 처리 중..."):
                sector_map = get_sector_map(market)
                df_sec = calc_sector_adr(close, sector_map, sec_period)

            if df_sec.empty:
                st.info("업종 분류 데이터를 가져올 수 없습니다.")
            else:
                st.plotly_chart(build_sector_chart(df_sec, market, sec_period), use_container_width=True)
                st.dataframe(
                    df_sec,
                    use_container_width=True, hide_index=True,
                    column_config={
                        "위비중": st.column_config.ProgressColumn("MA위 비중(%)", min_value=0, max_value=100, format="%.1f"),
                        "ADR": st.column_config.NumberColumn(format="%.3f"),
                    }
                )

        # ── ⑤ 다이버전스 ─────────────────────────────────────────────────────
        with tab_div:
            st.markdown("#### 지수 vs ADR 다이버전스 분석")

            if index_s is None:
                st.warning("지수 데이터가 없어 다이버전스 분석을 실행할 수 없습니다.")
            else:
                dv1, dv2 = st.columns(2)
                div_period = dv1.selectbox("ADR 기간", periods, key=f"div_p_{market}")
                div_window = dv2.slider("다이버전스 측정 윈도우 (거래일)", 5, 60, 20, key=f"div_w_{market}")

                df_div = calc_divergence(adr_all[div_period], index_s, div_window)

                # 현재 상황 요약
                latest_div = df_div.iloc[-1]
                bear_now = bool(latest_div["bear_div"])
                bull_now = bool(latest_div["bull_div"])
                div_str = latest_div["idx_chg"] - latest_div["adr_chg"]

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("현재 다이버전스 강도",
                             f"{div_str:+.1f}%p",
                             help="양수=약세다이버전스(지수↑ ADR↓), 음수=강세다이버전스(지수↓ ADR↑)")
                col_b.metric(f"지수 {div_window}일 변화율", f"{latest_div['idx_chg']:+.1f}%")
                col_c.metric(f"ADR {div_window}일 변화율", f"{latest_div['adr_chg']:+.1f}%")

                if bear_now:
                    st.error(f"⚠️ **약세 다이버전스 진행 중** — 지수 상승 + ADR 하락. "
                             f"내부 약화 신호. 소수 대형주가 지수를 끌어올리는 집중 장세.")
                elif bull_now:
                    st.success(f"✅ **강세 다이버전스 진행 중** — 지수 하락 + ADR 상승. "
                               f"내부 강화 신호. 바닥 다지기 가능성.")
                else:
                    st.info("현재 뚜렷한 다이버전스 없음 (지수와 ADR 방향 일치)")

                # 역사적 약세 다이버전스 에피소드 수
                bear_episodes = df_div["bear_div"].sum()
                bull_episodes = df_div["bull_div"].sum()
                st.caption(f"전체 기간 중 약세 다이버전스: {bear_episodes}일  |  강세 다이버전스: {bull_episodes}일")

                st.plotly_chart(
                    build_divergence_chart(df_div, market, div_period, date_from, date_to),
                    use_container_width=True
                )

        # ── ⑥ 종목 스크리너 ──────────────────────────────────────────────────
        with tab_screen:
            st.markdown("#### 종목 스크리너 — ADR 신호 기반 종목 필터")

            sr1, sr2, sr3 = st.columns(3)
            screen_p = sr1.selectbox("MA 기간", periods, key=f"sc_p_{market}")
            screen_mode = sr2.selectbox("스크리닝 조건", [
                "MA 위로 막 전환 (돌파 후보)",
                "MA 아래로 막 전환 (이탈 경고)",
                "MA 대비 많이 하락 (반등 후보)",
                "MA 대비 많이 상승 (과열 종목)",
            ], key=f"sc_m_{market}")
            screen_days = sr3.slider("전환 감지 기간 (일)", 1, 10, 3, key=f"sc_d_{market}")

            dir_map = {
                "MA 위로 막 전환 (돌파 후보)": "cross_up",
                "MA 아래로 막 전환 (이탈 경고)": "cross_down",
                "MA 대비 많이 하락 (반등 후보)": "far_below",
                "MA 대비 많이 상승 (과열 종목)": "far_above",
            }

            with st.spinner("종목 스크리닝 중..."):
                df_screen = screen_stocks(close, screen_p, dir_map[screen_mode], screen_days)
                names = get_stock_names(market)
                if not df_screen.empty:
                    df_screen.insert(1, "종목명", df_screen["종목코드"].map(names).fillna(""))

            if df_screen.empty:
                st.info("해당 조건에 맞는 종목이 없습니다.")
            else:
                st.markdown(f"**{len(df_screen)}개 종목** 해당")
                st.dataframe(
                    df_screen,
                    use_container_width=True, hide_index=True,
                    column_config={
                        "MA대비(%)": st.column_config.NumberColumn(format="%.1f%%"),
                        "현재가": st.column_config.NumberColumn(format="%d"),
                    }
                )

        # ── ⑦ 회복 시점 예측 ─────────────────────────────────────────────────
        with tab_recover:
            st.markdown("#### 🔮 ADR 평균 회귀 — 회복 시점 예측")
            st.caption(
                "과거 ADR이 극단 과매도 구간에 진입했을 때, 회복선까지 도달하는 데 걸린 시간을 분석합니다. "
                "현재 에피소드가 진행 중이라면 예상 회복 시점 범위를 제시합니다."
            )

            rv1, rv2, rv3 = st.columns(3)
            rec_period = rv1.selectbox("ADR 기간", periods, key=f"rv_p_{market}")
            rec_enter = rv2.slider("진입 기준 (%ile)", 1, 20, 10, key=f"rv_e_{market}")
            rec_exit = rv3.slider("회복 기준 (%ile)", 30, 70, 50, key=f"rv_x_{market}")

            rec = recovery_analysis(adr_all[rec_period], rec_enter, rec_exit)

            if not rec["durations"]:
                st.info("해당 기준으로 완료된 회복 에피소드가 없습니다.")
            else:
                # 현재 에피소드 예측
                if rec["current_in_episode"] and rec["current_enter"]:
                    today = adr_all[rec_period].index[-1]
                    days_in = (today - rec["current_enter"]).days
                    eta_median = max(0, rec["median_days"] - days_in)
                    eta_p75 = max(0, rec["p75_days"] - days_in)
                    pred_median = today + timedelta(days=eta_median)
                    pred_p75 = today + timedelta(days=eta_p75)

                    st.warning(
                        f"🔴 **현재 과매도 에피소드 진행 중** — "
                        f"진입일: {rec['current_enter'].strftime('%Y-%m-%d')} "
                        f"(경과 {days_in}일)\n\n"
                        f"과거 중앙값 기준 예상 회복: **{pred_median.strftime('%Y-%m-%d')}** "
                        f"(잔여 {eta_median}일) | "
                        f"상위 25% 지연 시나리오: **{pred_p75.strftime('%Y-%m-%d')}** "
                        f"(잔여 {eta_p75}일)"
                    )

                # 통계 카드
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("중앙값 회복 기간", f"{rec['median_days']}일")
                rc2.metric("평균 회복 기간", f"{rec['mean_days']}일")
                rc3.metric("25% 빠른 회복", f"{rec['p25_days']}일")
                rc4.metric("75% 느린 회복", f"{rec['p75_days']}일")

                st.plotly_chart(build_recovery_chart(rec, adr_all[rec_period], rec_period),
                                use_container_width=True)

                st.markdown("##### 과거 에피소드 목록")
                st.dataframe(rec["episode_records"], use_container_width=True, hide_index=True)

        # ── ⑧ 반등 잠재력 ────────────────────────────────────────────────────
        with tab_rebound:
            st.markdown("#### 🚀 쏠림 완화 시 반등 잠재력 종목 분석")
            st.caption(
                "ADR이 극단 과매도 구간에서 회복될 때 가장 많이 반등할 가능성이 높은 종목을 점수화합니다. "
                "**MA 이격(40점) + RSI 과매도(25점) + 업종 약세도(20점) + 소형주 베타(15점)** 합산."
            )

            rb1, rb2 = st.columns(2)
            rb_top_n = rb1.slider("상위 종목 수", 20, 100, 50, step=10, key=f"rb_n_{market}")
            rb_periods_label = rb2.selectbox("기준 이평 (이격 계산)", periods, key=f"rb_p_{market}")

            st.info("⏳ 전 종목 RSI·이격·업종 점수 계산 중... (약 10~30초)")

            with st.spinner("반등 잠재력 계산 중..."):
                sector_map_rb = get_sector_map(market)
                marcap_map = get_marcap_map(market)
                names_rb = get_stock_names(market)
                df_rb = calc_rebound_score(close, sector_map_rb, marcap_map, periods, top_n=rb_top_n)

            if df_rb.empty:
                st.warning("계산 실패 또는 데이터 부족")
            else:
                # 종목명 추가
                df_rb.insert(1, "종목명", df_rb["종목코드"].map(names_rb).fillna(""))

                # 상위 3 하이라이트
                st.markdown("##### 🥇 Top 3 반등 후보")
                top3 = df_rb.head(3)
                cols3 = st.columns(3)
                medals = ["🥇", "🥈", "🥉"]
                for col3, (_, r), medal in zip(cols3, top3.iterrows(), medals):
                    with col3:
                        st.metric(
                            label=f"{medal} {r['종목명']} ({r['종목코드']})",
                            value=f"{r['반등잠재력']:.0f}점",
                            delta=f"MA이격 {r['MA이격(%)']:+.1f}%  |  RSI {r['RSI14']:.0f}",
                        )
                        st.caption(f"업종: {r['업종']}  |  업종MA위비중: {r['업종MA위비중(%)']:.0f}%")

                st.divider()

                # 차트
                st.plotly_chart(build_rebound_chart(df_rb), use_container_width=True)

                # 전체 테이블
                st.markdown("##### 전체 반등 후보 목록")
                st.caption("종목코드 클릭 → 네이버 금융 연결은 브라우저에서 직접 검색하세요.")
                st.dataframe(
                    df_rb,
                    use_container_width=True,
                    column_config={
                        "반등잠재력": st.column_config.ProgressColumn(
                            "반등잠재력", min_value=0, max_value=100, format="%.0f점"),
                        "MA이격(%)": st.column_config.NumberColumn(format="%.1f%%"),
                        "RSI14": st.column_config.NumberColumn(format="%.1f"),
                        "업종MA위비중(%)": st.column_config.NumberColumn(format="%.1f%%"),
                        "현재가": st.column_config.NumberColumn(format="%d"),
                    }
                )

                # 업종별 반등 후보 분포
                st.markdown("##### 업종별 반등 후보 분포")
                sec_cnt = df_rb["업종"].value_counts().reset_index()
                sec_cnt.columns = ["업종", "종목수"]
                fig_sec_rb = go.Figure(go.Bar(
                    x=sec_cnt["종목수"], y=sec_cnt["업종"],
                    orientation="h",
                    marker_color="#FF9800",
                    text=sec_cnt["종목수"],
                    textposition="outside",
                ))
                fig_sec_rb.update_layout(
                    title="반등 후보 상위 종목의 업종 분포",
                    xaxis_title="종목수",
                    height=max(250, len(sec_cnt) * 28 + 60),
                    margin=dict(l=120, r=60, t=50, b=30),
                    plot_bgcolor="white", paper_bgcolor="white",
                )
                st.plotly_chart(fig_sec_rb, use_container_width=True)
