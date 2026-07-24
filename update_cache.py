# -*- coding: utf-8 -*-
"""
ADR 주가 데이터 증분 업데이트 (작업 스케줄러용)
매일 장 마감 후 실행 — 최신 하루치 데이터를 pickle 캐시에 추가
"""
import os, sys, pickle
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from datetime import datetime, timedelta
import pandas as pd
import FinanceDataReader as fdr

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".adr_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def update_market(market: str, force_today: bool = False):
    cache_file = os.path.join(CACHE_DIR, f"close_{market}.pkl")
    today = datetime.today().strftime("%Y-%m-%d")

    if not os.path.exists(cache_file):
        print(f"[{market}] 캐시 없음 — 앱 실행 시 전체 다운로드 예정")
        return

    with open(cache_file, "rb") as f:
        existing = pickle.load(f)

    cached_end = existing.index.max().strftime("%Y-%m-%d")

    # force_today: 오늘치가 이미 있어도 덮어쓰기 (장 마감 후 완전한 종가 반영)
    if cached_end >= today and not force_today:
        print(f"[{market}] 이미 최신 ({cached_end})")
        return

    # 오늘치 포함해서 재수집
    fetch_start = today if force_today and cached_end >= today \
        else (existing.index.max() + timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"[{market}] {fetch_start} ~ {today} 업데이트 중...")

    # 종목 리스트
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"{market.lower()}_tickers.csv")
    if os.path.exists(csv_path):
        tickers = pd.read_csv(csv_path)["Code"].astype(str).str.zfill(6).tolist()
    else:
        listing = fdr.StockListing('KRX-DESC')
        tickers = listing[listing['Market'] == market]["Code"].tolist()

    frames = {}
    for i, ticker in enumerate(tickers, 1):
        if i % 100 == 0:
            print(f"  {i}/{len(tickers)}...")
        try:
            df = fdr.DataReader(ticker, fetch_start, today)
            if not df.empty and "Close" in df.columns:
                frames[ticker] = df["Close"].rename(ticker)
        except Exception:
            pass

    if not frames:
        print(f"[{market}] 새 데이터 없음 (휴장일 가능성)")
        return

    new_df = pd.DataFrame(frames)
    new_df.index = pd.to_datetime(new_df.index, errors="coerce")
    new_df = new_df[new_df.index.notna()].sort_index()

    combined = pd.concat([existing, new_df])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()

    with open(cache_file, "wb") as f:
        pickle.dump(combined, f)

    print(f"[{market}] 완료 — {combined.index.max().date()}까지 {combined.shape[1]}종목")

if __name__ == "__main__":
    print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M')}] ADR 캐시 업데이트 시작")
    for market in ["KOSPI", "KOSDAQ"]:
        update_market(market)
    print("완료")
