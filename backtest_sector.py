# -*- coding: utf-8 -*-
import pickle, numpy as np, pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('.adr_cache/close_KOSPI.pkl','rb') as f:
    close = pickle.load(f)

import FinanceDataReader as fdr

# 앱과 동일한 섹터 키워드 매핑
_SECTOR_KEYWORDS = [
    ("전기전자", ["반도체","전자부품","통신장비","디스플레이","이차전지","전자","HBM","메모리","시스템반도체"]),
    ("화학",     ["화학","플라스틱","고무","도료","비료","정밀화학"]),
    ("바이오·의약",["의약","의료","바이오","헬스","제약","진단"]),
    ("철강·금속", ["철강","금속","비철","주조","금","은","알루미늄"]),
    ("기계·장비", ["기계","장비","자동화","로봇","공작","산업용"]),
    ("자동차",   ["자동차","부품","타이어","차량"]),
    ("건설·건재", ["건설","건재","시멘트","레미콘","인테리어"]),
    ("금융·보험", ["금융","은행","보험","증권","카드","캐피탈","저축"]),
    ("에너지",   ["에너지","석유","가스","전력","발전","원자력","신재생"]),
    ("유통·소비", ["유통","소비","도매","소매","음식료","식품","의류","패션"]),
    ("미디어·통신",["미디어","통신","방송","콘텐츠","게임","광고","출판"]),
    ("운수·물류", ["운수","물류","항공","해운","택배","운송"]),
    ("서비스·기타",["서비스","교육","부동산","임대","기타"]),
]

def map_industry(industry: str) -> str:
    if not isinstance(industry, str):
        return "기타"
    for sec, kws in _SECTOR_KEYWORDS:
        for kw in kws:
            if kw in industry:
                return sec
    return "서비스·기타"

desc = fdr.StockListing('KRX-DESC')
kospi = desc[desc['Market']=='KOSPI'][['Code','Industry']].copy()
kospi['Sector'] = kospi['Industry'].apply(map_industry)

sec_map = kospi.set_index('Code')['Sector'].to_dict()

sectors = {}
for t, s in sec_map.items():
    if t in close.columns:
        sectors.setdefault(s, []).append(t)
sectors = {s: t for s, t in sectors.items() if len(t) >= 5}

print(f'섹터 수: {len(sectors)}')
for s, tks in sorted(sectors.items()):
    print(f'  {s}: {len(tks)}개 종목')

period = 20
horizons = [20, 60, 120]
results = {h: {'bot10':[], 'bot30':[], 'normal':[], 'top30':[]} for h in horizons}

dates = close.index
ma = close.rolling(period, min_periods=period).mean()

print(f'\n분석 기간: {dates[0].date()} ~ {dates[-1].date()}, 총 {len(dates)}거래일')
print('백테스트 진행 중...')

for sec, tickers in sectors.items():
    tks = [t for t in tickers if t in close.columns]
    if len(tks) < 5:
        continue
    cl = close[tks]
    m = ma[tks]
    above = (cl > m).sum(axis=1) / cl.notna().sum(axis=1) * 100
    roll_pct = above.rolling(252, min_periods=120).rank(pct=True) * 100

    for i in range(252, len(dates)):
        pct = roll_pct.iloc[i]
        if np.isnan(pct):
            continue
        for h in horizons:
            if i + h >= len(dates):
                continue
            fwd_vals = (cl.iloc[i+h] / cl.iloc[i] - 1) * 100
            fwd = fwd_vals.dropna().mean()
            if np.isnan(fwd):
                continue
            bucket = 'bot10' if pct <= 10 else 'bot30' if pct <= 30 else 'top30' if pct >= 70 else 'normal'
            results[h][bucket].append(fwd)

print()
print('='*62)
print('  섹터ADR 백분위 구간별 이후 수익률 (KOSPI 10년 백테스트)')
print('='*62)
print(f'{"구간":<14} {"20일 평균":>9} {"60일 평균":>9} {"120일 평균":>10} {"표본수":>8}')
print('-'*56)
labels = [
    ('하위10% (극단)', 'bot10'),
    ('하위30%',        'bot30'),
    ('정상 30~70%',    'normal'),
    ('상위70%+',       'top30'),
]
for label, key in labels:
    r20  = results[20][key]
    r60  = results[60][key]
    r120 = results[120][key]
    a20  = np.mean(r20)  if r20  else np.nan
    a60  = np.mean(r60)  if r60  else np.nan
    a120 = np.mean(r120) if r120 else np.nan
    print(f'{label:<14} {a20:>+8.2f}%  {a60:>+8.2f}%  {a120:>+9.2f}%  {len(r20):>7,}')

print()
print('  중앙값 (극단값 영향 제거)')
print('-'*56)
for label, key in labels:
    r20  = results[20][key]
    r60  = results[60][key]
    r120 = results[120][key]
    m20  = np.median(r20)  if r20  else np.nan
    m60  = np.median(r60)  if r60  else np.nan
    m120 = np.median(r120) if r120 else np.nan
    print(f'{label:<14} {m20:>+8.2f}%  {m60:>+8.2f}%  {m120:>+9.2f}%')

print()
print('  하락 확률 (수익률 < 0)')
print('-'*56)
for label, key in labels:
    r20 = results[20][key]
    r60 = results[60][key]
    d20 = sum(1 for x in r20 if x < 0) / len(r20) * 100 if r20 else np.nan
    d60 = sum(1 for x in r60 if x < 0) / len(r60) * 100 if r60 else np.nan
    print(f'{label:<14}  20일 하락: {d20:.1f}%   60일 하락: {d60:.1f}%')
