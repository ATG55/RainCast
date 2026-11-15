import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor)
try:
    from xgboost import XGBRegressor            
except ImportError:
    XGBRegressor = None
    print("xgboost kurulu değil, atlanıyor (python -m pip install xgboost)")

warnings.filterwarnings("ignore")


#VERİ

dosya = "data2.csv"
df = pd.read_csv(dosya)
df["region"] = (df["region"].str.replace("İ", "i", regex=False)
                            .str.replace("I", "i", regex=False)
                            .str.lower())
df["month"] = pd.to_datetime(df["month"])
aylik = df.groupby(["region", "month"])["precipitation_mm"].mean().unstack(0).asfreq("MS")
isim = {"istanbul": "İstanbul", "izmir": "İzmir"}


#ÖZELLİKLER (log1p hedef; gecikmeler + mevsim)

LAGS = (1, 2, 3, 12)

def ozellik(gecmis, tarih):
    """gecmis: log1p serisi (tarihten önceki değerler). Tek satır özellik döndürür."""
    v = gecmis.values
    satir = {
        "ay_sin": np.sin(2 * np.pi * tarih.month / 12),
        "ay_cos": np.cos(2 * np.pi * tarih.month / 12),
        "roll12": v[-12:].mean(),
    }
    for L in LAGS:
        satir[f"lag{L}"] = v[-L]
    return satir

def egitim_tablosu(seri_log):
    satirlar, hedef = [], []
    for i in range(12, len(seri_log)):
        satirlar.append(ozellik(seri_log.iloc[:i], seri_log.index[i]))
        hedef.append(seri_log.iloc[i])
    return pd.DataFrame(satirlar), np.array(hedef)

def modeller():
    m = {
        "RandomForest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0),
    }
    if XGBRegressor is not None:
        m["XGBoost"] = XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)
    return m

def ozyinelemeli_tahmin(model, seri, adim):
    """Her adımda tahmini geçmişe ekleyip bir sonraki ayı tahmin eder (recursive)."""
    log = np.log1p(seri)
    X, y = egitim_tablosu(log)
    model.fit(X, y)
    gecmis = log.copy()
    for _ in range(adim):
        tarih = gecmis.index[-1] + pd.offsets.MonthBegin(1)
        x = pd.DataFrame([ozellik(gecmis, tarih)])
        gecmis.loc[tarih] = model.predict(x)[0]
    return np.expm1(gecmis.iloc[-adim:]).clip(lower=0)

rmse = lambda a, b: np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


#GERİYE DÖNÜK TEST (2001-2018 eğit -> 2019-2024 tahmin)

print("=== Geriye dönük test: RMSE (mm) ===")
sonuc = {}
for bolge in aylik.columns:
    egitim, test = aylik.loc[:"2018-12", bolge], aylik.loc["2019-01":, bolge]
    klima = egitim.groupby(egitim.index.month).mean()
    klima_tahmin = np.array([klima[m] for m in test.index.month])
    sonuc[bolge] = {"İklimatoloji": rmse(test, klima_tahmin)}
    for ad, m in modeller().items():
        tahmin = ozyinelemeli_tahmin(m, egitim, len(test))
        sonuc[bolge][ad] = rmse(test, tahmin)
print(pd.DataFrame(sonuc).round(1).rename(columns=isim).to_string())


#2025-2030 TAHMİNİ

adim = 72
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
for ax, bolge in zip(axes, aylik.columns):
    seri = aylik[bolge]
    ax.plot(seri.index, seri, lw=1, color="tab:blue", label="Gözlem (2001-2024)")
    for ad, m in modeller().items():
        t = ozyinelemeli_tahmin(m, seri, adim)
        ax.plot(t.index, t, lw=1.2, label=ad)
        print(f"{isim[bolge]} - {ad}: yıllık toplam ≈ {t.groupby(t.index.year).sum().mean():.0f} mm")
    ax.set_title(f"{isim[bolge]} - Aylık Yağış Tahmini (ML modelleri)")
    ax.set_ylabel("mm/ay"); ax.grid(alpha=0.3)
    for yil in (2010, 2020, 2030):
        ax.axvline(pd.Timestamp(f"{yil}-01-01"), color="gray", ls="--", lw=1)
    ax.axvline(pd.Timestamp("2025-01-01"), color="black", ls=":", lw=1)
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("ml_tahmin_2030.png", dpi=150)
plt.show()
import json

def _temiz(v):
    return None if pd.isna(v) else round(float(v), 2)

cikti = {"bolgeler": {}}
for bolge in aylik.columns:
    seri = aylik[bolge]
    kayit = {
        "ad": isim.get(bolge, bolge),
        "gozlem": {"tarih": [d.strftime("%Y-%m") for d in seri.index],
                   "deger": [_temiz(v) for v in seri.values]},
        "rmse": {k: round(float(v), 2) for k, v in sonuc[bolge].items()},
        "tahmin": {},
        "yillik": {},
    }
    for ad, m in modeller().items():
        t = ozyinelemeli_tahmin(m, seri, 72)
        kayit["tahmin"][ad] = {"tarih": [d.strftime("%Y-%m") for d in t.index],
                               "deger": [_temiz(v) for v in t.values]}
        kayit["yillik"][ad] = {str(y): round(float(s), 1)
                               for y, s in t.groupby(t.index.year).sum().items()}
    cikti["bolgeler"][bolge] = kayit

with open("forecast.json", "w", encoding="utf-8") as f:
    json.dump(cikti, f, ensure_ascii=False)
print("forecast.json yazıldı")
