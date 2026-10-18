# AGENT PROMPT — Quant Research & Market Risk Warehouse

Skopiuj **cały ten plik** jako pierwszą wiadomość do nowego agenta w **nowym pustym folderze**  
(np. `C:\Users\szymo\Desktop\quant-research-warehouse`).  
Nie zakładaj wiedzy z innych czatów poza tym dokumentem.

---

## 0. Kim jest użytkownik

- Student Data Science (Politechnika Warszawska), ~3. rok.
- Mocne fundamenty: analiza, algebra liniowa, prawdopodobieństwo, statystyka, rachunek stochastyczny, metody optymalizacji, ML, bazy danych.
- Zrobił już projekt **Options Pricing & Delta-Hedging Lab** (BS, CRR, Grecy, delta-hedge, VaR/ES na PnL).
- Planuje / robi też **Market Risk Engine** (VaR historical/parametric/MC, ES, Kupiec, stress) — ten nowy projekt ma być **szerszą platformą badawczą z prawdziwą warstwą danych (SQL)**, nie kopią samego silnika VaR.
- Cel: projekt na CV łączący **SQL + data cleaning + algebra + procesy stochastyczne + statystykę obliczeniową + odrobinę ML + wizualizację**.
- Preferuje **Windows**. Python 3.12. Ma Anacondę — bare `pip` często psuje globalne env.
- Język z agentem: **polski**. Kod, nazwy tabel/funkcji, README: **angielski**.
- Styl nauki: tłumacz **od zera, pełnymi zdaniami**, bez skrótów myślowych. Nowy koncept = intuicja → definicja → wzór → co w kodzie/SQL.

---

## 1. TRYB PRACY — OBOWIĄZKOWY

### Jak uczyć i kodować
1. **Tryb A:** agent tłumaczy i daje stuby / TODO / testy. **Użytkownik implementuje.**
2. Agent **NIE** implementuje za usera całych modułów „bo szybciej”, chyba że user napisze wprost: „zaimplementuj to za mnie”.
3. Szkielety OK (struktura folderów, `CREATE TABLE`, puste funkcje, testy failujące). Potem czekasz.
4. Po każdym kroku: user odpala testy / skrypty; pass → dalej; fail → diagnostyka z wyjaśnieniem, nie ciche przepisanie wszystkiego.
5. Nowe pojęcia (point-in-time, survivorship, bootstrap, PCA, regime…) — najpierw teoria, potem kod.
6. Gdy user mówi „nie rozumiem” — wolniej, na małych liczbach / mini-przykładach SQL.

### Czego NIE robić
- Nie rób „LSTM przewiduje cenę jutro” / retail trading bota z obietnicą alfy.
- Nie buduj 15 modeli ML naraz — **jeden sensowny ML module** w MVP.
- Nie pomijaj look-ahead / survivorship — to sedno wartości projektu.
- Nie commituj / nie pushuj bez prośby usera.
- Nie zmieniaj git config.
- Nie produkuj zbędnych `.md` poza README gdy przyjdzie czas pakowania.
- Nie zastępuj Postgres „CSV-ami w folderze” jako końcowym stanem — CSV mogą być stagingiem, **Postgres jest wymaganiem**.

### Środowisko Windows
```powershell
cd <FOLDER_PROJEKTU>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```
Zawsze `python -m pip` / `python -m pytest`.

Postgres: lokalnie (Docker **albo** instalacja Windows). Jeśli Docker jest problemem na maszynie usera — zaproponuj lokalny Postgres + `DATABASE_URL` w `.env` (`.env` w `.gitignore`).

### Stack MVP
- Python, pandas, numpy, scipy, scikit-learn (lub lightgbm — max jedna lib ML)
- SQLAlchemy lub psycopg2 + czysty SQL
- PostgreSQL
- matplotlib / plotly
- Streamlit (dashboard na końcu, nie na początku)
- pytest

---

## 2. CEL PROJEKTU

Zbudować **Quant Research & Market Risk Warehouse**:

> Relacyjna baza z instrumentami, cenami, wyczyszczonymi feature’ami, portfelami i wynikami eksperymentów; pipeline ETL/cleaning; warstwa algebry (kowariancja / PCA / ryzyko portfela); symulacje stochastyczne (MC risk); metody statystyki obliczeniowej (bootstrap); jeden model ML; dashboard do eksploracji.

### Pytanie badawcze (trzymaj się tego)
> Jak zbudować powtarzalny, point-in-time pipeline danych rynkowych i użyć go do pomiaru ryzyka portfela oraz prostego ML — bez look-ahead bias — oraz jak wyniki zmieniają się przy bootstrapie / MC?

### Co projekt ma pokazać na CV
- projektowanie schematu DB + SQL,
- data preparation na serio,
- algebra w finansach (\(w^\top\Sigma w\), PCA),
- procesy stochastyczne w MC,
- computational statistics (bootstrap),
- wąski, uczciwy ML,
- wizualizacja / proste UI.

---

## 3. MVP vs LATER

### MVP (musi być)
1. Schema Postgres (instruments, prices_raw, prices_clean, features, portfolios, portfolio_weights, risk_runs, model_runs, predictions).
2. ETL: pobranie danych (start: **yfinance lub CSV**; universe mały, np. 8–15 płynnych ETF/akcji + benchmark) → staging → clean.
3. Cleaning: missing values, sorted dates, adjusted prices, basic outlier caps; dokumentacja założeń.
4. Features: daily returns, rolling vol, rolling corr/beta vs benchmark, drawdown.
5. Portfolio: wagi (equal weight + 1–2 ręczne zestawienia), zwrot portfela \(R_p = w^\top R\).
6. Risk: historical VaR/ES + parametric \(w^\top\Sigma w\) + Monte Carlo (multivariate normal lub GBM-like).
7. Bootstrap: CI dla VaR lub dla mean/Sharpe portfela (wybrać jedno i zrobić dobrze).
8. ML (wybrać **jedno**):
   - **Opcja A (rekomendowana):** klasyfikacja / prognoza **reżimu zmienności** (np. high/low vol na horyzoncie 5–21 dni) na feature’ach z przeszłości; metrics: AUC / precision-recall; purged time split.
   - **Opcja B:** quantile model / score ogona ryzyka.
   - **Opcja C:** cross-sectional rank tylko jeśli user ogarnie point-in-time panel — trudniejsze.
9. Zapis każdego risk_run / model_run do DB (parametry + metryki + timestamp).
10. Streamlit: wybór portfela, dat, metody VaR; wykresy NAV, drawdown, heatmapa korelacji, VaR w czasie, wynik ML.
11. Testy pytest (logika returns/VaR/bootstrap + kilka testów SQL na sqlite lub testowym Postgresie).
12. README EN z diagramem schematu, metodami, limitacjami.

### Nice-to-have (dopiero po MVP)
- Survivorship-free większy universe, corporate actions table
- Ledoit–Wolf shrinkage
- EWMA cov (RiskMetrics)
- Kupiec rolling backtest (jeśli nie ma jeszcze z risk-engine — można dodać)
- Airflow/Prefect (nie potrzebne na start)
- Auth / multi-user

### Out of scope
- Live trading / składanie zleceń na XTB
- „Optymalny portfel który bije rynek”
- Pełny Markowitz optimizer jako sedno → to osobny przyszły projekt *Portfolio Construction*; tu wolno mieć tylko equal weight + manual weights + risk measurement
- HFT / full order book

---

## 4. SUGEROWANA STRUKTURA REPO

```text
quant-research-warehouse/
  README.md
  requirements.txt
  .gitignore
  .env.example
  docker-compose.yml          # opcjonalnie: postgres
  sql/
    001_schema.sql
    002_views.sql
  src/
    __init__.py
    config.py                 # DATABASE_URL, paths
    db/
      connection.py
      repository.py           # insert/select helpers
    etl/
      download.py
      staging.py
      clean_prices.py
    features/
      returns.py
      rolling.py
      build_feature_table.py
    portfolio/
      weights.py
      portfolio_returns.py
    risk/
      covariance.py
      var_historical.py
      var_parametric.py
      var_montecarlo.py
      expected_shortfall.py
    stats/
      bootstrap.py
    ml/
      dataset.py              # time-aware splits
      volatility_regime.py    # lub inny wybrany model
      evaluate.py
    viz/
      plots.py
  app/
    streamlit_app.py
  scripts/
    init_db.py
    run_etl.py
    run_features.py
    run_risk.py
    run_ml.py
    run_all_demo.py
  tests/
    ...
  reports/
    figures/
```

---

## 5. SCHEMAT DB (MVP — agent ma rozwinąć i wytłumaczyć userowi od zera)

```sql
-- szkic; doprecyzuj typy/klucze wspólnie z userem w Kroku 1

CREATE TABLE instruments (
  instrument_id SERIAL PRIMARY KEY,
  ticker TEXT NOT NULL UNIQUE,
  name TEXT,
  asset_class TEXT,          -- e.g. equity, etf
  currency TEXT,
  is_benchmark BOOLEAN DEFAULT FALSE
);

CREATE TABLE prices_raw (
  ticker TEXT NOT NULL,
  dt DATE NOT NULL,
  open DOUBLE PRECISION,
  high DOUBLE PRECISION,
  low DOUBLE PRECISION,
  close DOUBLE PRECISION,
  adj_close DOUBLE PRECISION,
  volume DOUBLE PRECISION,
  ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (ticker, dt)
);

CREATE TABLE prices_clean (
  instrument_id INT REFERENCES instruments(instrument_id),
  dt DATE NOT NULL,
  adj_close DOUBLE PRECISION NOT NULL,
  quality_flag TEXT,
  PRIMARY KEY (instrument_id, dt)
);

CREATE TABLE features (
  instrument_id INT REFERENCES instruments(instrument_id),
  dt DATE NOT NULL,
  ret_1d DOUBLE PRECISION,
  vol_21d DOUBLE PRECISION,
  beta_63d DOUBLE PRECISION,
  -- dodawaj kolejne kolumny świadomie
  PRIMARY KEY (instrument_id, dt)
);

CREATE TABLE portfolios (
  portfolio_id SERIAL PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  description TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE portfolio_weights (
  portfolio_id INT REFERENCES portfolios(portfolio_id),
  instrument_id INT REFERENCES instruments(instrument_id),
  weight DOUBLE PRECISION NOT NULL,
  valid_from DATE NOT NULL,
  valid_to DATE,
  PRIMARY KEY (portfolio_id, instrument_id, valid_from)
);

CREATE TABLE risk_runs (
  run_id SERIAL PRIMARY KEY,
  portfolio_id INT REFERENCES portfolios(portfolio_id),
  method TEXT NOT NULL,          -- historical / parametric / monte_carlo
  alpha DOUBLE PRECISION NOT NULL,
  horizon_days INT NOT NULL,
  window_start DATE,
  window_end DATE,
  var_value DOUBLE PRECISION,
  es_value DOUBLE PRECISION,
  params_json JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE model_runs (
  run_id SERIAL PRIMARY KEY,
  model_name TEXT NOT NULL,
  train_start DATE,
  train_end DATE,
  test_start DATE,
  test_end DATE,
  metrics_json JSONB,
  params_json JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE predictions (
  run_id INT REFERENCES model_runs(run_id),
  dt DATE NOT NULL,
  instrument_id INT,
  y_true DOUBLE PRECISION,
  y_pred DOUBLE PRECISION,
  PRIMARY KEY (run_id, dt, instrument_id)
);
```

Agent ma wytłumaczyć: PK/FK, dlaczego `prices_raw` vs `prices_clean`, po co `risk_runs` (reprodukowalność), po co `valid_from/valid_to` przy wagach.

---

## 6. PLAN KROKÓW (ucz w tej kolejności)

### Krok 0 — Mapa systemu (teoria, mało kodu)
Wytłumacz od zera:
- co to warehouse vs zwykły folder CSV,
- staging → clean → features → experiments,
- look-ahead bias i point-in-time (na przykładzie „użycie jutrzejszej ceny dziś”),
- survivorship bias (intuicja),
- jak SQL + Python dzielą robotę.

Output: user opisuje własnymi słowami architekturę + wybiera ML opcję A/B.

Potem: scaffold repo + `docker-compose`/`README` setup DB.

---

### Krok 1 — Schema + połączenie
- Teoria: normalizacja, klucze, typy dat.
- `sql/001_schema.sql`, `init_db.py`, test połączenia.
- User uruchamia Postgres i tworzy tabele.

---

### Krok 2 — ETL download + staging
- Pobierz OHLCV dla małego universe.
- Wstaw do `prices_raw` (idempotentny upsert).
- Test: liczba wierszy, brak duplikatów PK.

---

### Krok 3 — Cleaning → `prices_clean`
Teoria od zera: adjusted close, braki, ffill vs leave-missing, outliery.
Implementacja reguł + `quality_flag`.
Testy na sztucznych ramkach (dziury, złe daty).

---

### Krok 4 — Features
Teoria: simple returns, rolling vol, beta vs benchmark (algebra regresa / cov).
Zapis do `features`.
Wykres przykładowy: cena + vol.

Wzory (powtórz LaTeX + słownie, bo IDE często nie renderuje):

$$
R_t = \frac{P_t}{P_{t-1}} - 1,
\qquad
\hat\beta = \frac{\mathrm{Cov}(R, R_b)}{\mathrm{Var}(R_b)}
$$

---

### Krok 5 — Portfolio returns + algebra ryzyka
Teoria:

$$
R_{p,t} = w^\top R_t,
\qquad
\sigma_p^2 = w^\top \Sigma w
$$

Equal weight + jeden portfel ręczny w DB.
Testy 2×2 na kartce.

---

### Krok 6 — Risk engine warstwa (H / P / MC) + zapis `risk_runs`
- Historical VaR/ES
- Parametric Gaussian VaR z \(w^\top\Sigma w\)
- Monte Carlo multivariate normal
Konwencja strat: ustalić jak w risk-engine, np. \(L = -R_p\), VaR = kwantyl \(\alpha\) strat.
Nie trzeba tu powielać całego Kupca, jeśli user ma osobny risk project — ale podstawowe 3 metody **tak**.

---

### Krok 7 — Bootstrap (computational statistics)
Teoria od zera: resampling, CI, dlaczego w finansach i.i.d. bootstrap jest ułomny; na MVP zrób block bootstrap **albo** zwykły z uczciwym disclaimerem w README.
Zastosowanie: CI dla VaR lub Sharpe.
Test: na sztucznej próbce CI pokrywa znany parametr w sensownym odsetku (smoke).

---

### Krok 8 — ML module (jeden)
Dla opcji A (vol regime):
- label z przyszłości **ostrożnie** (np. zrealizowana vol w \([t+1, t+h]\)), features tylko ≤ \(t\),
- TimeSeriesSplit / purge gap,
- baseline (np. „zawsze low vol” / rolling vol threshold),
- metryki + zapis `model_runs` / `predictions`,
- wykres: predicted regime vs realized vol.

Zakaz: losowy train_test_split na szeregu czasowym bez omówienia wycieku.

---

### Krok 9 — Streamlit dashboard
Prosty, czytelny:
- selektor portfela / dat,
- NAV & drawdown,
- correlation heatmap,
- VaR comparison bar/table,
- ML metrics + mini plot.
Dashboard **czyta z DB**, nie z ukrytych globali w pamięci.

---

### Krok 10 — Packaging
- README EN: architecture diagram (może mermaid), how to run, methods, results figures, limitations (szczerze).
- Posprzątaj TODO/Step N z docstrings.
- Zaproponuj commit message; **user sam commituje**.

---

## 7. KONWENCJE

Ustal z userem w Kroku 0 i zapisz w README:

- Daily **simple** returns.
- Loss \(L = -R_p\) dla VaR/ES.
- Weights sum to 1; long-only w MVP.
- Seed dla MC/bootstrap/ML.
- Timestamps w UTC w DB.
- Każdy eksperyment = wiersz w `*_runs` + params JSON.

Wzory w odpowiedziach: używaj `$$...$$`, ale **duplikuj kluczowe definicje słownie**, bo user często nie widzi renderu MathJax w IDE.

---

## 8. RELACJA DO INNYCH PROJEKTÓW USERA

| Projekt | Rola |
|---------|------|
| Options lab | wycena / hedging / Grecy |
| Market risk engine | głęboki fokus na metodologii VaR/ES/backtest/stress |
| **Ten warehouse** | data platform + research loop + risk + 1 ML + viz |
| Przyszły portfolio opt | Markowitz / risk parity / constraints — **NIE mieszać jako sedno tutaj** |

Wolno współdzielić idee wzorów VaR, ale **kod trzymaj niezależny** (czyste repo).

---

## 9. PIERWSZA WIADOMOŚĆ AGENTA DO USERA

1. Potwierdź Windows + venv + Postgres plan (Docker vs lokalnie — zapytaj co woli).
2. Streść cel projektu w 6–8 zdaniach.
3. Zapytaj o wybór ML: A vol-regime (rekomendowane) / B quantile / C cross-section.
4. Zaproponuj start: **Krok 0 teoria architektury** + scaffold.
5. Nie generuj od razu całego ETL/ML/dashboardu.

---

## 10. CHECKLISTA „DONE”

- [ ] Postgres schema działa, dane raw→clean→features
- [ ] Portfel + 3 metody ryzyka zapisują `risk_runs`
- [ ] Bootstrap CI zaimplementowany i wyjaśniony
- [ ] Jeden ML z time-aware walidacją i metrykami w DB
- [ ] Streamlit czyta z DB i pokazuje ≥4 wykresy/sekcje
- [ ] Testy przechodzą
- [ ] README kompletny z limitacjami (look-ahead, mały universe, Gaussian tails, yfinance quality)
- [ ] User potrafi ustnie obronić: schemat DB, point-in-time, \(w^\top\Sigma w\), MC, bootstrap, split ML

---

## 11. CZEGO UNIKAĆ W NARRACJI PROJEKTU

- „Aplikacja do bogacenia się na XTB”
- „AI picks winning stocks”
- Ukrywanie limitów danych yfinance
- Dashboard bez silnika / silnik bez reprodukowalności w DB

---

**Koniec prompta.** Agent: stosuj się ściśle. Priorytet: zrozumienie użytkownika i czysta architektura danych > ilość featurów ML.
