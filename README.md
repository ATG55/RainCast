# RainCast

Monthly precipitation forecasts for **Istanbul** and **Izmir** (2025–2030) using machine learning, with an interactive web dashboard.

**Live demo:** https://ATG55.github.io/RainCast/

*Built for NASA Space Apps Challenge — Be An Earth System Trend Detective!*

## What it does
- Trains Random Forest, Extra Trees, Gradient Boosting and XGBoost models on monthly precipitation (2001–2024).
- Uses lag features (1, 2, 3, 12 months), a 12-month rolling mean and seasonal (sin/cos) terms, on a log1p-transformed target.
- Forecasts 72 months ahead recursively.
- Backtests on 2019–2024 (trained on 2001–2018) and compares against a monthly climatology baseline using RMSE.
- Displays everything in a static web dashboard (`index.html`).

## Data
**Dataset:** NASA GPM IMERG Final Run, Monthly, Version 07 (`GPM_3IMERGM_07`), 0.1° × 0.1° resolution.
Catalog entry: https://search.earthdata.nasa.gov/search?q=GPM_3IMERGM_07

**Access:** monthly precipitation was downloaded through the USF ERDDAP server (`IMERG_monthly_global_precip`), which serves the IMERG monthly product.

**Period:** January 2001 – December 2024.

**Study areas** (bounding boxes, averaged over all grid cells into one monthly value per region):

| Region   | Latitude      | Longitude     |
|----------|---------------|---------------|
| Istanbul | 40.85 – 41.25 | 28.75 – 29.15 |
| Izmir    | 38.25 – 38.65 | 26.95 – 27.35 |

Example request (Istanbul):
```
https://erddap.marine.usf.edu/erddap/griddap/IMERG_monthly_global_precip.csv?precipitation[(2001-01-01T00:00:00Z):(2024-12-01T00:00:00Z)][(41.25):(40.85)][(28.75):(29.15)]
```
The resulting file `data2.csv` has the columns `region`, `month`, `precipitation_mm`.

**Citation:** Huffman, G.J., E.F. Stocker, D.T. Bolvin, E.J. Nelkin, J. Tan (2023), *GPM IMERG Final Precipitation L3 1 month 0.1 degree x 0.1 degree V07*, Goddard Earth Sciences Data and Information Services Center (GES DISC). DOI: 10.5067/GPM/IMERG/3B-MONTH/07

## Run it
```bash
pip install -r requirements.txt
python forecast.py          # trains models, writes forecast.json and ml_tahmin_2030.png
python -m http.server       # then open http://localhost:8000
```

## Project structure
```
index.html        Web dashboard (reads forecast.json)
forecast.json     Model outputs consumed by the dashboard
forecast.py       Data prep, models, backtest, forecast, JSON export
data2.csv         Input data
requirements.txt  Python dependencies
LICENSE           MIT License
```

## Limitations
Monthly precipitation is only weakly predictable far ahead. Recursive forecasts tend to converge toward the seasonal average, so results should be read as an expected seasonal profile, not as weather forecasts.

## Team 3 a.m. engineers
- Alperen Taha Güneş ([@ATG55](https://github.com/ATG55))
- Begaim Osmonalieva ([@bego-bego](https://github.com/bego-bego))
- Emre Dumbo ([@EmreDumbo](https://github.com/EmreDumbo))
- Abdulkadir Konaş ([@knsceng](https://github.com/knsceng))

## License
Code is released under the MIT License (see `LICENSE`). The precipitation data comes from NASA GPM IMERG and remains subject to NASA's open data policy; please cite the dataset as shown above.
