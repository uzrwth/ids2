#!/usr/bin/env python

### Display
from IPython.display import display
import pandas as pd
import numpy as np
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pio.templates.default = 'plotly_white'
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, month_plot, seasonal_plot, plot_predict
from sktime.utils.plotting import plot_series, plot_lags
from statsmodels.tsa.api import STL
from statsmodels.tsa.api import ARIMA as StARIMA
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.base import ForecastingHorizon
from sktime.performance_metrics.forecasting import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# (a)	OVERVIEW OVER THE DATA

df = pd.read_csv("./time_series/ufo-sightings.csv", parse_dates=["Date_time"]).set_index('Date_time')
display(f"number of UFO sighting: {len(df)}")

display(f"date range: {df.index.min()} - {df.index.max()}")


frames = []

for i in df.groupby(["Year"]):
    year = i[0][0]
    lat=i[1]['latitude']
    lon=i[1]['longitude']

    frames.append(
        go.Frame(
            data=[
                go.Scattergeo(
                    lon=lon,
                    lat=lat,
                    marker=dict(
                        color='red',
                        size=10,
                    ),
                    mode="markers",
                )
            ],
            name=year
        )
    )


scattergeo = go.Scattergeo(mode="markers")

fig = go.Figure(data=[scattergeo], frames=frames)

fig.update_layout(
    geo=dict(
        projection=dict(type="orthographic"),
        showland=True,
        landcolor="white",
        countrycolor="black",
        showcountries=True,
        lataxis=dict(
            showgrid=True,
        ),
        lonaxis=dict(
            showgrid=True,
        ),
    ),
    updatemenus=[
        dict(
            type="buttons",
            showactive=True,
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=1000, redraw=True),
                            fromcurrent=True,
                            mode="immediate",
                        ),
                    ],
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[
                        [None],
                        dict(
                            mode="immediate",
                            transition=dict(duration=0),
                            frame=dict(duration=0, redraw=True),
                        ),
                    ],
                ),
            ],
        )
    ],
)

fig.show()


# There is a trend that the number of sightings increases every year.

years=[]
occurs=[]
g = df.groupby("Year").groups
for i in g:
	countries_num = len(df.loc[g[i]]["Country"].unique())
	occurrences_per_country = len(g[i])/countries_num
	years.append(i)
	occurs.append(occurrences_per_country)

sp = sns.scatterplot(data=pd.DataFrame({"year": years, "occurs": occurs}), x="year", y="occurs")
sp.set_xlabel("year")
sp.set_ylabel("occurrences per country")
sp.set_title("The sighting occurrences per Country over time")
plt.show()

# The most prominent trend is the number of sighting occurrences increases over the years.


# (b) PREPROCESSING THE DATA INTO CLEAN TIME SERIES

# generate default missing quarters
def gen_quarters(sy, sq, ey, eq):
	i = sy * 4 + (sq - 1)
	j = ey * 4 + (eq - 1)
	out = {}
	while i <= j:
		q = i % 4 + 1
		y = i // 4
		i += 1
		out[y * 10 + q] = 0
	return out


countries = df.groupby("Country").groups
country_time_count = {}
for i in countries:
	name=i
	idx = countries[name]
	cdf = df.loc[idx]
	start = min(idx)
	end = max(idx)
	m = gen_quarters(start.year, start.quarter, end.year, end.quarter)
	for j in cdf.index:
		k = j.year * 10 + j.quarter
		m[k] += 1
	country_time_count[name] = m

usa=country_time_count['United States']
usa_values=list(usa.values())
display(f"First 3/Last 3 rows of the time series for the USA: {usa_values[:3]}, {usa_values[-3:]}")

aus=country_time_count['Australia']
aus_values=list(aus.values())
display(f"First 3/Last 3 rows of the time series for the AUS: {aus_values[:3]}, {aus_values[-3:]}")


s = pd.Series(usa_values)
plot_series(s)
plt.show()


# Everytime the number hits a local peak, it goes down for two quarters and again rises up.

# (c) ts-yearly.csv and ts-monthly-10yrs.csv
ydf = pd.read_csv("./time_series/ts-yearly.csv", parse_dates=["time"]).set_index('time')
mdf = pd.read_csv("./time_series/ts-monthly-10yrs.csv", parse_dates=["time"]).set_index('time')
gydf = ydf.loc[ydf["country"]=="GLOBAL"]
gmdf = mdf.loc[mdf["country"]=="GLOBAL"]
plot_series(gydf["count"])
plot_series(gmdf["count"])
plt.show()

# The time series of sighting counts of yearly data is smooth, but that of the monthly data sees more fluctuations. In the plot for the monthly data, there seems to be a certain pattern of repeating trends. The sighting count will go up for a few months and then down for another few months, repeatedly.


# The montly data is more suitable for forecasting, because we can see repeating pattern in the plot. The yearly data is not suitabe for forecasting, since we don't see any pattern on how the sighting counts change. What we only see in the plot for the yearly data is that the counts keep increasing yearly.


# The dataset only includes records of the first 5 months of 2014, whereas other years contain data records for its corresponding whole year.


# (d) SERIES-SPECIFIC ANALYSIS
aus_mdf = mdf.loc[mdf["country"]=="AUS"]
usa_mdf = mdf.loc[mdf["country"]=="USA"]

def plot_ma(series, ms: list[int]):
    f = go.Figure()
    f.add_trace(go.Scatter(x=series.index, y=series, name='Original'))
    for m in ms:
        f.add_trace(go.Scatter(x=series.iloc[m:-m].index, y=series.rolling(m, center=True).mean(), name=f'{m}-MA'))
    return f

plot_ma(aus_mdf["count"], [3, 5, 7]).show()
plot_ma(usa_mdf["count"], [3, 5, 7]).show()

# px.line(x=usa_mdf.index.month, y=usa_mdf["count"], color=usa_mdf.index.year, title='Overlaid Seasonal Plot of Sighting Counts')

plus_trend = usa_mdf["count"] + np.linspace(0, 1, len(usa_mdf)) * usa_mdf["count"].mean() * 2

f, axs = plt.subplots(4)
plot_acf(plus_trend, lags=40, bartlett_confint=False, ax=axs[0])
for d in [1, 2, 3]:
    plot_acf(plus_trend.diff(d).dropna(), lags=40, bartlett_confint=False, ax=axs[d])
    axs[d].set_title(f'differenced {d} times')

plt.show()

# The lag of 12 is the most significant one.


# plot_month

g = mdf['country'].unique()
f, axs = plt.subplots(len(g))
j = 0
for name in g:
	month_plot(mdf.loc[mdf['country'] == name]['count'].diff(1).dropna(), ax=axs[j])
	axs[j].set_title(f'country: {name}')
	j += 1

plt.show()

# Infer: July may be the month more likely to see ufo sightings. USA has the largest numbers of sighting counts as compared to the other countries.

# STL (statsmodels)

for d in [0, 1, 2, 3]:
	count_diff = usa_mdf["count"].diff(d).dropna()
	if d == 0:
		count_diff = usa_mdf["count"]
	decomp = STL(count_diff, period=12).fit()
	decomp.plot()
	plt.title(f'differencing: {d}')
	px.line(count_diff - decomp.seasonal).show()

plt.show()

# No, none of these differenced versions is almost-stationary.

# (e) FORECASTING

train_df = pd.read_csv("./time_series/ts-train.csv", parse_dates=["time"]).set_index('time')
test_df = pd.read_csv("./time_series/ts-test.csv", parse_dates=["time"]).set_index('time')

i = train_df.index
train_df.index = pd.date_range(min(i), max(i), freq="MS")

i = test_df.index
test_df.index = pd.date_range(min(i), max(i), freq="MS")

def score(y_true, y_pred):
    return pd.Series({'rmse': mean_squared_error(y_true, y_pred, square_root=True),
                      'mae': mean_absolute_error(y_true, y_pred),
                      'mape': mean_absolute_percentage_error(y_true, y_pred)}, name='error')



orders = []
for p in range(12):
	for d in range(4):
		for q in range(12):
			orders.append((p+1, d, q+1))

countries = ["AUS", "CAN", "GBR", "GLOBAL", "IND", "USA"]

fh = np.arange(1,13)

country_preds = []
country_scores = []
best_orders = []


for i in countries:
	key = f'{i}_count'
	last_nf = NaiveForecaster(strategy="last")
	mean_nf = NaiveForecaster(strategy="mean")
	y = train_df[key]
	last_nf.fit(y)
	mean_nf.fit(y)
	y_last_pred = last_nf.predict(fh)
	y_mean_pred = mean_nf.predict(fh)
	y_true = test_df[key]
	mean_score = score(y_true, y_mean_pred)
	last_score = score(y_true, y_last_pred)
	min_rmse = min(mean_score.rmse, last_score.rmse)
	arima_score = pd.Series({'rmse': 0, 'mae': 0, 'mape': 0})
	for order in orders:
		sts_arima = StARIMA(y, order=order, freq='MS').fit()
		y_pred = sts_arima.predict(start=len(y), end=len(y) + len(y_true) - 1)
		arima_score = score(y_true, y_pred)
		if arima_score["rmse"] < min_rmse:
			break
	m, l, a = mean_score, last_score, arima_score
	country_scores.append([i, m.rmse, m.mae, m.mape, l.rmse, l.mae, l.mape, a.rmse, a.mae, a.mape, f"{order}"])
	country_preds.append([i, y_mean_pred, y_last_pred, y_pred])
	best_orders.append(order)

tbl = pd.DataFrame(country_scores, columns=["country", "mean rmse", "mean mae", "mean mape", "last rmse", "last mae", "last mape", "arima rmse", "arima mae", "arima mape", "order"])
display(tbl)
display(country_preds)
display(best_orders)


f, axs = plt.subplots(len(country_preds))
j = 0
for i in country_preds:
	name, m, l, a = i
	key = f'{name}_count'
	y = train_df[key]
	y_true = test_df[key]
	plot_series(y, y_true, m, l, a, labels=['train', 'test', 'mean forecast', 'last forecast', 'ARIMA forecast'], title=f'{name}', ax=axs[j])
	j += 1

plt.show()


# (ii) The USA and Canada have better forecasts than the other countries. Their shapes of the ARIMA forecasts resemble the true data than other countries.


# (iii) These are not very good results. Among the countries, only the training time series for USA and Canada show some seasonality. The shape of the time series for other countries appear very irregular, this attributes to the difficulty of forecasting.