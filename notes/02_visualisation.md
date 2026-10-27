# Data visualisation (Matplotlib and Seaborn)

## Choosing the chart
| Question | Chart |
|---|---|
| How is one variable distributed? | histogram / KDE (`sns.histplot(x, kde=True)`), box plot for outliers |
| How do two numeric variables relate? | scatter (`sns.scatterplot`), with `alpha` to fix over-plotting; `sns.regplot` adds a trend line |
| How do categories compare? | bar chart (sorted), or box/violin plot per category |
| How does something change over time? | line chart |
| Which features are correlated? | heat map of `df.corr()` (`sns.heatmap(annot=True, cmap="coolwarm")`) |
| Where is it happening? | scatter of longitude/latitude, coloured by value |

## Matplotlib object API (the one I use)
```python
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(df.longitude, df.latitude, c=df.median_house_value, s=df.population / 100, alpha=0.4)
ax[0].set(xlabel="Longitude", ylabel="Latitude", title="Prices by location")
ax[1].barh(names, values, xerr=stds)          # error bars = std over CV folds / seeds
fig.tight_layout(); fig.savefig("plot.png", dpi=130)
```

## Rules of thumb
- One message per chart. Put the number or comparison in the title.
- Sort bar charts, start bar axes at zero, and show uncertainty (error bars) when comparing models.
- Use colour for meaning (baseline grey, model blue), not decoration.

**Applied in:** the geographic price map, correlation bars, model-comparison bars with CV error bars, and predicted-vs-actual scatter in project 2. Project 1 has Hit@k bar charts with seed error bars.
