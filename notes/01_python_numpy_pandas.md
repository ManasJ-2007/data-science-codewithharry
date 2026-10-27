# Python, NumPy and Pandas for data science

## Python essentials I used most
- **Comprehensions:** `[u["id"] for u in users if u["friends"]]`, plus dict and set comprehensions for fast lookups.
- **`set` operations** are what make a recommender cheap: `mine & theirs` gives mutual friends, and `theirs - mine` gives candidates (project 1).
- **`collections`:** `Counter` for counting, `defaultdict(float)` for accumulating scores.
- **`json`** to load and save nested data. **`sorted(..., key=lambda kv: (-kv[1], kv[0]))`** sorts by score, then breaks ties by id.
- **conda / venv:** one environment per project, with dependencies pinned in `requirements.txt`.

## NumPy
NumPy arrays are typed, contiguous blocks of memory, so maths on a whole array runs in compiled C instead of a Python loop.

```python
a = np.arange(1_000_000)
a * 2            # vectorised: about 100x faster than [x * 2 for x in list]
a.reshape(1000, 1000).sum(axis=0)   # axis=0 collapses rows, giving one value per column
a[a % 2 == 0]    # boolean mask
a[[0, 10, 20]]   # fancy indexing (returns a copy)
a[10:20]         # slicing (returns a VIEW: writing to it changes `a`)
```
- **Broadcasting:** shapes are aligned from the right, and size-1 dimensions stretch. For example, `X - X.mean(axis=0)` centres every column.
- Useful functions: `np.linspace`, `np.where`, `np.unique`, `np.percentile`, `np.corrcoef`, `np.random.default_rng(seed)`.

## Pandas
A `Series` is a labelled 1-D array, and a `DataFrame` is a dictionary of Series that share one index.

| Task | Code |
|---|---|
| Explore | `df.info()`, `df.describe()`, `df["col"].value_counts()` |
| Select | `df.loc[rows, cols]` (by label), `df.iloc[i, j]` (by position), `df.query("age > 30 and city == 'Delhi'")` |
| Clean | `df.isna().sum()`, `fillna`, `dropna`, `drop_duplicates`, `astype`, `.str.strip().str.lower()` |
| Transform | `assign`, `apply` / `map`, `pd.cut` (binning) |
| Aggregate | `df.groupby("city")["salary"].agg(["mean", "median", "count"])`, `pivot_table` |
| Combine | `pd.merge(a, b, on="id", how="left")`, `pd.concat` |
| Time series | `pd.to_datetime`, `.dt.month`, `resample("M")`, `rolling(7).mean()` |

**Applied in:** project 2 uses `pd.cut` on income to build strata for the train/test split, plus `value_counts(normalize=True)` to check that the split preserved them.
