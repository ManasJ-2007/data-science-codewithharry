# Data collection and SQL

## Collecting data
- **Files:** `pd.read_csv`, `read_json`, `read_excel`, `read_parquet` (fast, typed, compressed).
- **APIs:** `requests.get(url, params=..., timeout=10).json()`. Respect rate limits, keep keys in environment variables, and never put them in code.
- **Web scraping:** `requests` plus `BeautifulSoup` (`soup.select("table tr")`), or `pd.read_html` for simple tables. Check the site's terms and `robots.txt` first.
- **Databases:** `pd.read_sql(query, connection)`.

## SQL I can write
```sql
-- top 5 cities by average salary, among cities with at least 20 employees
SELECT city, AVG(salary) AS avg_salary, COUNT(*) AS n
FROM employees
WHERE joined >= '2022-01-01'
GROUP BY city
HAVING COUNT(*) >= 20
ORDER BY avg_salary DESC
LIMIT 5;
```
| Concept | Note |
|---|---|
| Logical order | `FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT`, which is why `WHERE` can't use a `SELECT` alias |
| `WHERE` vs `HAVING` | filters rows before grouping vs groups after |
| Joins | `INNER` keeps matches only; `LEFT` keeps every row of the left table (no match → NULL); a self-join compares rows of the same table |
| NULL | `= NULL` is never true, so use `IS NULL` / `COALESCE(x, 0)` |
| Sub-queries and CTEs | `WITH recent AS (SELECT ...) SELECT ... FROM recent` keeps queries readable |
| Window functions | `ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC)` gives top-n per group; `SUM(x) OVER (ORDER BY day)` gives a running total |
| Keys and constraints | primary key, foreign key, `UNIQUE`, `NOT NULL`, and indexes for fast lookups |

**Pandas ↔ SQL:** `merge` = `JOIN`, `groupby().agg()` = `GROUP BY`, `query` = `WHERE`, `sort_values` = `ORDER BY`.
