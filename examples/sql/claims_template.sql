-- Template query for SQL Server (pyodbc)
-- Expected output columns for ClaimsCollection:
-- id, uw_year, period, paid, outstanding

SELECT
    CAST(c.claim_id AS varchar(100)) AS id,
    DATEFROMPARTS(c.uw_year, 1, 1) AS uw_year,
    EOMONTH(c.valuation_date) AS period,
    CAST(SUM(c.paid_amount) AS float) AS paid,
    CAST(SUM(c.outstanding_amount) AS float) AS outstanding
FROM dbo.claim_movements c
WHERE c.segment = ?
  AND c.uw_year BETWEEN ? AND ?
GROUP BY
    c.claim_id,
    c.uw_year,
    EOMONTH(c.valuation_date)
ORDER BY
    c.uw_year,
    period,
    id;
