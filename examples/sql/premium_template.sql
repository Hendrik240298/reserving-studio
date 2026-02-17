-- Template query for SQL Server (pyodbc)
-- Expected output columns for PremiumRepository canonical schema:
-- uw_year, period, Premium_selected

SELECT
    DATEFROMPARTS(p.uw_year, 1, 1) AS uw_year,
    EOMONTH(p.valuation_date) AS period,
    CAST(SUM(p.earned_premium) AS float) AS Premium_selected
FROM dbo.premium_triangle p
WHERE p.segment = ?
  AND p.uw_year BETWEEN ? AND ?
GROUP BY
    p.uw_year,
    EOMONTH(p.valuation_date)
ORDER BY
    p.uw_year,
    period;
