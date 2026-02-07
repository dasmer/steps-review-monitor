#!/usr/bin/env python3
"""Generate README.md with Mermaid chart and rate statistics from git history."""

import subprocess
import re
import math
from datetime import datetime, timedelta
from collections import OrderedDict

REPO_URL = "https://apps.apple.com/us/app/steps-simple-pedometer/id1602546738"


def get_review_commits():
    """Parse git log for review_count update commits."""
    result = subprocess.run(
        ["git", "log", "--format=%aI %s", "--reverse"],
        capture_output=True, text=True, check=True,
    )
    pattern = re.compile(
        r"^(\d{4}-\d{2}-\d{2})T\S+\s+chore\(state\): update review_count to (\d+)$"
    )
    entries = []
    for line in result.stdout.strip().split("\n"):
        m = pattern.match(line)
        if m:
            entries.append((m.group(1), int(m.group(2))))
    return entries


def compute_eod_counts(entries):
    """Return ordered dict of date_str -> last review count for that day."""
    eod = OrderedDict()
    for date_str, count in entries:
        eod[date_str] = count
    return eod


def generate_mermaid_chart(eod):
    """Generate Mermaid xychart-beta with thinned x-axis labels."""
    dates = list(eod.keys())
    counts = list(eod.values())

    # Thin labels: show ~13 evenly spaced labels
    n = len(dates)
    label_interval = max(1, n // 13)

    labels = []
    hidden_counter = 0
    for i, d in enumerate(dates):
        dt = datetime.strptime(d, "%Y-%m-%d")
        formatted = dt.strftime("%m/%d")
        if i % label_interval == 0 or i == n - 1:
            labels.append(f'"{formatted}"')
        else:
            # Each hidden label must be unique or Mermaid collapses them
            hidden_counter += 1
            zwsp = "\u200b" * hidden_counter
            labels.append(f'" {zwsp}"')

    y_min = (min(counts) // 100) * 100
    y_max = ((max(counts) // 100) + 1) * 100

    lines = [
        "```mermaid",
        "xychart-beta",
        '    title "EOD Review Count Per Day"',
        f'    x-axis "Date" [{", ".join(labels)}]',
        f"    y-axis \"Review Count\" {y_min} --> {y_max}",
        f"    line [{', '.join(str(c) for c in counts)}]",
        "```",
    ]
    return "\n".join(lines)


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2
    return s[mid]


def _stdev(values, mu=None):
    """Population standard deviation."""
    if len(values) < 2:
        return 0.0
    if mu is None:
        mu = _mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / len(values))


def _percentile(values, p):
    """Linear interpolation percentile (0-100)."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    if n == 1:
        return s[0]
    k = (p / 100) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def compute_daily_deltas(eod):
    """Compute per-day review deltas from EOD counts.

    For consecutive dates, delta = count[i] - count[i-1].
    For gaps (missing days), the total is spread evenly across the gap.
    Returns list of (date_str, delta) for every calendar day in range.
    """
    dates = list(eod.keys())
    counts = list(eod.values())
    deltas = []

    for i in range(1, len(dates)):
        d0 = datetime.strptime(dates[i - 1], "%Y-%m-%d")
        d1 = datetime.strptime(dates[i], "%Y-%m-%d")
        gap_days = (d1 - d0).days
        total_added = counts[i] - counts[i - 1]
        per_day = total_added / gap_days if gap_days > 0 else total_added

        for g in range(1, gap_days + 1):
            day = d0 + timedelta(days=g)
            deltas.append((day.strftime("%Y-%m-%d"), per_day))

    return deltas


def compute_rate_stats(eod):
    """Compute comprehensive rate statistics with distribution metrics."""
    dates = list(eod.keys())
    counts = list(eod.values())

    if len(dates) < 2:
        return None

    last_date = datetime.strptime(dates[-1], "%Y-%m-%d")
    first_date = datetime.strptime(dates[0], "%Y-%m-%d")
    total_days = (last_date - first_date).days

    deltas = compute_daily_deltas(eod)
    all_delta_values = [d for _, d in deltas]

    # Slice deltas by recency
    def deltas_for_last_n_days(n):
        cutoff = last_date - timedelta(days=n)
        return [v for d, v in deltas
                if datetime.strptime(d, "%Y-%m-%d") > cutoff]

    def deltas_for_window(end_days_ago, window_size):
        window_end = last_date - timedelta(days=end_days_ago)
        window_start = window_end - timedelta(days=window_size)
        return [v for d, v in deltas
                if window_start < datetime.strptime(d, "%Y-%m-%d") <= window_end]

    # Per-period distribution stats
    periods = {}
    for label, n in [("7d", 7), ("14d", 14), ("30d", 30)]:
        vals = deltas_for_last_n_days(n)
        if vals:
            mu = _mean(vals)
            periods[label] = {
                "mean": mu,
                "median": _median(vals),
                "stdev": _stdev(vals, mu),
                "min": min(vals),
                "max": max(vals),
                "total": sum(vals),
                "n": len(vals),
            }

    if all_delta_values:
        mu = _mean(all_delta_values)
        periods["all-time"] = {
            "mean": mu,
            "median": _median(all_delta_values),
            "stdev": _stdev(all_delta_values, mu),
            "min": min(all_delta_values),
            "max": max(all_delta_values),
            "total": sum(all_delta_values),
            "n": len(all_delta_values),
        }

    # Acceleration: compare windows + z-score for significance
    accel = {}
    for window in [7, 14]:
        recent_vals = deltas_for_window(0, window)
        prior_vals = deltas_for_window(window, window)
        if len(recent_vals) >= 2 and len(prior_vals) >= 2:
            recent_mu = _mean(recent_vals)
            prior_mu = _mean(prior_vals)
            recent_sd = _stdev(recent_vals, recent_mu)
            prior_sd = _stdev(prior_vals, prior_mu)

            pct_change = (
                ((recent_mu - prior_mu) / prior_mu) * 100
                if prior_mu > 0 else 0.0
            )

            # Welch's t-test approximation (z-score for large-ish samples)
            # SE of difference of means
            se = math.sqrt(
                (recent_sd ** 2 / len(recent_vals))
                + (prior_sd ** 2 / len(prior_vals))
            ) if (recent_sd > 0 or prior_sd > 0) else 0.0
            z_score = (recent_mu - prior_mu) / se if se > 0 else 0.0

            # Significance thresholds: |z| > 1.96 → p < 0.05, |z| > 2.58 → p < 0.01
            if abs(z_score) >= 2.58:
                significance = "p < 0.01"
            elif abs(z_score) >= 1.96:
                significance = "p < 0.05"
            elif abs(z_score) >= 1.64:
                significance = "p < 0.10"
            else:
                significance = "not significant"

            accel[window] = {
                "recent_mean": recent_mu,
                "prior_mean": prior_mu,
                "recent_stdev": recent_sd,
                "prior_stdev": prior_sd,
                "pct_change": pct_change,
                "z_score": z_score,
                "significance": significance,
            }

    # Z-score of recent 7d mean vs all-time distribution
    z_vs_alltime = None
    if "7d" in periods and "all-time" in periods:
        at = periods["all-time"]
        r7 = periods["7d"]
        if at["stdev"] > 0:
            # SE of the 7d sample mean under all-time distribution
            se = at["stdev"] / math.sqrt(r7["n"])
            z = (r7["mean"] - at["mean"]) / se
            z_vs_alltime = round(z, 2)

    # Percentile of recent 7d mean within all daily rates
    pct_rank = None
    if all_delta_values and "7d" in periods:
        r7_mean = periods["7d"]["mean"]
        below = sum(1 for v in all_delta_values if v < r7_mean)
        pct_rank = round((below / len(all_delta_values)) * 100, 0)

    return {
        "periods": periods,
        "acceleration": accel,
        "z_vs_alltime": z_vs_alltime,
        "pct_rank": pct_rank,
        "total_days": total_days,
    }


def _fmt(v, decimals=1):
    return f"{v:.{decimals}f}"


def generate_rate_section(stats):
    """Generate the rate statistics markdown section."""
    if stats is None:
        return ""

    lines = []

    # Headline: is recent rate significantly different from baseline?
    if 7 in stats["acceleration"]:
        a = stats["acceleration"][7]
        recent = a["recent_mean"]
        prior = a["prior_mean"]
        direction = "accelerating" if a["pct_change"] > 0 else "decelerating"
        sign = "+" if a["pct_change"] > 0 else ""
        sig = a["significance"]

        if sig == "not significant":
            verdict = (
                f"Last 7d averaged **{_fmt(recent)}/day** vs "
                f"**{_fmt(prior)}/day** prior 7d ({sign}{_fmt(a['pct_change'])}%) "
                f"— **not statistically significant** (z = {_fmt(a['z_score'], 2)})"
            )
        else:
            verdict = (
                f"Reviews are **{direction}**: last 7d averaged "
                f"**{_fmt(recent)}/day** vs **{_fmt(prior)}/day** prior 7d "
                f"({sign}{_fmt(a['pct_change'])}%) — "
                f"**statistically significant** ({sig}, z = {_fmt(a['z_score'], 2)})"
            )
        lines.append(verdict)

    # Context vs all-time
    if stats["z_vs_alltime"] is not None and stats["pct_rank"] is not None:
        z = stats["z_vs_alltime"]
        pct = int(stats["pct_rank"])
        at = stats["periods"]["all-time"]
        lines.append("")
        lines.append(
            f"Current 7d rate sits at the **{pct}th percentile** of all daily rates "
            f"(z = {z} vs all-time mean of {_fmt(at['mean'])}/day)"
        )

    # Distribution table
    lines.append("")
    lines.append("### Daily Rate Distribution")
    lines.append("")
    lines.append(
        "| Period | Mean/day | Median/day | Std Dev | Min | Max | N days |"
    )
    lines.append(
        "|--------|----------|------------|---------|-----|-----|--------|"
    )
    for label in ["7d", "14d", "30d", "all-time"]:
        if label in stats["periods"]:
            p = stats["periods"][label]
            lines.append(
                f"| {label} | {_fmt(p['mean'])} | {_fmt(p['median'])} | "
                f"{_fmt(p['stdev'])} | {_fmt(p['min'])} | {_fmt(p['max'])} | "
                f"{p['n']} |"
            )

    # Acceleration detail
    lines.append("")
    lines.append("### Acceleration Tests")
    lines.append("")
    lines.append(
        "| Window | Recent avg | Prior avg | Change | Std Dev (R) | "
        "Std Dev (P) | z-score | Significant? |"
    )
    lines.append(
        "|--------|-----------|-----------|--------|-------------|"
        "-------------|---------|--------------|"
    )
    for window in [7, 14]:
        if window in stats["acceleration"]:
            a = stats["acceleration"][window]
            sign = "+" if a["pct_change"] > 0 else ""
            lines.append(
                f"| {window}d vs prior {window}d | "
                f"{_fmt(a['recent_mean'])} | {_fmt(a['prior_mean'])} | "
                f"{sign}{_fmt(a['pct_change'])}% | "
                f"{_fmt(a['recent_stdev'])} | {_fmt(a['prior_stdev'])} | "
                f"{_fmt(a['z_score'], 2)} | {a['significance']} |"
            )

    # Interpretation guide
    lines.append("")
    lines.append(
        "> **Reading the stats:** z-score measures how many standard errors "
        "the recent mean is from the prior mean. "
        "|z| > 1.96 → significant at 95% confidence (p < 0.05). "
        "High std dev relative to the mean suggests noisy data where "
        "apparent trends may just be normal variance."
    )

    return "\n".join(lines)


def generate_weekly_breakdown(eod):
    """Generate a weekly breakdown table."""
    dates = list(eod.keys())
    counts = list(eod.values())

    if len(dates) < 2:
        return ""

    # Group by ISO week
    weeks = OrderedDict()
    for i, d in enumerate(dates):
        dt = datetime.strptime(d, "%Y-%m-%d")
        iso_year, iso_week, _ = dt.isocalendar()
        key = (iso_year, iso_week)
        if key not in weeks:
            weeks[key] = {"dates": [], "counts": []}
        weeks[key]["dates"].append(d)
        weeks[key]["counts"].append(counts[i])

    lines = ["| Week | Dates | +Reviews | Avg/day |"]
    lines.append("|------|-------|----------|---------|")

    week_keys = list(weeks.keys())
    prev_last_count = None

    for wk_key in week_keys:
        wk = weeks[wk_key]
        first_d = wk["dates"][0]
        last_d = wk["dates"][-1]
        date_range = f"{first_d} to {last_d}" if first_d != last_d else first_d

        last_count = wk["counts"][-1]
        if prev_last_count is not None:
            added = last_count - prev_last_count
        else:
            added = wk["counts"][-1] - wk["counts"][0]

        first_dt = datetime.strptime(first_d, "%Y-%m-%d")
        last_dt = datetime.strptime(last_d, "%Y-%m-%d")
        num_days = (last_dt - first_dt).days + 1
        avg_per_day = round(added / num_days, 1) if num_days > 0 else 0

        iso_year, iso_week = wk_key
        week_label = f"{iso_year}-W{iso_week:02d}"

        lines.append(
            f"| {week_label} | {date_range} | +{added} | {avg_per_day} |"
        )
        prev_last_count = last_count

    return "\n".join(lines)


def generate_readme(eod, chart, rate_section, weekly):
    """Assemble the full README.md content."""
    current_count = list(eod.values())[-1]
    last_date = list(eod.keys())[-1]

    readme = f"""# Steps Review Monitor

Tracks the App Store review count for [Steps - Simple Pedometer]({REPO_URL}) via a GitHub Actions cron job (every 5 minutes). When the count increases, it commits the new value and opens a GitHub Issue.

**Current count: {current_count}** (as of {last_date})

## EOD Review Count

{chart}

## Rate Statistics

{rate_section}

## Weekly Breakdown

{weekly}

---
*Auto-generated by `scripts/generate_readme.py` — updated twice daily via GitHub Actions.*
"""
    return readme


def main():
    entries = get_review_commits()
    if not entries:
        print("No review_count commits found in git history.")
        return

    eod = compute_eod_counts(entries)
    chart = generate_mermaid_chart(eod)
    stats = compute_rate_stats(eod)
    rate_section = generate_rate_section(stats)
    weekly = generate_weekly_breakdown(eod)
    readme = generate_readme(eod, chart, rate_section, weekly)

    with open("README.md", "w") as f:
        f.write(readme)

    print(f"README.md generated with {len(eod)} data points.")


if __name__ == "__main__":
    main()
