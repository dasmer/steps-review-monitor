#!/usr/bin/env python3
"""Generate README.md with Mermaid chart and rate statistics from git history."""

import subprocess
import re
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


def compute_rate_stats(eod):
    """Compute reviews/day rates for various periods and acceleration."""
    dates = list(eod.keys())
    counts = list(eod.values())

    if len(dates) < 2:
        return None

    last_date = datetime.strptime(dates[-1], "%Y-%m-%d")
    first_date = datetime.strptime(dates[0], "%Y-%m-%d")

    def rate_for_period(days):
        """Compute reviews/day for the last N days."""
        cutoff = last_date - timedelta(days=days)
        # Find the count at or just before the cutoff
        start_count = None
        for i, d in enumerate(dates):
            dt = datetime.strptime(d, "%Y-%m-%d")
            if dt <= cutoff:
                start_count = counts[i]
            else:
                break
        if start_count is None:
            # Use first available data point
            start_count = counts[0]
            actual_days = (last_date - first_date).days
            if actual_days == 0:
                return None
            return (counts[-1] - start_count) / actual_days
        actual_days = days
        if actual_days == 0:
            return None
        return (counts[-1] - start_count) / actual_days

    def rate_for_window(end_days_ago, window_size):
        """Compute reviews/day for a window ending N days ago."""
        window_end = last_date - timedelta(days=end_days_ago)
        window_start = window_end - timedelta(days=window_size)
        start_count = None
        end_count = None
        for i, d in enumerate(dates):
            dt = datetime.strptime(d, "%Y-%m-%d")
            if dt <= window_start:
                start_count = counts[i]
            if dt <= window_end:
                end_count = counts[i]
        if start_count is None or end_count is None:
            return None
        if window_size == 0:
            return None
        return (end_count - start_count) / window_size

    rates = {}
    for period in [7, 14, 30]:
        r = rate_for_period(period)
        if r is not None:
            rates[f"{period}d"] = round(r, 1)

    total_days = (last_date - first_date).days
    if total_days > 0:
        rates["all-time"] = round((counts[-1] - counts[0]) / total_days, 1)

    # Acceleration: compare recent window to prior window
    accel = {}
    for period in [7, 14]:
        recent = rate_for_window(0, period)
        prior = rate_for_window(period, period)
        if recent is not None and prior is not None and prior > 0:
            pct_change = ((recent - prior) / prior) * 100
            accel[period] = {
                "recent": round(recent, 1),
                "prior": round(prior, 1),
                "pct_change": round(pct_change, 1),
            }

    return {"rates": rates, "acceleration": accel}


def generate_rate_section(stats):
    """Generate the rate statistics markdown section."""
    if stats is None:
        return ""

    lines = []

    # Acceleration summary
    if 7 in stats["acceleration"]:
        a = stats["acceleration"][7]
        direction = "accelerating" if a["pct_change"] > 0 else "decelerating"
        sign = "+" if a["pct_change"] > 0 else ""
        lines.append(
            f"Reviews are **{direction}**: last 7d averaged "
            f"**{a['recent']}/day** vs **{a['prior']}/day** prior 7d "
            f"({sign}{a['pct_change']}%)"
        )
        lines.append("")

    # Period rates table
    lines.append("| Period | Reviews/day |")
    lines.append("|--------|-------------|")
    for key in ["7d", "14d", "30d", "all-time"]:
        if key in stats["rates"]:
            lines.append(f"| {key} | {stats['rates'][key]} |")

    # 14d acceleration detail
    if 14 in stats["acceleration"]:
        a = stats["acceleration"][14]
        sign = "+" if a["pct_change"] > 0 else ""
        lines.append("")
        lines.append(
            f"14-day trend: **{a['recent']}/day** vs **{a['prior']}/day** "
            f"prior 14d ({sign}{a['pct_change']}%)"
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
