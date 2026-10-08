"""Sensor-log MCP server for the Smart Hardware Diagnostic Agent.

Tools (all return JSON-friendly dicts, never raw DataFrames):
  - query_sensor_log          : min/max/mean of a metric over a time window
  - calculate_temperature_rate: how fast the temperature is changing
  - detect_anomaly            : find readings that deviate from normal
  - plot_signal               : save a chart of a metric as a PNG

Run directly (stdio transport):  python sensor_tool/sensor_tool.py
IMPORTANT: never use print() in this file. A stdio MCP server uses stdout
for its protocol messages, so stray prints corrupt the connection.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: save to file, never open a window
import matplotlib.pyplot as plt
import pandas as pd
from fastmcp import FastMCP

mcp = FastMCP("sensor_tool")

# Anchored to this file, so it works no matter where the server is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "Data" / "sensor_logs.csv"
PLOTS_DIR = PROJECT_ROOT / "Data" / "plots"

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
METRIC_UNITS = {"temperature": "°C", "pressure": "kPa", "humidity": "%"}


def load_sensor_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _metric_error(df: pd.DataFrame, metric: str) -> dict | None:
    """Return an error dict if `metric` isn't a numeric sensor column, else None."""
    valid = [c for c in df.columns if c != "timestamp"]
    if metric not in valid:
        return {"error": f"Unknown metric: {metric!r}. Available metrics: {valid}"}
    return None


def _window(df: pd.DataFrame, start_time: str | None, end_time: str | None) -> pd.DataFrame:
    """Filter to [start_time, end_time]; either bound may be None/empty for 'no limit'."""
    if start_time:
        df = df[df["timestamp"] >= pd.to_datetime(start_time)]
    if end_time:
        df = df[df["timestamp"] <= pd.to_datetime(end_time)]
    return df


def _ts(value) -> str:
    return pd.Timestamp(value).strftime(TIME_FORMAT)


@mcp.tool
def query_sensor_log(
    metric: str,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict:
    """Get summary statistics (count, min, max, mean) for one sensor metric
    over a time window, including when the min and max occurred.

    Args:
        start_time: Optional window start, e.g. "2026-10-08 13:00:00". Omit for the start of the log.
        end_time: Optional window end. Omit for the end of the log.
        metric: A numeric sensor column in the CSV. The sample CSV contains temperature, pressure, and humidity; unknown metrics return an error.
    """
    df = load_sensor_data()
    if (err := _metric_error(df, metric)):
        return err

    filtered = _window(df, start_time, end_time)
    if filtered.empty:
        return {
            "error": "No readings in that time window.",
            "log_covers": f"{_ts(df['timestamp'].min())} to {_ts(df['timestamp'].max())}",
        }

    values = filtered[metric]
    return {
        "metric": metric,
        "unit": METRIC_UNITS.get(metric, "unknown"),
        "start_time": start_time,
        "end_time": end_time,
        "count": int(len(values)),
        "min": float(values.min()),
        "min_at": _ts(filtered.loc[values.idxmin(), "timestamp"]),
        "max": float(values.max()),
        "max_at": _ts(filtered.loc[values.idxmax(), "timestamp"]),
        "mean": float(values.mean()),
    }


@mcp.tool
def calculate_temperature_rate(
    start_time: str | None = None,
    end_time: str | None = None,
    threshold_c_per_min: float = 1.0,
) -> dict:
    """Measure how quickly the temperature is rising (degrees C per minute,
    averaged over 3 readings to ignore single-reading noise) and report the
    longest sustained period where it rose faster than a threshold. Useful for
    spotting overheating ramps that a simple min/max would not reveal.

    Args:
        start_time: Optional window start, e.g. "2026-10-08 12:00:00".
        end_time: Optional window end.
        threshold_c_per_min: Rise rate (C/min) above which the temperature counts as rising fast.
    """
    df = _window(load_sensor_data(), start_time, end_time).copy()
    if len(df) < 4:
        return {"error": "Need at least 4 readings in the window to compute a smoothed rate."}

    minutes = df["timestamp"].diff().dt.total_seconds() / 60
    raw_rate = df["temperature"].diff() / minutes.where(minutes > 0)  # avoid divide-by-zero
    df["rate"] = raw_rate.rolling(3).mean()  # smooth out single-reading noise
    df = df.dropna(subset=["rate"])

    above = df["rate"] > threshold_c_per_min
    peak = df.loc[df["rate"].idxmax()]
    result = {
        "threshold_c_per_min": threshold_c_per_min,
        "max_rise_c_per_min": round(float(peak["rate"]), 2),
        "max_rise_at": _ts(peak["timestamp"]),
        "readings_above_threshold": int(above.sum()),
    }
    if not above.any():
        return result

    # Longest unbroken run of fast-rising readings = the real event, not stray blips.
    run_id = (above != above.shift()).cumsum()
    runs = [g for _, g in df[above].groupby(run_id[above])]
    longest = max(runs, key=len)
    result.update({
        "longest_rise_start": _ts(longest["timestamp"].iloc[0]),
        "longest_rise_end": _ts(longest["timestamp"].iloc[-1]),
        "longest_rise_readings": int(len(longest)),
        "temperature_at_rise_start": float(longest["temperature"].iloc[0]),
        "temperature_at_rise_end": float(longest["temperature"].iloc[-1]),
    })
    return result


@mcp.tool
def detect_anomaly(
    metric: str,
    method: str = "robust",
    threshold: float | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict:
    """Find readings of a metric that deviate abnormally from its normal level,
    and report when the abnormal period started, ended and how extreme it got.

    Args:
        metric: A numeric sensor column in the CSV. The sample CSV contains temperature, pressure, and humidity; unknown metrics return an error.
        method: "robust" (default) uses the median and MAD, which a large fault
            cannot distort. "zscore" uses mean and standard deviation, which a
            large fault inflates, so it only catches the most extreme readings.
        threshold: Score above which a reading is anomalous. Defaults to 3.5 for
            "robust" and 3.0 for "zscore".
        start_time: Optional inclusive analysis-window start timestamp.
        end_time: Optional inclusive analysis-window end timestamp.
    """
    all_rows = load_sensor_data()
    df = _window(all_rows, start_time, end_time).copy()
    if (err := _metric_error(df, metric)):
        return err
    if df.empty:
        return {
            "error": "No readings in that time window.",
            "log_covers": f"{_ts(all_rows['timestamp'].min())} to {_ts(all_rows['timestamp'].max())}",
        }
    if method not in ("robust", "zscore"):
        return {"error": f"Unknown method: {method!r}. Use 'robust' or 'zscore'."}

    # Score only the requested target window, but estimate "normal" from the
    # complete log. Otherwise a sustained fault inside the target window can
    # shift its own median and hide itself.
    values = df[metric]
    baseline_values = all_rows[metric]
    if method == "robust":
        center = float(baseline_values.median())
        mad = float((baseline_values - center).abs().median())
        spread = mad / 0.6745  # scales MAD to be comparable to a standard deviation
        threshold = 3.5 if threshold is None else threshold
    else:
        center = float(baseline_values.mean())
        spread = float(baseline_values.std())
        threshold = 3.0 if threshold is None else threshold

    if not spread or pd.isna(spread):
        return {"error": f"No variation in {metric}; cannot score anomalies."}

    df["score"] = (values - center) / spread
    anomalies = df[df["score"].abs() > threshold]

    result = {
        "metric": metric,
        "unit": METRIC_UNITS.get(metric, "unknown"),
        "method": method,
        "threshold": threshold,
        "normal_level": center,
        "spread": spread,
        "baseline_scope": "entire_log",
        "baseline_readings": int(len(baseline_values)),
        "readings_checked": int(len(df)),
        "anomaly_count": int(len(anomalies)),
        "window_start": start_time,
        "window_end": end_time,
    }
    if anomalies.empty:
        return result

    peak = anomalies.loc[anomalies["score"].abs().idxmax()]
    result.update({
        "first_anomaly_at": _ts(anomalies["timestamp"].iloc[0]),
        "last_anomaly_at": _ts(anomalies["timestamp"].iloc[-1]),
        "peak_value": float(peak[metric]),
        "peak_at": _ts(peak["timestamp"]),
        "sample_anomalies": [  # first 10 only, to keep the response short
            {"timestamp": _ts(r["timestamp"]), metric: float(r[metric]), "score": round(float(r["score"]), 2)}
            for _, r in anomalies.head(10).iterrows()
        ],
    })
    return result


@mcp.tool
def plot_signal(metric: str, start_time: str, end_time: str) -> dict:
    """Save a line chart of one sensor metric over a time window as a PNG
    image and return the file path.

    Args:
        metric: One of "temperature", "pressure", "humidity".
        start_time: Window start, e.g. "2026-10-08 12:30:00".
        end_time: Window end, e.g. "2026-10-08 14:00:00".
    """
    df = load_sensor_data()
    if (err := _metric_error(df, metric)):
        return err

    filtered = _window(df, start_time, end_time)
    if filtered.empty:
        return {"error": "No readings in that time window."}

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_start = pd.to_datetime(start_time).strftime("%Y%m%d_%H%M")
    safe_end = pd.to_datetime(end_time).strftime("%Y%m%d_%H%M")
    out_path = PLOTS_DIR / f"{metric}_{safe_start}_{safe_end}.png"

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(filtered["timestamp"], filtered[metric], marker="o", markersize=3, linestyle="-")
    ax.set_title(f"{metric.capitalize()} from {start_time} to {end_time}")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel(metric.capitalize())
    ax.grid(True)
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return {"metric": metric, "points_plotted": int(len(filtered)), "image_path": str(out_path)}


if __name__ == "__main__":
    mcp.run()
