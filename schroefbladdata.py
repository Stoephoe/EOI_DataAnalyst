import os
import re
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from tkinter import Tk, filedialog

# ==================== USER CONFIG ====================
COMBINE_ALL_FILES = True      # Only combined output is supported in this version
INCLUDE_SMOOTHED = True       # Add smoothed overlay traces (median rolling)
SMOOTH_WINDOW_SECONDS = 2     # Smoothing window in seconds
MAX_POINTS_PER_TRACE = 5000   # Downsample if too many points (None disables)
SECONDARY_Y_SIGNALS = ['duty_cycle_p', 'rpm']
# ================ END USER CONFIG ====================

COLOR_SEQ = {
    'rpm':              '#1f77b4',
    'battery_voltage_v':'#2ca02c',
    'battery_current_a':'#ff7f0e',
    'motor_current_a':  '#8c564b',
    'duty_cycle_p':     '#9467bd',
    'gnss_speed_kmh':   '#d62728',
    'motor_power_kw':   '#17becf'
}
LABELS = {
    'timestamp_s':      'Timestamp (s)',
    'rpm':              'Motor RPM',
    'battery_voltage_v':'Battery Voltage (V)',
    'battery_current_a':'Battery Current (A)',
    'motor_current_a':  'Motor Current (A)',
    'duty_cycle_p':     'Duty Cycle (%)',
    'gnss_speed_kmh':   'GPS Speed (km/h)',
    'motor_power_kw':   'Motor Power (kW)',
}
ALIASES = {
    'motor_rpm': 'rpm',
    'rpm_motor':'rpm',
    'vbat':'battery_voltage_v','v_bat':'battery_voltage_v','vbatt':'battery_voltage_v',
    'ibat':'battery_current_a','i_bat':'battery_current_a',
    'motor_current':'motor_current_a','motorcurrent':'motor_current_a',
    'duty':'duty_cycle_p','dutycycle':'duty_cycle_p',
    'speed':'gnss_speed_kmh','speed_kmh':'gnss_speed_kmh'
}

def select_folder():
    """Prompt user for folder selection."""
    root = Tk(); root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Folder with CSV Files")
    if not folder_path:
        raise SystemExit("No folder selected.")
    return folder_path

def parse_datetime_from_filename(filename):
    """Extract datetime from filename."""
    match = re.search(r'candump-(\d{4})-(\d{2})-(\d{2})_(\d{6})', filename)
    if not match:
        return None
    y, m, d, hms = match.groups()
    return datetime.strptime(f"{y}-{m}-{d} {hms[:2]}:{hms[2:4]}:{hms[4:]}", "%Y-%m-%d %H:%M:%S")

def clean_signal(series, window=1):
    """Clip outliers (±3*IQR) then rolling median."""
    if series.isnull().all():
        return series
    s = series.copy()
    clean = s.dropna()
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    clipped = clean.clip(lower=lower_bound, upper=upper_bound)
    smoothed = clipped.rolling(window=window, center=True, min_periods=1).median()
    s.loc[smoothed.index] = smoothed
    return s

def build_time_index(df, file_start_time):
    """Return pandas.DatetimeIndex aligned to first timestamp_s."""
    base = df['timestamp_s'].iloc[0]
    dt_series = pd.to_datetime(file_start_time) + pd.to_timedelta(df['timestamp_s'] - base, unit='s')
    return dt_series

def smooth_by_seconds(series, times, seconds):
    """Rolling median by approximate window length in seconds."""
    if not INCLUDE_SMOOTHED or seconds is None or seconds <= 0:
        return None
    if len(times) < 2:
        return series
    dt = np.median(np.diff(times.view('int64'))) / 1e9  # ns -> s
    if dt <= 0:
        dt = 1
    win = max(int(round(seconds / dt)), 1)
    return series.rolling(window=win, center=True, min_periods=1).median()

def downsample(x, y, max_points):
    if max_points is None or len(y) <= max_points:
        return x, y
    idx = np.linspace(0, len(y)-1, max_points).astype(int)
    return x.iloc[idx], y.iloc[idx]

def load_and_clean_csv_files(folder_path):
    """Load, clean, and parse all usable CSV files in the given folder."""
    csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No CSV files found.")

    datasets = []
    for file in sorted(csv_files):
        path = os.path.join(folder_path, file)
        try:
            df = pd.read_csv(path, sep=None, engine='python')
        except Exception as e:
            print(f"⚠ Skipping {file}: failed to read ({e})")
            continue

        df.columns = [c.strip().lower() for c in df.columns]
        df.rename(columns={k: v for k, v in ALIASES.items() if k in df.columns}, inplace=True)

        if df.empty or 'timestamp_s' not in df:
            print(f"⚠ Skipping {file}: empty or missing timestamp.")
            continue

        # Derived columns
        if 'motor_current_a' in df and 'battery_voltage_v' in df:
            df['motor_power_kw'] = (df['motor_current_a'] * df['battery_voltage_v']) / 1000.0

        # Clean data (skip timestamp)
        for col in df.columns:
            if col != 'timestamp_s':
                df[col] = clean_signal(df[col])

        start_dt = parse_datetime_from_filename(file)
        if start_dt is None:
            print(f"⚠ Skipping {file}: cannot parse start time from name.")
            continue

        try:
            time_index = build_time_index(df, start_dt)
        except Exception as e:
            print(f"⚠ Skipping {file}: {e}")
            continue

        candidate_cols = [c for c in df.columns if c != 'timestamp_s']
        has_data = any(df[c].dropna().abs().sum() > 0 for c in candidate_cols)
        if not has_data:
            print(f"⚠ Skipping {file}: no nonzero data in signals.")
            continue

        datasets.append(dict(
            name=file,
            df=df,
            time=time_index
        ))
        print(f"✅ Loaded {file}: {len(df)} rows, signals={candidate_cols}")

    if not datasets:
        raise SystemExit("No usable datasets found.")

    return datasets

def dataset_stats(d):
    """Generate stats summary string for a dataset."""
    df = d['df']
    duration_s = float(df['timestamp_s'].iloc[-1] - df['timestamp_s'].iloc[0])
    rpm_stats = ""
    if 'rpm' in df:
        rpm_stats = (f" | RPM min {df['rpm'].min():.0f} max {df['rpm'].max():.0f} avg {df['rpm'].mean():.0f}")
    return f"{d['name']} ({duration_s:.1f}s){rpm_stats}"

def generate_combined_plot(datasets, output_file):
    """Create a Plotly figure with dropdown to select among datasets."""
    all_signals = []
    for d in datasets:
        for c in d['df'].columns:
            if c != 'timestamp_s' and c not in all_signals:
                all_signals.append(c)
    # Canonical order
    def sig_sort_key(sig):
        try:
            return list(LABELS.keys()).index(sig)
        except ValueError:
            return 999
    all_signals.sort(key=sig_sort_key)

    fig = go.Figure()
    traces_by_dataset = [[] for _ in datasets]

    for i, data in enumerate(datasets):
        df = data['df']
        t = data['time']
        for sig in all_signals:
            if sig in df.columns:
                xs, ys = t, df[sig]
                ys_smooth = None
                if INCLUDE_SMOOTHED:
                    ys_smooth = smooth_by_seconds(ys, t.values.astype('datetime64[ns]'), SMOOTH_WINDOW_SECONDS)

                xs_ds, ys_ds = downsample(xs, ys, MAX_POINTS_PER_TRACE)
                fig.add_trace(go.Scatter(
                    x=xs_ds, y=ys_ds,
                    mode='lines',
                    name=f"{LABELS.get(sig, sig)} ({data['name']})",
                    line=dict(color=COLOR_SEQ.get(sig, '#888')),
                    visible=True if i == 0 else False,
                    legendgroup=f"{sig}_{i}",
                    hovertemplate="%{x|%H:%M:%S}<br>%{y:.3f}<extra>" + LABELS.get(sig, sig) + f" ({data['name']})</extra>",
                    yaxis='y2' if sig in SECONDARY_Y_SIGNALS else 'y'
                ))
                traces_by_dataset[i].append(len(fig.data) - 1)

                if ys_smooth is not None:
                    xs_ds2, ys_ds2 = downsample(xs, ys_smooth, MAX_POINTS_PER_TRACE)
                    fig.add_trace(go.Scatter(
                        x=xs_ds2, y=ys_ds2,
                        mode='lines',
                        name=f"{LABELS.get(sig, sig)} smoothed ({data['name']})",
                        line=dict(color=COLOR_SEQ.get(sig, '#888'), dash='dot', width=1),
                        visible=False,
                        legendgroup=f"{sig}_{i}",
                        showlegend=False,
                        yaxis='y2' if sig in SECONDARY_Y_SIGNALS else 'y'
                    ))
                    traces_by_dataset[i].append(len(fig.data) - 1)

    # Dropdown buttons to select dataset
    dropdown_buttons = []
    for i, d in enumerate(datasets):
        visible = [False] * len(fig.data)
        for idx in traces_by_dataset[i]:
            visible[idx] = True
        dropdown_buttons.append(dict(
            label=dataset_stats(d),
            method='update',
            args=[
                {'visible': visible},
                {'title': f"Dataset: {dataset_stats(d)}"}
            ]
        ))

    # Layout
    fig.update_layout(
        template='plotly_dark',
        title=f"Dataset: {dataset_stats(datasets[0])}",
        xaxis=dict(
            title="Time (HH:MM:SS)",
            tickformat="%H:%M:%S",
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            showgrid=True,
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=[
                    dict(count=60,  label="1m",  step="second", stepmode="backward"),
                    dict(count=300, label="5m",  step="second", stepmode="backward"),
                    dict(step="all", label="All")
                ]
            )
        ),
        yaxis=dict(
            title="Primary Signals",
            showgrid=True,
            ticksuffix=""
        ),
        yaxis2=dict(
            title="Secondary Signals",
            overlaying='y',
            side='right',
            showgrid=False,
            ticksuffix=""
        ),
        hovermode="x unified",
        updatemenus=[dict(
            active=0,
            buttons=dropdown_buttons,
            x=0.5, xanchor="center",
            y=1.12, yanchor="top"
        )],
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.40,           # below the bottom of the plot, adjust as needed
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.7)'
            ),
        height=800,
        margin=dict(t=160, b=100, l=70, r=80)  # t: top, b: bottom
    )

    # Global annotations (initial dataset)
    annotations = []
    d0 = datasets[0]
    if 'rpm' in d0['df']:
        annotations.append(dict(
            x=0.01, y=-1.10, xref='paper', yref='paper',
            text=f"Max RPM: {d0['df']['rpm'].max():.0f}",
            showarrow=False, font=dict(size=12, color="gray")
        ))
    if 'gnss_speed_kmh' in d0['df']:
        annotations.append(dict(
            x=0.25, y=-1.10, xref='paper', yref='paper',
            text=f"Avg Speed: {d0['df']['gnss_speed_kmh'].mean():.1f} km/h",
            showarrow=False, font=dict(size=12, color="gray")
        ))
    annotations.append(dict(
        x=0.6, y=-1.10, xref='paper', yref='paper',
        text=f"Duration: {d0['df']['timestamp_s'].iloc[-1] - d0['df']['timestamp_s'].iloc[0]:.1f} sec",
        showarrow=False, font=dict(size=12, color="gray")
    ))

    # data_point_annos = []

    # if 'rpm' in df:
    #     idx_max_rpm = df['rpm'].idxmax()
    #     t_max_rpm = t.iloc[idx_max_rpm]
    #     rpm_max_val = df['rpm'].max()
    #     print(f"Annotating max RPM at t={t_max_rpm}, value={rpm_max_val}")
    #     fig.add_annotation(
    #         x=t_max_rpm,   # This must be a pd.Timestamp (from t)
    #         y=rpm_max_val,
    #         xref='x', yref='y2',
    #         text=f"Max RPM: {rpm_max_val:.0f}",
    #         showarrow=True, arrowhead=2, ax=0, ay=-50,
    #         font=dict(color='white', size=13),
    #         bgcolor="rgba(31, 119, 180, 0.7)",
    #         bordercolor="white", borderwidth=1,
    #     )

    # if 'gnss_speed_kmh' in df:
    #     idx_max_spd = df['gnss_speed_kmh'].idxmax()
    #     t_max_spd = t.iloc[idx_max_spd]
    #     spd_max_val = df['gnss_speed_kmh'].max()
    #     data_point_annos.append(dict(
    #         x=t_max_spd, y=spd_max_val,
    #         xref='x', yref='y',
    #         text=f"Max Speed: {spd_max_val:.1f} km/h",
    #         showarrow=True, arrowhead=2, ax=0, ay=-60,
    #         font=dict(color='white', size=12),
    #         bgcolor="rgba(214, 39, 40, 0.7)"
    #     ))

    # Combine with previous layout (header) annotations:
    # for anno in data_point_annos:
    #     fig.add_annotation(**anno)
    


    fig.update_layout(annotations=annotations)

    # Save figure
    fig.write_html(output_file, include_plotlyjs='cdn')
    print(f"✅ Combined plot saved: {output_file}")

def inject_dark_mode_toggle(html_path):
    """Inject a dark mode toggle button and script into the HTML file."""
    dark_toggle_script = """
    <script>
    (function(){
        let isDark = true;
        const btn = document.createElement("button");
        btn.textContent = "🌙 Toggle Dark Mode";
        btn.style.position = "fixed";
        btn.style.top = "60px";
        btn.style.right = "12px";
        btn.style.zIndex = "9999";
        btn.style.padding = "6px 10px";
        btn.style.background = "#333";
        btn.style.color = "#fff";
        btn.style.border = "none";
        btn.style.borderRadius = "5px";
        btn.style.cursor = "pointer";

        const applyTheme = (dark) => {
            const bg = dark ? "#1e1e1e" : "#ffffff";
            const fg = dark ? "#eeeeee" : "#000000";
            document.body.style.backgroundColor = bg;
            document.body.style.color = fg;
            Plotly.relayout(document.querySelector(".js-plotly-plot"), {
                'paper_bgcolor': bg,
                'plot_bgcolor': bg,
                'font.color': fg,
                'xaxis.color': fg,
                'yaxis.color': fg,
                'yaxis2.color': fg
            });
        };

        btn.onclick = () => {
            isDark = !isDark;
            applyTheme(isDark);
        };

        document.body.appendChild(btn);
        applyTheme(isDark);
    })();
    </script>
    """
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    if "</body>" in html:
        html = html.replace("</body>", dark_toggle_script + "\n</body>")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print("✅ Dark mode toggle injected.")

def main():
    print("=== Interactive Sensor Plotter ===")
    folder_path = select_folder()
    print(f"Selected folder: {folder_path}")
    datasets = load_and_clean_csv_files(folder_path)
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(folder_path, f"combined_interactive_plot_{now_str}.html")
    generate_combined_plot(datasets, output_file)
    inject_dark_mode_toggle(output_file)
    # Open plot in browser
    import webbrowser
    webbrowser.open(output_file)

if __name__ == "__main__":
    main()
