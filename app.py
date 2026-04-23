import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from scipy.stats import linregress
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import plotly.graph_objects as go
import plotly.express as px

interp_points = np.linspace(0, 100, 100)


def render_plotly_line_reveal(
    fig,
    *,
    animate=False,
    use_container_width=True,
    chunk_count=40,
    frame_delay=0.02,
):
    """Render a Plotly figure, optionally revealing line traces in a fixed plot window."""
    trace_points = {}
    max_points = 0
    all_x_values = []
    all_y_values = []
    for trace_index, trace in enumerate(fig.data):
        if getattr(trace, "type", None) != "scatter":
            continue

        mode = getattr(trace, "mode", "") or ""
        if "lines" not in mode:
            continue

        x_values = list(trace.x) if trace.x is not None else []
        y_values = list(trace.y) if trace.y is not None else []
        point_count = min(len(x_values), len(y_values))
        if point_count <= 1:
            continue

        trace_points[trace_index] = (x_values, y_values, point_count)
        max_points = max(max_points, point_count)
        all_x_values.extend(val for val in x_values if pd.notna(val))
        all_y_values.extend(val for val in y_values if pd.notna(val))

    rendered_fig = go.Figure(fig)
    if all_x_values and rendered_fig.layout.xaxis.range is None:
        x_min = min(all_x_values)
        x_max = max(all_x_values)
        if x_min == x_max:
            x_min -= 1
            x_max += 1
        rendered_fig.update_xaxes(range=[x_min, x_max])

    if all_y_values and rendered_fig.layout.yaxis.range is None:
        y_min = min(all_y_values)
        y_max = max(all_y_values)
        if y_min == y_max:
            y_min -= 1
            y_max += 1
        else:
            y_padding = (y_max - y_min) * 0.05
            y_min -= y_padding
            y_max += y_padding
        rendered_fig.update_yaxes(range=[y_min, y_max])

    if not animate or not trace_points or max_points <= 1:
        st.plotly_chart(rendered_fig, use_container_width=use_container_width)
        return

    total_steps = max(2, min(int(chunk_count), int(max_points)))
    frame_duration_ms = max(10, int(float(frame_delay) * 1000))
    trace_indices = list(trace_points.keys())
    frames = []

    for active_trace_position, trace_index in enumerate(trace_indices):
        x_values, y_values, point_count = trace_points[trace_index]
        active_steps = max(2, min(int(chunk_count), int(point_count)))

        for step in range(1, active_steps + 1):
            progress_points = max(1, int(np.ceil((step / active_steps) * point_count)))
            frame_traces = []

            for candidate_position, candidate_trace_index in enumerate(trace_indices):
                candidate_x, candidate_y, candidate_count = trace_points[candidate_trace_index]

                if candidate_position < active_trace_position:
                    visible_points = candidate_count
                elif candidate_position == active_trace_position:
                    visible_points = min(progress_points, candidate_count)
                else:
                    visible_points = 1

                frame_traces.append(
                    go.Scatter(
                        x=candidate_x[:visible_points],
                        y=candidate_y[:visible_points],
                    )
                )

            frames.append(
                go.Frame(
                    data=frame_traces,
                    traces=trace_indices,
                    name=f"trace_{active_trace_position}_frame_{step}",
                )
            )

    first_frame = frames[0]
    for local_trace_index, trace_index in enumerate(trace_indices):
        rendered_fig.data[trace_index].x = first_frame.data[local_trace_index].x
        rendered_fig.data[trace_index].y = first_frame.data[local_trace_index].y

    rendered_fig.frames = frames
    rendered_fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=1.0,
                y=1.15,
                xanchor="right",
                yanchor="top",
                pad=dict(r=8, t=0),
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=frame_duration_ms, redraw=False),
                                transition=dict(duration=0),
                                fromcurrent=False,
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
                                frame=dict(duration=0, redraw=False),
                                transition=dict(duration=0),
                                mode="immediate",
                            ),
                        ],
                    ),
                ],
            )
        ]
    )

    st.plotly_chart(rendered_fig, use_container_width=use_container_width)

# Database connection settings
def get_connection():
    secrets = st.secrets["db"]
    return psycopg2.connect(
        host=secrets["host"],
        port=secrets["port"],
        dbname=secrets["dbname"],
        user=secrets["user"],
        password=secrets["password"],
    )

def normalize_time(df, start_frame, end_frame):
    df = df.copy()
    df["time_pct"] = (df["frame"] - start_frame) / (end_frame - start_frame) * 100
    return df

def interpolate_zero_crossing(frame0, value0, frame1, value1):
    """Linearly interpolate the frame where the signal crosses zero."""
    if value0 == 0:
        return float(frame0)
    if value1 == 0:
        return float(frame1)
    if value1 == value0:
        return float(frame0)
    return float(frame0 + ((0.0 - value0) * (frame1 - frame0) / (value1 - value0)))

def compute_positive_lobe_auc(df, min_frame=None, start_after_frame=None):
    """Return AUC for the first positive lobe bounded by zero crossings."""
    if df.empty:
        return np.nan, np.nan

    work = df.copy()
    work["x_data"] = pd.to_numeric(work["x_data"], errors="coerce")
    work = work.dropna(subset=["frame", "x_data"]).sort_values("frame").reset_index(drop=True)
    if min_frame is not None and not np.isnan(min_frame):
        work = work[work["frame"] >= float(min_frame)].reset_index(drop=True)
    if start_after_frame is not None and not np.isnan(start_after_frame):
        work = work[work["frame"] >= float(start_after_frame)].reset_index(drop=True)
    if len(work) < 2:
        return np.nan, np.nan

    frames = work["frame"].to_numpy(dtype=float)
    values = work["x_data"].to_numpy(dtype=float)

    for start_idx in range(1, len(work)):
        if values[start_idx - 1] <= 0 and values[start_idx] > 0:
            for end_idx in range(start_idx, len(work) - 1):
                if values[end_idx] > 0 and values[end_idx + 1] <= 0:
                    start_frame = interpolate_zero_crossing(
                        frames[start_idx - 1], values[start_idx - 1],
                        frames[start_idx], values[start_idx]
                    )
                    end_frame = interpolate_zero_crossing(
                        frames[end_idx], values[end_idx],
                        frames[end_idx + 1], values[end_idx + 1]
                    )
                    seg_frames = np.concatenate((
                        [start_frame],
                        frames[start_idx:end_idx + 1],
                        [end_frame]
                    ))
                    seg_values = np.concatenate((
                        [0.0],
                        values[start_idx:end_idx + 1],
                        [0.0]
                    ))
                    return float(np.trapezoid(seg_values, seg_frames)), float(end_frame)

    return np.nan, np.nan

def compute_negative_lobe_auc(df, threshold=-500, min_frame=None, start_after_frame=None):
    """Return AUC for the first negative lobe that crosses the threshold."""
    if df.empty:
        return np.nan, np.nan

    work = df.copy()
    work["x_data"] = pd.to_numeric(work["x_data"], errors="coerce")
    work = work.dropna(subset=["frame", "x_data"]).sort_values("frame").reset_index(drop=True)
    if min_frame is not None and not np.isnan(min_frame):
        work = work[work["frame"] >= float(min_frame)].reset_index(drop=True)
    if start_after_frame is not None and not np.isnan(start_after_frame):
        work = work[work["frame"] >= float(start_after_frame)].reset_index(drop=True)
    if len(work) < 2:
        return np.nan, np.nan

    frames = work["frame"].to_numpy(dtype=float)
    values = work["x_data"].to_numpy(dtype=float)

    for start_idx in range(1, len(work)):
        if values[start_idx - 1] >= 0 and values[start_idx] < 0:
            for end_idx in range(start_idx, len(work) - 1):
                if values[end_idx] < 0 and values[end_idx + 1] >= 0:
                    if np.nanmin(values[start_idx:end_idx + 1]) > threshold:
                        break
                    start_frame = interpolate_zero_crossing(
                        frames[start_idx - 1], values[start_idx - 1],
                        frames[start_idx], values[start_idx]
                    )
                    end_frame = interpolate_zero_crossing(
                        frames[end_idx], values[end_idx],
                        frames[end_idx + 1], values[end_idx + 1]
                    )
                    seg_frames = np.concatenate((
                        [start_frame],
                        frames[start_idx:end_idx + 1],
                        [end_frame]
                    ))
                    seg_values = np.concatenate((
                        [0.0],
                        values[start_idx:end_idx + 1],
                        [0.0]
                    ))
                    return float(np.trapezoid(seg_values, seg_frames)), float(end_frame)
            continue

    return np.nan, np.nan

def prepare_biodex_dataframe(uploaded_file):
    """Parse a Biodex CSV upload into a plottable time-series dataframe."""
    df = pd.read_csv(uploaded_file)
    df.columns = [str(col).strip() for col in df.columns]

    if df.empty:
        raise ValueError("The uploaded Biodex file is empty.")

    if "Time" not in df.columns:
        raise ValueError("The uploaded Biodex file must include a 'Time' column.")

    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)

    if df.empty:
        raise ValueError("The uploaded Biodex file does not contain any valid timestamps.")

    numeric_columns = []
    for col in df.columns:
        if col == "Time":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].notna().any():
            numeric_columns.append(col)

    if not numeric_columns:
        raise ValueError("The uploaded Biodex file does not contain any numeric measurement columns.")

    start_time = df["Time"].iloc[0]
    df["Elapsed Seconds"] = (df["Time"] - start_time).dt.total_seconds()
    return df, numeric_columns

def fetch_all_athletes(cur):
    cur.execute(
        """
        SELECT
            athlete_id,
            athlete_name,
            COALESCE(first_name, '') AS first_name,
            COALESCE(last_name, '') AS last_name,
            COALESCE(handedness, '') AS handedness
        FROM athletes
        ORDER BY athlete_name
        """
    )
    return cur.fetchall()

def insert_athlete(cur, conn, first_name, last_name, handedness):
    athlete_name = f"{first_name.strip()} {last_name.strip()}".strip()
    cur.execute(
        """
        INSERT INTO athletes (athlete_name, first_name, last_name, handedness)
        VALUES (%s, %s, %s, %s)
        RETURNING athlete_id
        """,
        (athlete_name, first_name.strip(), last_name.strip(), handedness.strip().upper()),
    )
    athlete_id = cur.fetchone()[0]
    conn.commit()
    return athlete_id, athlete_name

def get_first_existing_biodex_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None

def insert_biodex_test(
    cur,
    athlete_id,
    test_name,
    protocol_type,
    limb,
    movement,
    speed_deg_per_sec,
    test_date,
    source_file_name,
    notes,
):
    cur.execute(
        """
        INSERT INTO biodex_tests (
            athlete_id,
            test_name,
            protocol_type,
            limb,
            movement,
            speed_deg_per_sec,
            test_date,
            source_file_name,
            notes
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING biodex_test_id
        """,
        (
            int(athlete_id),
            test_name,
            protocol_type,
            limb,
            movement,
            int(speed_deg_per_sec) if speed_deg_per_sec is not None else None,
            test_date,
            source_file_name,
            notes,
        ),
    )
    return int(cur.fetchone()[0])

def insert_biodex_time_series(cur, biodex_test_id, biodex_df, chunk_size=2000, progress_callback=None):
    angular_velocity_col = get_first_existing_biodex_column(
        biodex_df,
        ["Angular_Velocity_Deg_Sec", "Angular_Velocity_Deg/Sec"],
    )
    rows_to_insert = []

    for sample_index, (_, row) in enumerate(biodex_df.iterrows(), start=1):
        time_raw = row.get("Time")
        if pd.isna(time_raw):
            time_raw = None
        elif hasattr(time_raw, "to_pydatetime"):
            time_raw = time_raw.to_pydatetime()

        time_seconds = row.get("Elapsed Seconds")
        if pd.isna(time_seconds):
            time_seconds = None
        else:
            time_seconds = float(time_seconds)

        angular_velocity = row.get(angular_velocity_col) if angular_velocity_col else None
        position_deg = row.get("Position_Deg")
        torque_nm = row.get("Torque_Nm")

        rows_to_insert.append((
            int(biodex_test_id),
            int(sample_index),
            time_seconds,
            time_raw,
            float(angular_velocity) if pd.notna(angular_velocity) else None,
            float(position_deg) if pd.notna(position_deg) else None,
            float(torque_nm) if pd.notna(torque_nm) else None,
        ))

    total_rows = len(rows_to_insert)
    insert_sql = """
        INSERT INTO biodex_time_series (
            biodex_test_id,
            sample_index,
            time_seconds,
            time_raw,
            angular_velocity_deg_sec,
            position_deg,
            torque_nm
        )
        VALUES %s
    """
    for start_idx in range(0, total_rows, int(chunk_size)):
        chunk = rows_to_insert[start_idx:start_idx + int(chunk_size)]
        execute_values(cur, insert_sql, chunk, page_size=len(chunk))
        if progress_callback is not None:
            progress_callback(min(total_rows, start_idx + len(chunk)), total_rows)
    return len(rows_to_insert)

def reset_postgres_sequence(cur, table_name, id_column):
    cur.execute(
        """
        SELECT setval(
            pg_get_serial_sequence(%s, %s),
            COALESCE((SELECT MAX(value_col) FROM (SELECT {id_column} AS value_col FROM {table_name}) seq_vals), 1),
            (SELECT COUNT(*) > 0 FROM {table_name})
        )
        """.format(table_name=table_name, id_column=id_column),
        (table_name, id_column),
    )

def delete_biodex_tests_by_ids(cur, conn, biodex_test_ids):
    biodex_test_ids = [int(test_id) for test_id in biodex_test_ids]
    if not biodex_test_ids:
        return 0

    cur.execute(
        "DELETE FROM biodex_time_series WHERE biodex_test_id = ANY(%s)",
        (biodex_test_ids,),
    )
    cur.execute(
        "DELETE FROM biodex_tests WHERE biodex_test_id = ANY(%s)",
        (biodex_test_ids,),
    )
    deleted_count = cur.rowcount
    reset_postgres_sequence(cur, "biodex_time_series", "biodex_time_series_id")
    reset_postgres_sequence(cur, "biodex_tests", "biodex_test_id")
    conn.commit()
    return deleted_count

def delete_biodex_tests_by_source_file_name(cur, conn, source_file_name):
    cur.execute(
        "SELECT biodex_test_id FROM biodex_tests WHERE source_file_name = %s ORDER BY biodex_test_id",
        (source_file_name,),
    )
    test_ids = [row[0] for row in cur.fetchall()]
    deleted_count = delete_biodex_tests_by_ids(cur, conn, test_ids)
    return deleted_count, test_ids

def get_most_recent_biodex_test(cur):
    cur.execute(
        """
        SELECT biodex_test_id, source_file_name
        FROM biodex_tests
        ORDER BY biodex_test_id DESC
        LIMIT 1
        """
    )
    return cur.fetchone()

def insert_biodex_processing_run(
    cur,
    biodex_test_id,
    processing_version,
    threshold,
    min_samples,
    buffer_samples,
    n_points,
    landmark_prominence_ratio,
    is_reviewed=False,
):
    cur.execute(
        """
        INSERT INTO biodex_processing_runs (
            biodex_test_id,
            processing_version,
            threshold,
            min_samples,
            buffer_samples,
            n_points,
            landmark_prominence_ratio,
            is_reviewed
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING biodex_processing_run_id
        """,
        (
            int(biodex_test_id),
            processing_version,
            float(threshold),
            int(min_samples),
            int(buffer_samples),
            int(n_points),
            float(landmark_prominence_ratio),
            bool(is_reviewed),
        ),
    )
    return int(cur.fetchone()[0])

def insert_biodex_rep_windows(cur, biodex_processing_run_id, rep_windows, rep_df_source):
    inserted_windows = []
    for rep_number, (start_idx, end_idx) in enumerate(rep_windows, start=1):
        start_time = float(rep_df_source.iloc[int(start_idx)]["Elapsed Seconds"])
        end_time = float(rep_df_source.iloc[int(end_idx)]["Elapsed Seconds"])
        cur.execute(
            """
            INSERT INTO biodex_rep_windows (
                biodex_processing_run_id,
                rep_number,
                start_sample_index,
                end_sample_index,
                start_time_seconds,
                end_time_seconds
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING biodex_rep_window_id
            """,
            (
                int(biodex_processing_run_id),
                int(rep_number),
                int(start_idx),
                int(end_idx),
                start_time,
                end_time,
            ),
        )
        inserted_windows.append({
            "rep_number": int(rep_number),
            "biodex_rep_window_id": int(cur.fetchone()[0]),
        })
    return inserted_windows

def insert_biodex_rep_landmarks(cur, rep_window_rows, aligned_rep_metadata):
    rep_window_id_by_number = {
        int(item["rep_number"]): int(item["biodex_rep_window_id"])
        for item in rep_window_rows
    }
    rows = []
    for rep_meta in aligned_rep_metadata:
        rep_window_id = rep_window_id_by_number.get(int(rep_meta["rep_number"]))
        if rep_window_id is None:
            continue

        kind_counts = {}
        for sample_index, kind, time_seconds, torque_nm in zip(
            rep_meta["landmark_indices"],
            rep_meta["landmark_kinds"],
            rep_meta["landmark_times"],
            rep_meta["landmark_torques"],
        ):
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
            landmark_name = f"{kind}{kind_counts[kind]}"
            rows.append((
                rep_window_id,
                landmark_name,
                int(sample_index),
                float(time_seconds),
                float(torque_nm),
            ))

    if rows:
        execute_values(
            cur,
            """
            INSERT INTO biodex_rep_landmarks (
                biodex_rep_window_id,
                landmark_name,
                sample_index,
                time_seconds,
                torque_nm
            )
            VALUES %s
            """,
            rows,
            page_size=len(rows),
        )
    return len(rows)

def insert_biodex_mean_curve(cur, biodex_processing_run_id, mean_df):
    rows = [
        (
            int(biodex_processing_run_id),
            float(row["movement_pct"]),
            float(row["mean_torque_nm"]),
            float(row["std_torque_nm"]),
            float(row["upper_band"]),
            float(row["lower_band"]),
        )
        for _, row in mean_df.iterrows()
    ]
    if rows:
        execute_values(
            cur,
            """
            INSERT INTO biodex_mean_curves (
                biodex_processing_run_id,
                movement_pct,
                mean_torque_nm,
                std_torque_nm,
                upper_band,
                lower_band
            )
            VALUES %s
            """,
            rows,
            page_size=len(rows),
        )
    return len(rows)

def fetch_biodex_processed_sessions(
    cur,
    athlete_id,
    protocol_type,
    movement,
    limb,
    speed_deg_per_sec,
):
    cur.execute(
        """
        SELECT
            pr.biodex_processing_run_id,
            bt.biodex_test_id,
            bt.test_name,
            bt.test_date,
            bt.protocol_type,
            bt.movement,
            bt.limb,
            bt.speed_deg_per_sec,
            bt.source_file_name,
            pr.processing_version,
            pr.created_at,
            pr.is_reviewed,
            COUNT(DISTINCT rw.biodex_rep_window_id) AS rep_count,
            MAX(mc.mean_torque_nm) AS peak_positive_mean_torque,
            MIN(mc.mean_torque_nm) AS peak_negative_mean_torque
        FROM biodex_processing_runs pr
        JOIN biodex_tests bt
            ON pr.biodex_test_id = bt.biodex_test_id
        LEFT JOIN biodex_rep_windows rw
            ON pr.biodex_processing_run_id = rw.biodex_processing_run_id
        LEFT JOIN biodex_mean_curves mc
            ON pr.biodex_processing_run_id = mc.biodex_processing_run_id
        WHERE bt.athlete_id = %s
          AND bt.protocol_type = %s
          AND bt.movement = %s
          AND bt.limb = %s
          AND bt.speed_deg_per_sec = %s
        GROUP BY
            pr.biodex_processing_run_id,
            bt.biodex_test_id,
            bt.test_name,
            bt.test_date,
            bt.protocol_type,
            bt.movement,
            bt.limb,
            bt.speed_deg_per_sec,
            bt.source_file_name,
            pr.processing_version,
            pr.created_at,
            pr.is_reviewed
        ORDER BY bt.test_date DESC NULLS LAST, pr.created_at DESC
        """,
        (
            int(athlete_id),
            protocol_type,
            movement,
            limb,
            int(speed_deg_per_sec),
        ),
    )
    columns = [
        "biodex_processing_run_id",
        "biodex_test_id",
        "test_name",
        "test_date",
        "protocol_type",
        "movement",
        "limb",
        "speed_deg_per_sec",
        "source_file_name",
        "processing_version",
        "created_at",
        "is_reviewed",
        "rep_count",
        "peak_positive_mean_torque",
        "peak_negative_mean_torque",
    ]
    return pd.DataFrame(cur.fetchall(), columns=columns)

def fetch_biodex_mean_curve(cur, biodex_processing_run_id):
    cur.execute(
        """
        SELECT
            movement_pct,
            mean_torque_nm,
            std_torque_nm,
            upper_band,
            lower_band
        FROM biodex_mean_curves
        WHERE biodex_processing_run_id = %s
        ORDER BY movement_pct
        """,
        (int(biodex_processing_run_id),),
    )
    return pd.DataFrame(
        cur.fetchall(),
        columns=[
            "movement_pct",
            "mean_torque_nm",
            "std_torque_nm",
            "upper_band",
            "lower_band",
        ],
    )

def fetch_all_biodex_processing_runs(cur):
    cur.execute(
        """
        SELECT
            pr.biodex_processing_run_id,
            bt.biodex_test_id,
            bt.test_name,
            bt.test_date,
            bt.protocol_type,
            bt.movement,
            bt.limb,
            bt.speed_deg_per_sec,
            bt.source_file_name,
            a.athlete_name,
            pr.processing_version,
            pr.created_at,
            pr.is_reviewed,
            pr.threshold,
            pr.min_samples,
            pr.buffer_samples,
            pr.n_points,
            pr.landmark_prominence_ratio,
            COUNT(DISTINCT rw.biodex_rep_window_id) AS rep_count
        FROM biodex_processing_runs pr
        JOIN biodex_tests bt
            ON pr.biodex_test_id = bt.biodex_test_id
        JOIN athletes a
            ON bt.athlete_id = a.athlete_id
        LEFT JOIN biodex_rep_windows rw
            ON pr.biodex_processing_run_id = rw.biodex_processing_run_id
        GROUP BY
            pr.biodex_processing_run_id,
            bt.biodex_test_id,
            bt.test_name,
            bt.test_date,
            bt.protocol_type,
            bt.movement,
            bt.limb,
            bt.speed_deg_per_sec,
            bt.source_file_name,
            a.athlete_name,
            pr.processing_version,
            pr.created_at,
            pr.is_reviewed,
            pr.threshold,
            pr.min_samples,
            pr.buffer_samples,
            pr.n_points,
            pr.landmark_prominence_ratio
        ORDER BY bt.test_date DESC NULLS LAST, pr.created_at DESC
        """
    )
    return pd.DataFrame(
        cur.fetchall(),
        columns=[
            "biodex_processing_run_id",
            "biodex_test_id",
            "test_name",
            "test_date",
            "protocol_type",
            "movement",
            "limb",
            "speed_deg_per_sec",
            "source_file_name",
            "athlete_name",
            "processing_version",
            "created_at",
            "is_reviewed",
            "threshold",
            "min_samples",
            "buffer_samples",
            "n_points",
            "landmark_prominence_ratio",
            "rep_count",
        ],
    )

def fetch_biodex_raw_time_series(cur, biodex_test_id):
    cur.execute(
        """
        SELECT
            sample_index,
            time_seconds,
            time_raw,
            angular_velocity_deg_sec,
            position_deg,
            torque_nm
        FROM biodex_time_series
        WHERE biodex_test_id = %s
        ORDER BY sample_index
        """,
        (int(biodex_test_id),),
    )
    return pd.DataFrame(
        cur.fetchall(),
        columns=[
            "sample_index",
            "time_seconds",
            "time_raw",
            "angular_velocity_deg_sec",
            "position_deg",
            "torque_nm",
        ],
    )

def fetch_biodex_rep_windows(cur, biodex_processing_run_id):
    cur.execute(
        """
        SELECT
            biodex_rep_window_id,
            rep_number,
            start_sample_index,
            end_sample_index,
            start_time_seconds,
            end_time_seconds
        FROM biodex_rep_windows
        WHERE biodex_processing_run_id = %s
        ORDER BY rep_number
        """,
        (int(biodex_processing_run_id),),
    )
    return pd.DataFrame(
        cur.fetchall(),
        columns=[
            "biodex_rep_window_id",
            "rep_number",
            "start_sample_index",
            "end_sample_index",
            "start_time_seconds",
            "end_time_seconds",
        ],
    )

def fetch_biodex_rep_landmarks(cur, biodex_processing_run_id):
    cur.execute(
        """
        SELECT
            rw.rep_number,
            rl.landmark_name,
            rl.sample_index,
            rl.time_seconds,
            rl.torque_nm
        FROM biodex_rep_landmarks rl
        JOIN biodex_rep_windows rw
            ON rl.biodex_rep_window_id = rw.biodex_rep_window_id
        WHERE rw.biodex_processing_run_id = %s
        ORDER BY rw.rep_number, rl.sample_index
        """,
        (int(biodex_processing_run_id),),
    )
    return pd.DataFrame(
        cur.fetchall(),
        columns=[
            "rep_number",
            "landmark_name",
            "sample_index",
            "time_seconds",
            "torque_nm",
        ],
    )

def mark_biodex_processing_run_reviewed(cur, conn, biodex_processing_run_id):
    cur.execute(
        """
        UPDATE biodex_processing_runs
        SET is_reviewed = TRUE
        WHERE biodex_processing_run_id = %s
        """,
        (int(biodex_processing_run_id),),
    )
    conn.commit()
    return cur.rowcount

def get_valid_savgol_window(window_length, series_length, polyorder):
    """Clamp the Savitzky-Golay window to a valid odd length for the series."""
    max_window = series_length if series_length % 2 == 1 else series_length - 1
    if max_window <= polyorder:
        return None

    valid_window = min(window_length, max_window)
    if valid_window % 2 == 0:
        valid_window -= 1
    if valid_window <= polyorder:
        valid_window = polyorder + 2 if polyorder % 2 == 1 else polyorder + 1
    if valid_window % 2 == 0:
        valid_window += 1

    return valid_window if valid_window <= max_window else None

def downsample_biodex_plot(plot_df, max_points):
    """Reduce plotted points while preserving overall curve shape for display."""
    if len(plot_df) <= max_points:
        return plot_df

    step = max(1, int(np.ceil(len(plot_df) / max_points)))
    return plot_df.iloc[::step].copy()

def smooth_biodex_display_curve(values, window_length=9, polyorder=3):
    """Lightly smooth a display-only curve without changing underlying stats."""
    arr = np.asarray(values, dtype=float)
    valid_window = get_valid_savgol_window(window_length, len(arr), polyorder)
    if valid_window is None:
        return arr
    return savgol_filter(arr, window_length=valid_window, polyorder=polyorder)

def _build_contiguous_regions(index_values):
    if not index_values:
        return []

    regions = []
    start = prev = int(index_values[0])
    for idx in index_values[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        regions.append((start, prev))
        start = prev = idx
    regions.append((start, prev))
    return regions

def _merge_close_regions(regions, max_gap):
    if not regions:
        return []

    merged = [regions[0]]
    for start_idx, end_idx in regions[1:]:
        prev_start, prev_end = merged[-1]
        if int(start_idx) <= int(prev_end) + int(max_gap) + 1:
            merged[-1] = (prev_start, max(prev_end, end_idx))
        else:
            merged.append((start_idx, end_idx))
    return merged

def detect_biodex_reps(df, value_col="Torque_Nm", threshold=20.0, min_samples=15, buffer_samples=20):
    if df.empty or value_col not in df.columns:
        return []

    signal = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    if len(signal) == 0:
        return []

    finite_mask = np.isfinite(signal)
    if not finite_mask.any():
        return []

    clean_signal = signal.copy()
    if not finite_mask.all():
        valid_idx = np.flatnonzero(finite_mask)
        clean_signal[~finite_mask] = np.interp(
            np.flatnonzero(~finite_mask),
            valid_idx,
            signal[finite_mask],
        )

    envelope_window = max(5, int(min_samples) * 2 + 1)
    envelope_window = min(envelope_window, len(clean_signal))
    if envelope_window % 2 == 0:
        envelope_window = max(1, envelope_window - 1)

    abs_signal = np.abs(clean_signal)
    envelope_series = pd.Series(abs_signal)
    smoothed_envelope = (
        envelope_series
        .rolling(window=envelope_window, center=True, min_periods=1)
        .max()
        .rolling(window=envelope_window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )

    active_idx = np.flatnonzero(smoothed_envelope >= float(threshold))
    if active_idx.size == 0:
        return []

    regions = _build_contiguous_regions(active_idx.tolist())
    merge_gap = max(int(buffer_samples) * 2, int(min_samples) * 2)
    regions = _merge_close_regions(regions, merge_gap)
    rep_windows = []
    n_rows = len(df)

    for start_idx, end_idx in regions:
        if (end_idx - start_idx + 1) < int(min_samples):
            continue
        buffered_start = max(0, int(start_idx) - int(buffer_samples))
        buffered_end = min(n_rows - 1, int(end_idx) + int(buffer_samples))
        rep_windows.append((buffered_start, buffered_end))

    return rep_windows

def detect_biodex_rep_landmarks(rep_df, value_col="Torque_Nm", prominence_ratio=0.12):
    if rep_df.empty or value_col not in rep_df.columns:
        return None

    torque_values = pd.to_numeric(rep_df[value_col], errors="coerce").to_numpy(dtype=float)
    if len(torque_values) < 7 or np.all(np.isnan(torque_values)):
        return None

    finite_mask = np.isfinite(torque_values)
    if not finite_mask.any():
        return None

    clean_values = torque_values.copy()
    if not finite_mask.all():
        valid_idx = np.flatnonzero(finite_mask)
        clean_values[~finite_mask] = np.interp(
            np.flatnonzero(~finite_mask),
            valid_idx,
            torque_values[finite_mask],
        )

    smooth_window = get_valid_savgol_window(11, len(clean_values), 3)
    if smooth_window is not None:
        smooth_values = savgol_filter(clean_values, window_length=smooth_window, polyorder=3)
    else:
        smooth_values = clean_values

    amplitude_span = float(np.nanmax(smooth_values) - np.nanmin(smooth_values))
    if amplitude_span <= 0:
        return None

    min_distance = max(1, len(smooth_values) // 12)
    prominence = max(1.0, amplitude_span * float(prominence_ratio))

    pos_peaks, pos_props = find_peaks(smooth_values, prominence=prominence, distance=min_distance)
    neg_peaks, neg_props = find_peaks(-smooth_values, prominence=prominence, distance=min_distance)

    if len(pos_peaks) < 2 or len(neg_peaks) < 2:
        return None

    top_pos_idx = np.argsort(pos_props["prominences"])[-2:]
    top_neg_idx = np.argsort(neg_props["prominences"])[-2:]

    candidates = []
    for idx in top_pos_idx:
        candidates.append((int(pos_peaks[idx]), "pos"))
    for idx in top_neg_idx:
        candidates.append((int(neg_peaks[idx]), "neg"))

    candidates = sorted(candidates, key=lambda item: item[0])
    landmark_indices = [idx for idx, _kind in candidates]

    if len(landmark_indices) != 4 or any(b <= a for a, b in zip(landmark_indices, landmark_indices[1:])):
        return None

    return {
        "indices": landmark_indices,
        "kinds": [kind for _idx, kind in candidates],
        "smooth_values": smooth_values,
    }

def extract_landmark_aligned_biodex_reps(
    df,
    rep_windows,
    time_col="Elapsed Seconds",
    value_col="Torque_Nm",
    n_points=101,
    prominence_ratio=0.12,
):
    if df.empty or not rep_windows:
        return pd.DataFrame(), pd.DataFrame(), []

    percent_axis = np.linspace(0.0, 100.0, int(n_points))
    valid_reps = []
    phase_fraction_rows = []

    for rep_number, (start_idx, end_idx) in enumerate(rep_windows, start=1):
        rep_df = df.iloc[int(start_idx):int(end_idx) + 1].copy()
        if rep_df.empty:
            continue

        rep_df[time_col] = pd.to_numeric(rep_df[time_col], errors="coerce")
        rep_df[value_col] = pd.to_numeric(rep_df[value_col], errors="coerce")
        rep_df = rep_df.dropna(subset=[time_col, value_col]).reset_index(drop=True)
        if len(rep_df) < 7:
            continue

        landmark_info = detect_biodex_rep_landmarks(
            rep_df,
            value_col=value_col,
            prominence_ratio=prominence_ratio,
        )
        if landmark_info is None:
            continue

        boundary_idx = [0] + landmark_info["indices"] + [len(rep_df) - 1]
        if any(b <= a for a, b in zip(boundary_idx, boundary_idx[1:])):
            continue

        phase_lengths = np.diff(boundary_idx).astype(float)
        if np.any(phase_lengths <= 0):
            continue

        phase_fraction_rows.append(phase_lengths / phase_lengths.sum())
        valid_reps.append({
            "rep_number": rep_number,
            "rep_df": rep_df,
            "boundary_idx": boundary_idx,
            "landmark_indices": landmark_info["indices"],
            "landmark_kinds": landmark_info["kinds"],
        })

    if not valid_reps:
        return pd.DataFrame(), pd.DataFrame(), []

    median_phase_fractions = np.nanmedian(np.vstack(phase_fraction_rows), axis=0)
    median_phase_fractions = median_phase_fractions / median_phase_fractions.sum()
    boundary_pct = np.concatenate(([0.0], np.cumsum(median_phase_fractions) * 100.0))

    normalized_curves = []
    rep_rows = []
    aligned_rep_metadata = []

    for rep_info in valid_reps:
        rep_df = rep_info["rep_df"]
        time_values = rep_df[time_col].to_numpy(dtype=float)
        torque_values = rep_df[value_col].to_numpy(dtype=float)
        sample_idx = np.arange(len(rep_df), dtype=float)
        mapped_pct = np.interp(sample_idx, rep_info["boundary_idx"], boundary_pct)
        interp_torque = np.interp(percent_axis, mapped_pct, torque_values)

        landmark_times = [float(time_values[idx]) for idx in rep_info["landmark_indices"]]
        landmark_torques = [float(torque_values[idx]) for idx in rep_info["landmark_indices"]]

        normalized_curves.append(interp_torque)
        rep_rows.append(pd.DataFrame({
            "rep_number": rep_info["rep_number"],
            "movement_pct": percent_axis,
            "torque_nm": interp_torque,
        }))
        aligned_rep_metadata.append({
            "rep_number": rep_info["rep_number"],
            "landmark_indices": rep_info["landmark_indices"],
            "landmark_kinds": rep_info["landmark_kinds"],
            "landmark_times": landmark_times,
            "landmark_torques": landmark_torques,
        })

    curves_arr = np.vstack(normalized_curves)
    reps_long_df = pd.concat(rep_rows, ignore_index=True)

    mean_df = pd.DataFrame({
        "movement_pct": percent_axis,
        "mean_torque_nm": np.nanmean(curves_arr, axis=0),
        "std_torque_nm": np.nanstd(curves_arr, axis=0),
    })
    mean_df["upper_band"] = mean_df["mean_torque_nm"] + mean_df["std_torque_nm"]
    mean_df["lower_band"] = mean_df["mean_torque_nm"] - mean_df["std_torque_nm"]

    landmark_labels = [
        f"{kind.upper()}{i + 1}"
        for i, kind in enumerate(aligned_rep_metadata[0]["landmark_kinds"])
    ]
    mean_df.attrs["landmark_boundary_pct"] = boundary_pct[1:-1].tolist()
    mean_df.attrs["landmark_labels"] = landmark_labels

    return reps_long_df, mean_df, aligned_rep_metadata

# Helper: Ball release frame by peak |hand CGVel X|
# Helper: Ball release frame by peak |hand CGVel X|
def get_ball_release_frame(take_id, handedness, cur):
    """Return ball release frame as the frame of peak |hand CGVel X|.

    This is robust to handedness / segment labeling by checking both LHA and RHA
    and choosing whichever hand has the largest absolute x_data value.
    """
    cur.execute(
        """
        SELECT ts.frame, ts.x_data
        FROM time_series_data ts
        JOIN categories c ON ts.category_id = c.category_id
        JOIN segments s   ON ts.segment_id  = s.segment_id
        WHERE ts.take_id = %s
          AND c.category_name = 'KINETIC_KINEMATIC_CGVel'
          AND s.segment_name IN ('LHA', 'RHA')
          AND ts.x_data IS NOT NULL
        ORDER BY ABS(ts.x_data) DESC
        LIMIT 1
        """,
        (take_id,),
    )
    row = cur.fetchone()
    if row:
        return int(row[0])
    return None

def get_ball_release_frame_pulldown(take_id, handedness, fp_frame, cur):
    """
    Pulldown BR:
    - Primary: peak |hand CGVel X| AFTER Foot Plant
    - Fallback: peak |hand CGVel X| AFTER pelvis angular velocity peak
    This prevents early-take CGVel artifacts from being mislabeled as BR.
    """

    # -----------------------------
    # Determine minimum valid frame
    # -----------------------------
    import numpy as np
    if fp_frame is not None and not np.isnan(fp_frame):
        min_frame = int(fp_frame)
    else:
        # Fallback anchor: pelvis angular velocity peak
        pelvis_peak = get_pelvis_angvel_peak_frame(take_id, handedness, cur)
        if pelvis_peak is None:
            return None
        min_frame = int(pelvis_peak)

    # -----------------------------
    # Peak hand CGVel X AFTER anchor
    # -----------------------------
    cur.execute(
        """
        SELECT ts.frame, ts.x_data
        FROM time_series_data ts
        JOIN categories c ON ts.category_id = c.category_id
        JOIN segments s   ON ts.segment_id  = s.segment_id
        WHERE ts.take_id = %s
          AND c.category_name = 'KINETIC_KINEMATIC_CGVel'
          AND s.segment_name IN ('LHA', 'RHA')
          AND ts.x_data IS NOT NULL
          AND ts.frame >= %s
        ORDER BY ABS(ts.x_data) DESC
        LIMIT 1
        """,
        (take_id, min_frame),
    )
    row = cur.fetchone()
    return int(row[0]) if row else None

def get_shoulder_er_max_frame(take_id, handedness, cur, throw_type=None):
    """
    Returns the frame of TRUE shoulder external rotation (layback),
    anchored to the FIRST significant arm energy peak (>100).

    Logic:
    1) Find first local arm-energy peak with x_data > 100 (RAR / LAR)
    2) Fallback to global arm-energy max if none found
    3) Find ER extreme within ±15 frames of that arm-energy frame

    - Uses SEGMENT_ENERGIES (RAR / LAR)
    - Uses JOINT_ANGLES (RT_SHOULDER / LT_SHOULDER)
    - Avoids BR, FP, and shoulder IR velocity entirely
    """

    import numpy as np
    from scipy.signal import find_peaks

    # -----------------------------------------------------------
    # Pulldown-only override:
    # MER must occur within ±30 frames of Foot Plant (FP)
    # -----------------------------------------------------------
    if throw_type == "Pulldown":
        fp_frame = get_foot_plant_frame(take_id, handedness, cur)
        if fp_frame is None:
            return None

        shoulder_segment = "RT_SHOULDER" if handedness == "R" else "LT_SHOULDER"
        start_frame = int(fp_frame) - 30
        end_frame   = int(fp_frame) + 30

        cur.execute("""
            SELECT ts.frame, ts.z_data
            FROM time_series_data ts
            JOIN categories c ON ts.category_id = c.category_id
            JOIN segments s   ON ts.segment_id  = s.segment_id
            WHERE ts.take_id = %s
              AND c.category_name = 'JOINT_ANGLES'
              AND s.segment_name = %s
              AND ts.frame BETWEEN %s AND %s
              AND ts.z_data IS NOT NULL
            ORDER BY ts.frame
        """, (int(take_id), shoulder_segment, int(start_frame), int(end_frame)))

        rows = cur.fetchall()
        if not rows:
            return None

        frames = np.array([r[0] for r in rows], dtype=int)
        z_vals = np.array([r[1] for r in rows], dtype=float)

        # Directional ER selection
        if handedness == "R":
            idx = int(np.nanargmin(z_vals))
        else:
            idx = int(np.nanargmax(z_vals))

        return int(frames[idx])

    # -----------------------------
    # 1) Load arm energy time series
    # -----------------------------
    arm_segment = "RAR" if handedness == "R" else "LAR"

    cur.execute("""
        SELECT ts.frame, ts.x_data
        FROM time_series_data ts
        JOIN categories c ON ts.category_id = c.category_id
        JOIN segments s   ON ts.segment_id  = s.segment_id
        WHERE ts.take_id = %s
          AND c.category_name = 'SEGMENT_ENERGIES'
          AND s.segment_name = %s
          AND ts.x_data IS NOT NULL
        ORDER BY ts.frame
    """, (int(take_id), arm_segment))

    rows = cur.fetchall()
    if not rows:
        return None

    frames = np.array([r[0] for r in rows], dtype=int)
    energy = np.array([r[1] for r in rows], dtype=float)

    # -----------------------------------------------------------
    # Find FIRST relevant arm-energy peak (>100) that then drops
    # -----------------------------------------------------------
    peak_arm_energy_frame = None

    # Mild smoothing to reduce jitter without shifting timing
    try:
        import pandas as pd
        energy_s = pd.Series(energy).rolling(window=5, center=True, min_periods=1).median().to_numpy()
    except Exception:
        energy_s = energy

    # First crossing above threshold
    idx0_candidates = np.where(energy_s > 100)[0]
    if idx0_candidates.size > 0:
        idx0 = int(idx0_candidates[0])

        lookahead = 6      # frames to confirm drop
        min_drop = 10.0    # require real decrease

        for i in range(idx0 + 1, len(energy_s) - lookahead - 1):
            prev_v = energy_s[i - 1]
            v      = energy_s[i]
            next_v = energy_s[i + 1]

            # Turning point: rising into i, then flat/down
            if (v > 100) and (prev_v < v) and (v >= next_v):
                future = energy_s[i + 1:i + 1 + lookahead]
                if future.size > 0 and (v - np.nanmin(future) >= min_drop):
                    peak_arm_energy_frame = int(frames[i])
                    break

    # -----------------------------------------------
    # 3) Fallback: global arm-energy max
    # -----------------------------------------------
    if peak_arm_energy_frame is None:
        idx = int(np.nanargmax(energy))
        peak_arm_energy_frame = int(frames[idx])

    # ------------------------------------
    # 4) Query shoulder ER angle (z_data)
    # ------------------------------------
    shoulder_segment = "RT_SHOULDER" if handedness == "R" else "LT_SHOULDER"
    start_frame = peak_arm_energy_frame - 15
    end_frame = peak_arm_energy_frame + 15


    cur.execute("""
        SELECT ts.frame, ts.z_data
        FROM time_series_data ts
        JOIN categories c ON ts.category_id = c.category_id
        JOIN segments s   ON ts.segment_id  = s.segment_id
        WHERE ts.take_id = %s
          AND c.category_name = 'JOINT_ANGLES'
          AND s.segment_name = %s
          AND ts.frame BETWEEN %s AND %s
          AND ts.z_data IS NOT NULL
        ORDER BY ts.frame
    """, (int(take_id), shoulder_segment, int(start_frame), int(end_frame)))

    rows = cur.fetchall()
    if not rows:
        return None

    sh_frames = np.array([r[0] for r in rows], dtype=int)
    z_vals    = np.array([r[1] for r in rows], dtype=float)

    # ------------------------------------
    # 5) Directional ER selection
    # ------------------------------------
    if handedness == "R":
        idx = int(np.nanargmin(z_vals))   # RHP: most negative = ER
    else:
        idx = int(np.nanargmax(z_vals))   # LHP: most positive = ER

    return int(sh_frames[idx])

# Helper: Lead ankle proximal X-velocity peak frame (prox_x_peak_frame)
def get_lead_ankle_prox_x_peak_frame(take_id, handedness, cur):
    """
    Returns the frame of the peak proximal X velocity of the lead ankle.
    Uses LFT (lead foot for RHP) and RFT (lead foot for LHP).
    """
    lead_ankle_segment = "LFT" if handedness == "R" else "RFT"

    cur.execute("""
        SELECT ts.frame, ts.x_data
        FROM time_series_data ts
        JOIN categories c ON ts.category_id = c.category_id
        JOIN segments s   ON ts.segment_id  = s.segment_id
        WHERE ts.take_id = %s
          AND c.category_name = 'KINETIC_KINEMATIC_ProxEndVel'
          AND s.segment_name = %s
          AND ts.x_data IS NOT NULL
        ORDER BY ts.x_data DESC
        LIMIT 1
    """, (take_id, lead_ankle_segment))

    row = cur.fetchone()
    return int(row[0]) if row else None

# Helper: deepest dip in ankle Z velocity (ankle_min SQL logic)
def get_ankle_min_frame(take_id, handedness, prox_x_peak_frame, sh_er_max_frame, cur):
    lead_ankle_segment = "LFT" if handedness == "R" else "RFT"

    cur.execute("""
        SELECT ts.frame, ts.z_data
        FROM time_series_data ts
        JOIN categories c ON ts.category_id = c.category_id
        JOIN segments s ON ts.segment_id = s.segment_id
        WHERE ts.take_id = %s
          AND c.category_name = 'KINETIC_KINEMATIC_DistEndVel'
          AND s.segment_name = %s
          AND ts.z_data IS NOT NULL
          AND ts.frame BETWEEN %s AND %s
        ORDER BY ts.z_data ASC
        LIMIT 1
    """, (take_id, lead_ankle_segment, prox_x_peak_frame, sh_er_max_frame))

    row = cur.fetchone()
    return int(row[0]) if row else None

# Helper: zero-cross foot plant detection (SQL zero_cross AS)
def get_zero_cross_frame(take_id, handedness, ankle_min_frame, sh_er_max_frame, cur):
    lead_ankle_segment = "LFT" if handedness == "R" else "RFT"

    cur.execute("""
        SELECT ts.frame, ts.z_data
        FROM time_series_data ts
        JOIN categories c ON ts.category_id = c.category_id
        JOIN segments s ON ts.segment_id = s.segment_id
        WHERE ts.take_id = %s
          AND c.category_name = 'KINETIC_KINEMATIC_DistEndVel'
          AND s.segment_name = %s
          AND ts.z_data >= -0.05
          AND ts.frame >= %s
          AND ts.frame <= %s
        ORDER BY ts.frame ASC
        LIMIT 1
    """, (take_id, lead_ankle_segment, ankle_min_frame, sh_er_max_frame))

    row = cur.fetchone()
    return int(row[0] + 1) if row else None

def get_max_rear_knee_flexion_frame_with_heel(take_id, handedness, cur):
    """
    Returns (frame, value) of max rear knee flexion.

    FP-aware + take-specific heel-floor gating:

    Window: [FP − 100, FP]

    1) Compute heel_floor = MIN(heel_z) in the window
    2) Compute heel_ceil  = MAX(heel_z) in the window
    3) Define heel_thresh = heel_floor + 0.40 * (heel_ceil - heel_floor)
       (i.e., within the lowest 40% of that take's heel excursion)
    4) Find max rear knee flexion ONLY on frames where heel_z <= heel_thresh
    5) Fallback: if no frames qualify, use the absolute heel-min frame and select
       knee at that same frame.

    Rear knee:
      - RHP -> RT_KNEE
      - LHP -> LT_KNEE

    Heel landmark:
      - RHP -> R_HEEL
      - LHP -> L_HEEL
    """

    knee_segment = "RT_KNEE" if handedness == "R" else "LT_KNEE"
    heel_segment = "R_HEEL" if handedness == "R" else "L_HEEL"

    # -------------------------------------------------
    # 0) Foot Plant (FP)
    # -------------------------------------------------
    fp_frame = get_foot_plant_frame(take_id, handedness, cur)
    if fp_frame is None:
        return None, None
    fp_frame = int(fp_frame)

    drive_start = fp_frame - 90

    # -------------------------------------------------
    # 1) Heel floor and ceiling in window
    # -------------------------------------------------
    cur.execute("""
        SELECT MIN(h.z_data) AS heel_floor,
               MAX(h.z_data) AS heel_ceil
        FROM time_series_data h
        JOIN categories c ON h.category_id = c.category_id
        JOIN segments s   ON h.segment_id = s.segment_id
        WHERE h.take_id = %s
          AND c.category_name = 'LANDMARK_ORIGINAL'
          AND s.segment_name = %s
          AND h.frame BETWEEN %s AND %s
          AND h.z_data IS NOT NULL
    """, (int(take_id), heel_segment, int(drive_start), int(fp_frame)))

    row = cur.fetchone()
    if not row or row[0] is None or row[1] is None:
        return None, None

    heel_floor = float(row[0])
    heel_ceil  = float(row[1])

    # If range is degenerate, fall back to heel_floor only
    heel_range = heel_ceil - heel_floor
    if heel_range <= 1e-9:
        heel_thresh = heel_floor
    else:
        heel_thresh = heel_floor + 0.40 * heel_range

    # -------------------------------------------------
    # 2) Knee min on frames where heel is near floor
    # -------------------------------------------------
    cur.execute("""
        SELECT k.frame, k.x_data
        FROM time_series_data k
        JOIN categories ck ON k.category_id = ck.category_id
        JOIN segments sk   ON k.segment_id = sk.segment_id
        JOIN time_series_data h
          ON h.take_id = k.take_id
         AND h.frame   = k.frame
        JOIN categories ch ON h.category_id = ch.category_id
        JOIN segments sh   ON h.segment_id = sh.segment_id
        WHERE k.take_id = %s
          AND ck.category_name = 'JOINT_ANGLES'
          AND sk.segment_name = %s
          AND ch.category_name = 'LANDMARK_ORIGINAL'
          AND sh.segment_name = %s
          AND k.frame BETWEEN %s AND %s
          AND k.x_data IS NOT NULL
          AND h.z_data IS NOT NULL
          AND h.z_data <= %s
        ORDER BY k.x_data ASC
        LIMIT 1
    """, (
        int(take_id),
        knee_segment,
        heel_segment,
        int(drive_start),
        int(fp_frame),
        float(heel_thresh)
    ))

    row = cur.fetchone()
    if row:
        return int(row[0]), float(row[1])

    # -------------------------------------------------
    # 3) Fallback: absolute heel-min frame, knee at same frame
    # -------------------------------------------------
    cur.execute("""
        SELECT h.frame, h.z_data
        FROM time_series_data h
        JOIN categories c ON h.category_id = c.category_id
        JOIN segments s   ON h.segment_id = s.segment_id
        WHERE h.take_id = %s
          AND c.category_name = 'LANDMARK_ORIGINAL'
          AND s.segment_name = %s
          AND h.frame BETWEEN %s AND %s
          AND h.z_data IS NOT NULL
        ORDER BY h.z_data ASC
        LIMIT 1
    """, (int(take_id), heel_segment, int(drive_start), int(fp_frame)))

    row = cur.fetchone()
    if not row:
        return None, None

    heel_min_frame = int(row[0])

    cur.execute("""
        SELECT k.frame, k.x_data
        FROM time_series_data k
        JOIN categories ck ON k.category_id = ck.category_id
        JOIN segments sk   ON k.segment_id = sk.segment_id
        WHERE k.take_id = %s
          AND ck.category_name = 'JOINT_ANGLES'
          AND sk.segment_name = %s
          AND k.frame = %s
          AND k.x_data IS NOT NULL
        LIMIT 1
    """, (int(take_id), knee_segment, int(heel_min_frame)))

    row = cur.fetchone()
    if row:
        return int(row[0]), float(row[1])

    return None, None

def get_pulldown_peak_knee_height_frame(take_id, handedness, br_frame, cur, heel_thresh=0.1, min_consecutive_frames=5):
    """
    Pulldown Peak Knee Height event (Tab 3 override):
    - Throwing-side knee X (RT/LT_KNEE.x_data) + throwing-side heel Z (R/L_HEEL.z_data)
    - Identify the first sustained heel-contact block where heel_z <= heel_thresh
      for at least min_consecutive_frames
    - Return frame of MIN knee_x (most negative flexion) within that block
    Fallbacks:
      1) min knee_x on any frame with heel_z <= heel_thresh
      2) min knee_x across the full queried window
    """

    knee_segment = "RT_KNEE" if handedness == "R" else "LT_KNEE"
    heel_segment = "R_HEEL" if handedness == "R" else "L_HEEL"

    fp_frame = get_foot_plant_frame(take_id, handedness, cur)
    start_frame = int(fp_frame) - 120 if fp_frame is not None else 0
    end_frame = int(br_frame) if br_frame is not None else 10**9

    cur.execute("""
        SELECT k.frame, k.x_data, h.z_data
        FROM time_series_data k
        JOIN categories ck ON k.category_id = ck.category_id
        JOIN segments sk   ON k.segment_id = sk.segment_id
        JOIN time_series_data h
          ON h.take_id = k.take_id
         AND h.frame   = k.frame
        JOIN categories ch ON h.category_id = ch.category_id
        JOIN segments sh   ON h.segment_id = sh.segment_id
        WHERE k.take_id = %s
          AND ck.category_name = 'JOINT_ANGLES'
          AND sk.segment_name = %s
          AND ch.category_name = 'LANDMARK_ORIGINAL'
          AND sh.segment_name = %s
          AND k.frame BETWEEN %s AND %s
          AND k.x_data IS NOT NULL
          AND h.z_data IS NOT NULL
        ORDER BY k.frame ASC
    """, (int(take_id), knee_segment, heel_segment, int(start_frame), int(end_frame)))

    rows = cur.fetchall()
    if not rows:
        return None

    # Find first sustained block where heel is below threshold.
    block_start = None
    block_end = None
    for i, (_frm, _kx, heel_z) in enumerate(rows):
        in_contact = float(heel_z) <= float(heel_thresh)
        if in_contact and block_start is None:
            block_start = i
        elif (not in_contact) and block_start is not None:
            if i - block_start >= int(min_consecutive_frames):
                block_end = i - 1
                break
            block_start = None

    if block_start is not None and block_end is None:
        if len(rows) - block_start >= int(min_consecutive_frames):
            block_end = len(rows) - 1

    if block_start is not None and block_end is not None:
        block_rows = rows[block_start:block_end + 1]
        peak_row = min(block_rows, key=lambda r: float(r[1]))
        return int(peak_row[0])

    # Fallback 1: any frame meeting heel threshold.
    contact_rows = [r for r in rows if float(r[2]) <= float(heel_thresh)]
    if contact_rows:
        peak_row = min(contact_rows, key=lambda r: float(r[1]))
        return int(peak_row[0])

    # Fallback 2: min knee x in full queried range.
    peak_row = min(rows, key=lambda r: float(r[1]))
    return int(peak_row[0])

# --- Pulldown helper: Pelvis angular velocity peak frame ---
def get_pelvis_angvel_peak_frame(take_id, handedness, cur):
    """
    Returns pelvis angular velocity Z peak frame.
    Constraint:
      - Peak must occur >= (min_frame + 20)
    Direction:
      - RHP -> max positive Z
      - LHP -> max negative Z
    """
    # Find earliest frame available for this take/category/segment
    cur.execute("""
        SELECT MIN(ts.frame)
        FROM time_series_data ts
        JOIN categories c ON ts.category_id = c.category_id
        JOIN segments s   ON ts.segment_id = s.segment_id
        WHERE ts.take_id = %s
          AND c.category_name = 'ORIGINAL'
          AND s.segment_name = 'PELVIS_ANGULAR_VELOCITY'
    """, (int(take_id),))
    row0 = cur.fetchone()
    if not row0 or row0[0] is None:
        return None

    # Find latest frame available (to exclude last 30 frames)
    cur.execute("""
        SELECT MAX(ts.frame)
        FROM time_series_data ts
        JOIN categories c ON ts.category_id = c.category_id
        JOIN segments s   ON ts.segment_id = s.segment_id
        WHERE ts.take_id = %s
          AND c.category_name = 'ORIGINAL'
          AND s.segment_name = 'PELVIS_ANGULAR_VELOCITY'
    """, (int(take_id),))
    row_end = cur.fetchone()
    if not row_end or row_end[0] is None:
        return None

    min_allowed_frame = int(row0[0]) + 20
    max_allowed_frame = int(row_end[0]) - 30

    order_clause = "ORDER BY ts.z_data DESC" if handedness == "R" else "ORDER BY ts.z_data ASC"

    cur.execute(f"""
        SELECT ts.frame
        FROM time_series_data ts
        JOIN categories c ON ts.category_id = c.category_id
        JOIN segments s   ON ts.segment_id = s.segment_id
        WHERE ts.take_id = %s
          AND c.category_name = 'ORIGINAL'
          AND s.segment_name = 'PELVIS_ANGULAR_VELOCITY'
          AND ts.z_data IS NOT NULL
          AND ts.frame >= %s
          AND ts.frame <= %s
        {order_clause}
        LIMIT 1
    """, (int(take_id), min_allowed_frame, max_allowed_frame))

    row = cur.fetchone()
    return int(row[0]) if row else None
# --- Helper: Pelvis angular velocity peak frame AND value ---
def get_pelvis_angvel_peak(take_id, handedness, cur):
    """
    Returns (frame, value) of pelvis angular velocity Z peak.
    Constraint:
      - Peak must occur >= (min_frame + 20)
    Direction:
      - RHP -> max positive Z
      - LHP -> max negative Z
    """
    # Find earliest frame available
    cur.execute("""
        SELECT MIN(ts.frame)
        FROM time_series_data ts
        JOIN categories c ON ts.category_id = c.category_id
        JOIN segments s   ON ts.segment_id = s.segment_id
        WHERE ts.take_id = %s
          AND c.category_name = 'ORIGINAL'
          AND s.segment_name = 'PELVIS_ANGULAR_VELOCITY'
    """, (int(take_id),))
    row0 = cur.fetchone()
    if not row0 or row0[0] is None:
        return None, None

    # Find latest frame available (to exclude last 30 frames)
    cur.execute("""
        SELECT MAX(ts.frame)
        FROM time_series_data ts
        JOIN categories c ON ts.category_id = c.category_id
        JOIN segments s   ON ts.segment_id = s.segment_id
        WHERE ts.take_id = %s
          AND c.category_name = 'ORIGINAL'
          AND s.segment_name = 'PELVIS_ANGULAR_VELOCITY'
    """, (int(take_id),))
    row_end = cur.fetchone()
    if not row_end or row_end[0] is None:
        return None, None

    min_allowed_frame = int(row0[0]) + 20
    max_allowed_frame = int(row_end[0]) - 30

    order_clause = "ORDER BY ts.z_data DESC" if handedness == "R" else "ORDER BY ts.z_data ASC"

    cur.execute(f"""
        SELECT ts.frame, ts.z_data
        FROM time_series_data ts
        JOIN categories c ON ts.category_id = c.category_id
        JOIN segments s   ON ts.segment_id = s.segment_id
        WHERE ts.take_id = %s
          AND c.category_name = 'ORIGINAL'
          AND s.segment_name = 'PELVIS_ANGULAR_VELOCITY'
          AND ts.z_data IS NOT NULL
          AND ts.frame >= %s
          AND ts.frame <= %s
        {order_clause}
        LIMIT 1
    """, (int(take_id), min_allowed_frame, max_allowed_frame))

    row = cur.fetchone()
    if row:
        return int(row[0]), float(row[1])
    return None, None
# --- Pulldown Foot Plant (Pelvis-anchored) ---
def get_foot_plant_frame(take_id, handedness, cur):
    """
    Pulldown Foot Plant (Pelvis-anchored):
    1) Find max pelvis angular velocity (z_data)
    2) Search ±15 frames around that for:
       - largest negative dip in lead foot DistEndVel Z
    3) Foot Plant = first return to ~zero AFTER that dip
    """

    lead_foot = "LFT" if handedness == "R" else "RFT"

    # -------------------------------------------------
    # 1) Pelvis angular velocity peak (anchor)
    # -------------------------------------------------
    pelvis_peak_frame = get_pelvis_angvel_peak_frame(take_id, handedness, cur)
    if pelvis_peak_frame is None:
        return None

    search_start = pelvis_peak_frame - 30
    search_end   = pelvis_peak_frame + 30

    # -------------------------------------------------
    # 2) Largest negative dip in DistEndVel Z (windowed)
    # -------------------------------------------------
    cur.execute("""
        SELECT ts.frame, ts.z_data
        FROM time_series_data ts
        JOIN categories c ON ts.category_id = c.category_id
        JOIN segments s   ON ts.segment_id = s.segment_id
        WHERE ts.take_id = %s
          AND c.category_name = 'KINETIC_KINEMATIC_DistEndVel'
          AND s.segment_name = %s
          AND ts.z_data IS NOT NULL
          AND ts.frame BETWEEN %s AND %s
        ORDER BY ts.z_data ASC
        LIMIT 1
    """, (int(take_id), lead_foot, int(search_start), int(search_end)))

    row = cur.fetchone()
    if not row:
        return None

    trough_frame = int(row[0])

    # -------------------------------------------------
    # 3) Back-to-zero AFTER trough (unchanged logic)
    # -------------------------------------------------
    cur.execute("""
        SELECT ts.frame
        FROM time_series_data ts
        JOIN categories c ON ts.category_id = c.category_id
        JOIN segments s   ON ts.segment_id = s.segment_id
        WHERE ts.take_id = %s
          AND c.category_name = 'KINETIC_KINEMATIC_DistEndVel'
          AND s.segment_name = %s
          AND ts.frame > %s
          AND ts.z_data >= -0.05
        ORDER BY ts.frame ASC
        LIMIT 1
    """, (int(take_id), lead_foot, trough_frame))

    row2 = cur.fetchone()
    return int(row2[0]) if row2 else None

def get_pulldown_window(take_id, handedness, cur):
    """
    Pulldown-only bounded window for drive (RTA_DIST), peak arm energy, and AUC searches.
    Window = [FP - 80, BR]
      - FP: pelvis-anchored foot plant
      - BR: pulldown ball release (peak |hand CGVel X| after FP; pelvis-anchored fallback)
    Returns: (start_frame, end_frame, fp_frame)
    """
    fp_frame = get_foot_plant_frame(take_id, handedness, cur)
    if fp_frame is None:
        return None, None, None

    br_frame = get_ball_release_frame_pulldown(take_id, handedness, fp_frame, cur)
    if br_frame is None:
        return None, None, int(fp_frame)

    return int(fp_frame) - 80, int(br_frame), int(fp_frame)
# Title
st.title("Biomechanics Viewer")

# Connect to DB
conn = get_connection()
cur = conn.cursor()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Compensation Analysis",
    "Session Comparison",
    "0-10 Report",
    "Time-Series",
    "Biodex",
    "Biodex (Test)",
])
def load_session_data(pitcher, date, rear_knee, torso_segment, shoulder_segment, arm_segment, velocity_min, velocity_max):
    # Get all take_ids on that date within velocity range
    cur.execute("""
        SELECT t.take_id FROM takes t
        JOIN athletes a ON t.athlete_id = a.athlete_id
        WHERE a.athlete_name = %s AND t.take_date = %s
        AND t.pitch_velo BETWEEN %s AND %s
        ORDER BY t.file_name
    """, (pitcher, date, velocity_min, velocity_max))
    take_rows = cur.fetchall()
    if not take_rows:
        return None

    dfs_shoulder = []
    dfs_torso = []

    for (take_id,) in take_rows:
        # Query torso power to get drive start
        cur.execute("""
            SELECT frame, x_data FROM time_series_data ts
            JOIN segments s ON ts.segment_id = s.segment_id
            JOIN categories c ON ts.category_id = c.category_id
            WHERE ts.take_id = %s AND c.category_name = 'SEGMENT_POWERS' AND s.segment_name = %s
            ORDER BY frame
        """, (take_id, torso_segment))
        df_power = pd.DataFrame(cur.fetchall(), columns=["frame", "x_data"])
        df_power["x_data"] = pd.to_numeric(df_power["x_data"], errors="coerce").fillna(0)
        drive_start_frame = df_power[df_power["x_data"] < -3000]["frame"].min()
        if pd.isna(drive_start_frame):
            continue

        # Rear knee peak flexion
        cur.execute("""
            SELECT frame, x_data FROM time_series_data ts
            JOIN segments s ON ts.segment_id = s.segment_id
            JOIN categories c ON ts.category_id = c.category_id
            WHERE ts.take_id = %s AND c.category_name = 'JOINT_ANGLES' AND s.segment_name = %s
            ORDER BY frame
        """, (take_id, rear_knee))
        df_knee = pd.DataFrame(cur.fetchall(), columns=["frame", "x_data"])
        if df_knee.empty:
            continue
        # --- Rear knee anchor (Pulldown-safe) ---
        df_knee["x_data"] = pd.to_numeric(df_knee["x_data"], errors="coerce")
        df_knee = df_knee.dropna(subset=["x_data"])

        if throw_type == "Pulldown":
            fp_frame = get_foot_plant_frame(take_id, handedness, cur)

            if fp_frame is not None:
                knee_window = df_knee[
                    (df_knee["frame"] >= fp_frame - 80) &
                    (df_knee["frame"] < fp_frame)
                    ]
            else:
                # fallback (rare)
                knee_window = df_knee[
                    (df_knee["frame"] >= drive_start_frame - 80) &
                    (df_knee["frame"] < drive_start_frame)
                    ]
        else:
            knee_window = df_knee[df_knee["frame"] < drive_start_frame]

        if knee_window.empty:
            continue

        idx = knee_window["x_data"].idxmin()
        if pd.isna(idx):
            continue

        max_knee_frame = int(knee_window.loc[idx, "frame"])

        # Arm energy to get end frame
        cur.execute("""
            SELECT frame, x_data FROM time_series_data ts
            JOIN segments s ON ts.segment_id = s.segment_id
            JOIN categories c ON ts.category_id = c.category_id
            WHERE ts.take_id = %s AND c.category_name = 'SEGMENT_ENERGIES' AND s.segment_name = %s
            ORDER BY frame
        """, (take_id, arm_segment))
        df_arm = pd.DataFrame(cur.fetchall(), columns=["frame", "x_data"])
        if df_arm.empty:
            continue
        peak_arm_frame = df_arm.loc[df_arm["x_data"].idxmax(), "frame"]
        end_frame = peak_arm_frame + 50

        # Shoulder rotation (z_data)
        cur.execute("""
            SELECT frame, z_data FROM time_series_data ts
            JOIN segments s ON ts.segment_id = s.segment_id
            JOIN categories c ON ts.category_id = c.category_id
            WHERE ts.take_id = %s AND c.category_name = 'JOINT_ANGLES' AND s.segment_name = %s
            ORDER BY frame
        """, (take_id, shoulder_segment))
        df_shoulder = pd.DataFrame(cur.fetchall(), columns=["frame", "z_data"])
        df_shoulder = df_shoulder[(df_shoulder["frame"] >= max_knee_frame) & (df_shoulder["frame"] <= end_frame)]
        if df_shoulder.empty:
            continue
        df_shoulder = normalize_time(df_shoulder, max_knee_frame, end_frame)

        # Torso power
        cur.execute("""
            SELECT frame, x_data FROM time_series_data ts
            JOIN segments s ON ts.segment_id = s.segment_id
            JOIN categories c ON ts.category_id = c.category_id
            WHERE ts.take_id = %s AND c.category_name = 'SEGMENT_POWERS' AND s.segment_name = %s
            ORDER BY frame
        """, (take_id, torso_segment))
        df_torso = pd.DataFrame(cur.fetchall(), columns=["frame", "x_data"])
        df_torso = df_torso[(df_torso["frame"] >= max_knee_frame) & (df_torso["frame"] <= end_frame)]
        if df_torso.empty:
            continue
        df_torso = normalize_time(df_torso, max_knee_frame, end_frame)

        dfs_shoulder.append(df_shoulder)
        dfs_torso.append(df_torso)

    if not dfs_shoulder or not dfs_torso:
        return None

    # Merge averages
    df_shoulder_all = pd.concat(dfs_shoulder).groupby("time_pct").mean().reset_index()
    df_torso_all = pd.concat(dfs_torso).groupby("time_pct").mean().reset_index()

    return {
        "take_ids": [row[0] for row in take_rows],
        "df_shoulder": df_shoulder_all,
        "df_torso": df_torso_all
    }

with tab1:
    # --- Four-column controls ---
    cur.execute("""
        SELECT DISTINCT a.athlete_name
        FROM athletes a
        JOIN takes t ON a.athlete_id = t.athlete_id
        ORDER BY a.athlete_name
    """)
    pitchers = [row[0] for row in cur.fetchall()]
    selected_pitcher = st.selectbox("Pitcher", pitchers)

    # --- Get handedness from DB ---
    cur.execute("SELECT handedness FROM athletes WHERE athlete_name = %s", (selected_pitcher,))
    handedness_row = cur.fetchone()
    handedness = handedness_row[0] if handedness_row else "R"
    rear_knee = "RT_KNEE" if handedness == "R" else "LT_KNEE"
    torso_segment = "RTA_DIST_R" if handedness == "R" else "RTA_DIST_L"
    arm_segment = "RAR" if handedness == "R" else "LAR"
    shoulder_stp_segment = "RTA_RAR" if handedness == "R" else "RTA_LAR"

    # --- Throw type options ---
    cur.execute("""
        SELECT DISTINCT COALESCE(t.throw_type, 'Mound') AS throw_type
        FROM takes t
        JOIN athletes a ON t.athlete_id = a.athlete_id
        WHERE a.athlete_name = %s
        ORDER BY throw_type
    """, (selected_pitcher,))
    throw_type_options = [r[0] for r in cur.fetchall()] or ["Mound", "Pulldown"]
    default_throw_types = ["Mound"] if "Mound" in throw_type_options else [throw_type_options[0]]

    selected_throw_types = st.multiselect(
        "Throw Type",
        options=throw_type_options,
        default=default_throw_types,
        key="throw_types"
    )
    # --- Guard: never allow empty throw type list (prevents IN ()) ---
    if not selected_throw_types:
        selected_throw_types = default_throw_types

    # --- Session date options ---
    cur.execute("""
        SELECT DISTINCT t.take_date
        FROM takes t
        JOIN athletes a ON a.athlete_id = t.athlete_id
        WHERE a.athlete_name = %s
        ORDER BY t.take_date
    """, (selected_pitcher,))
    dates = [row[0].strftime("%Y-%m-%d") for row in cur.fetchall()]
    dates.insert(0, "All Dates")

    selected_dates = st.multiselect(
        "Session Date",
        options=dates,
        default=["All Dates"],
        key="tab1_dates"
    )

    # --- Get takes for selected pitcher/date ---
    if "All Dates" in selected_dates or not selected_dates:
        placeholders_tt = ",".join(["%s"] * len(selected_throw_types))
        cur.execute(f"""
                    SELECT
                        t.take_id,
                        t.file_name,
                        t.pitch_velo,
                        t.take_date,
                        COALESCE(t.throw_type, 'Mound') AS throw_type
                    FROM takes t
                             JOIN athletes a ON t.athlete_id = a.athlete_id
                    WHERE a.athlete_name = %s
                      AND COALESCE(t.throw_type, 'Mound') IN ({placeholders_tt})
                    ORDER BY t.take_date, t.file_name
                    """, (selected_pitcher, *selected_throw_types))
        take_rows = cur.fetchall()
    else:
        placeholders = ",".join(["%s"] * len(selected_dates))
        placeholders_tt = ",".join(["%s"] * len(selected_throw_types))
        cur.execute(f"""
            SELECT
                t.take_id,
                t.file_name,
                t.pitch_velo,
                t.take_date,
                COALESCE(t.throw_type, 'Mound') AS throw_type
            FROM takes t
            JOIN athletes a ON t.athlete_id = a.athlete_id
            WHERE a.athlete_name = %s
              AND t.take_date IN ({placeholders})
              AND COALESCE(t.throw_type, 'Mound') IN ({placeholders_tt})
            ORDER BY t.take_date, t.file_name
        """, (selected_pitcher, *selected_dates, *selected_throw_types))
        take_rows = cur.fetchall()

    if not take_rows:
        st.warning("No takes found for this pitcher and date.")
        st.stop()

    energy_plot_options = st.multiselect(
        "Energy Flow Type",
        [
            "Torso Power",
            "STP Elevation",
            "STP Horizontal Abduction",
            "STP Rotational",
            "STP Rotational into Layback",
            "STP Rotational into Ball",
        ],
        default=["Torso Power"],
        key="tab1_energy_plot_options"
    )

    # ---- Build session-scoped pitch numbers (reset per session date) ----
    from collections import defaultdict

    pitch_number_map = {}  # take_id -> pitch_number (per session date)
    takes_by_date = defaultdict(list)

    for tid, file_name, velo, take_date, throw_type in take_rows:
        takes_by_date[take_date].append((tid, file_name))

    for take_date in sorted(takes_by_date.keys()):
        ordered = sorted(takes_by_date[take_date], key=lambda x: x[1])  # file_name order
        for idx, (tid, _fname) in enumerate(ordered, start=1):
            pitch_number_map[int(tid)] = idx

    # ---- Exclude Takes (Tab 1) ----
    # Build stable labels from take_rows (ordered by date, filename), with pitch numbers from pitch_number_map
    exclude_pairs = []
    for tid, file_name, velo, take_date, throw_type in take_rows:
        lbl = (
            f"{take_date.strftime('%Y-%m-%d')} | {throw_type} | "
            f"Pitch {pitch_number_map[int(tid)]} ({velo:.1f} mph)"
        )
        exclude_pairs.append((lbl, int(tid)))

    exclude_label_options = [l for l, _ in exclude_pairs]

    excluded_labels = st.multiselect(
        "Exclude Takes",
        options=exclude_label_options,
        default=[],
        key="tab1_exclude_takes"
    )

    exclude_take_ids = {tid for (lbl, tid) in exclude_pairs if lbl in excluded_labels}



    # --- Ensure empty selections are still guarded ---
    if not selected_throw_types:
        selected_throw_types = default_throw_types
    if not energy_plot_options:
        energy_plot_options = ["Torso Power"]

    # --- Get takes for selected pitcher/date ---
    if "All Dates" in selected_dates or not selected_dates:
        placeholders_tt = ",".join(["%s"] * len(selected_throw_types))
        cur.execute(f"""
                    SELECT
                        t.take_id,
                        t.file_name,
                        t.pitch_velo,
                        t.take_date,
                        COALESCE(t.throw_type, 'Mound') AS throw_type
                    FROM takes t
                             JOIN athletes a ON t.athlete_id = a.athlete_id
                    WHERE a.athlete_name = %s
                      AND COALESCE(t.throw_type, 'Mound') IN ({placeholders_tt})
                    ORDER BY t.take_date, t.file_name
                    """, (selected_pitcher, *selected_throw_types))
        take_rows = cur.fetchall()
    else:
        placeholders = ",".join(["%s"] * len(selected_dates))
        placeholders_tt = ",".join(["%s"] * len(selected_throw_types))
        cur.execute(f"""
            SELECT
                t.take_id,
                t.file_name,
                t.pitch_velo,
                t.take_date,
                COALESCE(t.throw_type, 'Mound') AS throw_type
            FROM takes t
            JOIN athletes a ON t.athlete_id = a.athlete_id
            WHERE a.athlete_name = %s
              AND t.take_date IN ({placeholders})
              AND COALESCE(t.throw_type, 'Mound') IN ({placeholders_tt})
            ORDER BY t.take_date, t.file_name
        """, (selected_pitcher, *selected_dates, *selected_throw_types))
        take_rows = cur.fetchall()

    if not take_rows:
        st.warning("No takes found for this pitcher and date.")
        st.stop()

    rows = []
    for tid, file_name, velo, take_date, throw_type in take_rows:
        if int(tid) in exclude_take_ids:
            continue
        pitch_number = pitch_number_map[int(tid)]
        label = f"{take_date.strftime('%Y-%m-%d')} | {throw_type} | Pitch {pitch_number} ({velo:.1f} mph)"
        # ---- Defaults: always include take; metrics may be NaN if detection fails ----
        max_knee_frame = np.nan
        max_knee_value = np.nan
        drive_start_frame = np.nan
        torso_end_frame = np.nan
        auc_total = np.nan
        arm_peak_frame = np.nan
        arm_peak_value = np.nan
        auc_to_peak = np.nan
        auc_pct = np.nan
        auc_stp_total = np.nan
        auc_stp_to_peak = np.nan
        auc_stp_habd_total = np.nan
        auc_stp_habd_to_peak = np.nan
        auc_stp_rot_total = np.nan
        auc_stp_rot_to_peak = np.nan
        auc_stp_rot_layback = np.nan
        auc_stp_rot_ball = np.nan
        pelvis_peak_frame = np.nan
        pelvis_peak_value = np.nan
        fp_frame = np.nan
        mer_frame = np.nan
        mer_value = np.nan
        # --- Foot Plant frame (pelvis-anchored) ---
        fp = get_foot_plant_frame(tid, handedness, cur)
        if fp is not None:
            fp_frame = float(fp)

        # --- Pelvis angular velocity peak frame and value ---
        pp_frame, pp_value = get_pelvis_angvel_peak(tid, handedness, cur)
        if pp_frame is not None:
            pelvis_peak_frame = float(pp_frame)
            pelvis_peak_value = float(pp_value)

        # Query torso power
        cur.execute("""
            SELECT frame, x_data FROM time_series_data ts
            JOIN segments s ON ts.segment_id = s.segment_id
            JOIN categories c ON ts.category_id = c.category_id
            WHERE ts.take_id = %s AND c.category_name = 'SEGMENT_POWERS' AND s.segment_name = %s
            ORDER BY frame
        """, (tid, torso_segment))
        df_power = pd.DataFrame(cur.fetchall(), columns=["frame", "x_data"])
        df_power["x_data"] = pd.to_numeric(df_power["x_data"], errors="coerce").fillna(0)

        # --- Drive start: within 50 frames BEFORE MER ---
        peak_shoulder_frame = get_shoulder_er_max_frame(tid, handedness, cur, throw_type=throw_type)
        # Keep None as None (do NOT coerce to NaN) so we never attempt int(np.nan)

        # --- MER frame and value columns ---
        if peak_shoulder_frame is not None and not (isinstance(peak_shoulder_frame, float) and np.isnan(peak_shoulder_frame)):
            mer_frame = float(int(peak_shoulder_frame))

            shoulder_segment = "RT_SHOULDER" if handedness == "R" else "LT_SHOULDER"
            cur.execute("""
                SELECT ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s   ON ts.segment_id  = s.segment_id
                WHERE ts.take_id = %s
                  AND c.category_name = 'JOINT_ANGLES'
                  AND s.segment_name = %s
                  AND ts.frame = %s
                  AND ts.z_data IS NOT NULL
                LIMIT 1
            """, (int(tid), shoulder_segment, int(mer_frame)))
            r = cur.fetchone()
            if r and r[0] is not None:
                mer_value = float(r[0])

        # Only attempt drive start detection if peak_shoulder_frame is valid
        if peak_shoulder_frame is not None and not (isinstance(peak_shoulder_frame, float) and np.isnan(peak_shoulder_frame)):
            df_power_window = df_power[
                (df_power["frame"] >= peak_shoulder_frame - 50) &
                (df_power["frame"] < peak_shoulder_frame)
            ]
            drive_start_frame = df_power_window[df_power_window["x_data"] < -3000]["frame"].min()

        if pd.isna(drive_start_frame):
            drive_start_frame = np.nan

        # --- Max Rear Knee Flexion (HEEL-constrained) ---
        rk_frame, rk_value = get_max_rear_knee_flexion_frame_with_heel(tid, handedness, cur)
        if rk_frame is not None:
            max_knee_frame = float(rk_frame)
        if rk_value is not None:
            max_knee_value = float(rk_value)

        # Only compute torso_end_frame and auc_total if max_knee_frame valid and df_power not empty
        if not np.isnan(max_knee_frame) and not df_power.empty:
            df_after = df_power[df_power["frame"] > max_knee_frame].copy()
            if not df_after.empty:
                neg_peak_idx = df_after["x_data"].idxmin()
                neg_peak_frame = df_after.loc[neg_peak_idx, "frame"]
                df_after_peak = df_after[df_after["frame"] > neg_peak_frame]
                zero_cross = df_after_peak[df_after_peak["x_data"] >= 0]

                torso_end_frame = (
                    float(int(zero_cross.iloc[0]["frame"]) - 1)
                    if not zero_cross.empty else float(int(df_after["frame"].iloc[-1]))
                )

                df_segment = df_power[
                    (df_power["frame"] >= max_knee_frame) &
                    (df_power["frame"] <= torso_end_frame)
                ]
                if not df_segment.empty:
                    auc_total = float(np.trapezoid(df_segment["x_data"], df_segment["frame"]))

        # Peak Arm Energy (MER-windowed max: ±30 frames around MER)
        cur.execute("""
            SELECT frame, x_data FROM time_series_data ts
            JOIN segments s ON ts.segment_id = s.segment_id
            JOIN categories c ON ts.category_id = c.category_id
            WHERE ts.take_id = %s AND c.category_name = 'SEGMENT_ENERGIES' AND s.segment_name = %s
            ORDER BY frame
        """, (tid, arm_segment))
        df_arm = pd.DataFrame(cur.fetchall(), columns=["frame", "x_data"])

        if not df_arm.empty and not np.isnan(max_knee_frame):
            mer_frame_local = int(peak_shoulder_frame) if not np.isnan(peak_shoulder_frame) else None
            df_arm["x_data"] = pd.to_numeric(df_arm["x_data"], errors="coerce")

            if mer_frame_local is not None:
                df_arm_window = df_arm[
                    (df_arm["frame"] >= mer_frame_local - 30) &
                    (df_arm["frame"] <= mer_frame_local + 30)
                ].copy()
            else:
                df_arm_window = df_arm.copy()

            if not df_arm_window.empty:
                windowed_energy = df_arm_window["x_data"]
                if not windowed_energy.isna().all():
                    peak_idx = int(windowed_energy.idxmax())
                    arm_peak_frame = float(int(df_arm.loc[peak_idx, "frame"]))
                    arm_peak_value = float(df_arm.loc[peak_idx, "x_data"])

                    # AUC to peak only if torso_end_frame and auc_total exist
                    if not np.isnan(torso_end_frame) and not np.isnan(auc_total):
                        df_to_peak = df_power[
                            (df_power["frame"] >= max_knee_frame) &
                            (df_power["frame"] <= arm_peak_frame)
                        ]
                        if not df_to_peak.empty:
                            auc_to_peak = float(np.trapezoid(df_to_peak["x_data"], df_to_peak["frame"]))
                            if auc_total != 0:
                                auc_pct = float(auc_to_peak / auc_total * 100.0)

        # ---------------- Shoulder STP Elev/Dep AUC ----------------
        cur.execute("""
            SELECT frame, x_data
            FROM time_series_data ts
            JOIN segments s ON ts.segment_id = s.segment_id
            JOIN categories c ON ts.category_id = c.category_id
            WHERE ts.take_id = %s
              AND c.category_name = 'JCS_STP_ELEV'
              AND s.segment_name = %s
            ORDER BY frame
        """, (tid, shoulder_stp_segment))
        df_stp = pd.DataFrame(cur.fetchall(), columns=["frame", "x_data"])
        if not df_stp.empty and not np.isnan(max_knee_frame) and not np.isnan(torso_end_frame):
            df_stp["x_data"] = pd.to_numeric(df_stp["x_data"], errors="coerce").fillna(0)
            df_stp_seg = df_stp[
                (df_stp["frame"] >= max_knee_frame) &
                (df_stp["frame"] <= torso_end_frame)
            ]
            if not df_stp_seg.empty:
                auc_stp_total = float(np.trapezoid(df_stp_seg["x_data"], df_stp_seg["frame"]))
            if not np.isnan(arm_peak_frame):
                df_stp_to_peak = df_stp[
                    (df_stp["frame"] >= max_knee_frame) &
                    (df_stp["frame"] <= arm_peak_frame)
                ]
                if not df_stp_to_peak.empty:
                    auc_stp_to_peak = float(np.trapezoid(df_stp_to_peak["x_data"], df_stp_to_peak["frame"]))

        # ---------------- Shoulder STP HorizAbd/Add AUC ----------------
        cur.execute("""
            SELECT frame, x_data
            FROM time_series_data ts
            JOIN segments s ON ts.segment_id = s.segment_id
            JOIN categories c ON ts.category_id = c.category_id
            WHERE ts.take_id = %s
              AND c.category_name = 'JCS_STP_HORIZABD'
              AND s.segment_name = %s
            ORDER BY frame
        """, (tid, shoulder_stp_segment))
        df_stp_habd = pd.DataFrame(cur.fetchall(), columns=["frame", "x_data"])
        if not df_stp_habd.empty and not np.isnan(max_knee_frame) and not np.isnan(torso_end_frame):
            df_stp_habd["x_data"] = pd.to_numeric(df_stp_habd["x_data"], errors="coerce").fillna(0)
            df_habd_seg = df_stp_habd[
                (df_stp_habd["frame"] >= max_knee_frame) &
                (df_stp_habd["frame"] <= torso_end_frame)
            ]
            if not df_habd_seg.empty:
                auc_stp_habd_total = float(np.trapezoid(df_habd_seg["x_data"], df_habd_seg["frame"]))
            if not np.isnan(arm_peak_frame):
                df_habd_to_peak = df_stp_habd[
                    (df_stp_habd["frame"] >= max_knee_frame) &
                    (df_stp_habd["frame"] <= arm_peak_frame)
                ]
                if not df_habd_to_peak.empty:
                    auc_stp_habd_to_peak = float(np.trapezoid(df_habd_to_peak["x_data"], df_habd_to_peak["frame"]))

        # ---------------- Shoulder STP Rotational AUC ----------------
        cur.execute("""
            SELECT frame, x_data
            FROM time_series_data ts
            JOIN segments s ON ts.segment_id = s.segment_id
            JOIN categories c ON ts.category_id = c.category_id
            WHERE ts.take_id = %s
              AND c.category_name = 'JCS_STP_ROT'
              AND s.segment_name = %s
            ORDER BY frame
        """, (tid, arm_segment))
        df_stp_rot = pd.DataFrame(cur.fetchall(), columns=["frame", "x_data"])
        if not df_stp_rot.empty and not np.isnan(max_knee_frame) and not np.isnan(torso_end_frame):
            df_stp_rot["x_data"] = pd.to_numeric(df_stp_rot["x_data"], errors="coerce").fillna(0)
            df_rot_seg = df_stp_rot[
                (df_stp_rot["frame"] >= max_knee_frame) &
                (df_stp_rot["frame"] <= torso_end_frame)
            ]
            if not df_rot_seg.empty:
                auc_stp_rot_total = float(np.trapezoid(df_rot_seg["x_data"], df_rot_seg["frame"]))
            if not np.isnan(arm_peak_frame):
                df_rot_to_peak = df_stp_rot[
                    (df_stp_rot["frame"] >= max_knee_frame) &
                    (df_stp_rot["frame"] <= arm_peak_frame)
                ]
                if not df_rot_to_peak.empty:
                    auc_stp_rot_to_peak = float(np.trapezoid(df_rot_to_peak["x_data"], df_rot_to_peak["frame"]))

            auc_stp_rot_layback, layback_end_frame = compute_negative_lobe_auc(
                df_stp_rot,
                threshold=-500,
                min_frame=max_knee_frame
            )
            auc_stp_rot_ball, _ = compute_positive_lobe_auc(
                df_stp_rot,
                min_frame=max_knee_frame,
                start_after_frame=layback_end_frame
            )

        rows.append({
            "take_id": tid,
            "label": label,
            "Pitch Number": pitch_number,
            "Session Date": take_date.strftime("%Y-%m-%d"),
            "Throw Type": throw_type,
            "Velocity": velo,
            "AUC (Drive → 0)": (round(auc_total, 2) if pd.notna(auc_total) else np.nan),
            "AUC (Drive → Peak Arm Energy)": (round(auc_to_peak, 2) if pd.notna(auc_to_peak) else np.nan),
            "Peak Arm Energy": (round(arm_peak_value, 2) if pd.notna(arm_peak_value) else np.nan),
            "% Total Energy Into Layback": (round(auc_pct, 1) if pd.notna(auc_pct) else np.nan),
            "STP Elevation AUC (Drive → 0)": (round(auc_stp_total, 2) if pd.notna(auc_stp_total) else np.nan),
            "STP Elevation AUC (Drive → Peak Arm Energy)": (round(auc_stp_to_peak, 2) if pd.notna(auc_stp_to_peak) else np.nan),
            "STP HorizAbd AUC (Drive → 0)": (round(auc_stp_habd_total, 2) if pd.notna(auc_stp_habd_total) else np.nan),
            "STP HorizAbd AUC (Drive → Peak Arm Energy)": (round(auc_stp_habd_to_peak, 2) if pd.notna(auc_stp_habd_to_peak) else np.nan),
            "STP Rotational AUC (Drive → 0)": (round(auc_stp_rot_total, 2) if pd.notna(auc_stp_rot_total) else np.nan),
            "STP Rotational AUC (Drive → Peak Arm Energy)": (round(auc_stp_rot_to_peak, 2) if pd.notna(auc_stp_rot_to_peak) else np.nan),
            "STP Rotational AUC (Into Layback)": (round(auc_stp_rot_layback, 2) if pd.notna(auc_stp_rot_layback) else np.nan),
            "STP Rotational AUC (Into Ball)": (round(auc_stp_rot_ball, 2) if pd.notna(auc_stp_rot_ball) else np.nan),
        })

# Guard for empty rows
if rows:
    df_tab1 = pd.DataFrame(rows)

    # Normalize Velocity column name if pitch_velo is used
    if "pitch_velo" in df_tab1.columns and "Velocity" not in df_tab1.columns:
        df_tab1 = df_tab1.rename(columns={"pitch_velo": "Velocity"})

    # Prepare regressions
    df_tab1["Velocity"] = pd.to_numeric(df_tab1["Velocity"], errors="coerce")
    df_tab1["AUC (Drive → 0)"] = pd.to_numeric(df_tab1["AUC (Drive → 0)"], errors="coerce")
    df_tab1["AUC (Drive → Peak Arm Energy)"] = pd.to_numeric(df_tab1["AUC (Drive → Peak Arm Energy)"], errors="coerce")
    # Keep ALL takes; regressions will use valid subsets later
    df_tab1_all = df_tab1.copy()
    df_tab1["STP Elevation AUC (Drive → 0)"] = pd.to_numeric(df_tab1["STP Elevation AUC (Drive → 0)"], errors="coerce")
    df_tab1["STP Elevation AUC (Drive → Peak Arm Energy)"] = pd.to_numeric(df_tab1["STP Elevation AUC (Drive → Peak Arm Energy)"], errors="coerce")
    df_tab1["STP HorizAbd AUC (Drive → 0)"] = pd.to_numeric(
        df_tab1["STP HorizAbd AUC (Drive → 0)"], errors="coerce"
    )
    df_tab1["STP HorizAbd AUC (Drive → Peak Arm Energy)"] = pd.to_numeric(
        df_tab1["STP HorizAbd AUC (Drive → Peak Arm Energy)"], errors="coerce"
    )
    df_tab1["STP Rotational AUC (Drive → 0)"] = pd.to_numeric(
        df_tab1["STP Rotational AUC (Drive → 0)"], errors="coerce"
    )
    df_tab1["STP Rotational AUC (Drive → Peak Arm Energy)"] = pd.to_numeric(
        df_tab1["STP Rotational AUC (Drive → Peak Arm Energy)"], errors="coerce"
    )
    df_tab1["STP Rotational AUC (Into Layback)"] = pd.to_numeric(
        df_tab1["STP Rotational AUC (Into Layback)"], errors="coerce"
    )
    df_tab1["STP Rotational AUC (Into Ball)"] = pd.to_numeric(
        df_tab1["STP Rotational AUC (Into Ball)"], errors="coerce"
    )
    # --- Normalize Throw Type: ensure always present and filled ---
    df_tab1["Throw Type"] = df_tab1["Throw Type"].fillna("Mound")

    # ---- Add customdata for POINT scatter traces (Tab 1 Energy Flow) ----
    # Ensure customdata includes: Session Date, Throw Type, Pitch Number, Velocity
    customdata_tab1_points = df_tab1[[
        "Session Date",
        "Throw Type",
        "Pitch Number",
        "Velocity"
    ]]
else:
    st.warning("No valid data found for this pitcher/date.")
    st.stop()

with tab1:
    # --- Date-based color map for Tab 1 ---
    date_color_cycle = px.colors.qualitative.Bold

    fig = go.Figure()
    # For marker symbols per metric
    metric_symbol_map = {
        "Torso Power": ["circle", "triangle-up"],
        "STP Elevation": ["diamond"],
        "STP Horizontal Abduction": ["square"],
        "STP Rotational": ["pentagon"],
        "STP Rotational into Layback": ["hexagon"],
        "STP Rotational into Ball": ["hexagon-open"],
    }
    # For regression line dashes per metric
    metric_dash_map = {
        "Torso Power": ["dash", "dot"],
        "STP Elevation": ["longdash"],
        "STP Horizontal Abduction": [None],
        "STP Rotational": ["dot"],
        "STP Rotational into Layback": ["dash"],
        "STP Rotational into Ball": ["dashdot"],
    }
    # For legend names per metric
    metric_trace_names = {
        "Torso Power": ["Torso AUC → 0", "Torso AUC → Peak"],
        "STP Elevation": ["STP Elev Peak"],
        "STP Horizontal Abduction": ["STP HABD Peak"],
        "STP Rotational": ["STP ROT Peak"],
    }
    metric_reg_names = {
        "Torso Power": ["R²", "Peak R²"],
        "STP Elevation": ["R²"],
        "STP Horizontal Abduction": ["R²"],
        "STP Rotational": ["R²"],
    }

    # --- Group by Session Date and Throw Type ---
    for i, ((date, throw_type), sub) in enumerate(df_tab1.groupby(["Session Date", "Throw Type"])):
        sub = sub.dropna(subset=["Velocity"]).copy()
        if len(sub) < 2:
            continue
        color = date_color_cycle[i % len(date_color_cycle)]
        x = sub["Velocity"]
        # customdata for these points (must match sub index)
        sub_customdata_df = sub[["Session Date", "Throw Type", "Pitch Number", "Velocity"]]
        for energy_plot_option in energy_plot_options:
            # Torso Power: plot both AUC → 0 and AUC → Peak
            if energy_plot_option == "Torso Power":
                # --- AUC Drive → 0 ---
                y0 = sub["AUC (Drive → 0)"].dropna()
                if y0.size >= 2:
                    slope0, intercept0, r0, _, _ = linregress(x.loc[y0.index], y0)
                    x_fit = np.linspace(x.min(), x.max(), 100)
                    y_fit0 = slope0 * x_fit + intercept0
                    fig.add_trace(go.Scatter(
                        x=x.loc[y0.index], y=y0, mode="markers",
                        marker=dict(color=color, symbol=metric_symbol_map[energy_plot_option][0]),
                        name=f"{date} | {throw_type} | {metric_trace_names[energy_plot_option][0]}",
                        # POINTS hover: customdata includes Pitch Number after Throw Type
                        customdata=sub_customdata_df.loc[y0.index].values,
                        hovertemplate=(
                            "%{customdata[0]} | %{customdata[1]} | Pitch %{customdata[2]}<br>"
                            "Value: %{y:.2f}<br>"
                            "Velocity: %{x:.1f} mph<br>"
                            "<extra></extra>"
                        ),
                    ))
                    fig.add_trace(go.Scatter(
                        x=x_fit, y=y_fit0, mode="lines",
                        line=dict(color=color, dash=metric_dash_map[energy_plot_option][0]),
                        name=f"R²={r0**2:.2f}",
                        hovertext=[f"{date} | {throw_type} | R²={r0**2:.2f}"] * len(x_fit),
                        hovertemplate="%{hovertext}<br>Velocity: %{x:.1f} mph<br>Predicted: %{y:.2f}<extra></extra>",
                    ))
                # --- AUC Drive → Peak Arm Energy ---
                y1 = sub["AUC (Drive → Peak Arm Energy)"].dropna()
                if y1.size >= 2:
                    slope1, intercept1, r1, _, _ = linregress(x.loc[y1.index], y1)
                    y_fit1 = slope1 * x_fit + intercept1
                    fig.add_trace(go.Scatter(
                        x=x.loc[y1.index], y=y1, mode="markers",
                        marker=dict(color=color, symbol=metric_symbol_map[energy_plot_option][1]),
                        name=f"{date} | {throw_type} | {metric_trace_names[energy_plot_option][1]}",
                        customdata=sub_customdata_df.loc[y1.index].values,
                        hovertemplate=(
                            "%{customdata[0]} | %{customdata[1]} | Pitch %{customdata[2]}<br>"
                            "Value: %{y:.2f}<br>"
                            "Velocity: %{x:.1f} mph<br>"
                            "<extra></extra>"
                        ),
                    ))
                    fig.add_trace(go.Scatter(
                        x=x_fit, y=y_fit1, mode="lines",
                        line=dict(color=color, dash=metric_dash_map[energy_plot_option][1]),
                        name=f"R²={r1**2:.2f}",
                        hovertext=[f"{date} | {throw_type} | R²={r1**2:.2f}"] * len(x_fit),
                        hovertemplate="%{hovertext}<br>Velocity: %{x:.1f} mph<br>Predicted: %{y:.2f}<extra></extra>",
                    ))
            elif energy_plot_option == "STP Elevation":
                # ---- Drive → 0 ----
                y0 = sub["STP Elevation AUC (Drive → 0)"].dropna()
                if y0.size >= 2:
                    slope0, intercept0, r0, _, _ = linregress(x.loc[y0.index], y0)
                    x_fit = np.linspace(x.min(), x.max(), 100)
                    y_fit0 = slope0 * x_fit + intercept0
                    fig.add_trace(go.Scatter(
                        x=x.loc[y0.index],
                        y=y0,
                        mode="markers",
                        marker=dict(color=color, symbol="diamond"),
                        name=f"{date} | {throw_type} | STP Elev → 0",
                        hovertext=[f"{date} | {throw_type} | STP Elev → 0"] * len(y0),
                        hovertemplate="%{hovertext}<br>Velocity: %{x:.1f} mph<br>Value: %{y:.2f}<extra></extra>",
                    ))
                    fig.add_trace(go.Scatter(
                        x=x_fit,
                        y=y_fit0,
                        mode="lines",
                        line=dict(color=color, dash="dash"),
                        name=f"R²={r0**2:.2f}",
                        hovertext=[f"{date} | {throw_type} | R²={r0**2:.2f}"] * len(x_fit),
                        hovertemplate="%{hovertext}<br>Velocity: %{x:.1f} mph<br>Predicted: %{y:.2f}<extra></extra>",
                    ))

                # ---- Drive → Peak Arm Energy ----
                y1 = sub["STP Elevation AUC (Drive → Peak Arm Energy)"].dropna()
                if y1.size >= 2:
                    slope1, intercept1, r1, _, _ = linregress(x.loc[y1.index], y1)
                    y_fit1 = slope1 * x_fit + intercept1
                    fig.add_trace(go.Scatter(
                        x=x.loc[y1.index],
                        y=y1,
                        mode="markers",
                        marker=dict(color=color, symbol="diamond-open"),
                        name=f"{date} | {throw_type} | STP Elev → Peak",
                        hovertext=[f"{date} | {throw_type} | STP Elev → Peak"] * len(y1),
                        hovertemplate="%{hovertext}<br>Velocity: %{x:.1f} mph<br>Value: %{y:.2f}<extra></extra>",
                    ))
                    fig.add_trace(go.Scatter(
                        x=x_fit,
                        y=y_fit1,
                        mode="lines",
                        line=dict(color=color, dash="dot"),
                        name=f"R²={r1**2:.2f}",
                        hovertext=[f"{date} | {throw_type} | R²={r1**2:.2f}"] * len(x_fit),
                        hovertemplate="%{hovertext}<br>Velocity: %{x:.1f} mph<br>Predicted: %{y:.2f}<extra></extra>",
                    ))
            elif energy_plot_option == "STP Horizontal Abduction":
                # ---- Drive → 0 ----
                y0 = sub["STP HorizAbd AUC (Drive → 0)"].dropna()
                if y0.size >= 2:
                    slope0, intercept0, r0, _, _ = linregress(x.loc[y0.index], y0)
                    x_fit = np.linspace(x.min(), x.max(), 100)
                    y_fit0 = slope0 * x_fit + intercept0
                    fig.add_trace(go.Scatter(
                        x=x.loc[y0.index],
                        y=y0,
                        mode="markers",
                        marker=dict(color=color, symbol="square"),
                        name=f"{date} | {throw_type} | STP HorizAbd → 0",
                        hovertext=[f"{date} | {throw_type} | STP HorizAbd → 0"] * len(y0),
                        hovertemplate="%{hovertext}<br>Velocity: %{x:.1f} mph<br>Value: %{y:.2f}<extra></extra>",
                    ))
                    fig.add_trace(go.Scatter(
                        x=x_fit,
                        y=y_fit0,
                        mode="lines",
                        line=dict(color=color, dash="dash"),
                        name=f"R²={r0**2:.2f}",
                        hovertext=[f"{date} | {throw_type} | R²={r0**2:.2f}"] * len(x_fit),
                        hovertemplate="%{hovertext}<br>Velocity: %{x:.1f} mph<br>Predicted: %{y:.2f}<extra></extra>",
                    ))

                # ---- Drive → Peak Arm Energy ----
                y1 = sub["STP HorizAbd AUC (Drive → Peak Arm Energy)"].dropna()
                if y1.size >= 2:
                    slope1, intercept1, r1, _, _ = linregress(x.loc[y1.index], y1)
                    y_fit1 = slope1 * x_fit + intercept1
                    fig.add_trace(go.Scatter(
                        x=x.loc[y1.index],
                        y=y1,
                        mode="markers",
                        marker=dict(color=color, symbol="square-open"),
                        name=f"{date} | {throw_type} | STP HorizAbd → Peak",
                        hovertext=[f"{date} | {throw_type} | STP HorizAbd → Peak"] * len(y1),
                        hovertemplate="%{hovertext}<br>Velocity: %{x:.1f} mph<br>Value: %{y:.2f}<extra></extra>",
                    ))
                    fig.add_trace(go.Scatter(
                        x=x_fit,
                        y=y_fit1,
                        mode="lines",
                        line=dict(color=color, dash="dot"),
                        name=f"R²={r1**2:.2f}",
                        hovertext=[f"{date} | {throw_type} | R²={r1**2:.2f}"] * len(x_fit),
                        hovertemplate="%{hovertext}<br>Velocity: %{x:.1f} mph<br>Predicted: %{y:.2f}<extra></extra>",
                    ))
            elif energy_plot_option == "STP Rotational":
                # ---- Drive → 0 ----
                y0 = sub["STP Rotational AUC (Drive → 0)"].dropna()
                if y0.size >= 2:
                    slope0, intercept0, r0, _, _ = linregress(x.loc[y0.index], y0)
                    x_fit = np.linspace(x.min(), x.max(), 100)
                    y_fit0 = slope0 * x_fit + intercept0
                    fig.add_trace(go.Scatter(
                        x=x.loc[y0.index],
                        y=y0,
                        mode="markers",
                        marker=dict(color=color, symbol="pentagon"),
                        name=f"{date} | {throw_type} | STP Rot → 0",
                        hovertext=[f"{date} | {throw_type} | STP Rot → 0"] * len(y0),
                        hovertemplate="%{hovertext}<br>Velocity: %{x:.1f} mph<br>Value: %{y:.2f}<extra></extra>",
                    ))
                    fig.add_trace(go.Scatter(
                        x=x_fit,
                        y=y_fit0,
                        mode="lines",
                        line=dict(color=color, dash="dash"),
                        name=f"R²={r0**2:.2f}",
                        hovertext=[f"{date} | {throw_type} | R²={r0**2:.2f}"] * len(x_fit),
                        hovertemplate="%{hovertext}<br>Velocity: %{x:.1f} mph<br>Predicted: %{y:.2f}<extra></extra>",
                    ))

                # ---- Drive → Peak Arm Energy ----
                y1 = sub["STP Rotational AUC (Drive → Peak Arm Energy)"].dropna()
                if y1.size >= 2:
                    slope1, intercept1, r1, _, _ = linregress(x.loc[y1.index], y1)
                    y_fit1 = slope1 * x_fit + intercept1
                    fig.add_trace(go.Scatter(
                        x=x.loc[y1.index],
                        y=y1,
                        mode="markers",
                        marker=dict(color=color, symbol="pentagon-open"),
                        name=f"{date} | {throw_type} | STP Rot → Peak",
                        hovertext=[f"{date} | {throw_type} | STP Rot → Peak"] * len(y1),
                        hovertemplate="%{hovertext}<br>Velocity: %{x:.1f} mph<br>Value: %{y:.2f}<extra></extra>",
                    ))
                    fig.add_trace(go.Scatter(
                        x=x_fit,
                        y=y_fit1,
                        mode="lines",
                        line=dict(color=color, dash="dot"),
                        name=f"R²={r1**2:.2f}",
                        hovertext=[f"{date} | {throw_type} | R²={r1**2:.2f}"] * len(x_fit),
                        hovertemplate="%{hovertext}<br>Velocity: %{x:.1f} mph<br>Predicted: %{y:.2f}<extra></extra>",
                    ))
            elif energy_plot_option == "STP Rotational into Layback":
                y0 = sub["STP Rotational AUC (Into Layback)"].dropna()
                if y0.size >= 2:
                    slope0, intercept0, r0, _, _ = linregress(x.loc[y0.index], y0)
                    x_fit = np.linspace(x.min(), x.max(), 100)
                    y_fit0 = slope0 * x_fit + intercept0
                    fig.add_trace(go.Scatter(
                        x=x.loc[y0.index],
                        y=y0,
                        mode="markers",
                        marker=dict(color=color, symbol=metric_symbol_map[energy_plot_option][0]),
                        name=f"{date} | {throw_type} | STP Rot → Layback",
                        hovertext=[f"{date} | {throw_type} | STP Rot → Layback"] * len(y0),
                        hovertemplate="%{hovertext}<br>Velocity: %{x:.1f} mph<br>Value: %{y:.2f}<extra></extra>",
                    ))
                    fig.add_trace(go.Scatter(
                        x=x_fit,
                        y=y_fit0,
                        mode="lines",
                        line=dict(color=color, dash=metric_dash_map[energy_plot_option][0]),
                        name=f"R²={r0**2:.2f}",
                        hovertext=[f"{date} | {throw_type} | R²={r0**2:.2f}"] * len(x_fit),
                        hovertemplate="%{hovertext}<br>Velocity: %{x:.1f} mph<br>Predicted: %{y:.2f}<extra></extra>",
                    ))
            elif energy_plot_option == "STP Rotational into Ball":
                y0 = sub["STP Rotational AUC (Into Ball)"].dropna()
                if y0.size >= 2:
                    slope0, intercept0, r0, _, _ = linregress(x.loc[y0.index], y0)
                    x_fit = np.linspace(x.min(), x.max(), 100)
                    y_fit0 = slope0 * x_fit + intercept0
                    fig.add_trace(go.Scatter(
                        x=x.loc[y0.index],
                        y=y0,
                        mode="markers",
                        marker=dict(color=color, symbol=metric_symbol_map[energy_plot_option][0]),
                        name=f"{date} | {throw_type} | STP Rot → Ball",
                        hovertext=[f"{date} | {throw_type} | STP Rot → Ball"] * len(y0),
                        hovertemplate="%{hovertext}<br>Velocity: %{x:.1f} mph<br>Value: %{y:.2f}<extra></extra>",
                    ))
                    fig.add_trace(go.Scatter(
                        x=x_fit,
                        y=y_fit0,
                        mode="lines",
                        line=dict(color=color, dash=metric_dash_map[energy_plot_option][0]),
                        name=f"R²={r0**2:.2f}",
                        hovertext=[f"{date} | {throw_type} | R²={r0**2:.2f}"] * len(x_fit),
                        hovertemplate="%{hovertext}<br>Velocity: %{x:.1f} mph<br>Predicted: %{y:.2f}<extra></extra>",
                    ))

    # Build dynamic title from energy_plot_options
    title_metric = ", ".join(energy_plot_options)
    dynamic_title = f"Velocity vs. {title_metric}"
    fig.update_layout(
        title=dynamic_title,
        xaxis_title="Velocity (mph)",
        yaxis_title="Energy / AUC",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5
        ),
        height=600
    )
    fig.update_layout(
        hoverlabel=dict(
            namelength=-1,
            font=dict(size=13)
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    def estimate_table_height(df, row_px=35, header_px=35, buffer_px=2):
        return len(df) * row_px + header_px + buffer_px

    height = estimate_table_height(df_tab1)
    display_cols = [col for col in df_tab1.columns if col not in ("take_id", "label")]

    priority_cols = [
        "Session Date",
        "Throw Type",
        "Pitch Number",
        "Velocity",
        "Peak Arm Energy Frame"
    ]

    # Keep only priority columns that exist
    priority_cols = [c for c in priority_cols if c in display_cols]

    # Final ordered columns
    display_cols = priority_cols + [c for c in display_cols if c not in priority_cols]

    st.dataframe(df_tab1[display_cols], height=height)

@st.cache_data
def load_reference_curves_player_mean(mode, pitcher_name, velo_min, velo_max, comp_col, use_abs, throw_types=None):
    if mode == "Selected Pitcher":
        if throw_types:
            placeholders_tt = ",".join(["%s"] * len(throw_types))
            cur.execute(f"""
                SELECT t.take_id, t.pitch_velo, t.athlete_id, a.athlete_name
                FROM takes t
                JOIN athletes a ON t.athlete_id = a.athlete_id
                WHERE a.athlete_name = %s
                  AND COALESCE(t.throw_type, 'Mound') IN ({placeholders_tt})
                  AND t.pitch_velo BETWEEN %s AND %s
                ORDER BY t.file_name
            """, (pitcher_name, *throw_types, velo_min, velo_max))
        else:
            cur.execute("""
                SELECT t.take_id, t.pitch_velo, t.athlete_id, a.athlete_name
                FROM takes t
                JOIN athletes a ON t.athlete_id = a.athlete_id
                WHERE a.athlete_name = %s
                  AND t.pitch_velo BETWEEN %s AND %s
                ORDER BY t.file_name
            """, (pitcher_name, velo_min, velo_max))
    else:
        if throw_types:
            placeholders_tt = ",".join(["%s"] * len(throw_types))
            cur.execute(f"""
                SELECT t.take_id, t.pitch_velo, t.athlete_id, a.handedness, a.athlete_name
                FROM takes t
                JOIN athletes a ON t.athlete_id = a.athlete_id
                WHERE COALESCE(t.throw_type, 'Mound') IN ({placeholders_tt})
                  AND t.pitch_velo BETWEEN %s AND %s
                ORDER BY t.file_name
            """, (*throw_types, velo_min, velo_max))
        else:
            cur.execute("""
                SELECT t.take_id, t.pitch_velo, t.athlete_id, a.handedness, a.athlete_name
                FROM takes t
                JOIN athletes a ON t.athlete_id = a.athlete_id
                WHERE t.pitch_velo BETWEEN %s AND %s
                ORDER BY t.file_name
            """, (velo_min, velo_max))
    rows = cur.fetchall()
    if not rows:
        return None, None, []
    from collections import defaultdict
    per_athlete_shoulders = defaultdict(list)
    per_athlete_torsos    = defaultdict(list)
    pitcher_names = set()
    for row in rows:
        if mode == "Selected Pitcher":
            take_id, _velo, athlete_id, athlete_name = row
            pitcher_names.add(athlete_name)
            cur.execute("SELECT handedness FROM athletes WHERE athlete_id = %s", (athlete_id,))
            handedness_row = cur.fetchone()
            handedness = handedness_row[0] if handedness_row else "R"
        else:
            take_id, _velo, athlete_id, handedness, athlete_name = row
            pitcher_names.add(athlete_name)
        rear_knee = "RT_KNEE" if handedness == "R" else "LT_KNEE"
        torso_segment = "RTA_DIST_R" if handedness == "R" else "RTA_DIST_L"
        shoulder_segment = "RT_SHOULDER" if handedness == "R" else "LT_SHOULDER"
        arm_segment = "RAR" if handedness == "R" else "LAR"
        cur.execute("""
            SELECT frame, x_data FROM time_series_data ts
            JOIN segments s ON ts.segment_id = s.segment_id
            JOIN categories c ON ts.category_id = c.category_id
            WHERE ts.take_id = %s AND c.category_name = 'SEGMENT_POWERS' AND s.segment_name = %s
            ORDER BY frame
        """, (take_id, torso_segment))
        df_power = pd.DataFrame(cur.fetchall(), columns=["frame", "x_data"])
        if df_power.empty:
            continue
        df_power["x_data"] = pd.to_numeric(df_power["x_data"], errors="coerce").fillna(0)
        # Determine take throw_type (needed for pulldown alignment)
        cur.execute("SELECT COALESCE(throw_type, 'Mound') FROM takes WHERE take_id = %s", (int(take_id),))
        _tt_row = cur.fetchone()
        take_throw_type = _tt_row[0] if _tt_row else "Mound"

        # Drive start anchor
        # - Mound: torso power threshold (< -3000)
        # - Pulldown: pulldown window start (FP - 80)
        if take_throw_type == "Pulldown":
            pd_start, pd_end, pd_fp = get_pulldown_window(take_id, handedness, cur)
            if pd_start is None or pd_end is None or pd_fp is None:
                continue
            drive_start_frame = int(pd_start)
        else:
            drive_start_frame = df_power[df_power["x_data"] < -3000]["frame"].min()
            if pd.isna(drive_start_frame):
                continue
        cur.execute("""
            SELECT frame, x_data FROM time_series_data ts
            JOIN segments s ON ts.segment_id = s.segment_id
            JOIN categories c ON ts.category_id = c.category_id
            WHERE ts.take_id = %s AND c.category_name = 'JOINT_ANGLES' AND s.segment_name = %s
            ORDER BY frame
        """, (take_id, rear_knee))
        df_knee = pd.DataFrame(cur.fetchall(), columns=["frame", "x_data"])
        if df_knee.empty:
            continue
        # Rear knee anchor
        # - Mound: knee min before drive_start_frame
        # - Pulldown: heel-gated rear knee flexion (Tab 1 helper)
        if take_throw_type == "Pulldown":
            rk_frame, _rk_val = get_max_rear_knee_flexion_frame_with_heel(take_id, handedness, cur)
            if rk_frame is None:
                continue
            max_knee_frame = int(rk_frame)
        else:
            df_knee_pre = df_knee[df_knee["frame"] < drive_start_frame]
            if df_knee_pre.empty:
                continue
            max_knee_frame = int(df_knee_pre.loc[df_knee_pre["x_data"].idxmin(), "frame"])
        cur.execute("""
            SELECT frame, x_data, y_data, z_data FROM time_series_data ts
            JOIN segments s ON ts.segment_id = s.segment_id
            JOIN categories c ON ts.category_id = c.category_id
            WHERE ts.take_id = %s AND c.category_name = 'JOINT_ANGLES' AND s.segment_name = %s
            ORDER BY frame
        """, (take_id, shoulder_segment))
        df_sh = pd.DataFrame(cur.fetchall(), columns=["frame", "x_data", "y_data", "z_data"])
        if df_sh.empty:
            continue
        df_sh["z_data"] = pd.to_numeric(df_sh["z_data"], errors="coerce")

        # MER frame
        # - Mound: legacy (max abs z)
        # - Pulldown: FP-windowed MER helper
        if take_throw_type == "Pulldown":
            peak_shoulder_frame = get_shoulder_er_max_frame(take_id, handedness, cur, throw_type="Pulldown")
            if peak_shoulder_frame is None:
                continue
            peak_shoulder_frame = int(peak_shoulder_frame)
        else:
            peak_shoulder_frame = int(df_sh.loc[df_sh["z_data"].abs().idxmax(), "frame"])
        cur.execute("""
            SELECT frame, x_data FROM time_series_data ts
            JOIN segments s ON ts.segment_id = s.segment_id
            JOIN categories c ON ts.category_id = c.category_id
            WHERE ts.take_id = %s AND c.category_name = 'SEGMENT_ENERGIES' AND s.segment_name = %s
            ORDER BY frame
        """, (take_id, arm_segment))
        df_arm = pd.DataFrame(cur.fetchall(), columns=["frame", "x_data"])
        if df_arm.empty:
            continue
        df_arm["x_data"] = pd.to_numeric(df_arm["x_data"], errors="coerce")
        if take_throw_type == "Pulldown":
            # End at BR for pulldowns (Tab 1 window)
            pd_start, pd_end, pd_fp = get_pulldown_window(take_id, handedness, cur)
            if pd_start is None or pd_end is None:
                continue
            end_frame = int(pd_end)
        else:
            end_frame = int(df_arm.loc[df_arm["x_data"].idxmax(), "frame"]) + 50
        df_sh_seg = df_sh[(df_sh["frame"] >= max_knee_frame) & (df_sh["frame"] <= end_frame)].copy()
        df_sh_seg["time_pct"] = np.where(
            df_sh_seg["frame"] <= peak_shoulder_frame,
            (df_sh_seg["frame"] - max_knee_frame) / (peak_shoulder_frame - max_knee_frame) * 50.0,
            50.0 + (df_sh_seg["frame"] - peak_shoulder_frame) / (end_frame - peak_shoulder_frame) * 50.0
        )
        _src = pd.to_numeric(df_sh_seg[comp_col], errors="coerce")
        if use_abs:
            _src = _src.abs()
        shoulder_curve = np.interp(interp_points, df_sh_seg["time_pct"], _src)
        df_pw_seg = df_power[(df_power["frame"] >= max_knee_frame) & (df_power["frame"] <= end_frame)].copy()
        df_pw_seg["time_pct"] = np.where(
            df_pw_seg["frame"] <= peak_shoulder_frame,
            (df_pw_seg["frame"] - max_knee_frame) / (peak_shoulder_frame - max_knee_frame) * 50.0,
            50.0 + (df_pw_seg["frame"] - peak_shoulder_frame) / (end_frame - peak_shoulder_frame) * 50.0
        )
        torso_curve = np.interp(interp_points, df_pw_seg["time_pct"], df_pw_seg["x_data"])
        per_athlete_shoulders[athlete_id].append(shoulder_curve)
        per_athlete_torsos[athlete_id].append(torso_curve)
    if not per_athlete_shoulders:
        return None, None, list(pitcher_names)
    per_player_sh = []
    per_player_to = []
    for aid in per_athlete_shoulders.keys():
        sh_arr = np.vstack(per_athlete_shoulders[aid])
        to_arr = np.vstack(per_athlete_torsos[aid])
        per_player_sh.append(np.nanmean(sh_arr, axis=0))
        per_player_to.append(np.nanmean(to_arr, axis=0))
    ref_shoulder = np.nanmean(np.vstack(per_player_sh), axis=0)
    ref_torso    = np.nanmean(np.vstack(per_player_to), axis=0)
    return ref_shoulder, ref_torso, sorted(pitcher_names)

with tab2:
    # TAB 2 — SESSION COMPARISON ONLY (NO VELOCITY PLOTS)
    # Two columns: controls (left), charts (right)
    left, right = st.columns([0.4, 1.4], vertical_alignment="top")

    # --- Session-date color map (Tab 2) ---
    session_colors = [
        "#1f77b4",  # blue
        "#2ca02c",  # green
        "#9467bd",  # purple
        "#17becf",  # teal
        "#8c564b",  # brown
        "#e377c2",  # pink (non-red)
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#aec7e8",  # light blue
        "#98df8a",  # light green
    ]
    _session_color_idx = {}

    def get_session_color(date_str):
        if date_str not in _session_color_idx:
            _session_color_idx[date_str] = len(_session_color_idx)
        return session_colors[_session_color_idx[date_str] % len(session_colors)]
    with left:
        # --- Select pitcher ---
        selected_pitcher_comp = st.selectbox("Select Pitcher", pitchers, key="comp_pitcher")
        selected_throw_types_comp = st.multiselect(
            "Throw Type(s)",
            options=["Mound", "Pulldown"],
            default=["Mound"],
            key="comp_throw_types"
        )
        if not selected_throw_types_comp:
            selected_throw_types_comp = ["Mound"]
        selected_throw_types_label = ", ".join(selected_throw_types_comp)
        reference_throw_types_comp = ["Mound"]  # Always exclude Pulldown from reference group
        display_mode_tab2 = st.radio(
            "Display Mode",
            ["Grouped Average", "Individual"],
            horizontal=True,
            key="comp_display_mode"
        )
        # --- Get handedness from DB ---
        cur.execute("SELECT handedness FROM athletes WHERE athlete_name = %s", (selected_pitcher_comp,))
        handedness_row_comp = cur.fetchone()
        handedness_comp = handedness_row_comp[0] if handedness_row_comp else "R"
        rear_knee_comp = "RT_KNEE" if handedness_comp == "R" else "LT_KNEE"
        torso_segment_comp = "RTA_DIST_R" if handedness_comp == "R" else "RTA_DIST_L"
        shoulder_segment_comp = "RT_SHOULDER" if handedness_comp == "R" else "LT_SHOULDER"
        arm_segment_comp = "RAR" if handedness_comp == "R" else "LAR"

        # --- Select session dates ---
        cur.execute("""
            SELECT DISTINCT t.take_date
            FROM takes t
            JOIN athletes a ON a.athlete_id = t.athlete_id
            WHERE a.athlete_name = %s
            ORDER BY t.take_date
        """, (selected_pitcher_comp,))
        dates_comp = [row[0].strftime("%Y-%m-%d") for row in cur.fetchall()]
        session1_date = st.selectbox("Select First Session Date", dates_comp, key="comp_date1")
        session2_date = st.selectbox("Select Second Session Date", dates_comp, key="comp_date2")

        # --- Get min and max velocity for session1 (selected throw type) ---
        placeholders_tt = ",".join(["%s"] * len(selected_throw_types_comp))
        cur.execute(f"""
            SELECT MIN(t.pitch_velo), MAX(t.pitch_velo)
            FROM takes t
            JOIN athletes a ON t.athlete_id = a.athlete_id
            WHERE a.athlete_name = %s
              AND t.take_date = %s
              AND COALESCE(t.throw_type, 'Mound') IN ({placeholders_tt})
        """, (selected_pitcher_comp, session1_date, *selected_throw_types_comp))
        velo_min1, velo_max1 = cur.fetchone()
        if velo_min1 is None or velo_max1 is None:
            st.warning(f"No {selected_throw_types_label} throws found for {session1_date}")
            st.stop()
        velo_min1 = float(f"{velo_min1:.1f}")
        velo_max1 = float(f"{velo_max1:.1f}")
        if velo_min1 >= velo_max1:
            velo_max1 = velo_min1 + 0.1
        velo_range1 = st.slider(
            f"Velocity Range for {session1_date}",
            min_value=velo_min1,
            max_value=velo_max1,
            value=(velo_min1, velo_max1),
            step=0.1,
            key="velo1"
        )

        # --- Get min and max velocity for session2 (selected throw type) ---
        cur.execute(f"""
            SELECT MIN(t.pitch_velo), MAX(t.pitch_velo)
            FROM takes t
            JOIN athletes a ON t.athlete_id = a.athlete_id
            WHERE a.athlete_name = %s
              AND t.take_date = %s
              AND COALESCE(t.throw_type, 'Mound') IN ({placeholders_tt})
        """, (selected_pitcher_comp, session2_date, *selected_throw_types_comp))
        velo_min2, velo_max2 = cur.fetchone()
        if velo_min2 is None or velo_max2 is None:
            st.warning(f"No {selected_throw_types_label} throws found for {session2_date}")
            st.stop()
        velo_min2 = float(f"{float(velo_min2):.1f}")
        velo_max2 = float(f"{float(velo_max2):.1f}")
        if velo_min2 >= velo_max2:
            velo_max2 = velo_min2 + 0.1
        velo_range2 = st.slider(
            f"Velocity Range for {session2_date}",
            min_value=velo_min2,
            max_value=velo_max2,
            value=(velo_min2, velo_max2),
            step=0.1,
            key="velo2"
        )

        def _build_take_options_for_session(date_str, velo_range):
            cur.execute(f"""
                SELECT t.take_id, t.file_name, COALESCE(t.throw_type, 'Mound') AS throw_type
                FROM takes t
                JOIN athletes a ON t.athlete_id = a.athlete_id
                WHERE a.athlete_name = %s
                  AND t.take_date = %s
                  AND COALESCE(t.throw_type, 'Mound') IN ({placeholders_tt})
                  AND t.pitch_velo BETWEEN %s AND %s
                ORDER BY t.file_name
            """, (selected_pitcher_comp, date_str, *selected_throw_types_comp, velo_range[0], velo_range[1]))
            rows_local = cur.fetchall()
            out = []
            for idx, (tid_local, _fname_local, tt_local) in enumerate(rows_local, start=1):
                out.append({
                    "take_id": int(tid_local),
                    "date": date_str,
                    "player": selected_pitcher_comp,
                    "throw_type": tt_local,
                    "take_number": idx,
                    "label": f"{date_str} | {selected_pitcher_comp} | {tt_local} | Take {idx}"
                })
            return out

        session1_take_options = _build_take_options_for_session(session1_date, velo_range1)
        session2_take_options = _build_take_options_for_session(session2_date, velo_range2)
        take_options_all = session1_take_options + session2_take_options
        take_options_by_label = {row["label"]: row for row in take_options_all}

        excluded_take_labels = st.multiselect(
            "Exclude Take(s)",
            options=list(take_options_by_label.keys()),
            default=[],
            key="comp_exclude_takes"
        )
        exclude_take_ids = {
            int(take_options_by_label[lbl]["take_id"])
            for lbl in excluded_take_labels
            if lbl in take_options_by_label
        }
        take_meta_by_id = {int(row["take_id"]): row for row in take_options_all}

        # Shoulder component selector
        component = st.selectbox("Shoulder Axis", ["Horizontal Abduction/Adduction (X)", "Abduction/Adduction (Y)", "Internal/External Rotation (Z)"], index=2, key="shoulder_component")
        comp_col = {"Horizontal Abduction/Adduction (X)": "x_data", "Abduction/Adduction (Y)": "y_data", "Internal/External Rotation (Z)": "z_data"}[component]
        use_abs = st.checkbox("Absolute Value", value=True, key="shoulder_abs")

        # --- Reference group controls (Grouped Average only) ---
        include_ref = False
        ref_mode = "All Pitchers"
        velo_range_ref = (95, 100)
        if display_mode_tab2 == "Grouped Average":
            include_ref = st.checkbox("Reference Group", value=True, key="ref_include")

            default_ref_velo_min = 95.0
            ref_mode = st.radio(
                "Reference Population",
                ["Selected Pitcher", "All Pitchers"],
                index=0 if st.session_state.get("ref_mode") == "Selected Pitcher" else 1,
                horizontal=True, key="ref_mode"
            )

            if ref_mode == "Selected Pitcher":
                cur.execute("""
                    SELECT MIN(t.pitch_velo), MAX(t.pitch_velo)
                    FROM takes t
                    JOIN athletes a ON t.athlete_id = a.athlete_id
                    WHERE a.athlete_name = %s
                      AND COALESCE(t.throw_type, 'Mound') = 'Mound'
                """, (selected_pitcher_comp,))
            else:
                cur.execute("""
                    SELECT MIN(t.pitch_velo), MAX(t.pitch_velo)
                    FROM takes t
                    WHERE COALESCE(t.throw_type, 'Mound') = 'Mound'
                """)
            rvmin, rvmax = cur.fetchone() or (None, None)
            if rvmin is None or rvmax is None:
                rvmin, rvmax = 70.0, 100.0
            rvmin = float(f"{rvmin:.1f}")
            rvmax = float(f"{rvmax:.1f}")
            if rvmin >= rvmax:
                rvmax = rvmin + 0.1

            velo_range_ref = st.slider(
                "Velocity Range for Reference Group",
                min_value=int(rvmin), max_value=int(rvmax),
                value=(int(st.session_state.get("ref_velo", (default_ref_velo_min, rvmax))[0]), int(st.session_state.get("ref_velo", (default_ref_velo_min, rvmax))[1])),
                step=1,
                key="ref_velo"
            )

            if include_ref:
                _, _, ref_pitchers = load_reference_curves_player_mean(
                    ref_mode, selected_pitcher_comp,
                    velo_range_ref[0], velo_range_ref[1],
                    comp_col, use_abs,
                    throw_types=reference_throw_types_comp
                )
                st.markdown("**Pitchers in Reference Group:**")
                if ref_pitchers:
                    if len(ref_pitchers) == 1:
                        st.info("Only one pitcher found in reference group for the selected settings.")
                    else:
                        st.code("\n".join(ref_pitchers), language="text")
                else:
                    st.info("No pitchers found in reference group for the selected settings.")

    def load_and_interpolate_curves(date, velo_min, velo_max, comp_col, use_abs):
        """
        Returns:
            shoulder_curves (list[np.ndarray])  # one per take, length 100
            torso_curves    (list[np.ndarray])  # one per take, length 100
            peak_arm_time_pcts (list[float])    # one per take
            curve_throw_types (list[str])       # one per take
            curve_meta (list[dict])             # one per take
        """
        if date is not None:
            cur.execute(f"""
                SELECT t.take_id, t.pitch_velo, COALESCE(t.throw_type, 'Mound') AS throw_type
                FROM takes t
                JOIN athletes a ON t.athlete_id = a.athlete_id
                WHERE a.athlete_name = %s
                  AND t.take_date = %s
                  AND COALESCE(t.throw_type, 'Mound') IN ({placeholders_tt})
                  AND t.pitch_velo BETWEEN %s AND %s
                ORDER BY t.file_name
            """, (selected_pitcher_comp, date, *selected_throw_types_comp, velo_min, velo_max))
        else:
            cur.execute(f"""
                SELECT t.take_id, t.pitch_velo, COALESCE(t.throw_type, 'Mound') AS throw_type
                FROM takes t
                JOIN athletes a ON t.athlete_id = a.athlete_id
                WHERE COALESCE(t.throw_type, 'Mound') IN ({placeholders_tt})
                  AND t.pitch_velo BETWEEN %s AND %s
                ORDER BY t.file_name
            """, (*selected_throw_types_comp, velo_min, velo_max))

        takes = cur.fetchall()
        if not takes:
            return None, None, None, None, None

        shoulder_curves, torso_curves = [], []
        peak_arm_time_pcts = []
        curve_throw_types = []
        curve_meta = []

        for tid, _velo, take_throw_type in takes:
            if int(tid) in exclude_take_ids:
                continue
            # Torso power (for torso curve and Mound drive-start anchor)
            cur.execute("""
                SELECT frame, x_data FROM time_series_data ts
                JOIN segments s ON ts.segment_id = s.segment_id
                JOIN categories c ON ts.category_id = c.category_id
                WHERE ts.take_id = %s AND c.category_name = 'SEGMENT_POWERS' AND s.segment_name = %s
                ORDER BY frame
            """, (tid, torso_segment_comp))
            df_power = pd.DataFrame(cur.fetchall(), columns=["frame", "x_data"])
            if df_power.empty:
                continue
            df_power["x_data"] = pd.to_numeric(df_power["x_data"], errors="coerce").fillna(0)
            # Start anchor for normalization:
            # - Mound: max rear knee flexion before drive start
            # - Pulldown: Peak Knee Height frame (same Tab 3 helper/logic)
            if take_throw_type == "Pulldown":
                pd_fp = get_foot_plant_frame(tid, handedness_comp, cur)
                pd_br = get_ball_release_frame_pulldown(tid, handedness_comp, pd_fp, cur)
                max_knee_frame = get_pulldown_peak_knee_height_frame(
                    tid,
                    handedness_comp,
                    pd_br,
                    cur
                )
                if max_knee_frame is None:
                    continue
                max_knee_frame = int(max_knee_frame)
            else:
                drive_start_frame = df_power[df_power["x_data"] < -3000]["frame"].min()
                if pd.isna(drive_start_frame):
                    continue
                cur.execute("""
                    SELECT frame, x_data FROM time_series_data ts
                    JOIN segments s ON ts.segment_id = s.segment_id
                    JOIN categories c ON ts.category_id = c.category_id
                    WHERE ts.take_id = %s AND c.category_name = 'JOINT_ANGLES' AND s.segment_name = %s
                    ORDER BY frame
                """, (tid, rear_knee_comp))
                df_knee = pd.DataFrame(cur.fetchall(), columns=["frame", "x_data"])
                if df_knee.empty:
                    continue
                df_knee_pre = df_knee[df_knee["frame"] < drive_start_frame]
                if df_knee_pre.empty:
                    continue
                max_knee_frame = int(df_knee_pre.loc[df_knee_pre["x_data"].idxmin(), "frame"])

            # Shoulder joint angles (x, y, z) to find layback (peak |z|)
            cur.execute("""
                SELECT frame, x_data, y_data, z_data FROM time_series_data ts
                JOIN segments s ON ts.segment_id = s.segment_id
                JOIN categories c ON ts.category_id = c.category_id
                WHERE ts.take_id = %s AND c.category_name = 'JOINT_ANGLES' AND s.segment_name = %s
                ORDER BY frame
            """, (tid, shoulder_segment_comp))
            df_sh = pd.DataFrame(cur.fetchall(), columns=["frame", "x_data", "y_data", "z_data"])
            if df_sh.empty:
                continue
            df_sh["z_data"] = pd.to_numeric(df_sh["z_data"], errors="coerce")

            # MER frame (Mound only)
            peak_shoulder_frame = int(df_sh.loc[df_sh["z_data"].abs().idxmax(), "frame"])
            cur.execute("""
                SELECT frame, x_data FROM time_series_data ts
                JOIN segments s ON ts.segment_id = s.segment_id
                JOIN categories c ON ts.category_id = c.category_id
                WHERE ts.take_id = %s AND c.category_name = 'SEGMENT_ENERGIES' AND s.segment_name = %s
                ORDER BY frame
            """, (tid, arm_segment))
            df_arm = pd.DataFrame(cur.fetchall(), columns=["frame", "x_data"])
            if df_arm.empty:
                continue
            df_arm["x_data"] = pd.to_numeric(df_arm["x_data"], errors="coerce")
            end_frame = int(df_arm.loc[df_arm["x_data"].idxmax(), "frame"]) + 50

            # Peak arm energy frame (MER-windowed max, constrained by Ball Release)
            mer_frame = int(peak_shoulder_frame)
            MER_WINDOW = 20

            br_frame = get_ball_release_frame(tid, handedness_comp, cur)

            # Find row index in df_arm closest to MER
            mer_row_idx = df_arm["frame"].sub(mer_frame).abs().idxmin()
            start_idx = max(0, mer_row_idx - MER_WINDOW)
            end_idx = min(len(df_arm) - 1, mer_row_idx + MER_WINDOW)

            # Windowed energy (MER ± window), additionally constrained to <= Ball Release
            df_arm_window = df_arm.iloc[start_idx:end_idx + 1].copy()
            if br_frame is not None:
                df_arm_window = df_arm_window[df_arm_window["frame"] <= br_frame]

            if df_arm_window.empty:
                continue

            windowed_energy = df_arm_window["x_data"]

            if windowed_energy.isna().all():
                peak_idx = int(df_arm_window["frame"].sub(mer_frame).abs().idxmin())
            else:
                peak_idx = int(windowed_energy.idxmax())

            peak_arm_frame = int(df_arm.loc[peak_idx, "frame"])

            # Build normalized time_pct for shoulder and torso segments:
            # 0..50: max_knee_frame -> peak_shoulder_frame
            # 50..100: peak_shoulder_frame -> end_frame
            def to_time_pct(frames):
                frames = np.asarray(frames, dtype=float)
                out = np.empty_like(frames, dtype=float)
                # Avoid division-by-zero
                pre_denom = max(1, (peak_shoulder_frame - max_knee_frame))
                post_denom = max(1, (end_frame - peak_shoulder_frame))
                mask_pre = frames <= peak_shoulder_frame
                out[mask_pre] = (frames[mask_pre] - max_knee_frame) / pre_denom * 50.0
                out[~mask_pre] = 50.0 + (frames[~mask_pre] - peak_shoulder_frame) / post_denom * 50.0
                return out

            # Shoulder curve
            df_sh_seg = df_sh[(df_sh["frame"] >= max_knee_frame) & (df_sh["frame"] <= end_frame)].copy()
            if df_sh_seg.empty:
                continue
            df_sh_seg["time_pct"] = to_time_pct(df_sh_seg["frame"])
            src = pd.to_numeric(df_sh_seg[comp_col], errors="coerce")
            if use_abs:
                src = src.abs()
            try:
                shoulder_curve = np.interp(interp_points, df_sh_seg["time_pct"].values, src.values)
            except Exception:
                continue

            # Torso curve
            df_pw_seg = df_power[(df_power["frame"] >= max_knee_frame) & (df_power["frame"] <= end_frame)].copy()
            if df_pw_seg.empty:
                continue
            df_pw_seg["time_pct"] = to_time_pct(df_pw_seg["frame"])
            try:
                torso_curve = np.interp(interp_points, df_pw_seg["time_pct"].values, df_pw_seg["x_data"].values)
            except Exception:
                continue

            # Peak arm energy normalized time pct
            pre_denom = max(1, (peak_shoulder_frame - max_knee_frame))
            post_denom = max(1, (end_frame - peak_shoulder_frame))
            if peak_arm_frame <= peak_shoulder_frame:
                peak_arm_time_pct = (peak_arm_frame - max_knee_frame) / pre_denom * 50.0
            else:
                peak_arm_time_pct = 50.0 + (peak_arm_frame - peak_shoulder_frame) / post_denom * 50.0

            shoulder_curves.append(shoulder_curve)
            torso_curves.append(torso_curve)
            peak_arm_time_pcts.append(peak_arm_time_pct)
            curve_throw_types.append(take_throw_type)
            _meta = take_meta_by_id.get(int(tid), {
                "take_id": int(tid),
                "date": date if date is not None else "All Dates",
                "player": selected_pitcher_comp,
                "throw_type": take_throw_type,
                "take_number": "?"
            })
            curve_meta.append(_meta)

        if not shoulder_curves or not torso_curves:
            return None, None, None, None, None

        return shoulder_curves, torso_curves, peak_arm_time_pcts, curve_throw_types, curve_meta

    # ---- Compute curves and render charts ----
    with right:
        # --- Load curves for each session (selected throw type) ---
        s1_sh_curves, s1_to_curves, s1_peak_arm_times, s1_throw_types, s1_meta = load_and_interpolate_curves(
            session1_date, velo_range1[0], velo_range1[1], comp_col, use_abs
        )
        s2_sh_curves, s2_to_curves, s2_peak_arm_times, s2_throw_types, s2_meta = load_and_interpolate_curves(
            session2_date, velo_range2[0], velo_range2[1], comp_col, use_abs
        )

        curves_s1 = dict(shoulder=s1_sh_curves, torso=s1_to_curves, throw_types=s1_throw_types, meta=s1_meta)
        curves_s2 = dict(shoulder=s2_sh_curves, torso=s2_to_curves, throw_types=s2_throw_types, meta=s2_meta)

        mean_ref_shoulder, mean_ref_torso = None, None
        if include_ref and display_mode_tab2 == "Grouped Average":
            mean_ref_shoulder, mean_ref_torso, _ = load_reference_curves_player_mean(
                ref_mode, selected_pitcher_comp,
                velo_range_ref[0], velo_range_ref[1],
                comp_col, use_abs,
                throw_types=reference_throw_types_comp
            )

        # Only plot if at least one session has at least one valid curve
        has_any_curve = (
            curves_s1["shoulder"] not in (None, [])
            and curves_s2["shoulder"] not in (None, [])
        )
        if has_any_curve:
            def _group_curves_by_throw_type(curves, throw_types, meta):
                grouped = {}
                for curve, tt, m in zip(curves or [], throw_types or [], meta or []):
                    if tt not in grouped:
                        grouped[tt] = {"curves": [], "meta": []}
                    grouped[tt]["curves"].append(curve)
                    grouped[tt]["meta"].append(m)
                return grouped

            pulldown_color = "#ff7f0e"

            # ===================== SHOULDER =====================
            fig_shoulder = go.Figure()
            if display_mode_tab2 == "Individual":
                seen_legend_s1 = set()
                seen_legend_s2 = set()
                for curve, tt, m in zip(curves_s1["shoulder"] or [], curves_s1["throw_types"] or [], curves_s1["meta"] or []):
                    color_s1 = get_session_color(session1_date) if tt == "Mound" else pulldown_color
                    key_s1 = f"{session1_date}_{tt}"
                    hover_label = f"{m['player']} | {tt} | Take {m['take_number']}"
                    fig_shoulder.add_trace(go.Scatter(
                        x=interp_points, y=curve,
                        mode="lines",
                        line=dict(color=color_s1, width=1.5),
                        opacity=0.45,
                        name=f"{session1_date} | {tt}",
                        legendgroup=f"s1_{tt.lower()}",
                        showlegend=(key_s1 not in seen_legend_s1),
                        hovertemplate=f"{hover_label}<br>Time: %{{x:.1f}}%<br>Value: %{{y:.2f}}<extra></extra>"
                    ))
                    seen_legend_s1.add(key_s1)
                for curve, tt, m in zip(curves_s2["shoulder"] or [], curves_s2["throw_types"] or [], curves_s2["meta"] or []):
                    color_s2 = get_session_color(session2_date) if tt == "Mound" else pulldown_color
                    key_s2 = f"{session2_date}_{tt}"
                    hover_label = f"{m['player']} | {tt} | Take {m['take_number']}"
                    fig_shoulder.add_trace(go.Scatter(
                        x=interp_points, y=curve,
                        mode="lines",
                        line=dict(color=color_s2, width=1.5, dash="dash" if tt == "Pulldown" else "solid"),
                        opacity=0.45,
                        name=f"{session2_date} | {tt}",
                        legendgroup=f"s2_{tt.lower()}",
                        showlegend=(key_s2 not in seen_legend_s2),
                        hovertemplate=f"{hover_label}<br>Time: %{{x:.1f}}%<br>Value: %{{y:.2f}}<extra></extra>"
                    ))
                    seen_legend_s2.add(key_s2)
            else:
                grouped_s1_sh = _group_curves_by_throw_type(curves_s1["shoulder"], curves_s1["throw_types"], curves_s1["meta"])
                for tt, payload in grouped_s1_sh.items():
                    color_s1 = get_session_color(session1_date) if tt == "Mound" else pulldown_color
                    mean_curve = np.nanmean(np.vstack(payload["curves"]), axis=0)
                    fig_shoulder.add_trace(go.Scatter(
                        x=interp_points, y=mean_curve,
                        mode="lines",
                        line=dict(color=color_s1, width=3, dash="solid"),
                        name=f"{session1_date} | {tt}",
                        legendgroup=f"s1_{tt.lower()}",
                        hovertemplate=f"{selected_pitcher_comp} | {tt} | Take Grouped<br>Time: %{{x:.1f}}%<br>Value: %{{y:.2f}}<extra></extra>"
                    ))
                grouped_s2_sh = _group_curves_by_throw_type(curves_s2["shoulder"], curves_s2["throw_types"], curves_s2["meta"])
                for tt, payload in grouped_s2_sh.items():
                    color_s2 = get_session_color(session2_date) if tt == "Mound" else pulldown_color
                    mean_curve = np.nanmean(np.vstack(payload["curves"]), axis=0)
                    fig_shoulder.add_trace(go.Scatter(
                        x=interp_points, y=mean_curve,
                        mode="lines",
                        line=dict(color=color_s2, width=3, dash="dash" if tt == "Pulldown" else "solid"),
                        name=f"{session2_date} | {tt}",
                        legendgroup=f"s2_{tt.lower()}",
                        hovertemplate=f"{selected_pitcher_comp} | {tt} | Take Grouped<br>Time: %{{x:.1f}}%<br>Value: %{{y:.2f}}<extra></extra>"
                    ))
            # Reference group
            if mean_ref_shoulder is not None:
                fig_shoulder.add_trace(go.Scatter(
                    x=interp_points, y=mean_ref_shoulder,
                    mode='lines', name="Reference (mean)",
                    line=dict(color='red', width=3),
                    hoverinfo="skip"
                ))

            # y-bounds for vertical markers
            yvals_sh = []
            if curves_s1["shoulder"]:
                for c in curves_s1["shoulder"]:
                    yvals_sh.extend(list(c))
            if curves_s2["shoulder"]:
                for c in curves_s2["shoulder"]:
                    yvals_sh.extend(list(c))
            if mean_ref_shoulder is not None:
                yvals_sh += list(mean_ref_shoulder)
            y0_sh, y1_sh = float(np.nanmin(yvals_sh)), float(np.nanmax(yvals_sh))

            # Legend entries for vertical lines
            fig_shoulder.add_trace(go.Scatter(
                x=[None], y=[None], mode='lines',
                line=dict(dash='dot', color='gray', width=2), name="Max Layback"
            ))
            fig_shoulder.add_trace(go.Scatter(
                x=[None], y=[None], mode='lines',
                line=dict(dash='dot', color='green', width=2), name="Peak Arm Energy"
            ))

            # Draw Max Layback (x=50) and Peak Arm Energy for session 1 (use mean across all session1 curves)
            shapes_list = [dict(type="line", x0=50, x1=50, y0=y0_sh, y1=y1_sh, line=dict(dash="dot", color="gray", width=2))]
            # For peak arm energy, use mean across all session1 curves
            peak_arm_time_pct = None
            if s1_peak_arm_times:
                vals = [v for v in s1_peak_arm_times if v is not None]
                if vals:
                    peak_arm_time_pct = float(np.nanmean(vals))
            if peak_arm_time_pct is not None:
                shapes_list.append(dict(type="line", x0=peak_arm_time_pct, x1=peak_arm_time_pct,
                                        y0=y0_sh, y1=y1_sh, line=dict(dash="dot", color="green", width=2)))
            fig_shoulder.update_layout(
                title=f"Shoulder {component}",
                xaxis_title="Normalized Time (%)",
                yaxis_title=f"Shoulder {component}",
                height=500,
                legend=dict(orientation="h", y=1, x=0.5, xanchor="center", yanchor="bottom"),
                shapes=shapes_list
            )

            # ===================== TORSO =====================
            fig_torso = go.Figure()
            if display_mode_tab2 == "Individual":
                seen_legend_s1_to = set()
                seen_legend_s2_to = set()
                for curve, tt, m in zip(curves_s1["torso"] or [], curves_s1["throw_types"] or [], curves_s1["meta"] or []):
                    color_s1 = get_session_color(session1_date) if tt == "Mound" else pulldown_color
                    key_s1_to = f"{session1_date}_{tt}"
                    hover_label = f"{m['player']} | {tt} | Take {m['take_number']}"
                    fig_torso.add_trace(go.Scatter(
                        x=interp_points, y=curve,
                        mode="lines",
                        line=dict(color=color_s1, width=1.5),
                        opacity=0.45,
                        name=f"{session1_date} | {tt}",
                        legendgroup=f"s1_{tt.lower()}",
                        showlegend=(key_s1_to not in seen_legend_s1_to),
                        hovertemplate=f"{hover_label}<br>Time: %{{x:.1f}}%<br>Value: %{{y:.2f}}<extra></extra>"
                    ))
                    seen_legend_s1_to.add(key_s1_to)
                for curve, tt, m in zip(curves_s2["torso"] or [], curves_s2["throw_types"] or [], curves_s2["meta"] or []):
                    color_s2 = get_session_color(session2_date) if tt == "Mound" else pulldown_color
                    key_s2_to = f"{session2_date}_{tt}"
                    hover_label = f"{m['player']} | {tt} | Take {m['take_number']}"
                    fig_torso.add_trace(go.Scatter(
                        x=interp_points, y=curve,
                        mode="lines",
                        line=dict(color=color_s2, width=1.5, dash="dash" if tt == "Pulldown" else "solid"),
                        opacity=0.45,
                        name=f"{session2_date} | {tt}",
                        legendgroup=f"s2_{tt.lower()}",
                        showlegend=(key_s2_to not in seen_legend_s2_to),
                        hovertemplate=f"{hover_label}<br>Time: %{{x:.1f}}%<br>Value: %{{y:.2f}}<extra></extra>"
                    ))
                    seen_legend_s2_to.add(key_s2_to)
            else:
                grouped_s1_to = _group_curves_by_throw_type(curves_s1["torso"], curves_s1["throw_types"], curves_s1["meta"])
                for tt, payload in grouped_s1_to.items():
                    color_s1 = get_session_color(session1_date) if tt == "Mound" else pulldown_color
                    mean_curve = np.nanmean(np.vstack(payload["curves"]), axis=0)
                    fig_torso.add_trace(go.Scatter(
                        x=interp_points, y=mean_curve,
                        mode="lines",
                        line=dict(color=color_s1, width=3, dash="solid"),
                        name=f"{session1_date} | {tt}",
                        legendgroup=f"s1_{tt.lower()}",
                        hovertemplate=f"{selected_pitcher_comp} | {tt} | Take Grouped<br>Time: %{{x:.1f}}%<br>Value: %{{y:.2f}}<extra></extra>"
                    ))
                grouped_s2_to = _group_curves_by_throw_type(curves_s2["torso"], curves_s2["throw_types"], curves_s2["meta"])
                for tt, payload in grouped_s2_to.items():
                    color_s2 = get_session_color(session2_date) if tt == "Mound" else pulldown_color
                    mean_curve = np.nanmean(np.vstack(payload["curves"]), axis=0)
                    fig_torso.add_trace(go.Scatter(
                        x=interp_points, y=mean_curve,
                        mode="lines",
                        line=dict(color=color_s2, width=3, dash="dash" if tt == "Pulldown" else "solid"),
                        name=f"{session2_date} | {tt}",
                        legendgroup=f"s2_{tt.lower()}",
                        hovertemplate=f"{selected_pitcher_comp} | {tt} | Take Grouped<br>Time: %{{x:.1f}}%<br>Value: %{{y:.2f}}<extra></extra>"
                    ))
            # Reference group
            if mean_ref_torso is not None:
                fig_torso.add_trace(go.Scatter(
                    x=interp_points, y=mean_ref_torso,
                    mode='lines', name="Reference (mean)",
                    line=dict(color='red', width=3),
                    hoverinfo="skip"
                ))

            yvals_to = []
            if curves_s1["torso"]:
                for c in curves_s1["torso"]:
                    yvals_to.extend(list(c))
            if curves_s2["torso"]:
                for c in curves_s2["torso"]:
                    yvals_to.extend(list(c))
            if mean_ref_torso is not None:
                yvals_to += list(mean_ref_torso)
            y0_to, y1_to = float(np.nanmin(yvals_to)), float(np.nanmax(yvals_to))

            fig_torso.add_trace(go.Scatter(
                x=[None], y=[None], mode='lines',
                line=dict(dash='dot', color='gray', width=2), name="Max Layback"
            ))
            fig_torso.add_trace(go.Scatter(
                x=[None], y=[None], mode='lines',
                line=dict(dash='dot', color='green', width=2), name="Peak Arm Energy"
            ))

            shapes_list_torso = [dict(type="line", x0=50, x1=50, y0=y0_to, y1=y1_to, line=dict(dash="dot", color="gray", width=2))]
            # For peak arm energy, use mean across all session1 curves
            peak_arm_time_pct_to = None
            if s1_peak_arm_times:
                vals = [v for v in s1_peak_arm_times if v is not None]
                if vals:
                    peak_arm_time_pct_to = float(np.nanmean(vals))
            if peak_arm_time_pct_to is not None:
                shapes_list_torso.append(dict(type="line", x0=peak_arm_time_pct_to, x1=peak_arm_time_pct_to,
                                              y0=y0_to, y1=y1_to, line=dict(dash="dot", color="green", width=2)))
            fig_torso.update_layout(
                title="Torso Power",
                xaxis_title="Normalized Time (%)",
                yaxis_title="Torso Power",
                height=500,
                legend=dict(orientation="h", y=1, x=0.5, xanchor="center", yanchor="bottom"),
                shapes=shapes_list_torso
            )

            st.plotly_chart(fig_shoulder, use_container_width=True)
            st.plotly_chart(fig_torso, use_container_width=True)
        else:
            st.warning("Could not load enough data for one or both sessions / velocity ranges.")

    # if knees1.size and knees2.size:
    #     mean_knee1 = knees1.mean(axis=0)
    #     mean_knee2 = knees2.mean(axis=0)

    #     fig_knee = go.Figure()
    #     fig_knee.add_trace(go.Scatter(x=interp_points, y=mean_knee1,
    #         mode='lines', name=f"Rear Knee {session1_date}", line=dict(color='blue')))
    #     fig_knee.add_trace(go.Scatter(x=interp_points, y=mean_knee2,
    #         mode='lines', name=f"Rear Knee {session2_date}", line=dict(color='orange')))
    #     fig_knee.update_layout(
    #         title="Mean Rear Knee Flexion (X)",
    #         xaxis_title="Normalized Time (%)", yaxis_title="Rear Knee Flexion Angle",
    #         height=400, legend=dict(orientation="h", y=1, x=0.5, xanchor="center", yanchor="bottom")
    #     )
    #     st.plotly_chart(fig_knee, use_container_width=True)
    # else:
    #     st.warning("Could not load enough rear knee data for one or both sessions / velocity ranges.")


with tab3:
    # --- Pitcher selection ---
    cur.execute("""
        SELECT DISTINCT a.athlete_name
        FROM athletes a
        JOIN takes t ON a.athlete_id = t.athlete_id
        ORDER BY a.athlete_name
    """)
    pitchers_010 = [row[0] for row in cur.fetchall()]
    # Allow selecting multiple pitchers
    selected_pitchers_010 = st.multiselect("Select Pitcher(s)", pitchers_010, key="010_pitcher")

    if not selected_pitchers_010:
        st.warning("Select one or more pitchers to view data.")

    # --- Throw type selection (default = Mound) ---
    cur.execute("""
        SELECT DISTINCT COALESCE(t.throw_type, 'Mound') AS throw_type
        FROM takes t
        ORDER BY throw_type
    """)
    throw_type_options_010 = [row[0] for row in cur.fetchall()] or ["Mound", "Pulldown"]

    default_throw_types_010 = ["Mound"] if "Mound" in throw_type_options_010 else [throw_type_options_010[0]]

    selected_throw_types_010 = st.multiselect(
        "Throw Type(s)",
        options=throw_type_options_010,
        default=default_throw_types_010,
        key="throw_types_010"
    )

    if not selected_throw_types_010:
        selected_throw_types_010 = default_throw_types_010

    # --- Per‑pitcher date selection ---
    pitcher_dates_010 = {}

    for pitcher in selected_pitchers_010:
        cur.execute("""
            SELECT DISTINCT t.take_date
            FROM takes t
            JOIN athletes a ON a.athlete_id = t.athlete_id
            WHERE a.athlete_name = %s
            ORDER BY t.take_date
        """, (pitcher,))
        dates = [row[0].strftime("%Y-%m-%d") for row in cur.fetchall()]
        dates.insert(0, "All Dates")

        pitcher_dates_010[pitcher] = st.multiselect(
            f"{pitcher} — Session Dates",
            options=dates,
            default=["All Dates"],
            key=f"010_dates_{pitcher}"
        )

    # --- Metric selection ---
    metric_group_options_010 = [
        "All Metrics",
        "Throwing Arm ROM",
        "Throwing Arm Angular Velocities",
        "Pelvis and Torso Angular Velocities",
        "COG Velocities",
        "Torso and Pelvis ROM",
    ]
    selected_metric_group_010 = st.selectbox(
        "Metric Group",
        metric_group_options_010,
        key="010_metric_group",
    )

    metric_options_010 = [
        "Max Shoulder Internal Rotation Velocity",
        "Max Shoulder External Rotation Velocity",
        "Max Elbow Extension Velocity",
        "Peak Rotational Torque into Layback",
        "Max Shoulder Horizontal Abduction Velocity into Max Scap Retraction",
        "Max Shoulder Horizontal Abduction/Adduction Velocity",
        "Max Shoulder Horizontal Abduction",
        "Max Shoulder External Rotation",
        "Shoulder External Rotation at Max Shoulder Horizontal Abduction",
        "Max Torso Angular Velocity",
        "Max Torso–Pelvis Angular Velocity",
        "Max Pelvis Angular Velocity",
        "Max Pelvis Angle (Z)",
        "Max COM Velocity",
        "Max Hip Extension Velocity",
        "Max Knee Extension Velocity",
        "Max Lead Knee Extension Velocity",
        "Max Ankle Extension Velocity",
        "Max Torso–Pelvis Angle (Z)",
        "Max Torso–Pelvis Angle (X-Extended)",
        "Max Torso–Pelvis Angle (X-Flexed)",
        "Max Torso–Pelvis Angle (Y-Glove Side)",
        "Max Torso–Pelvis Angle (Y-Arm Side)",
        "Pelvis Posterior Tilt at Peak Knee Height",
        "Pelvis Arm Side Tilt at Peak Knee Height",
        "Pelvis Counter Rotation at Peak Knee Height",
        "Pelvis Anterior Tilt at Ball Release",
        "Max Hand Speed",
    ]

    group_to_metrics_010 = {
        "All Metrics": metric_options_010,

        "Throwing Arm ROM": [
            "Max Shoulder Horizontal Abduction",
            "Max Shoulder External Rotation",
            "Shoulder External Rotation at Max Shoulder Horizontal Abduction",
        ],

        "Throwing Arm Angular Velocities": [
            "Max Shoulder Internal Rotation Velocity",
            "Max Shoulder External Rotation Velocity",
            "Max Elbow Extension Velocity",
            "Peak Rotational Torque into Layback",
            "Max Shoulder Horizontal Abduction Velocity into Max Scap Retraction",
            "Max Shoulder Horizontal Abduction/Adduction Velocity",
            "Max Hand Speed",
        ],

        "Pelvis and Torso Angular Velocities": [
            "Max Torso Angular Velocity",
            "Max Torso–Pelvis Angular Velocity",
            "Max Pelvis Angular Velocity",
        ],

        "COG Velocities": [
            "Max COM Velocity",
            "Max Hip Extension Velocity",
            "Max Knee Extension Velocity",
            "Max Ankle Extension Velocity"
        ],
        "Torso and Pelvis ROM": [
            "Max Torso–Pelvis Angle (Z)",
            "Max Torso–Pelvis Angle (X-Extended)",
            "Max Torso–Pelvis Angle (X-Flexed)",
            "Max Torso–Pelvis Angle (Y-Glove Side)",
            "Max Torso–Pelvis Angle (Y-Arm Side)",
            "Pelvis Posterior Tilt at Peak Knee Height",
            "Pelvis Arm Side Tilt at Peak Knee Height",
            "Pelvis Counter Rotation at Peak Knee Height",
            "Pelvis Anterior Tilt at Ball Release",
        ],
    }

    current_metric_options_010 = group_to_metrics_010.get(
        selected_metric_group_010,
        metric_options_010,
    )

    selected_metric_010 = st.selectbox(
        "Select Metric",
        current_metric_options_010,
        key="010_metric",
    )

    torso_axis = None
    if selected_metric_010 == "Max Torso Angular Velocity":
        torso_axis = st.selectbox("Select Torso Axis", ["X (Extension)", "X (Flexion)", "Y", "Z"], key="torso_axis")

    torso_pelvis_axis = None
    if selected_metric_010 == "Max Torso–Pelvis Angular Velocity":
        torso_pelvis_axis = st.selectbox("Select Torso–Pelvis Axis", ["X (Extension)", "X (Flexion)", "Y", "Z"], key="torso_pelvis_axis")

    pelvis_axis = None
    if selected_metric_010 == "Max Pelvis Angular Velocity":
        pelvis_axis = st.selectbox("Select Pelvis Axis", ["X", "Z"], key="pelvis_axis")

    com_axis = None
    if selected_metric_010 == "Max COM Velocity":
        com_axis = st.selectbox("Select COM Axis", ["X", "Y", "Z"], key="com_axis")

    # --- Query takes for selected pitchers and their selected dates ---
    take_rows_010 = []

    for pitcher, selected_dates in pitcher_dates_010.items():

        # Handle All Dates
        if "All Dates" in selected_dates or not selected_dates:
            placeholders_tt = ",".join(["%s"] * len(selected_throw_types_010))
            cur.execute(f"""
                SELECT t.take_id, t.pitch_velo, a.handedness, COALESCE(t.throw_type, 'Mound') AS throw_type
                FROM takes t
                JOIN athletes a ON t.athlete_id = a.athlete_id
                WHERE a.athlete_name = %s
                  AND COALESCE(t.throw_type, 'Mound') IN ({placeholders_tt})
                ORDER BY t.file_name
            """, (pitcher, *selected_throw_types_010))
            take_rows_010.extend(cur.fetchall())

        else:
            placeholders_dates = ",".join(["%s"] * len(selected_dates))
            placeholders_tt = ",".join(["%s"] * len(selected_throw_types_010))
            cur.execute(f"""
                SELECT t.take_id, t.pitch_velo, a.handedness, COALESCE(t.throw_type, 'Mound') AS throw_type
                FROM takes t
                JOIN athletes a ON t.athlete_id = a.athlete_id
                WHERE a.athlete_name = %s
                  AND t.take_date IN ({placeholders_dates})
                  AND COALESCE(t.throw_type, 'Mound') IN ({placeholders_tt})
                ORDER BY t.file_name
            """, (pitcher, *selected_dates, *selected_throw_types_010))
            take_rows_010.extend(cur.fetchall())

    rows_010 = []
    for take_id_010, pitch_velo_010, handedness_local, throw_type_local in take_rows_010:
        scap_zero_frame_010 = np.nan
        max_scap_retraction_frame_010 = np.nan
        torque_window_start_frame_010 = np.nan
        torque_window_end_frame_010 = np.nan
        # Determine handedness-specific segment names for this take
        if handedness_local == "R":
            shoulder_velo_segment = "RT_SHOULDER_ANGULAR_VELOCITY"
            shoulder_rta_velo_segment = "RT_SHOULDER_RTA_ANGULAR_VELOCITY"
            hip_velo_segment   = "RT_HIP_ANGULAR_VELOCITY"
            knee_velo_segment  = "RT_KNEE_ANGULAR_VELOCITY"
            ankle_velo_segment = "RT_ANKLE_ANGULAR_VELOCITY"
            lead_knee_velo_segment = "LT_KNEE_ANGULAR_VELOCITY"
            elbow_velo_segment = "RT_ELBOW_ANGULAR_VELOCITY"
            shank_seg_name = "LSK"
            hand_segment = "RHA"
        else:
            shoulder_velo_segment = "LT_SHOULDER_ANGULAR_VELOCITY"
            shoulder_rta_velo_segment = "LT_SHOULDER_RTA_ANGULAR_VELOCITY"
            hip_velo_segment   = "LT_HIP_ANGULAR_VELOCITY"
            knee_velo_segment  = "LT_KNEE_ANGULAR_VELOCITY"
            ankle_velo_segment = "LT_ANKLE_ANGULAR_VELOCITY"
            lead_knee_velo_segment = "RT_KNEE_ANGULAR_VELOCITY"
            elbow_velo_segment = "LT_ELBOW_ANGULAR_VELOCITY"
            shank_seg_name = "RSK"
            hand_segment = "LHA"

        # Get ball release frame for this take
        br_frame_010 = get_ball_release_frame(take_id_010, handedness_local, cur)

        # --- Compute Peak Knee Height event frame ---
        # Mound: glove-side shank peak height pre-BR (existing logic)
        # Pulldown: heel-gated throwing-side knee-x max event
        knee_peak_frame_pre_br_010 = None
        if throw_type_local == "Pulldown":
            pd_fp = get_foot_plant_frame(take_id_010, handedness_local, cur)
            pd_br = get_ball_release_frame_pulldown(take_id_010, handedness_local, pd_fp, cur)
            knee_peak_frame_pre_br_010 = get_pulldown_peak_knee_height_frame(
                take_id_010,
                handedness_local,
                pd_br,
                cur
            )
        elif br_frame_010 is not None:
            cur.execute("""
                SELECT ts.frame, ts.z_data
                FROM time_series_data ts
                JOIN segments s ON ts.segment_id = s.segment_id
                JOIN categories c ON ts.category_id = c.category_id
                WHERE ts.take_id = %s
                  AND c.category_name = 'KINETIC_KINEMATIC_ProxEndPos'
                  AND s.segment_name = %s
                  AND ts.frame < %s
                ORDER BY ts.z_data DESC, ts.frame ASC
                LIMIT 1
            """, (take_id_010, shank_seg_name, br_frame_010))
            row = cur.fetchone()
            if row is not None:
                knee_peak_frame_pre_br_010 = int(row[0])

        # Get max shoulder ER frame for this take
        sh_er_max_frame_010 = get_shoulder_er_max_frame(take_id_010, handedness_local, cur)
        # Updated Foot Plant Logic (prox_x_peak → ankle_min → zero_cross)
        fp_start_candidate = get_lead_ankle_prox_x_peak_frame(take_id_010, handedness_local, cur)

        ankle_min_frame_010 = None
        if fp_start_candidate is not None and sh_er_max_frame_010 is not None:
            ankle_min_frame_010 = get_ankle_min_frame(
                take_id_010, handedness_local,
                fp_start_candidate,
                sh_er_max_frame_010,
                cur
            )

        zero_cross_frame_010 = None
        if ankle_min_frame_010 is not None:
            zero_cross_frame_010 = get_zero_cross_frame(
                take_id_010, handedness_local,
                ankle_min_frame_010,
                sh_er_max_frame_010,
                cur
            )

        if zero_cross_frame_010 is not None:
            fp_frame_010 = zero_cross_frame_010
        elif ankle_min_frame_010 is not None:
            fp_frame_010 = ankle_min_frame_010
        else:
            fp_frame_010 = fp_start_candidate

        # Determine segment for metric (handedness-aware)
        if selected_metric_010 == "Max Shoulder Internal Rotation Velocity":
            velo_segment = shoulder_velo_segment
        elif selected_metric_010 == "Max Shoulder External Rotation Velocity":
            velo_segment = shoulder_velo_segment
        elif selected_metric_010 == "Max Elbow Extension Velocity":
            velo_segment = elbow_velo_segment
        elif selected_metric_010 == "Peak Rotational Torque into Layback":
            # Explicit handedness mapping:
            # RHP -> RT_SHOULDER_RAR_MMT, LHP -> LT_SHOULDER_LAR_MMT
            velo_segment = "RT_SHOULDER_RAR_MMT" if handedness_local == "R" else "LT_SHOULDER_LAR_MMT"
        elif selected_metric_010 == "Max Torso Angular Velocity":
            velo_segment = "TORSO_ANGULAR_VELOCITY"
        elif selected_metric_010 == "Max Torso–Pelvis Angular Velocity":
            velo_segment = "TORSO_PELVIS_ANGULAR_VELOCITY"
        elif selected_metric_010 == "Max Pelvis Angular Velocity":
            velo_segment = "PELVIS_ANGULAR_VELOCITY"
        elif selected_metric_010 == "Max Pelvis Angle (Z)":
            velo_segment = "PELVIS_ANGLE"
        elif selected_metric_010 == "Max COM Velocity":
            velo_segment = "CenterOfMass_VELO"
        elif selected_metric_010 == "Max Hip Extension Velocity":
            velo_segment = hip_velo_segment
        elif selected_metric_010 == "Max Knee Extension Velocity":
            velo_segment = knee_velo_segment
        elif selected_metric_010 == "Max Lead Knee Extension Velocity":
            velo_segment = lead_knee_velo_segment
        elif selected_metric_010 == "Max Ankle Extension Velocity":
            velo_segment = ankle_velo_segment
        elif selected_metric_010 == "Max Torso–Pelvis Angle (Z)":
            velo_segment = "TORSO_PELVIS_ANGLE"
        elif selected_metric_010 == "Max Torso–Pelvis Angle (X-Extended)":
            velo_segment = "TORSO_PELVIS_ANGLE"
        elif selected_metric_010 == "Max Torso–Pelvis Angle (X-Flexed)":
            velo_segment = "TORSO_PELVIS_ANGLE"
        elif selected_metric_010 == "Max Torso–Pelvis Angle (Y-Glove Side)":
            velo_segment = "TORSO_PELVIS_ANGLE"
        elif selected_metric_010 == "Max Torso–Pelvis Angle (Y-Arm Side)":
            velo_segment = "TORSO_PELVIS_ANGLE"
        elif selected_metric_010 == "Pelvis Posterior Tilt at Peak Knee Height":
            velo_segment = "PELVIS_ANGLE"
        elif selected_metric_010 == "Pelvis Arm Side Tilt at Peak Knee Height":
            velo_segment = "PELVIS_ANGLE"
        elif selected_metric_010 == "Pelvis Counter Rotation at Peak Knee Height":
            velo_segment = "PELVIS_ANGLE"
        elif selected_metric_010 == "Pelvis Anterior Tilt at Ball Release":
            velo_segment = "PELVIS_ANGLE"
        elif selected_metric_010 == "Max Shoulder Horizontal Abduction/Adduction Velocity":
            velo_segment = shoulder_rta_velo_segment
        elif selected_metric_010 == "Max Shoulder Horizontal Abduction Velocity into Max Scap Retraction":
            velo_segment = shoulder_rta_velo_segment
        elif selected_metric_010 == "Max Shoulder Horizontal Abduction":
            velo_segment = "RT_SHOULDER_ANGLE" if handedness_local == "R" else "LT_SHOULDER_ANGLE"
        elif selected_metric_010 == "Max Shoulder External Rotation":
            velo_segment = "RT_SHOULDER" if handedness_local == "R" else "LT_SHOULDER"
        elif selected_metric_010 == "Shoulder External Rotation at Max Shoulder Horizontal Abduction":
            velo_segment = "RT_SHOULDER_ANGLE" if handedness_local == "R" else "LT_SHOULDER_ANGLE"
        elif selected_metric_010 == "Max Hand Speed":
            velo_segment = hand_segment
        else:
            velo_segment = None

        # --- Query velocity data for this take ---
        if velo_segment is None:
            continue
        cur.execute("""
            SELECT ts.frame, ts.x_data, ts.y_data, ts.z_data
            FROM time_series_data ts
            JOIN segments s ON ts.segment_id = s.segment_id
            JOIN categories c ON ts.category_id = c.category_id
            WHERE ts.take_id = %s
              AND c.category_name = CASE
                    WHEN %s = 'CenterOfMass_VELO' THEN 'PROCESSED'
                    WHEN %s IN ('RT_SHOULDER', 'LT_SHOULDER') THEN 'JOINT_ANGLES'
                    WHEN %s IN ('RHA', 'LHA') THEN 'KINETIC_KINEMATIC_CGVel'
                    ELSE 'ORIGINAL'
                END
              AND s.segment_name = %s
            ORDER BY ts.frame
        """, (take_id_010, velo_segment, velo_segment, velo_segment, velo_segment))
        data = cur.fetchall()
        if not data:
            continue
        # Convert to array
        arr = np.array(data, dtype=float)
        frames = arr[:, 0].astype(int)
        arr = arr[:, 1:]   # now x = arr[:,0], y = arr[:,1], z = arr[:,2]

        # -------------------------------------------------
        # Tab 3 (0-10 Report): Max Shoulder IR Velo
        # Use the SAME pulldown-safe logic as Tab 1
        # -------------------------------------------------
        if selected_metric_010 == "Max Shoulder Internal Rotation Velocity":
            z_vals = arr[:, 2]

            # -------------------------------------------------
            # BR-anchored windowing (pulldown-safe)
            # Window: BR +/- 20 frames
            # -------------------------------------------------
            if throw_type_local == "Pulldown":
                br_anchor = get_ball_release_frame_pulldown(take_id_010, handedness_local, fp_frame_010, cur)
            else:
                br_anchor = br_frame_010

            if br_anchor is not None:
                brf = int(br_anchor)
                win_mask = (
                    (frames >= brf - 20) &
                    (frames <= brf + 20)
                )
                if np.any(win_mask):
                    frames_w = frames[win_mask]
                    z_w = z_vals[win_mask]
                else:
                    frames_w = frames
                    z_w = z_vals
            else:
                frames_w = frames
                z_w = z_vals

            # Handedness-aware IR peak
            if handedness_local == "R":
                # RHP IR velocity = most positive
                raw_val = np.nanmax(z_w)
            else:
                # LHP IR velocity = most negative
                raw_val = np.nanmin(z_w)

            vals = np.array([raw_val])
        elif selected_metric_010 == "Max Shoulder External Rotation Velocity":
            z_vals = arr[:, 2]

            # -------------------------------------------------
            # BR-anchored windowing (pulldown-safe)
            # Window: BR - 50 frames -> BR
            # -------------------------------------------------
            if throw_type_local == "Pulldown":
                br_anchor = get_ball_release_frame_pulldown(take_id_010, handedness_local, fp_frame_010, cur)
            else:
                br_anchor = br_frame_010

            if br_anchor is not None:
                brf = int(br_anchor)
                win_mask = (
                    (frames >= brf - 50) &
                    (frames <= brf)
                )
                if np.any(win_mask):
                    z_w = z_vals[win_mask]
                else:
                    z_w = z_vals
            else:
                z_w = z_vals

            # Handedness-aware ER peak
            if handedness_local == "R":
                # RHP ER velocity = most negative
                raw_val = np.nanmin(z_w)
            else:
                # LHP ER velocity = most positive
                raw_val = np.nanmax(z_w)

            vals = np.array([raw_val])
        elif selected_metric_010 == "Max Elbow Extension Velocity":
            x_vals = arr[:, 0]
            # ---------------------------------------------
            # Windowing for throwing arm metrics
            # - Pulldown: BR - 30 -> BR + 10
            # - Mound: ER-centered (existing logic)
            # ---------------------------------------------
            if throw_type_local == "Pulldown":
                br_anchor = get_ball_release_frame_pulldown(take_id_010, handedness_local, fp_frame_010, cur)
                if br_anchor is not None:
                    brf = int(br_anchor)
                    win_mask = (
                        (frames >= brf - 30) &
                        (frames <= brf + 10)
                    )
                    if np.any(win_mask):
                        x_vals = x_vals[win_mask]
            else:
                # ER-centered windowing for Mound throws
                if sh_er_max_frame_010 is not None:
                    er_frame = int(sh_er_max_frame_010)
                    win_mask = (
                        (frames >= er_frame - 50) &
                        (frames <= er_frame + 30)
                    )
                    # Only apply window if it yields data
                    if np.any(win_mask):
                        x_vals = x_vals[win_mask]
            # Elbow extension velocity: always the most negative value (both handedness)
            vals = np.array([np.nanmin(x_vals)])
        elif selected_metric_010 == "Peak Rotational Torque into Layback":
            # Rotational torque uses SHOULDER_LAR_MMT Z.
            torque_z = arr[:, 2]

            # Pulldown-safe MER anchor for identifying max scap retraction frame.
            if throw_type_local == "Pulldown":
                mer_anchor = get_shoulder_er_max_frame(
                    take_id_010,
                    handedness_local,
                    cur,
                    throw_type="Pulldown"
                )
            else:
                mer_anchor = sh_er_max_frame_010

            # Find max scap retraction frame from shoulder angle X (same system).
            scap_angle_segment = "RT_SHOULDER_ANGLE" if handedness_local == "R" else "LT_SHOULDER_ANGLE"
            cur.execute("""
                SELECT ts.frame, ts.x_data
                FROM time_series_data ts
                JOIN segments s ON ts.segment_id = s.segment_id
                JOIN categories c ON ts.category_id = c.category_id
                WHERE ts.take_id = %s
                  AND c.category_name = 'ORIGINAL'
                  AND s.segment_name = %s
                  AND ts.x_data IS NOT NULL
                ORDER BY ts.frame
            """, (take_id_010, scap_angle_segment))
            scap_rows = cur.fetchall()

            max_scap_frame = None
            if scap_rows:
                scap_arr = np.array(scap_rows, dtype=float)
                scap_frames = scap_arr[:, 0].astype(int)
                scap_x = scap_arr[:, 1]
                if mer_anchor is not None:
                    merf = int(mer_anchor)
                    scap_win = (scap_frames >= merf - 50) & (scap_frames <= merf)
                    if np.any(scap_win):
                        scap_frames_w = scap_frames[scap_win]
                        scap_x_w = scap_x[scap_win]
                    else:
                        scap_frames_w = scap_frames
                        scap_x_w = scap_x
                else:
                    scap_frames_w = scap_frames
                    scap_x_w = scap_x

                if handedness_local == "R":
                    peak_idx = int(np.nanargmin(scap_x_w))
                else:
                    peak_idx = int(np.nanargmax(scap_x_w))
                max_scap_frame = int(scap_frames_w[peak_idx])
            elif mer_anchor is not None:
                max_scap_frame = int(mer_anchor)

            if max_scap_frame is not None:
                torque_window_start_frame_010 = float(max_scap_frame)
                after_mask = frames >= max_scap_frame
                after_frames = frames[after_mask]
                after_torque = torque_z[after_mask]
                if after_torque.size > 0:
                    neg_peak_local_idx = int(np.nanargmin(after_torque))
                    neg_peak_frame = int(after_frames[neg_peak_local_idx])

                    post_peak_mask = frames >= neg_peak_frame
                    post_frames = frames[post_peak_mask]
                    post_torque = torque_z[post_peak_mask]

                    if post_torque.size > 1:
                        zc_idx = np.where((post_torque[:-1] <= 0) & (post_torque[1:] > 0))[0]
                        if zc_idx.size > 0:
                            end_frame = int(post_frames[int(zc_idx[0] + 1)])
                        else:
                            end_frame = int(post_frames[-1])
                    elif post_torque.size == 1:
                        end_frame = int(post_frames[0])
                    else:
                        end_frame = int(after_frames[-1])
                    torque_window_end_frame_010 = float(end_frame)

                    win_mask = (frames >= max_scap_frame) & (frames <= end_frame)
                    if np.any(win_mask):
                        vals = np.array([np.nanmin(torque_z[win_mask])])
                    else:
                        vals = np.array([np.nanmin(after_torque)])
                else:
                    vals = np.array([np.nan])
            else:
                vals = np.array([np.nanmin(torque_z)])
        elif selected_metric_010 == "Max Torso Angular Velocity":
            if torso_axis == "X (Extension)":
                x_vals = arr[:, 0]

                # FP-windowed Torso X (Extension) angular velocity
                # Window: FP - 50 frames -> FP + 20 frames
                # - Pulldown: use Tab 1 pulldown FP helper
                # - Mound: use Tab 3 mound FP anchor
                if throw_type_local == "Pulldown":
                    fp_anchor = get_foot_plant_frame(take_id_010, handedness_local, cur)
                else:
                    fp_anchor = fp_frame_010

                if fp_anchor is not None:
                    fpf = int(fp_anchor)
                    win_mask = (
                        (frames >= fpf - 50) &
                        (frames <= fpf + 20)
                    )
                    if np.any(win_mask):
                        x_w = x_vals[win_mask]
                    else:
                        x_w = x_vals
                else:
                    x_w = x_vals

                # Extension = most positive X
                vals = np.array([np.nanmax(x_w)])

            elif torso_axis == "X (Flexion)":
                x_vals = arr[:, 0]

                # MER-windowed Torso X (Flexion) angular velocity
                # Window: MER +/- 30 frames
                # - Pulldown: pulldown-aware MER anchor
                # - Mound: existing MER anchor
                if throw_type_local == "Pulldown":
                    mer_anchor = get_shoulder_er_max_frame(
                        take_id_010,
                        handedness_local,
                        cur,
                        throw_type="Pulldown"
                    )
                else:
                    mer_anchor = sh_er_max_frame_010

                if mer_anchor is not None:
                    merf = int(mer_anchor)
                    win_mask = (
                        (frames >= merf - 30) &
                        (frames <= merf + 30)
                    )
                    if np.any(win_mask):
                        x_w = x_vals[win_mask]
                    else:
                        x_w = x_vals
                else:
                    x_w = x_vals

                # Flexion = most negative X
                vals = np.array([np.nanmin(x_w)])

            elif torso_axis == "Y":
                y_vals = arr[:, 1]

                # FP-windowed Torso Y angular velocity
                # Window: FP +/- 30 frames
                # - Pulldown: use Tab 1 pulldown FP helper
                # - Mound: use Tab 3 mound FP anchor
                if throw_type_local == "Pulldown":
                    fp_anchor = get_foot_plant_frame(take_id_010, handedness_local, cur)
                else:
                    fp_anchor = fp_frame_010

                if fp_anchor is not None:
                    fpf = int(fp_anchor)
                    win_mask = (
                        (frames >= fpf - 30) &
                        (frames <= fpf + 30)
                    )
                    if np.any(win_mask):
                        y_w = y_vals[win_mask]
                    else:
                        y_w = y_vals
                else:
                    y_w = y_vals

                # Right-handed: most negative Y
                if handedness_local == "R":
                    vals = np.array([np.nanmin(y_w)])

                # Left-handed: most positive Y
                else:
                    vals = np.array([np.nanmax(y_w)])

            elif torso_axis == "Z":
                z_vals = arr[:, 2]

                # FP-windowed Torso Z angular velocity
                # Window: FP +/- 30 frames
                # - Pulldown: use Tab 1 pulldown FP helper
                # - Mound: use Tab 3 mound FP anchor
                if throw_type_local == "Pulldown":
                    fp_anchor = get_foot_plant_frame(take_id_010, handedness_local, cur)
                else:
                    fp_anchor = fp_frame_010

                if fp_anchor is not None:
                    fpf = int(fp_anchor)
                    win_mask = (
                        (frames >= fpf - 30) &
                        (frames <= fpf + 30)
                    )
                    if np.any(win_mask):
                        z_w = z_vals[win_mask]
                    else:
                        z_w = z_vals
                else:
                    z_w = z_vals

                if handedness_local == "R":
                    # RHP: most positive Z
                    vals = np.array([np.nanmax(z_w)])
                else:
                    # LHP: most negative Z
                    vals = np.array([np.nanmin(z_w)])

            else:
                vals = np.array([])
        elif selected_metric_010 == "Max Torso–Pelvis Angular Velocity":

            # X-axis split into Extension (positive) and Flexion (negative)
            if torso_pelvis_axis == "X (Extension)":
                x_vals = arr[:, 0]

                # FP-windowed torso-pelvis X extension
                # Window: FP +/- 30 frames
                if throw_type_local == "Pulldown":
                    fp_anchor = get_foot_plant_frame(take_id_010, handedness_local, cur)
                else:
                    fp_anchor = fp_frame_010

                if fp_anchor is not None:
                    fpf = int(fp_anchor)
                    mask = (
                        (frames >= fpf - 30) &
                        (frames <= fpf + 30)
                    )
                    if np.any(mask):
                        x_vals = x_vals[mask]

                raw_val = np.nanmax(x_vals)
                vals = np.array([raw_val])


            elif torso_pelvis_axis == "X (Flexion)":
                x_vals = arr[:, 0]

                # FP-windowed torso-pelvis X flexion
                # Window: FP - 20 frames -> FP + 50 frames
                if throw_type_local == "Pulldown":
                    fp_anchor = get_foot_plant_frame(take_id_010, handedness_local, cur)
                else:
                    fp_anchor = fp_frame_010

                if fp_anchor is not None:
                    fpf = int(fp_anchor)
                    mask = (
                        (frames >= fpf - 20) &
                        (frames <= fpf + 50)
                    )
                    if np.any(mask):
                        x_vals = x_vals[mask]

                raw_val = np.nanmin(x_vals)
                vals = np.array([abs(raw_val)])  # normalize flexion to positive

            # Y-axis: handedness-aware, FP-windowed logic
            elif torso_pelvis_axis == "Y":
                y_vals = arr[:, 1]

                # FP-windowed torso-pelvis Y
                # Window: FP +/- 30 frames
                if throw_type_local == "Pulldown":
                    fp_anchor = get_foot_plant_frame(take_id_010, handedness_local, cur)
                else:
                    fp_anchor = fp_frame_010

                if fp_anchor is not None:
                    fpf = int(fp_anchor)
                    mask = (
                        (frames >= fpf - 30) &
                        (frames <= fpf + 30)
                    )
                    y_window = y_vals[mask]
                else:
                    y_window = y_vals

                # Fallback if window empty
                if y_window.size == 0:
                    y_window = y_vals

                if handedness_local == "R":
                    # RHP: glove-side = most negative
                    raw_val = np.nanmin(y_window)
                else:
                    # LHP: glove-side = most positive
                    raw_val = np.nanmax(y_window)

                # Normalize to positive magnitude
                vals = np.array([abs(raw_val)])

            # Z-axis: FP-windowed, positive maxima
            elif torso_pelvis_axis == "Z":
                z_vals = arr[:, 2]

                # Window: FP - 50 frames -> FP + 20 frames
                if throw_type_local == "Pulldown":
                    fp_anchor = get_foot_plant_frame(take_id_010, handedness_local, cur)
                else:
                    fp_anchor = fp_frame_010

                if fp_anchor is not None:
                    fpf = int(fp_anchor)
                    mask = (
                        (frames >= fpf - 50) &
                        (frames <= fpf + 20)
                    )
                    z_window = z_vals[mask]
                else:
                    z_window = z_vals

                # Fallback if window empty
                if z_window.size == 0:
                    z_window = z_vals

                # Handedness-separated Z peak:
                # - RHP: most negative
                # - LHP: most positive
                if handedness_local == "R":
                    vals = np.array([np.nanmin(z_window)])
                else:
                    vals = np.array([np.nanmax(z_window)])

            else:
                vals = np.array([])
        elif selected_metric_010 == "Max Pelvis Angular Velocity":
            # Foot-plant windowing for both throw types:
            # - Pulldown: use Tab 1 pulldown FP helper
            # - Mound: use Tab 3 mound FP anchor
            if throw_type_local == "Pulldown":
                fp_anchor = get_foot_plant_frame(take_id_010, handedness_local, cur)
            else:
                fp_anchor = fp_frame_010

            if fp_anchor is not None:
                fpf = int(fp_anchor)
                mask = (
                    (frames >= fpf - 40) &
                    (frames <= fpf + 40)
                )
                if np.any(mask):
                    arr_w = arr[mask]
                else:
                    arr_w = arr
            else:
                arr_w = arr

            # X-axis: always most negative for both handedness
            if pelvis_axis == "X":
                vals = np.array([np.nanmin(arr_w[:, 0])])
            # Z-axis: use absolute maxima regardless of handedness
            elif pelvis_axis == "Z":
                z_vals = arr_w[:, 2]
                idx = np.nanargmax(np.abs(z_vals))
                vals = np.array([z_vals[idx]])
            else:
                vals = np.array([])
        elif selected_metric_010 == "Max Pelvis Angle (Z)":
            z_vals = arr[:, 2]

            # Windowing:
            # - Pulldown: Peak Knee Height frame -> Pulldown BR
            # - Mound: FP - 70 -> FP + 20 (existing logic)
            if throw_type_local == "Pulldown":
                pd_fp = get_foot_plant_frame(take_id_010, handedness_local, cur)
                pd_br = get_ball_release_frame_pulldown(
                    take_id_010,
                    handedness_local,
                    pd_fp,
                    cur
                )
                knee_event = knee_peak_frame_pre_br_010

                if knee_event is not None and pd_br is not None:
                    start = int(min(knee_event, pd_br))
                    end = int(max(knee_event, pd_br))
                    mask = (
                        (frames >= start) &
                        (frames <= end)
                    )
                    if np.any(mask):
                        z_vals = z_vals[mask]
            else:
                fp_anchor = fp_frame_010
                if fp_anchor is not None:
                    fpf = int(fp_anchor)
                    mask = (
                        (frames >= fpf - 70) &
                        (frames <= fpf + 20)
                    )
                    if np.any(mask):
                        z_vals = z_vals[mask]

            # Match SQL-handedness transform:
            # CASE WHEN handedness = 'R' THEN z + 90 ELSE 90 - z END
            if handedness_local == "R":
                transformed = z_vals + 90.0
            else:
                transformed = 90.0 - z_vals

            # Max pelvis angle (Z) in the selected window
            raw_val = np.nanmax(transformed)
            vals = np.array([raw_val])
        elif selected_metric_010 == "Max COM Velocity":
            # Event window for COM metrics:
            # Peak Knee Height frame -> Foot Plant frame
            if throw_type_local == "Pulldown":
                fp_anchor = get_foot_plant_frame(take_id_010, handedness_local, cur)
            else:
                fp_anchor = fp_frame_010

            knee_anchor = knee_peak_frame_pre_br_010

            if knee_anchor is not None and fp_anchor is not None:
                start = int(min(knee_anchor, fp_anchor))
                end = int(max(knee_anchor, fp_anchor))
                com_mask = (
                    (frames >= start) &
                    (frames <= end)
                )
            else:
                com_mask = None

            if com_axis == "X":
                x_vals = arr[:, 0]
                if com_mask is not None and np.any(com_mask):
                    x_vals = x_vals[com_mask]
                vals = x_vals
            elif com_axis == "Y":
                y_vals = arr[:, 1]

                # Event-windowed Y (fallback to prior ER-based behavior if window unavailable)
                if com_mask is not None and np.any(com_mask):
                    y_window = y_vals[com_mask]
                # Restrict to frames before max shoulder ER (layback), if available
                elif sh_er_max_frame_010 is not None:
                    mask = frames < sh_er_max_frame_010
                    y_window = y_vals[mask]
                else:
                    y_window = y_vals

                # Fallback: if the window is empty, use the full series
                if y_window.size == 0:
                    y_window = y_vals

                # Most negative (negative maxima) before ER max
                vals = np.array([np.nanmin(y_window)])
            elif com_axis == "Z":
                z_vals = arr[:, 2]
                if com_mask is not None and np.any(com_mask):
                    z_vals = z_vals[com_mask]
                vals = np.array([np.nanmin(z_vals)])  # COM Z = most negative max for both handedness
            else:
                vals = np.array([])
        elif selected_metric_010 == "Max Hip Extension Velocity":
            x_vals = arr[:, 0]

            # Foot-plant windowing for both throw types:
            # - Pulldown: use Tab 1 pulldown FP helper
            # - Mound: use Tab 3 mound FP anchor
            if throw_type_local == "Pulldown":
                fp_anchor = get_foot_plant_frame(take_id_010, handedness_local, cur)
            else:
                fp_anchor = fp_frame_010

            if fp_anchor is not None:
                fpf = int(fp_anchor)
                mask = (
                    (frames >= fpf - 50) &
                    (frames <= fpf + 10)
                )
                window_vals = x_vals[mask]
            else:
                window_vals = x_vals

            # Fallback if window empty
            if window_vals.size == 0:
                window_vals = x_vals

            # Most negative peak in FP window (true peak extension)
            vals = np.array([np.nanmin(window_vals)])
        elif selected_metric_010 == "Max Knee Extension Velocity":
            x_vals = arr[:, 0]  # knee extension velocity typically in X

            # Foot-plant windowing for both throw types:
            # - Pulldown: use Tab 1 pulldown FP helper
            # - Mound: use Tab 3 mound FP anchor
            if throw_type_local == "Pulldown":
                fp_anchor = get_foot_plant_frame(take_id_010, handedness_local, cur)
            else:
                fp_anchor = fp_frame_010

            if fp_anchor is not None:
                fpf = int(fp_anchor)
                mask = (
                    (frames >= fpf - 50) &
                    (frames <= fpf)
                )
                window_vals = x_vals[mask]
            else:
                window_vals = x_vals

            # Fallback if window empty
            if window_vals.size == 0:
                window_vals = x_vals

            # Most positive peak in FP window
            vals = np.array([np.nanmax(window_vals)])
        elif selected_metric_010 == "Max Lead Knee Extension Velocity":
            # Simplified window: BR ± 25 frames
            if throw_type_local == "Pulldown":
                pd_fp = get_foot_plant_frame(take_id_010, handedness_local, cur)
                br_anchor = get_ball_release_frame_pulldown(
                    take_id_010,
                    handedness_local,
                    pd_fp,
                    cur
                )
            else:
                br_anchor = br_frame_010

            if br_anchor is not None:
                start = max(0, int(br_anchor) - 25)
                end = int(br_anchor) + 25
                mask = (frames >= start) & (frames <= end)

                # x_data = knee extension velocity (using LT/RT knee angular velocity depending on handedness)
                window_vals = arr[mask, 0]

                if window_vals.size > 0:
                    # Use absolute value to capture magnitude
                    vals = np.array([np.nanmax(np.abs(window_vals))])
                else:
                    vals = np.array([np.nanmax(np.abs(arr[:, 0]))])
            else:
                vals = np.array([np.nanmax(np.abs(arr[:, 0]))])
        elif selected_metric_010 == "Max Ankle Extension Velocity":
            x_vals = arr[:, 0]  # ankle extension velocity typically in X

            # Foot-plant windowing for both throw types:
            # - Pulldown: use Tab 1 pulldown FP helper
            # - Mound: use Tab 3 mound FP anchor
            if throw_type_local == "Pulldown":
                fp_anchor = get_foot_plant_frame(take_id_010, handedness_local, cur)
            else:
                fp_anchor = fp_frame_010

            if fp_anchor is not None:
                fpf = int(fp_anchor)
                mask = (
                    (frames >= fpf - 40) &
                    (frames <= fpf + 10)
                )
                window_vals = x_vals[mask]
            else:
                window_vals = x_vals

            # Fallback if window empty
            if window_vals.size == 0:
                window_vals = x_vals

            # Most negative peak in FP window (true ankle extension peak)
            vals = np.array([np.nanmin(window_vals)])
        elif selected_metric_010 == "Pelvis Posterior Tilt at Peak Knee Height":
            # Pelvis X-angle at peak glove-side knee height pre-BR
            x_vals = arr[:, 0]          # pelvis X (from ORIGINAL / PELVIS_ANGLE)
            frame_vals = frames         # pelvis frames

            if knee_peak_frame_pre_br_010 is not None and frame_vals.size > 0:
                # nearest-frame match
                idx = np.argmin(np.abs(frame_vals - knee_peak_frame_pre_br_010))
                pel_x_at_event = x_vals[idx]
                vals = np.array([pel_x_at_event])   # signed value (Option A)
            else:
                vals = np.array([np.nan])

        elif selected_metric_010 == "Pelvis Arm Side Tilt at Peak Knee Height":
            # Pelvis Y-angle at peak glove-side knee height pre-BR
            y_vals = arr[:, 1]      # pelvis Y
            frame_vals = frames
            if knee_peak_frame_pre_br_010 is not None and frame_vals.size > 0:
                idx = np.argmin(np.abs(frame_vals - knee_peak_frame_pre_br_010))
                pel_y_at_event = y_vals[idx]
                vals = np.array([pel_y_at_event])
            else:
                vals = np.array([np.nan])

        elif selected_metric_010 == "Pelvis Counter Rotation at Peak Knee Height":
            # Pelvis Z-angle at peak glove-side knee height pre-BR
            z_vals = arr[:, 2]      # pelvis Z
            frame_vals = frames
            if knee_peak_frame_pre_br_010 is not None and frame_vals.size > 0:
                idx = np.argmin(np.abs(frame_vals - knee_peak_frame_pre_br_010))
                pel_z_at_event = z_vals[idx]
                vals = np.array([pel_z_at_event])
            else:
                vals = np.array([np.nan])
        elif selected_metric_010 == "Pelvis Anterior Tilt at Ball Release":
            # Pelvis anterior tilt = PELVIS_ANGLE X at BR frame
            # Pull directly from ORIGINAL/PELVIS_ANGLE to avoid any segment mismatch.

            if throw_type_local == "Pulldown":
                br_anchor = get_ball_release_frame_pulldown(
                    take_id_010,
                    handedness_local,
                    fp_frame_010,
                    cur
                )
            else:
                br_anchor = br_frame_010

            if br_anchor is not None:
                brf = int(br_anchor)
                cur.execute("""
                    SELECT ts.x_data
                    FROM time_series_data ts
                    JOIN segments s ON ts.segment_id = s.segment_id
                    JOIN categories c ON ts.category_id = c.category_id
                    WHERE ts.take_id = %s
                      AND c.category_name = 'ORIGINAL'
                      AND s.segment_name = 'PELVIS_ANGLE'
                      AND ts.x_data IS NOT NULL
                    ORDER BY
                      CASE WHEN ts.frame = %s THEN 0 ELSE 1 END,
                      ABS(ts.frame - %s)
                    LIMIT 1
                """, (take_id_010, brf, brf))
                row = cur.fetchone()
                if row is not None:
                    vals = np.array([float(row[0])])
                else:
                    vals = np.array([np.nan])
            else:
                vals = np.array([np.nan])
        elif selected_metric_010 == "Max Torso–Pelvis Angle (Z)":
            z_vals = arr[:, 2]

            # Foot-plant windowing for both throw types:
            # - Pulldown: use Tab 1 pulldown FP helper
            # - Mound: use Tab 3 mound FP anchor
            if throw_type_local == "Pulldown":
                fp_anchor = get_foot_plant_frame(take_id_010, handedness_local, cur)
            else:
                fp_anchor = fp_frame_010

            if fp_anchor is not None:
                fpf = int(fp_anchor)
                win_mask = (
                    (frames >= fpf - 30) &
                    (frames <= fpf + 30)
                )
                if np.any(win_mask):
                    z_vals = z_vals[win_mask]

            if handedness_local == "R":
                # Right-handed: most negative Z
                vals = np.array([np.nanmin(z_vals)])
            else:
                # Left-handed: most positive Z
                vals = np.array([np.nanmax(z_vals)])
        elif selected_metric_010 == "Max Torso–Pelvis Angle (X-Extended)":
            x_vals = arr[:, 0]

            # Foot-plant windowing for both throw types:
            # - Pulldown: use Tab 1 pulldown FP helper
            # - Mound: use Tab 3 mound FP anchor
            if throw_type_local == "Pulldown":
                fp_anchor = get_foot_plant_frame(take_id_010, handedness_local, cur)
            else:
                fp_anchor = fp_frame_010

            if fp_anchor is not None:
                fpf = int(fp_anchor)
                win_mask = (
                    (frames >= fpf - 20) &
                    (frames <= fpf + 20)
                )
                if np.any(win_mask):
                    x_w = x_vals[win_mask]
                else:
                    x_w = x_vals
            else:
                x_w = x_vals

            # Most positive peak within the FP window
            raw_val = np.nanmax(x_w)

            # Normalize for UI
            vals = np.array([raw_val])

        elif selected_metric_010 == "Max Torso–Pelvis Angle (X-Flexed)":
            x_vals = arr[:, 0]

            # Ball-release windowing for both throw types:
            # - Pulldown: use pulldown BR helper
            # - Mound: use standard BR frame
            if throw_type_local == "Pulldown":
                pd_fp = get_foot_plant_frame(take_id_010, handedness_local, cur)
                br_anchor = get_ball_release_frame_pulldown(
                    take_id_010,
                    handedness_local,
                    pd_fp,
                    cur
                )
            else:
                br_anchor = br_frame_010

            if br_anchor is not None:
                brf = int(br_anchor)
                win_mask = (
                    (frames >= brf - 10) &
                    (frames <= brf + 30)
                )
                if np.any(win_mask):
                    x_w = x_vals[win_mask]
                else:
                    x_w = x_vals
            else:
                x_w = x_vals

            # Most negative peak within the BR window
            raw_val = np.nanmin(x_w)

            # Normalize for UI (absolute)
            vals = np.array([abs(raw_val)])
        elif selected_metric_010 == "Max Torso–Pelvis Angle (Y-Glove Side)":
            y_vals = arr[:, 1]

            # Ball-release windowing for both throw types:
            # - Pulldown: use pulldown BR helper
            # - Mound: use standard BR frame
            if throw_type_local == "Pulldown":
                pd_fp = get_foot_plant_frame(take_id_010, handedness_local, cur)
                br_anchor = get_ball_release_frame_pulldown(
                    take_id_010,
                    handedness_local,
                    pd_fp,
                    cur
                )
            else:
                br_anchor = br_frame_010

            if br_anchor is not None:
                brf = int(br_anchor)
                win_mask = (
                    (frames >= brf - 10) &
                    (frames <= brf + 10)
                )
                if np.any(win_mask):
                    y_vals = y_vals[win_mask]

            if handedness_local == "R":
                # Right-handed: Glove Side = most negative Y
                vals = np.array([np.nanmin(y_vals)])
            else:
                # Left-handed: Glove Side = most positive Y
                vals = np.array([np.nanmax(y_vals)])

        elif selected_metric_010 == "Max Torso–Pelvis Angle (Y-Arm Side)":
            y_vals = arr[:, 1]

            # Foot-plant windowing for both throw types:
            # - Pulldown: use Tab 1 pulldown FP helper
            # - Mound: use Tab 3 mound FP anchor
            if throw_type_local == "Pulldown":
                fp_anchor = get_foot_plant_frame(take_id_010, handedness_local, cur)
            else:
                fp_anchor = fp_frame_010

            if fp_anchor is not None:
                fpf = int(fp_anchor)
                win_mask = (
                    (frames >= fpf - 40) &
                    (frames <= fpf + 10)
                )
                if np.any(win_mask):
                    y_vals = y_vals[win_mask]

            if handedness_local == "R":
                # Right-handed: Arm Side = most positive Y
                vals = np.array([np.nanmax(y_vals)])
            else:
                # Left-handed: Arm Side = most negative Y
                vals = np.array([np.nanmin(y_vals)])
        elif selected_metric_010 == "Max Shoulder Horizontal Abduction/Adduction Velocity":
            x_vals = arr[:, 0]
            # ---------------------------------------------
            # ER-centered windowing for throwing arm metrics
            # ---------------------------------------------
            if throw_type_local == "Pulldown":
                mer_anchor = get_shoulder_er_max_frame(
                    take_id_010,
                    handedness_local,
                    cur,
                    throw_type="Pulldown"
                )
            else:
                mer_anchor = sh_er_max_frame_010

            if mer_anchor is not None:
                er_frame = int(mer_anchor)
                win_mask = (
                    (frames >= er_frame - 50) &
                    (frames <= er_frame + 30)
                )
                # Only apply window if it yields data
                if np.any(win_mask):
                    x_vals = x_vals[win_mask]
            # Always take the most negative value (both handedness)
            vals = np.array([np.nanmin(x_vals)])
        elif selected_metric_010 == "Max Shoulder Horizontal Abduction Velocity into Max Scap Retraction":
            # Use shoulder horizontal abduction/adduction velocity (x_data) and
            # compute peak negative velocity in the lead-up window from zero-cross
            # to max scap retraction frame.
            x_vel = arr[:, 0]

            # Pulldown-safe MER anchor
            if throw_type_local == "Pulldown":
                mer_anchor = get_shoulder_er_max_frame(
                    take_id_010,
                    handedness_local,
                    cur,
                    throw_type="Pulldown"
                )
            else:
                mer_anchor = sh_er_max_frame_010

            # Query shoulder angle to find max scap retraction frame in the same window.
            scap_angle_segment = "RT_SHOULDER_ANGLE" if handedness_local == "R" else "LT_SHOULDER_ANGLE"
            cur.execute("""
                SELECT ts.frame, ts.x_data
                FROM time_series_data ts
                JOIN segments s ON ts.segment_id = s.segment_id
                JOIN categories c ON ts.category_id = c.category_id
                WHERE ts.take_id = %s
                  AND c.category_name = 'ORIGINAL'
                  AND s.segment_name = %s
                  AND ts.x_data IS NOT NULL
                ORDER BY ts.frame
            """, (take_id_010, scap_angle_segment))
            scap_rows = cur.fetchall()

            if scap_rows:
                scap_arr = np.array(scap_rows, dtype=float)
                scap_frames = scap_arr[:, 0].astype(int)
                scap_x = scap_arr[:, 1]

                if mer_anchor is not None:
                    merf = int(mer_anchor)
                    scap_win = (scap_frames >= merf - 50) & (scap_frames <= merf)
                    if np.any(scap_win):
                        scap_frames_w = scap_frames[scap_win]
                        scap_x_w = scap_x[scap_win]
                    else:
                        scap_frames_w = scap_frames
                        scap_x_w = scap_x
                else:
                    scap_frames_w = scap_frames
                    scap_x_w = scap_x

                if handedness_local == "R":
                    scap_peak_idx = int(np.nanargmin(scap_x_w))
                else:
                    scap_peak_idx = int(np.nanargmax(scap_x_w))
                max_scap_frame = int(scap_frames_w[scap_peak_idx])
                max_scap_retraction_frame_010 = float(max_scap_frame)
            else:
                max_scap_frame = int(mer_anchor) if mer_anchor is not None else None
                if max_scap_frame is not None:
                    max_scap_retraction_frame_010 = float(max_scap_frame)

            if max_scap_frame is not None:
                pre_mask = frames <= max_scap_frame
                pre_frames = frames[pre_mask]
                pre_vel = x_vel[pre_mask]
                if pre_vel.size > 0:
                    if scap_rows:
                        # Use shoulder-angle zero approach for all:
                        # before max scap retraction, use the zero crossing
                        # closest to the peak (last crossing before peak).
                        # Use full angle timeline before max_scap_frame (not MER-limited)
                        # so the true early zero-cross is not clipped.
                        scap_pre_mask = scap_frames < max_scap_frame
                        scap_pre_frames = scap_frames[scap_pre_mask]
                        scap_pre_x = scap_x[scap_pre_mask]

                        if scap_pre_x.size > 1:
                            # Crossing direction follows peak sign:
                            # positive peak -> neg->pos, negative peak -> pos->neg
                            if handedness_local == "L":
                                zc_idx = np.where((scap_pre_x[:-1] < 0) & (scap_pre_x[1:] >= 0))[0]
                            else:
                                zc_idx = np.where((scap_pre_x[:-1] > 0) & (scap_pre_x[1:] <= 0))[0]
                            if zc_idx.size > 0:
                                # Use crossing nearest to max scap frame.
                                start_frame = int(scap_pre_frames[int(zc_idx[-1] + 1)])
                            else:
                                # Fallback: no negative before peak; closest-to-zero frame.
                                start_frame = int(scap_pre_frames[int(np.argmin(np.abs(scap_pre_x)))])
                        elif scap_pre_x.size == 1:
                            start_frame = int(scap_pre_frames[0])
                        else:
                            start_frame = int(pre_frames[0])
                    else:
                        # Fallback if angle stream unavailable.
                        start_frame = int(pre_frames[0])
                    scap_zero_frame_010 = float(start_frame)

                    leadup_mask = (frames >= start_frame) & (frames <= max_scap_frame)
                    if np.any(leadup_mask):
                        x_w = x_vel[leadup_mask]
                    else:
                        x_w = pre_vel
                else:
                    x_w = x_vel
            else:
                x_w = x_vel

            # Metric value is the most positive velocity within the
            # handedness-specific zero-cross -> max scap retraction window.
            vals = np.array([np.nanmax(x_w)])
        elif selected_metric_010 == "Max Shoulder Horizontal Abduction":
            x_vals = arr[:, 0]

            # ---------------------------------------------
            # MER-windowed Horizontal Abduction
            # Window: MER - 50 frames -> MER
            # - Pulldown: pulldown-aware MER anchor
            # - Mound: existing MER anchor
            # ---------------------------------------------
            if throw_type_local == "Pulldown":
                mer_anchor = get_shoulder_er_max_frame(
                    take_id_010,
                    handedness_local,
                    cur,
                    throw_type="Pulldown"
                )
            else:
                mer_anchor = sh_er_max_frame_010

            if mer_anchor is not None:
                merf = int(mer_anchor)
                win_mask = (
                    (frames >= merf - 50) &
                    (frames <= merf)
                )
                if np.any(win_mask):
                    x_w = x_vals[win_mask]
                else:
                    x_w = x_vals
            else:
                x_w = x_vals

            if handedness_local == "R":
                # Right-handed: HA = most negative
                raw_val = np.nanmin(x_w)
            else:
                # Left-handed: HA = most positive
                raw_val = np.nanmax(x_w)

            # Normalize for UI
            vals = np.array([abs(raw_val)])

        elif selected_metric_010 == "Max Shoulder External Rotation":
            z_vals = arr[:, 2]

            # -------------------------------------------------
            # Pulldown-safe MER anchor (match Tab 1 logic)
            # -------------------------------------------------
            if throw_type_local == "Pulldown":
                mer_anchor = get_shoulder_er_max_frame(
                    take_id_010,
                    handedness_local,
                    cur,
                    throw_type="Pulldown"
                )
            else:
                mer_anchor = sh_er_max_frame_010

            # Sample z at MER frame
            if mer_anchor is not None and frames.size > 0:
                nearest_idx = int(np.argmin(np.abs(frames - int(mer_anchor))))
                raw_val = float(z_vals[nearest_idx])
            else:
                # Fallback only if MER unavailable
                if handedness_local == "R":
                    raw_val = float(np.nanmin(z_vals))
                else:
                    raw_val = float(np.nanmax(z_vals))

            vals = np.array([abs(raw_val)])

        elif selected_metric_010 == "Shoulder External Rotation at Max Shoulder Horizontal Abduction":

            # x_data = horizontal abduction angle
            # z_data = ER/IR angle
            x_vals = arr[:, 0]
            z_vals = arr[:, 2]

            # ---------------------------------------------
            # MER-windowed HA peak -> sample ER at that HA
            # Window: MER - 50 frames -> MER
            # - Pulldown: pulldown-aware MER anchor
            # - Mound: existing MER anchor
            # ---------------------------------------------
            if throw_type_local == "Pulldown":
                mer_anchor = get_shoulder_er_max_frame(
                    take_id_010,
                    handedness_local,
                    cur,
                    throw_type="Pulldown"
                )
            else:
                mer_anchor = sh_er_max_frame_010

            if mer_anchor is not None:
                merf = int(mer_anchor)
                win_mask = (
                    (frames >= merf - 50) &
                    (frames <= merf)
                )
                if np.any(win_mask):
                    x_w = x_vals[win_mask]
                    z_w = z_vals[win_mask]
                else:
                    x_w = x_vals
                    z_w = z_vals
            else:
                x_w = x_vals
                z_w = z_vals

            # Determine HA peak index based on handedness (within window)
            if handedness_local == "R":
                # Right-handed: HA = most negative X
                ha_idx = int(np.nanargmin(x_w))
            else:
                # Left-handed: HA = most positive X
                ha_idx = int(np.nanargmax(x_w))

            # Extract ER at that HA index
            raw_er = float(z_w[ha_idx])

            # Normalize for UI (absolute)
            vals = np.array([abs(raw_er)])

        elif selected_metric_010 == "Max Hand Speed":
            # Hand CG speed magnitude
            # arr columns: x, y, z CG velocity components
            x_vals = arr[:, 0]
            y_vals = arr[:, 1]
            z_vals = arr[:, 2]
            speed = np.sqrt(x_vals**2 + y_vals**2 + z_vals**2)

            # -------------------------------------------------
            # BR-anchored windowing (pulldown-safe)
            # Window: BR - 30 frames -> BR + 10 frames
            # -------------------------------------------------
            if throw_type_local == "Pulldown":
                br_anchor = get_ball_release_frame_pulldown(take_id_010, handedness_local, fp_frame_010, cur)
            else:
                br_anchor = br_frame_010

            if br_anchor is not None:
                brf = int(br_anchor)
                win_mask = (
                    (frames >= brf - 30) &
                    (frames <= brf + 10)
                )
                if np.any(win_mask):
                    speed_w = speed[win_mask]
                else:
                    speed_w = speed
            else:
                speed_w = speed

            vals = np.array([np.nanmax(speed_w)])
        # Convert vals to a scalar for storage
        raw_value = float(np.nanmax(vals)) if vals.size > 0 else np.nan

        # For most metrics we report absolute magnitude, but for
        # Max Torso–Pelvis Angular Velocity we want the signed value
        if selected_metric_010 == "Max Torso–Pelvis Angular Velocity":
            metric_value = raw_value
        else:
            metric_value = abs(raw_value)

        row_payload = {
            "take_id": take_id_010,
            "Throw Type": (throw_type_local if throw_type_local is not None else "Mound"),
            "Velocity": pitch_velo_010,
            selected_metric_010: metric_value
        }
        if selected_metric_010 == "Peak Rotational Torque into Layback":
            row_payload["Window Start Frame (Max Scap Retraction)"] = torque_window_start_frame_010
            row_payload["Window End Frame (Post-Peak Zero Cross)"] = torque_window_end_frame_010
        rows_010.append(row_payload)

    if rows_010:
        # Rebuild dataframe including take_id so we can filter correctly
        df_010 = pd.DataFrame(rows_010)

        # Attach pitcher names and dates to df_010 (for labels & plotting)
        cur.execute(f"""
            SELECT t.take_id, a.athlete_name, t.take_date, COALESCE(t.throw_type, 'Mound') AS throw_type
            FROM takes t
            JOIN athletes a ON t.athlete_id = a.athlete_id
            WHERE t.take_id IN ({",".join(["%s"] * len(df_010))})
        """, tuple(df_010["take_id"].tolist()))

        take_lookup = {
            row[0]: (row[1], row[2].strftime("%Y-%m-%d"), row[3])
            for row in cur.fetchall()
        }

        df_010["Pitcher"] = df_010["take_id"].map(lambda x: take_lookup[x][0])
        df_010["Date"]    = df_010["take_id"].map(lambda x: take_lookup[x][1])
        df_010["Throw Type"] = df_010["take_id"].map(lambda x: take_lookup[x][2])

        # Build rich, human-readable exclude labels
        def make_exclude_label(row):
            return (
                f"{row['Pitcher']} | "
                f"{row['Date']} | "
                f"{row['Throw Type']} | "
                f"{row['Velocity']:.1f} mph | "
                f"{row[selected_metric_010]:.2f}"
            )

        df_010["label"] = df_010.apply(make_exclude_label, axis=1)

        # Multiselect now using readable labels but still returns take_id
        exclude_labels = st.multiselect(
            "Exclude Takes",
            options=df_010["label"].tolist(),
            key="exclude_takes_010"
        )

        # Map labels back to take_ids for filtering
        exclude_take_ids = df_010[df_010["label"].isin(exclude_labels)]["take_id"].tolist()

        # Filter out excluded takes
        df_010 = df_010[~df_010["take_id"].isin(exclude_take_ids)]
        # Remove any rows with missing values
        df_010 = df_010.dropna(subset=["Velocity", selected_metric_010])
        if len(df_010) == 0:
            st.warning("No valid data found for this selection.")
        else:
            # -----------------------------------------
            # Per‑player scatter + regression lines
            # -----------------------------------------
            fig_010 = go.Figure()

            import plotly.express as px
            color_cycle = px.colors.qualitative.Plotly
            color_idx = 0
            for (pitcher_name, date_str, tt), sub in df_010.groupby(["Pitcher", "Date", "Throw Type"]):
                x = pd.to_numeric(sub["Velocity"], errors="coerce")
                y = pd.to_numeric(sub[selected_metric_010], errors="coerce")
                mask = (~x.isna()) & (~y.isna())
                x = x[mask]
                y = y[mask]

                if len(x) < 2:
                    color_idx += 1
                    continue

                slope, intercept, r_value, _, _ = linregress(x, y)
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit = slope * x_fit + intercept

                color = color_cycle[color_idx % len(color_cycle)]
                marker_symbol = "circle" if tt == "Mound" else "diamond"
                marker_color = color if tt == "Mound" else "#ff7f0e"

                # Hover label: match marker color, readable font
                _font_color = "white"
                try:
                    if marker_color in ("#aec7e8", "#98df8a"):
                        _font_color = "black"
                except Exception:
                    pass

                # Scatter points
                fig_010.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(color=marker_color, symbol=marker_symbol, size=10),
                    name=f"{pitcher_name} | {date_str} | {tt}",
                    hovertemplate=(
                        "Pitcher: %{customdata[0]}<br>"
                        "Date: %{customdata[1]}<br>"
                        "Throw Type: %{customdata[2]}<br>"
                        "Velocity: %{customdata[3]:.1f} mph<br>"
                        f"{selected_metric_010}: %{{customdata[4]:.2f}}<extra></extra>"
                    ),
                    customdata=np.column_stack([
                        np.array([pitcher_name] * len(x), dtype=object),
                        np.array([date_str] * len(x), dtype=object),
                        np.array([tt] * len(x), dtype=object),
                        x.values,
                        y.values
                    ]),
                    hoverlabel=dict(
                        bgcolor=marker_color,
                        font=dict(color=_font_color, size=13),
                        align="left",
                        namelength=-1
                    )
                ))

                # Regression line
                fig_010.add_trace(go.Scatter(
                    x=x_fit,
                    y=y_fit,
                    mode="lines",
                    line=dict(color=marker_color, dash="dash"),
                    name=f"{pitcher_name} | {date_str} | {tt} | R² = {r_value**2:.2f}"
                ))
                color_idx += 1

            fig_010.update_layout(
                title=f"Velocity vs {selected_metric_010} — Per Pitcher",
                xaxis_title="Velocity",
                yaxis_title=selected_metric_010,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1,
                    xanchor="center",
                    x=0.5
                ),
                height=520,
                margin=dict(l=60, r=140, t=90, b=60),
                hovermode="closest",
                hoverlabel=dict(
                    font_size=13,
                    align="left",
                    namelength=-1
                )
            )

            st.plotly_chart(fig_010, use_container_width=True)
            # Show table
            def estimate_table_height(df, row_px=35, header_px=35, buffer_px=2):
                return len(df) * row_px + header_px + buffer_px
            height_010 = estimate_table_height(df_010)
            st.dataframe(
                df_010.drop(columns=["take_id", "label"]),
                height=height_010
            )
    else:
        st.warning("No data found for the selected pitchers and date.")

with tab4:
    st.subheader("Time-Series")

    # --- Pitcher selector (local to Time-Series tab) ---
    cur.execute("""
        SELECT DISTINCT a.athlete_name
        FROM athletes a
        JOIN takes t ON a.athlete_id = t.athlete_id
        ORDER BY a.athlete_name
    """)
    pitchers_ts = [row[0] for row in cur.fetchall()]

    selected_pitcher_ts = st.selectbox(
        "Select Pitcher",
        pitchers_ts,
        key="ts_pitcher"
    )

    if not selected_pitcher_ts:
        st.info("Select a pitcher to view time-series data.")
        st.stop()

    # --- Throw type selection (default = Mound) ---
    cur.execute("""
        SELECT DISTINCT COALESCE(t.throw_type, 'Mound') AS throw_type
        FROM takes t
        ORDER BY throw_type
    """)
    throw_type_options_ts = [row[0] for row in cur.fetchall()] or ["Mound", "Pulldown"]

    default_throw_types_ts = ["Mound"] if "Mound" in throw_type_options_ts else [throw_type_options_ts[0]]

    selected_throw_types_ts = st.multiselect(
        "Throw Type(s)",
        options=throw_type_options_ts,
        default=default_throw_types_ts,
        key="throw_types_ts"
    )

    if not selected_throw_types_ts:
        selected_throw_types_ts = default_throw_types_ts

    # --- Determine handedness for selected pitcher ---
    cur.execute(
        "SELECT handedness FROM athletes WHERE athlete_name = %s LIMIT 1",
        (selected_pitcher_ts,)
    )
    handedness_row = cur.fetchone()
    handedness_ts = handedness_row[0] if handedness_row else "R"

    # Segment selection based on handedness
    arm_prox_segment = "LAR_PROX" if handedness_ts == "L" else "RAR_PROX"

    # --- Select Date(s) ---
    cur.execute(
        """
        SELECT DISTINCT t.take_date
        FROM takes t
        JOIN athletes a ON t.athlete_id = a.athlete_id
        WHERE a.athlete_name = %s
        ORDER BY t.take_date
        """,
        (selected_pitcher_ts,)
    )
    available_dates = [row[0].strftime("%Y-%m-%d") for row in cur.fetchall()]
    available_dates.insert(0, "All Dates")

    selected_dates_ts = st.multiselect(
        "Select Date(s)",
        options=available_dates,
        default=["All Dates"],
        key="ts_dates"
    )

    # Resolve dates for query
    if "All Dates" in selected_dates_ts or not selected_dates_ts:
        date_filter_clause = ""
        date_filter_params = ()
    else:
        date_filter_clause = "AND t.take_date IN ({})".format(
            ",".join(["%s"] * len(selected_dates_ts))
        )
        date_filter_params = tuple(selected_dates_ts)

    # --- Metric selector (user-facing label mapped to internal category) ---
    metric_label_to_category_ts = {
        "Arm Proximal Energy Transfer": "SEGMENT_POWERS"
    }

    selected_metric_labels_ts = st.multiselect(
        "Select Metric(s)",
        options=list(metric_label_to_category_ts.keys()),
        default=list(metric_label_to_category_ts.keys()),
        key="ts_metrics"
    )

    # Resolve selected categories (future-proofed for multiple metrics)
    selected_metric_categories_ts = [
        metric_label_to_category_ts[label] for label in selected_metric_labels_ts
    ]

    # --- Get all takes for this pitcher (with date filter) ---
    placeholders_tt = ",".join(["%s"] * len(selected_throw_types_ts))
    cur.execute(
        f"""
        SELECT t.take_id, t.take_date, COALESCE(t.throw_type, 'Mound') AS throw_type
        FROM takes t
        JOIN athletes a ON t.athlete_id = a.athlete_id
        WHERE a.athlete_name = %s
          AND COALESCE(t.throw_type, 'Mound') IN ({placeholders_tt})
        {date_filter_clause}
        ORDER BY t.take_date, t.file_name
        """,
        (selected_pitcher_ts, *selected_throw_types_ts, *date_filter_params)
    )
    takes_ts = cur.fetchall()

    if not takes_ts:
        st.warning("No takes found for this pitcher.")
    else:
        import plotly.graph_objects as go
        import plotly.express as px
        color_cycle = px.colors.qualitative.Plotly
        date_colors = {}

        fig_ts = go.Figure()
        fp_aligned_ms = []
        er_aligned_ms = []
        ms_per_frame = 1000.0 / 250.0  # capture sampled at 250 Hz

        fp_aligned_frames = []
        mer_aligned_frames = []

        def _pick_mer_frame_ts(take_id, br_frame, fp_frame=None, throw_type="Mound"):
            shoulder_segment = "RT_SHOULDER" if handedness_ts == "R" else "LT_SHOULDER"

            if throw_type == "Pulldown" and fp_frame is not None:
                start_frame = int(fp_frame) - 30
                end_frame = int(fp_frame) + 30
                cur.execute(
                    """
                    SELECT ts.frame, ts.z_data
                    FROM time_series_data ts
                    JOIN categories c ON ts.category_id = c.category_id
                    JOIN segments s   ON ts.segment_id  = s.segment_id
                    WHERE ts.take_id = %s
                      AND c.category_name = 'JOINT_ANGLES'
                      AND s.segment_name = %s
                      AND ts.frame BETWEEN %s AND %s
                      AND ts.z_data IS NOT NULL
                    ORDER BY ts.frame
                    """,
                    (take_id, shoulder_segment, start_frame, end_frame)
                )
                rows = cur.fetchall()
                if rows:
                    frames = np.array([r[0] for r in rows], dtype=int)
                    z_vals = np.array([r[1] for r in rows], dtype=float)
                    idx = int(np.nanargmin(z_vals)) if handedness_ts == "R" else int(np.nanargmax(z_vals))
                    return int(frames[idx])

            if br_frame is None:
                return None

            cur.execute(
                """
                SELECT ts.frame, ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s   ON ts.segment_id  = s.segment_id
                WHERE ts.take_id = %s
                  AND c.category_name = 'JOINT_ANGLES'
                  AND s.segment_name = %s
                  AND ts.frame <= %s
                  AND ts.z_data IS NOT NULL
                ORDER BY ts.frame
                """,
                (take_id, shoulder_segment, int(br_frame))
            )
            rows = cur.fetchall()
            if not rows:
                return None

            frames = np.array([r[0] for r in rows], dtype=int)
            z_vals = np.array([r[1] for r in rows], dtype=float)
            idx = int(np.nanargmin(z_vals)) if handedness_ts == "R" else int(np.nanargmax(z_vals))
            return int(frames[idx])

        for take_id, take_date, throw_type in takes_ts:
            throw_type = throw_type or "Mound"

            # 1) Preliminary BR from hand CGVel peak
            br_prelim = get_ball_release_frame(take_id, handedness_ts, cur)
            if br_prelim is None:
                continue

            # 2) Preliminary MER (pre-BR) to bound refined FP search
            mer_prelim = _pick_mer_frame_ts(take_id, br_prelim, fp_frame=None, throw_type="Mound")

            # 3) Refined FP via ankle prox-x peak -> ankle z min -> zero-cross (Terra-style flow)
            fp_start_candidate = get_lead_ankle_prox_x_peak_frame(take_id, handedness_ts, cur)
            ankle_min_frame = None
            if fp_start_candidate is not None and mer_prelim is not None:
                ankle_min_frame = get_ankle_min_frame(
                    take_id, handedness_ts, fp_start_candidate, mer_prelim, cur
                )

            zero_cross_frame = None
            if ankle_min_frame is not None and mer_prelim is not None:
                zero_cross_frame = get_zero_cross_frame(
                    take_id, handedness_ts, ankle_min_frame, mer_prelim, cur
                )

            if zero_cross_frame is not None:
                fp_frame = zero_cross_frame
            elif ankle_min_frame is not None:
                fp_frame = ankle_min_frame
            else:
                fp_frame = fp_start_candidate

            # 4) Throw-type-aware BR (Pulldown: anchor after FP)
            if throw_type == "Pulldown":
                br_frame = get_ball_release_frame_pulldown(take_id, handedness_ts, fp_frame, cur)
                if br_frame is None:
                    br_frame = br_prelim
            else:
                br_frame = br_prelim

            # 5) Throw-type-aware MER (Pulldown: FP ±30; Mound: pre-BR)
            sh_er_max_frame = _pick_mer_frame_ts(
                take_id, br_frame, fp_frame=fp_frame, throw_type=throw_type
            )

            if fp_frame is not None:
                fp_aligned_frames.append(fp_frame - br_frame)
                fp_aligned_ms.append((fp_frame - br_frame) * ms_per_frame)
            if sh_er_max_frame is not None:
                mer_aligned_frames.append(sh_er_max_frame - br_frame)
                er_aligned_ms.append((sh_er_max_frame - br_frame) * ms_per_frame)

            cur.execute(
                """
                SELECT ts.frame, ts.x_data
                FROM time_series_data ts
                JOIN segments s ON ts.segment_id = s.segment_id
                JOIN categories c ON ts.category_id = c.category_id
                WHERE ts.take_id = %s
                  AND c.category_name = %s
                  AND s.segment_name = %s
                ORDER BY ts.frame
                """,
                (take_id, selected_metric_categories_ts[0], arm_prox_segment)
            )

            rows = cur.fetchall()
            if not rows:
                continue

            df_ts = pd.DataFrame(rows, columns=["frame", "x_data"])
            df_ts["x_data"] = pd.to_numeric(df_ts["x_data"], errors="coerce")

            frames = df_ts["frame"].values
            vals   = df_ts["x_data"].values

            # Align time so that Ball Release maps to 0 ms
            t_aligned_ms = (frames - br_frame) * ms_per_frame

            date_str = take_date.strftime("%Y-%m-%d")

            # Assign consistent color per date
            if date_str not in date_colors:
                date_colors[date_str] = color_cycle[len(date_colors) % len(color_cycle)]

            fig_ts.add_trace(
                go.Scatter(
                    x=t_aligned_ms,
                    y=vals,
                    mode="lines",
                    line=dict(color=date_colors[date_str]),
                    name=date_str,
                    legendgroup=date_str,
                    opacity=0.4,
                    showlegend=(date_str not in [t.name for t in fig_ts.data])
                )
            )

        # Terra-style event lines/annotations
        fig_ts.add_vline(x=0, line_width=3, line_dash="dash", line_color="blue")
        fig_ts.add_annotation(
            x=0, y=1.06, xref="x", yref="paper",
            text="BR", showarrow=False,
            font=dict(color="blue", size=13, family="Arial"),
            align="center"
        )

        # --- Median Foot Plant ---
        if fp_aligned_ms:
            median_fp = float(np.median(fp_aligned_ms))
            fig_ts.add_vline(x=median_fp, line_width=3, line_dash="dash", line_color="green")
            fig_ts.add_annotation(
                x=median_fp, y=1.06, xref="x", yref="paper",
                text="FP", showarrow=False,
                font=dict(color="green", size=13, family="Arial"),
                align="center"
            )
        # --- Median Shoulder ER ---
        if er_aligned_ms:
            median_er = float(np.median(er_aligned_ms))
            fig_ts.add_vline(x=median_er, line_width=3, line_dash="dash", line_color="red")
            fig_ts.add_annotation(
                x=median_er, y=1.06, xref="x", yref="paper",
                text="MER", showarrow=False,
                font=dict(color="red", size=13, family="Arial"),
                align="center"
            )

        # Terra-style window: start at median FP - 50 frames, end at +50 frames
        if fp_aligned_frames:
            window_start_frame = int(np.median(fp_aligned_frames)) - 50
        else:
            window_start_frame = -100
        window_end_frame = 50
        x_start = window_start_frame * ms_per_frame
        x_end = window_end_frame * ms_per_frame

        fig_ts.update_layout(
            title=f"Arm Proximal Energy Transfer — {arm_prox_segment} (Aligned to Ball Release, ms)",
            xaxis_title="Time Relative to Ball Release (ms)",
            yaxis_title="Power",
            xaxis=dict(range=[x_start, x_end]),
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )

        st.plotly_chart(fig_ts, use_container_width=True)

with tab5:
    st.subheader("Biodex")
    st.caption("Upload Biodex CSV exports to visualize a measurement over time.")

    athlete_rows = fetch_all_athletes(cur)
    athlete_options = {}
    athlete_labels = {}
    for athlete_id, athlete_name, first_name, last_name, handedness in athlete_rows:
        display_name = athlete_name or " ".join(part for part in [first_name, last_name] if part).strip() or f"Athlete {athlete_id}"
        handedness_suffix = f" ({handedness})" if handedness else ""
        athlete_options[int(athlete_id)] = {
            "athlete_name": display_name,
            "first_name": first_name,
            "last_name": last_name,
            "handedness": handedness,
        }
        athlete_labels[int(athlete_id)] = f"{display_name}{handedness_suffix}"

    if "show_biodex_add_athlete" not in st.session_state:
        st.session_state["show_biodex_add_athlete"] = False
    if "biodex_pending_selected_athlete_id" in st.session_state:
        st.session_state["biodex_selected_athlete_id"] = st.session_state.pop("biodex_pending_selected_athlete_id")
    if "biodex_selected_athlete_id" not in st.session_state:
        st.session_state["biodex_selected_athlete_id"] = next(iter(athlete_options), None)

    selector_col, add_col = st.columns([1.0, 0.28], vertical_alignment="bottom")
    with selector_col:
        if athlete_options:
            selected_athlete_id = st.selectbox(
                "Athlete",
                options=list(athlete_options.keys()),
                format_func=lambda athlete_id: athlete_labels.get(athlete_id, f"Athlete {athlete_id}"),
                key="biodex_selected_athlete_id",
                index=(
                    list(athlete_options.keys()).index(st.session_state["biodex_selected_athlete_id"])
                    if st.session_state["biodex_selected_athlete_id"] in athlete_options
                    else 0
                ),
            )
        else:
            selected_athlete_id = None
            st.text_input("Athlete", value="No athletes found yet", disabled=True)
    with add_col:
        if st.button("Add New Athlete", key="biodex_add_athlete_button", use_container_width=True):
            st.session_state["show_biodex_add_athlete"] = not st.session_state["show_biodex_add_athlete"]

    if st.session_state["show_biodex_add_athlete"]:
        with st.form("biodex_add_athlete_form", clear_on_submit=True):
            add_col1, add_col2, add_col3 = st.columns([1.0, 1.0, 0.6])
            with add_col1:
                new_first_name = st.text_input("First Name")
            with add_col2:
                new_last_name = st.text_input("Last Name")
            with add_col3:
                new_handedness = st.selectbox("Handedness", options=["R", "L"], index=0)

            submitted_new_athlete = st.form_submit_button("Add Athlete", use_container_width=True)
            if submitted_new_athlete:
                first_name_clean = new_first_name.strip()
                last_name_clean = new_last_name.strip()
                handedness_clean = new_handedness.strip().upper()

                if not first_name_clean or not last_name_clean:
                    st.error("Enter both a first name and a last name before adding the athlete.")
                else:
                    new_athlete_name = f"{first_name_clean} {last_name_clean}"
                    existing_names = {details["athlete_name"].strip().lower() for details in athlete_options.values()}
                    if new_athlete_name.strip().lower() in existing_names:
                        st.error("That athlete already exists in the database.")
                    else:
                        try:
                            new_athlete_id, inserted_athlete_name = insert_athlete(
                                cur,
                                conn,
                                first_name_clean,
                                last_name_clean,
                                handedness_clean,
                            )
                        except Exception as exc:
                            conn.rollback()
                            st.error(f"Could not add athlete: {exc}")
                        else:
                            st.session_state["biodex_pending_selected_athlete_id"] = int(new_athlete_id)
                            st.session_state["show_biodex_add_athlete"] = False
                            st.success(f"Added {inserted_athlete_name}.")
                            st.rerun()

    selected_athlete = athlete_options.get(selected_athlete_id) if athlete_options else None
    if selected_athlete:
        st.caption(
            f"Uploading Biodex data for {selected_athlete['athlete_name']}"
            + (f" ({selected_athlete['handedness']})" if selected_athlete["handedness"] else "")
        )
    else:
        st.info("Select an athlete or add a new athlete before uploading a Biodex file.")

    protocol_type_options = [
        "aerobic",
        "reactive_eccentric",
        "speed",
        "strength",
    ]
    selected_protocol_type = st.selectbox(
        "Protocol Type",
        options=protocol_type_options,
        format_func=lambda value: value.replace("_", " ").title(),
        key="biodex_protocol_type",
        disabled=selected_athlete is None,
    )
    limb_options = [
        "right",
        "left",
    ]
    selected_limb = st.selectbox(
        "Limb",
        options=limb_options,
        format_func=lambda value: value.title(),
        key="biodex_limb",
        disabled=selected_athlete is None,
    )
    movement_options = [
        "d2_shoulder_pattern",
        "shoulder_er_ir",
    ]
    selected_movement = st.selectbox(
        "Movement",
        options=movement_options,
        format_func=lambda value: {
            "d2_shoulder_pattern": "D2 Shoulder Pattern",
            "shoulder_er_ir": "Shoulder ER/IR",
        }.get(value, value.replace("_", " ").title()),
        key="biodex_movement",
        disabled=selected_athlete is None,
    )
    selected_speed_deg_per_sec = st.number_input(
        "Speed (deg/s)",
        min_value=0,
        max_value=1000,
        value=75,
        step=1,
        key="biodex_speed_deg_per_sec",
        disabled=selected_athlete is None,
    )
    selected_test_date = st.date_input(
        "Test Date",
        key="biodex_test_date",
        disabled=selected_athlete is None,
    )
    entered_biodex_notes = st.text_area(
        "Notes",
        key="biodex_notes",
        placeholder="Optional notes about the Biodex session",
        disabled=selected_athlete is None,
    )
    biodex_plot_label_parts = []
    if selected_athlete:
        biodex_plot_label_parts.append(selected_athlete["athlete_name"])
    if selected_protocol_type:
        biodex_plot_label_parts.append(selected_protocol_type.replace("_", " ").title())
    if selected_limb:
        biodex_plot_label_parts.append(selected_limb.title())
    if selected_movement:
        biodex_plot_label_parts.append({
            "d2_shoulder_pattern": "D2 Shoulder Pattern",
            "shoulder_er_ir": "Shoulder ER/IR",
        }.get(selected_movement, selected_movement.replace("_", " ").title()))
    if selected_speed_deg_per_sec:
        biodex_plot_label_parts.append(f"{selected_speed_deg_per_sec} deg/s")
    if selected_test_date:
        biodex_plot_label_parts.append(selected_test_date.strftime("%Y-%m-%d"))
    biodex_plot_label = " | ".join(biodex_plot_label_parts) or "Biodex Upload"

    uploaded_biodex_files = st.file_uploader(
        "Upload Biodex CSV file(s)",
        type=["csv"],
        accept_multiple_files=True,
        key="biodex_upload",
        disabled=selected_athlete is None,
    )

    if selected_athlete is None:
        uploaded_biodex_files = []

    if not uploaded_biodex_files:
        st.info("Upload one or more Biodex CSV files to display a time-series plot.")
    else:
        biodex_data = []

        for uploaded_file in uploaded_biodex_files:
            try:
                biodex_df, numeric_columns = prepare_biodex_dataframe(uploaded_file)
                biodex_data.append({
                    "name": uploaded_file.name,
                    "df": biodex_df,
                    "numeric_columns": numeric_columns,
                })
            except Exception as exc:
                st.error(f"{uploaded_file.name}: {exc}")

        if biodex_data:
            torque_ready_files = [
                item for item in biodex_data
                if "Torque_Nm" in item["numeric_columns"]
            ]

            if not torque_ready_files:
                st.warning("The uploaded file(s) do not contain a `Torque_Nm` column to plot.")
            else:
                animation_controls_col, _ = st.columns([0.35, 1.0], vertical_alignment="center")
                with animation_controls_col:
                    animate_biodex_lines = st.toggle(
                        "Animate Biodex line draw",
                        value=True,
                        key="biodex_animate_lines",
                    )
                    biodex_animation_speed = st.slider(
                        "Animation speed",
                        min_value=0.005,
                        max_value=0.08,
                        value=0.02,
                        step=0.005,
                        format="%.3f s/frame",
                        key="biodex_animation_speed",
                        disabled=not animate_biodex_lines,
                    )
                    if animate_biodex_lines:
                        st.caption("Use the chart's Play/Pause controls to run the line animation.")
                    smooth_biodex_first_plot = st.toggle(
                        "Smooth first plot for display only",
                        value=False,
                        key="biodex_smooth_first_plot",
                    )
                    biodex_first_plot_window = st.slider(
                        "First plot smoothing window",
                        min_value=5,
                        max_value=31,
                        value=9,
                        step=2,
                        key="biodex_first_plot_window",
                        disabled=not smooth_biodex_first_plot,
                    )
                    biodex_first_plot_polyorder = st.slider(
                        "First plot smoothing polynomial order",
                        min_value=2,
                        max_value=5,
                        value=3,
                        step=1,
                        key="biodex_first_plot_polyorder",
                        disabled=not smooth_biodex_first_plot,
                    )

                fig_biodex = go.Figure()
                for item_index, item in enumerate(torque_ready_files, start=1):
                    plot_df = item["df"].dropna(subset=["Torque_Nm"])
                    if plot_df.empty:
                        continue

                    y_values = plot_df["Torque_Nm"].to_numpy(dtype=float)
                    if smooth_biodex_first_plot:
                        y_values = smooth_biodex_display_curve(
                            y_values,
                            window_length=int(biodex_first_plot_window),
                            polyorder=int(biodex_first_plot_polyorder),
                        )

                    trace_name = biodex_plot_label
                    if len(torque_ready_files) > 1:
                        trace_name = f"{trace_name} ({item_index})"
                    if smooth_biodex_first_plot:
                        trace_name = f"{trace_name} (Display-Smoothed)"

                    fig_biodex.add_trace(
                        go.Scatter(
                            x=plot_df["Elapsed Seconds"],
                            y=y_values,
                            mode="lines",
                            name=trace_name,
                        )
                    )

                fig_biodex.update_layout(
                    title="Biodex Time-Series: Torque_Nm",
                    xaxis_title="Elapsed Time (s)",
                    yaxis_title="Torque_Nm",
                    height=600,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5,
                    ),
                )

                render_plotly_line_reveal(
                    fig_biodex,
                    animate=animate_biodex_lines,
                    use_container_width=True,
                    frame_delay=float(biodex_animation_speed),
                )

                if torque_ready_files:
                    st.markdown("### Biodex Rep Averaging")
                    st.caption(
                        "Detect full torque bursts from a smoothed |torque| envelope, align reps by repeated peak landmarks, and compute an average torque curve."
                    )

                    rep_controls_col, rep_plot_col = st.columns([0.35, 1.0], vertical_alignment="top")

                    with rep_controls_col:
                        selected_rep_file_name = st.selectbox(
                            "Rep averaging file",
                            options=[item["name"] for item in torque_ready_files],
                            key="biodex_rep_file",
                        )
                        threshold = st.number_input(
                            "Rep detection threshold (smoothed |Torque| envelope)",
                            min_value=1.0,
                            max_value=500.0,
                            value=20.0,
                            step=1.0,
                            key="biodex_threshold",
                        )
                        min_samples = st.number_input(
                            "Minimum active samples per rep",
                            min_value=1,
                            max_value=500,
                            value=15,
                            step=1,
                            key="biodex_min_samples",
                        )
                        buffer_samples = st.number_input(
                            "Buffer samples before/after rep",
                            min_value=0,
                            max_value=500,
                            value=20,
                            step=1,
                            key="biodex_buffer_samples",
                        )
                        n_points = st.number_input(
                            "Normalized points per rep",
                            min_value=25,
                            max_value=500,
                            value=101,
                            step=1,
                            key="biodex_n_points",
                        )
                        landmark_prominence = st.slider(
                            "Landmark prominence ratio",
                            min_value=0.05,
                            max_value=0.40,
                            value=0.12,
                            step=0.01,
                            key="biodex_landmark_prominence",
                        )
                        smooth_mean_display = st.toggle(
                            "Smooth mean curve for display only",
                            value=False,
                            key="biodex_smooth_mean_display",
                        )
                        mean_display_window = st.slider(
                            "Display smoothing window",
                            min_value=5,
                            max_value=31,
                            value=9,
                            step=2,
                            key="biodex_mean_display_window",
                            disabled=not smooth_mean_display,
                        )
                        mean_display_polyorder = st.slider(
                            "Display smoothing polynomial order",
                            min_value=2,
                            max_value=5,
                            value=3,
                            step=1,
                            key="biodex_mean_display_polyorder",
                            disabled=not smooth_mean_display,
                        )

                    selected_rep_item = next(
                        item for item in torque_ready_files
                        if item["name"] == selected_rep_file_name
                    )
                    rep_df_source = selected_rep_item["df"].copy()

                    rep_windows = detect_biodex_reps(
                        rep_df_source,
                        value_col="Torque_Nm",
                        threshold=float(threshold),
                        min_samples=int(min_samples),
                        buffer_samples=int(buffer_samples),
                    )

                    reps_long_df, mean_df, aligned_rep_metadata = extract_landmark_aligned_biodex_reps(
                        rep_df_source,
                        rep_windows,
                        time_col="Elapsed Seconds",
                        value_col="Torque_Nm",
                        n_points=int(n_points),
                        prominence_ratio=float(landmark_prominence),
                    )

                    with rep_plot_col:
                        raw_fig = go.Figure()
                        raw_fig.add_trace(go.Scatter(
                            x=rep_df_source["Elapsed Seconds"],
                            y=rep_df_source["Torque_Nm"],
                            mode="lines",
                            name=f"{selected_rep_file_name} (Raw)",
                        ))

                        shapes = []
                        for rep_number, (start_idx, end_idx) in enumerate(rep_windows, start=1):
                            x0 = float(rep_df_source.iloc[start_idx]["Elapsed Seconds"])
                            x1 = float(rep_df_source.iloc[end_idx]["Elapsed Seconds"])

                            shapes.append(dict(
                                type="rect",
                                xref="x",
                                yref="paper",
                                x0=x0,
                                x1=x1,
                                y0=0,
                                y1=1,
                                fillcolor="rgba(0, 123, 255, 0.12)",
                                line=dict(width=0),
                                layer="below",
                            ))

                            raw_fig.add_annotation(
                                x=(x0 + x1) / 2.0,
                                y=1.02,
                                xref="x",
                                yref="paper",
                                text=f"Rep {rep_number}",
                                showarrow=False,
                            )

                        landmark_symbol_map = {
                            "pos": "triangle-up",
                            "neg": "triangle-down",
                        }
                        landmark_color_map = {
                            "pos": "#f59e0b",
                            "neg": "#ef4444",
                        }
                        for rep_meta in aligned_rep_metadata:
                            for idx, kind, x_val, y_val in zip(
                                rep_meta["landmark_indices"],
                                rep_meta["landmark_kinds"],
                                rep_meta["landmark_times"],
                                rep_meta["landmark_torques"],
                            ):
                                raw_fig.add_trace(go.Scatter(
                                    x=[x_val],
                                    y=[y_val],
                                    mode="markers",
                                    marker=dict(
                                        size=10,
                                        symbol=landmark_symbol_map.get(kind, "circle"),
                                        color=landmark_color_map.get(kind, "#ffffff"),
                                    ),
                                    name=f"{kind.upper()} landmark",
                                    legendgroup=f"{kind}_landmark",
                                    showlegend=False,
                                    hovertemplate=(
                                        f"Rep {rep_meta['rep_number']}<br>"
                                        f"{kind.upper()} landmark<br>"
                                        "Time: %{x:.2f}s<br>"
                                        "Torque: %{y:.2f}<extra></extra>"
                                    ),
                                ))

                        raw_fig.update_layout(
                            title="Detected Torque Reps",
                            xaxis_title="Elapsed Time (s)",
                            yaxis_title="Torque_Nm",
                            shapes=shapes,
                            height=500,
                        )
                        render_plotly_line_reveal(
                            raw_fig,
                            animate=animate_biodex_lines,
                            use_container_width=True,
                            frame_delay=float(biodex_animation_speed),
                        )

                        if reps_long_df.empty or mean_df.empty:
                            st.warning("No visible reps were detected with the current settings.")
                        else:
                            avg_fig = go.Figure()
                            mean_display_values = mean_df["mean_torque_nm"].to_numpy(dtype=float)
                            if smooth_mean_display:
                                mean_display_values = smooth_biodex_display_curve(
                                    mean_display_values,
                                    window_length=int(mean_display_window),
                                    polyorder=int(mean_display_polyorder),
                                )

                            for rep_number, rep_df in reps_long_df.groupby("rep_number"):
                                avg_fig.add_trace(go.Scatter(
                                    x=rep_df["movement_pct"],
                                    y=rep_df["torque_nm"],
                                    mode="lines",
                                    line=dict(width=1),
                                    opacity=0.35,
                                    name=f"Rep {rep_number}",
                                ))

                            avg_fig.add_trace(go.Scatter(
                                x=mean_df["movement_pct"],
                                y=mean_df["upper_band"],
                                mode="lines",
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo="skip",
                            ))
                            avg_fig.add_trace(go.Scatter(
                                x=mean_df["movement_pct"],
                                y=mean_df["lower_band"],
                                mode="lines",
                                line=dict(width=0),
                                fill="tonexty",
                                name="±1 SD",
                            ))
                            if smooth_mean_display:
                                avg_fig.add_trace(go.Scatter(
                                    x=mean_df["movement_pct"],
                                    y=mean_df["mean_torque_nm"],
                                    mode="lines",
                                    line=dict(width=2, dash="dash"),
                                    opacity=0.55,
                                    name="Mean Torque (Raw)",
                                ))
                            avg_fig.add_trace(go.Scatter(
                                x=mean_df["movement_pct"],
                                y=mean_display_values,
                                mode="lines",
                                line=dict(width=4),
                                name="Mean Torque" if not smooth_mean_display else "Mean Torque (Display-Smoothed)",
                            ))

                            for boundary_pct, label in zip(
                                mean_df.attrs.get("landmark_boundary_pct", []),
                                mean_df.attrs.get("landmark_labels", []),
                            ):
                                avg_fig.add_vline(
                                    x=float(boundary_pct),
                                    line_width=2,
                                    line_dash="dot",
                                    line_color="rgba(255,255,255,0.45)",
                                )
                                avg_fig.add_annotation(
                                    x=float(boundary_pct),
                                    y=1.03,
                                    xref="x",
                                    yref="paper",
                                    text=label,
                                    showarrow=False,
                                    font=dict(size=11),
                                )

                            avg_fig.update_layout(
                                title="Landmark-Aligned Average Torque Curve Across Detected Reps",
                                xaxis_title="Movement Cycle (%)",
                                yaxis_title="Torque_Nm",
                                height=500,
                            )
                            render_plotly_line_reveal(
                                avg_fig,
                                animate=animate_biodex_lines,
                                use_container_width=True,
                                frame_delay=float(biodex_animation_speed),
                            )

                    if rep_windows:
                        summary_rows = []
                        for rep_number, (start_idx, end_idx) in enumerate(rep_windows, start=1):
                            rep_df = rep_df_source.iloc[start_idx:end_idx + 1].copy()
                            rep_df["Elapsed Seconds"] = pd.to_numeric(rep_df["Elapsed Seconds"], errors="coerce")
                            rep_df["Torque_Nm"] = pd.to_numeric(rep_df["Torque_Nm"], errors="coerce")
                            rep_df = rep_df.dropna(subset=["Elapsed Seconds", "Torque_Nm"])
                            if rep_df.empty:
                                continue

                            summary_rows.append({
                                "Rep": rep_number,
                                "Start Time (s)": float(rep_df["Elapsed Seconds"].iloc[0]),
                                "End Time (s)": float(rep_df["Elapsed Seconds"].iloc[-1]),
                                "Duration (s)": float(rep_df["Elapsed Seconds"].iloc[-1] - rep_df["Elapsed Seconds"].iloc[0]),
                                "Peak Positive Torque": float(rep_df["Torque_Nm"].max()),
                                "Peak Negative Torque": float(rep_df["Torque_Nm"].min()),
                                "Torque Impulse": float(np.trapezoid(rep_df["Torque_Nm"], rep_df["Elapsed Seconds"])),
                            })

                            rep_meta = next(
                                (item for item in aligned_rep_metadata if item["rep_number"] == rep_number),
                                None,
                            )
                            if rep_meta is not None:
                                for landmark_idx, (kind, x_val, y_val) in enumerate(
                                    zip(
                                        rep_meta["landmark_kinds"],
                                        rep_meta["landmark_times"],
                                        rep_meta["landmark_torques"],
                                    ),
                                    start=1,
                                ):
                                    summary_rows[-1][f"{kind.upper()}{landmark_idx} Time (s)"] = x_val
                                    summary_rows[-1][f"{kind.upper()}{landmark_idx} Torque"] = y_val

                        if summary_rows:
                            st.markdown("### Rep Summary")
                            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

                    if not mean_df.empty:
                        st.markdown("### Mean Curve Preview")
                        st.dataframe(mean_df, use_container_width=True)

                show_biodex_table = st.checkbox("Show uploaded data table", key="biodex_show_table")
                if show_biodex_table:
                    preview_frames = []
                    for item in biodex_data:
                        preview_df = item["df"][["Time", "Elapsed Seconds"] + item["numeric_columns"]].copy()
                        preview_df.insert(0, "File", item["name"])
                        preview_frames.append(preview_df)

                    if preview_frames:
                        st.dataframe(pd.concat(preview_frames, ignore_index=True), use_container_width=True)

with tab6:
    st.subheader("Biodex (Test)")
    st.caption("Separate workspace for designing the long-term Biodex upload, processing, and comparison flow.")

    biodex_test_tab1, biodex_test_tab2, biodex_test_tab3 = st.tabs([
        "Upload & Process",
        "Compare Sessions",
        "Review Reps",
    ])

    with biodex_test_tab1:
        st.markdown("### Upload & Process")
        st.caption("Use this area for new Biodex uploads, metadata capture, raw storage, rep detection, and saving processed outputs.")
        athlete_rows_test = fetch_all_athletes(cur)
        athlete_options_test = {}
        athlete_labels_test = {}
        for athlete_id, athlete_name, first_name, last_name, handedness in athlete_rows_test:
            display_name = athlete_name or " ".join(part for part in [first_name, last_name] if part).strip() or f"Athlete {athlete_id}"
            handedness_suffix = f" ({handedness})" if handedness else ""
            athlete_options_test[int(athlete_id)] = {
                "athlete_name": display_name,
                "handedness": handedness,
            }
            athlete_labels_test[int(athlete_id)] = f"{display_name}{handedness_suffix}"

        upload_col1, upload_col2 = st.columns(2)
        with upload_col1:
            if athlete_options_test:
                selected_biodex_test_athlete_id = st.selectbox(
                    "Athlete",
                    options=list(athlete_options_test.keys()),
                    format_func=lambda athlete_id: athlete_labels_test.get(athlete_id, f"Athlete {athlete_id}"),
                    key="biodex_test_upload_athlete",
                )
            else:
                selected_biodex_test_athlete_id = None
                st.text_input("Athlete", value="No athletes found yet", disabled=True)

            selected_biodex_test_protocol = st.selectbox(
                "Protocol Type",
                options=["aerobic", "reactive_eccentric", "speed", "strength"],
                format_func=lambda value: value.replace("_", " ").title(),
                key="biodex_test_upload_protocol",
                disabled=selected_biodex_test_athlete_id is None,
            )
            selected_biodex_test_movement = st.selectbox(
                "Movement",
                options=["d2_shoulder_pattern", "shoulder_er_ir"],
                format_func=lambda value: {
                    "d2_shoulder_pattern": "D2 Shoulder Pattern",
                    "shoulder_er_ir": "Shoulder ER/IR",
                }.get(value, value.replace("_", " ").title()),
                key="biodex_test_upload_movement",
                disabled=selected_biodex_test_athlete_id is None,
            )
        with upload_col2:
            selected_biodex_test_limb = st.selectbox(
                "Limb",
                options=["right", "left"],
                format_func=lambda value: value.title(),
                key="biodex_test_upload_limb",
                disabled=selected_biodex_test_athlete_id is None,
            )
            selected_biodex_test_speed = st.number_input(
                "Speed (deg/s)",
                min_value=0,
                value=75,
                step=1,
                key="biodex_test_upload_speed",
                disabled=selected_biodex_test_athlete_id is None,
            )
            selected_biodex_test_date = st.date_input(
                "Test Date",
                key="biodex_test_upload_date",
                disabled=selected_biodex_test_athlete_id is None,
            )

        entered_biodex_test_notes = st.text_area(
            "Notes",
            key="biodex_test_upload_notes",
            placeholder="Optional notes about the Biodex session",
            disabled=selected_biodex_test_athlete_id is None,
        )
        uploaded_biodex_test_files = st.file_uploader(
            "Upload Biodex CSV file(s)",
            type=["csv"],
            accept_multiple_files=True,
            key="biodex_test_upload_files",
            disabled=selected_biodex_test_athlete_id is None,
        )

        if st.button(
            "Store Raw Upload(s) & Preview Detection",
            key="biodex_test_store_uploads",
            disabled=selected_biodex_test_athlete_id is None or not uploaded_biodex_test_files,
            use_container_width=True,
        ):
            stored_uploads = []
            upload_progress = st.progress(0, text="Preparing Biodex upload...")
            upload_status = st.empty()
            total_files = len(uploaded_biodex_test_files)

            for file_index, uploaded_file in enumerate(uploaded_biodex_test_files, start=1):
                file_base_progress = (file_index - 1) / total_files
                file_weight = 1 / total_files
                try:
                    upload_status.info(f"Reading file {file_index} of {total_files}: {uploaded_file.name}")
                    upload_progress.progress(
                        min(100, int(file_base_progress * 100 + file_weight * 8)),
                        text=f"Reading file {file_index} of {total_files}: {uploaded_file.name}",
                    )
                    biodex_df, numeric_columns = prepare_biodex_dataframe(uploaded_file)
                    selected_athlete_name = athlete_options_test[int(selected_biodex_test_athlete_id)]["athlete_name"]
                    test_name = " | ".join([
                        selected_athlete_name,
                        selected_biodex_test_protocol.replace("_", " ").title(),
                        {
                            "d2_shoulder_pattern": "D2 Shoulder Pattern",
                            "shoulder_er_ir": "Shoulder ER/IR",
                        }.get(selected_biodex_test_movement, selected_biodex_test_movement.replace("_", " ").title()),
                        str(int(selected_biodex_test_speed)) + " deg/s",
                    ])
                    upload_status.info(f"Creating biodex_tests row for {uploaded_file.name}")
                    upload_progress.progress(
                        min(100, int(file_base_progress * 100 + file_weight * 18)),
                        text=f"Creating metadata row for {uploaded_file.name}",
                    )
                    biodex_test_id = insert_biodex_test(
                        cur,
                        athlete_id=int(selected_biodex_test_athlete_id),
                        test_name=test_name,
                        protocol_type=selected_biodex_test_protocol,
                        limb=selected_biodex_test_limb,
                        movement=selected_biodex_test_movement,
                        speed_deg_per_sec=int(selected_biodex_test_speed),
                        test_date=selected_biodex_test_date,
                        source_file_name=uploaded_file.name,
                        notes=entered_biodex_test_notes,
                    )
                    upload_status.info(f"Inserting raw time-series rows for {uploaded_file.name}")
                    def _update_insert_progress(inserted_rows, total_rows, *, _file_name=uploaded_file.name, _base=file_base_progress, _weight=file_weight):
                        inner_fraction = inserted_rows / total_rows if total_rows else 1.0
                        overall_fraction = _base + _weight * (0.18 + inner_fraction * 0.72)
                        upload_progress.progress(
                            min(100, int(overall_fraction * 100)),
                            text=f"Inserting {_file_name}: {inserted_rows:,} / {total_rows:,} rows",
                        )

                    inserted_row_count = insert_biodex_time_series(
                        cur,
                        biodex_test_id,
                        biodex_df,
                        progress_callback=_update_insert_progress,
                    )
                    upload_status.info(f"Committing {uploaded_file.name} to the database")
                    upload_progress.progress(
                        min(100, int(file_base_progress * 100 + file_weight * 96)),
                        text=f"Committing {uploaded_file.name}",
                    )
                    conn.commit()
                except Exception as exc:
                    conn.rollback()
                    st.error(f"Could not store {uploaded_file.name}: {exc}")
                    continue

                stored_uploads.append({
                    "biodex_test_id": biodex_test_id,
                    "name": uploaded_file.name,
                    "row_count": inserted_row_count,
                    "df": biodex_df,
                    "numeric_columns": numeric_columns,
                    "test_name": test_name,
                })

            if stored_uploads:
                st.session_state["biodex_test_uploaded_previews"] = stored_uploads
                upload_progress.progress(100, text="Biodex upload complete.")
                upload_status.success("Raw uploads stored successfully. Preview is ready below.")
                st.success(f"Stored {len(stored_uploads)} Biodex upload(s) in the database and loaded them for preview below.")
            else:
                upload_progress.empty()
                upload_status.empty()

        with st.expander("Admin Test Tools"):
            st.caption("Testing utilities for deleting Biodex uploads from `biodex_tests` and `biodex_time_series`.")

            delete_file_name = st.text_input(
                "Delete by source_file_name",
                key="biodex_test_delete_source_file_name",
                placeholder="Enter exact uploaded file name",
            )
            if st.button(
                "Delete Upload(s) by File Name",
                key="biodex_test_delete_by_file_name",
                disabled=not delete_file_name.strip(),
                use_container_width=True,
            ):
                try:
                    deleted_count, deleted_ids = delete_biodex_tests_by_source_file_name(
                        cur,
                        conn,
                        delete_file_name.strip(),
                    )
                except Exception as exc:
                    conn.rollback()
                    st.error(f"Could not delete uploads for {delete_file_name}: {exc}")
                else:
                    st.session_state["biodex_test_uploaded_previews"] = [
                        item for item in st.session_state.get("biodex_test_uploaded_previews", [])
                        if int(item["biodex_test_id"]) not in set(deleted_ids)
                    ]
                    if deleted_count:
                        st.success(f"Deleted {deleted_count} Biodex upload(s) matching `{delete_file_name}` and reset the sequences.")
                    else:
                        st.info(f"No Biodex uploads were found for `{delete_file_name}`.")

            most_recent_biodex_test = get_most_recent_biodex_test(cur)
            if most_recent_biodex_test:
                st.caption(
                    f"Most recent upload: ID {most_recent_biodex_test[0]} | {most_recent_biodex_test[1]}"
                )
                if st.button(
                    "Delete Most Recent Upload",
                    key="biodex_test_delete_most_recent",
                    use_container_width=True,
                ):
                    try:
                        deleted_count = delete_biodex_tests_by_ids(
                            cur,
                            conn,
                            [most_recent_biodex_test[0]],
                        )
                    except Exception as exc:
                        conn.rollback()
                        st.error(f"Could not delete the most recent upload: {exc}")
                    else:
                        st.session_state["biodex_test_uploaded_previews"] = [
                            item for item in st.session_state.get("biodex_test_uploaded_previews", [])
                            if int(item["biodex_test_id"]) != int(most_recent_biodex_test[0])
                        ]
                        if deleted_count:
                            st.success(
                                f"Deleted most recent Biodex upload `{most_recent_biodex_test[1]}` (ID {most_recent_biodex_test[0]}) and reset the sequences."
                            )
            else:
                st.caption("No Biodex uploads are currently stored.")

        uploaded_previews = st.session_state.get("biodex_test_uploaded_previews", [])
        if uploaded_previews:
            summary_df = pd.DataFrame([
                {
                    "biodex_test_id": item["biodex_test_id"],
                    "source_file_name": item["name"],
                    "stored_rows": item["row_count"],
                    "test_name": item["test_name"],
                }
                for item in uploaded_previews
            ])
            st.markdown("### Stored Upload Summary")
            st.dataframe(summary_df, use_container_width=True)

            preview_file_name = st.selectbox(
                "Preview stored upload",
                options=[item["name"] for item in uploaded_previews],
                key="biodex_test_preview_file",
            )
            preview_item = next(
                item for item in uploaded_previews
                if item["name"] == preview_file_name
            )
            preview_df = preview_item["df"].copy()

            if "Torque_Nm" in preview_item["numeric_columns"]:
                preview_fig = go.Figure()
                preview_fig.add_trace(go.Scatter(
                    x=preview_df["Elapsed Seconds"],
                    y=preview_df["Torque_Nm"],
                    mode="lines",
                    name=preview_item["test_name"],
                ))
                preview_fig.update_layout(
                    title="Stored Raw Torque Preview",
                    xaxis_title="Elapsed Time (s)",
                    yaxis_title="Torque_Nm",
                    height=500,
                )
                st.plotly_chart(preview_fig, use_container_width=True)

                st.markdown("### Rep Detection Preview")
                preview_controls_col, preview_plot_col = st.columns([0.35, 1.0], vertical_alignment="top")

                with preview_controls_col:
                    preview_threshold = st.number_input(
                        "Rep detection threshold (smoothed |Torque| envelope)",
                        min_value=1.0,
                        max_value=500.0,
                        value=20.0,
                        step=1.0,
                        key="biodex_test_preview_threshold",
                    )
                    preview_min_samples = st.number_input(
                        "Minimum active samples per rep",
                        min_value=1,
                        max_value=500,
                        value=15,
                        step=1,
                        key="biodex_test_preview_min_samples",
                    )
                    preview_buffer_samples = st.number_input(
                        "Buffer samples before/after rep",
                        min_value=0,
                        max_value=500,
                        value=20,
                        step=1,
                        key="biodex_test_preview_buffer",
                    )
                    preview_n_points = st.number_input(
                        "Normalized points per rep",
                        min_value=25,
                        max_value=500,
                        value=101,
                        step=1,
                        key="biodex_test_preview_n_points",
                    )
                    preview_landmark_prominence = st.slider(
                        "Landmark prominence ratio",
                        min_value=0.05,
                        max_value=0.40,
                        value=0.12,
                        step=0.01,
                        key="biodex_test_preview_prominence",
                    )

                preview_rep_windows = detect_biodex_reps(
                    preview_df,
                    value_col="Torque_Nm",
                    threshold=float(preview_threshold),
                    min_samples=int(preview_min_samples),
                    buffer_samples=int(preview_buffer_samples),
                )
                preview_reps_long_df, preview_mean_df, preview_aligned_rep_metadata = extract_landmark_aligned_biodex_reps(
                    preview_df,
                    preview_rep_windows,
                    time_col="Elapsed Seconds",
                    value_col="Torque_Nm",
                    n_points=int(preview_n_points),
                    prominence_ratio=float(preview_landmark_prominence),
                )

                with preview_plot_col:
                    preview_raw_fig = go.Figure()
                    preview_raw_fig.add_trace(go.Scatter(
                        x=preview_df["Elapsed Seconds"],
                        y=preview_df["Torque_Nm"],
                        mode="lines",
                        name=f"{preview_item['name']} (Raw)",
                    ))

                    preview_shapes = []
                    for rep_number, (start_idx, end_idx) in enumerate(preview_rep_windows, start=1):
                        x0 = float(preview_df.iloc[start_idx]["Elapsed Seconds"])
                        x1 = float(preview_df.iloc[end_idx]["Elapsed Seconds"])
                        preview_shapes.append(dict(
                            type="rect",
                            xref="x",
                            yref="paper",
                            x0=x0,
                            x1=x1,
                            y0=0,
                            y1=1,
                            fillcolor="rgba(0, 123, 255, 0.12)",
                            line=dict(width=0),
                            layer="below",
                        ))
                        preview_raw_fig.add_annotation(
                            x=(x0 + x1) / 2.0,
                            y=1.02,
                            xref="x",
                            yref="paper",
                            text=f"Rep {rep_number}",
                            showarrow=False,
                        )

                    preview_raw_fig.update_layout(
                        title="Detected Torque Reps",
                        xaxis_title="Elapsed Time (s)",
                        yaxis_title="Torque_Nm",
                        shapes=preview_shapes,
                        height=500,
                    )
                    st.plotly_chart(preview_raw_fig, use_container_width=True)

                    if preview_reps_long_df.empty or preview_mean_df.empty:
                        st.warning("No visible reps were detected with the current settings.")
                    else:
                        preview_avg_fig = go.Figure()
                        for rep_number, rep_df in preview_reps_long_df.groupby("rep_number"):
                            preview_avg_fig.add_trace(go.Scatter(
                                x=rep_df["movement_pct"],
                                y=rep_df["torque_nm"],
                                mode="lines",
                                line=dict(width=1),
                                opacity=0.35,
                                name=f"Rep {rep_number}",
                            ))
                        preview_avg_fig.add_trace(go.Scatter(
                            x=preview_mean_df["movement_pct"],
                            y=preview_mean_df["upper_band"],
                            mode="lines",
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo="skip",
                        ))
                        preview_avg_fig.add_trace(go.Scatter(
                            x=preview_mean_df["movement_pct"],
                            y=preview_mean_df["lower_band"],
                            mode="lines",
                            line=dict(width=0),
                            fill="tonexty",
                            name="±1 SD",
                        ))
                        preview_avg_fig.add_trace(go.Scatter(
                            x=preview_mean_df["movement_pct"],
                            y=preview_mean_df["mean_torque_nm"],
                            mode="lines",
                            line=dict(width=4),
                            name="Mean Torque",
                        ))
                        preview_avg_fig.update_layout(
                            title="Landmark-Aligned Average Torque Curve Across Detected Reps",
                            xaxis_title="Movement Cycle (%)",
                            yaxis_title="Torque_Nm",
                            height=500,
                        )
                        st.plotly_chart(preview_avg_fig, use_container_width=True)

                        if st.button(
                            "Save Detection Results",
                            key=f"biodex_test_save_detection_{preview_item['biodex_test_id']}",
                            use_container_width=True,
                        ):
                            try:
                                biodex_processing_run_id = insert_biodex_processing_run(
                                    cur,
                                    biodex_test_id=int(preview_item["biodex_test_id"]),
                                    processing_version="shoulder_er_ir_landmark_v1",
                                    threshold=float(preview_threshold),
                                    min_samples=int(preview_min_samples),
                                    buffer_samples=int(preview_buffer_samples),
                                    n_points=int(preview_n_points),
                                    landmark_prominence_ratio=float(preview_landmark_prominence),
                                    is_reviewed=False,
                                )
                                rep_window_rows = insert_biodex_rep_windows(
                                    cur,
                                    biodex_processing_run_id=biodex_processing_run_id,
                                    rep_windows=preview_rep_windows,
                                    rep_df_source=preview_df,
                                )
                                inserted_landmark_count = insert_biodex_rep_landmarks(
                                    cur,
                                    rep_window_rows=rep_window_rows,
                                    aligned_rep_metadata=preview_aligned_rep_metadata,
                                )
                                inserted_mean_curve_points = insert_biodex_mean_curve(
                                    cur,
                                    biodex_processing_run_id=biodex_processing_run_id,
                                    mean_df=preview_mean_df,
                                )
                                conn.commit()
                            except Exception as exc:
                                conn.rollback()
                                st.error(f"Could not save detection results: {exc}")
                            else:
                                st.success(
                                    "Saved detection results "
                                    f"(processing run {biodex_processing_run_id}, "
                                    f"{len(rep_window_rows)} rep windows, "
                                    f"{inserted_landmark_count} landmarks, "
                                    f"{inserted_mean_curve_points} mean-curve points)."
                                )
            else:
                st.warning("The stored upload does not contain a `Torque_Nm` column for rep detection preview.")

    with biodex_test_tab2:
        st.markdown("### Compare Sessions")
        st.caption("Use this area to compare saved Biodex sessions from different days for the same exercise type.")

        athlete_rows_compare = fetch_all_athletes(cur)
        athlete_options_compare = {}
        athlete_labels_compare = {}
        for athlete_id, athlete_name, first_name, last_name, handedness in athlete_rows_compare:
            display_name = athlete_name or " ".join(part for part in [first_name, last_name] if part).strip() or f"Athlete {athlete_id}"
            handedness_suffix = f" ({handedness})" if handedness else ""
            athlete_options_compare[int(athlete_id)] = display_name
            athlete_labels_compare[int(athlete_id)] = f"{display_name}{handedness_suffix}"

        compare_col1, compare_col2 = st.columns(2)
        with compare_col1:
            if athlete_options_compare:
                selected_compare_athlete_id = st.selectbox(
                    "Athlete",
                    options=list(athlete_options_compare.keys()),
                    format_func=lambda athlete_id: athlete_labels_compare.get(athlete_id, f"Athlete {athlete_id}"),
                    key="biodex_test_compare_athlete",
                )
            else:
                selected_compare_athlete_id = None
                st.text_input("Athlete", value="No athletes found yet", disabled=True)

            selected_compare_protocol = st.selectbox(
                "Protocol Type",
                options=["aerobic", "reactive_eccentric", "speed", "strength"],
                format_func=lambda value: value.replace("_", " ").title(),
                index=0,
                key="biodex_test_compare_protocol",
                disabled=selected_compare_athlete_id is None,
            )
            selected_compare_movement = st.selectbox(
                "Movement",
                options=["d2_shoulder_pattern", "shoulder_er_ir"],
                format_func=lambda value: {
                    "d2_shoulder_pattern": "D2 Shoulder Pattern",
                    "shoulder_er_ir": "Shoulder ER/IR",
                }.get(value, value.replace("_", " ").title()),
                index=0,
                key="biodex_test_compare_movement",
                disabled=selected_compare_athlete_id is None,
            )
        with compare_col2:
            selected_compare_limb = st.selectbox(
                "Limb",
                options=["right", "left"],
                format_func=lambda value: value.title(),
                index=0,
                key="biodex_test_compare_limb",
                disabled=selected_compare_athlete_id is None,
            )
            selected_compare_speed = st.number_input(
                "Speed (deg/s)",
                min_value=0,
                value=75,
                step=1,
                key="biodex_test_compare_speed",
                disabled=selected_compare_athlete_id is None,
            )

        if selected_compare_athlete_id is None:
            st.info("Select an athlete to compare saved Biodex processing runs.")
        else:
            processed_sessions_df = fetch_biodex_processed_sessions(
                cur,
                athlete_id=int(selected_compare_athlete_id),
                protocol_type=selected_compare_protocol,
                movement=selected_compare_movement,
                limb=selected_compare_limb,
                speed_deg_per_sec=int(selected_compare_speed),
            )

            if processed_sessions_df.empty:
                st.info("No saved processing runs match these filters yet.")
            else:
                session_labels = {}
                for _, row in processed_sessions_df.iterrows():
                    test_date = row["test_date"].strftime("%Y-%m-%d") if pd.notna(row["test_date"]) else "No date"
                    reviewed_suffix = "Reviewed" if row["is_reviewed"] else "Auto"
                    session_labels[int(row["biodex_processing_run_id"])] = (
                        f"{test_date} | Run {int(row['biodex_processing_run_id'])} | "
                        f"{reviewed_suffix} | {int(row['rep_count'])} reps"
                    )

                selected_compare_run_ids = st.multiselect(
                    "Sessions",
                    options=list(session_labels.keys()),
                    default=list(session_labels.keys())[:2],
                    format_func=lambda run_id: session_labels.get(run_id, f"Run {run_id}"),
                    key="biodex_test_compare_sessions",
                )

                if not selected_compare_run_ids:
                    st.info("Select at least one saved session to display its mean curve.")
                else:
                    compare_fig = go.Figure()
                    selected_summary_rows = []

                    for run_id in selected_compare_run_ids:
                        curve_df = fetch_biodex_mean_curve(cur, run_id)
                        if curve_df.empty:
                            continue

                        session_row = processed_sessions_df[
                            processed_sessions_df["biodex_processing_run_id"] == run_id
                        ].iloc[0]
                        label = session_labels.get(run_id, f"Run {run_id}")

                        compare_fig.add_trace(go.Scatter(
                            x=curve_df["movement_pct"],
                            y=curve_df["mean_torque_nm"],
                            mode="lines",
                            line=dict(width=4),
                            name=label,
                        ))
                        compare_fig.add_trace(go.Scatter(
                            x=curve_df["movement_pct"],
                            y=curve_df["upper_band"],
                            mode="lines",
                            line=dict(width=0),
                            legendgroup=label,
                            showlegend=False,
                            hoverinfo="skip",
                        ))
                        compare_fig.add_trace(go.Scatter(
                            x=curve_df["movement_pct"],
                            y=curve_df["lower_band"],
                            mode="lines",
                            line=dict(width=0),
                            fill="tonexty",
                            opacity=0.18,
                            legendgroup=label,
                            name=f"{label} ±1 SD",
                        ))

                        selected_summary_rows.append({
                            "Processing Run": int(run_id),
                            "Test ID": int(session_row["biodex_test_id"]),
                            "Test Date": session_row["test_date"],
                            "Source File": session_row["source_file_name"],
                            "Rep Count": int(session_row["rep_count"]),
                            "Peak Positive Mean Torque": float(session_row["peak_positive_mean_torque"]),
                            "Peak Negative Mean Torque": float(session_row["peak_negative_mean_torque"]),
                            "Processing Version": session_row["processing_version"],
                            "Reviewed": bool(session_row["is_reviewed"]),
                        })

                    if not compare_fig.data:
                        st.warning("Selected sessions do not have saved mean-curve points.")
                    else:
                        compare_fig.update_layout(
                            title="Saved Landmark-Aligned Biodex Mean Curves",
                            xaxis_title="Movement Cycle (%)",
                            yaxis_title="Torque_Nm",
                            height=600,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="center",
                                x=0.5,
                            ),
                        )
                        st.plotly_chart(compare_fig, use_container_width=True)

                    if selected_summary_rows:
                        st.markdown("### Selected Session Summary")
                        st.dataframe(pd.DataFrame(selected_summary_rows), use_container_width=True)

    with biodex_test_tab3:
        st.markdown("### Review Reps")
        st.caption("Use this area to inspect auto-detected reps, adjust windows or landmarks, and save reviewed processing for future comparisons.")

        review_runs_df = fetch_all_biodex_processing_runs(cur)

        if review_runs_df.empty:
            st.info("No saved Biodex processing runs are available to review yet.")
        else:
            review_labels = {}
            for _, row in review_runs_df.iterrows():
                test_date = row["test_date"].strftime("%Y-%m-%d") if pd.notna(row["test_date"]) else "No date"
                reviewed_suffix = "Reviewed" if row["is_reviewed"] else "Needs Review"
                review_labels[int(row["biodex_processing_run_id"])] = (
                    f"{row['athlete_name']} | {test_date} | Run {int(row['biodex_processing_run_id'])} | "
                    f"{row['movement']} | {row['protocol_type']} | {reviewed_suffix}"
                )

            selected_review_run_id = st.selectbox(
                "Processed Biodex Test",
                options=list(review_labels.keys()),
                format_func=lambda run_id: review_labels.get(run_id, f"Run {run_id}"),
                key="biodex_test_review_run",
            )

            selected_review_row = review_runs_df[
                review_runs_df["biodex_processing_run_id"] == selected_review_run_id
            ].iloc[0]

            review_summary_col1, review_summary_col2, review_summary_col3 = st.columns(3)
            with review_summary_col1:
                st.metric("Processing Run", int(selected_review_row["biodex_processing_run_id"]))
                st.metric("Rep Count", int(selected_review_row["rep_count"]))
            with review_summary_col2:
                st.metric("Biodex Test", int(selected_review_row["biodex_test_id"]))
                st.metric("Reviewed", "Yes" if selected_review_row["is_reviewed"] else "No")
            with review_summary_col3:
                st.metric("Speed", f"{int(selected_review_row['speed_deg_per_sec'])} deg/s")
                st.metric("Version", selected_review_row["processing_version"])

            st.dataframe(
                pd.DataFrame([{
                    "Athlete": selected_review_row["athlete_name"],
                    "Test Date": selected_review_row["test_date"],
                    "Protocol": selected_review_row["protocol_type"],
                    "Movement": selected_review_row["movement"],
                    "Limb": selected_review_row["limb"],
                    "Source File": selected_review_row["source_file_name"],
                    "Threshold": selected_review_row["threshold"],
                    "Min Samples": selected_review_row["min_samples"],
                    "Buffer Samples": selected_review_row["buffer_samples"],
                    "N Points": selected_review_row["n_points"],
                    "Landmark Prominence": selected_review_row["landmark_prominence_ratio"],
                }]),
                use_container_width=True,
            )

            raw_review_df = fetch_biodex_raw_time_series(
                cur,
                int(selected_review_row["biodex_test_id"]),
            )
            windows_review_df = fetch_biodex_rep_windows(cur, selected_review_run_id)
            landmarks_review_df = fetch_biodex_rep_landmarks(cur, selected_review_run_id)
            mean_review_df = fetch_biodex_mean_curve(cur, selected_review_run_id)

            if raw_review_df.empty:
                st.warning("No raw time-series rows were found for this Biodex test.")
            else:
                raw_review_df["plot_time_seconds"] = pd.to_numeric(raw_review_df["time_seconds"], errors="coerce")
                if raw_review_df["plot_time_seconds"].isna().all():
                    raw_review_df["time_raw"] = pd.to_datetime(raw_review_df["time_raw"], errors="coerce")
                    if raw_review_df["time_raw"].notna().any():
                        raw_review_df["plot_time_seconds"] = (
                            raw_review_df["time_raw"] - raw_review_df["time_raw"].dropna().iloc[0]
                        ).dt.total_seconds()
                    else:
                        raw_review_df["plot_time_seconds"] = pd.to_numeric(raw_review_df["sample_index"], errors="coerce")

                review_raw_fig = go.Figure()
                review_raw_fig.add_trace(go.Scatter(
                    x=raw_review_df["plot_time_seconds"],
                    y=raw_review_df["torque_nm"],
                    mode="lines",
                    name="Raw Torque",
                ))

                review_shapes = []
                for _, window_row in windows_review_df.iterrows():
                    x0 = float(window_row["start_time_seconds"])
                    x1 = float(window_row["end_time_seconds"])
                    review_shapes.append(dict(
                        type="rect",
                        xref="x",
                        yref="paper",
                        x0=x0,
                        x1=x1,
                        y0=0,
                        y1=1,
                        fillcolor="rgba(0, 123, 255, 0.12)",
                        line=dict(width=0),
                        layer="below",
                    ))
                    review_raw_fig.add_annotation(
                        x=(x0 + x1) / 2.0,
                        y=1.02,
                        xref="x",
                        yref="paper",
                        text=f"Rep {int(window_row['rep_number'])}",
                        showarrow=False,
                    )

                if not landmarks_review_df.empty:
                    for landmark_name, landmark_df in landmarks_review_df.groupby("landmark_name"):
                        review_raw_fig.add_trace(go.Scatter(
                            x=landmark_df["time_seconds"],
                            y=landmark_df["torque_nm"],
                            mode="markers",
                            marker=dict(
                                size=10,
                                symbol="triangle-up" if str(landmark_name).startswith("pos") else "triangle-down",
                            ),
                            name=str(landmark_name),
                        ))

                review_raw_fig.update_layout(
                    title="Saved Rep Windows and Landmarks",
                    xaxis_title="Elapsed Time (s)",
                    yaxis_title="Torque_Nm",
                    shapes=review_shapes,
                    height=500,
                )
                st.plotly_chart(review_raw_fig, use_container_width=True)

            if mean_review_df.empty:
                st.warning("No saved mean curve was found for this processing run.")
            else:
                review_mean_fig = go.Figure()
                review_mean_fig.add_trace(go.Scatter(
                    x=mean_review_df["movement_pct"],
                    y=mean_review_df["upper_band"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ))
                review_mean_fig.add_trace(go.Scatter(
                    x=mean_review_df["movement_pct"],
                    y=mean_review_df["lower_band"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    name="±1 SD",
                ))
                review_mean_fig.add_trace(go.Scatter(
                    x=mean_review_df["movement_pct"],
                    y=mean_review_df["mean_torque_nm"],
                    mode="lines",
                    line=dict(width=4),
                    name="Mean Torque",
                ))
                review_mean_fig.update_layout(
                    title="Saved Landmark-Aligned Mean Curve",
                    xaxis_title="Movement Cycle (%)",
                    yaxis_title="Torque_Nm",
                    height=500,
                )
                st.plotly_chart(review_mean_fig, use_container_width=True)

            if selected_review_row["is_reviewed"]:
                st.success("This processing run is already marked as reviewed.")
            else:
                if st.button(
                    "Mark as Reviewed",
                    key=f"biodex_test_mark_reviewed_{selected_review_run_id}",
                    use_container_width=True,
                ):
                    try:
                        updated_count = mark_biodex_processing_run_reviewed(
                            cur,
                            conn,
                            selected_review_run_id,
                        )
                    except Exception as exc:
                        conn.rollback()
                        st.error(f"Could not mark this processing run as reviewed: {exc}")
                    else:
                        if updated_count:
                            st.success("Processing run marked as reviewed.")
                            st.rerun()
