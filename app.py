from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    /* Keep the sidebar logo visible in both Streamlit light and dark themes. */
    [data-testid="stSidebar"] img {
        mix-blend-mode: difference;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from scipy.stats import linregress
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

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

def format_biodex_movement_label(value):
    return {
        "d2_shoulder_pattern": "D2 Shoulder Pattern",
        "shoulder_er_ir": "Shoulder ER/IR",
        "posterior_cuff": "Posterior Cuff",
    }.get(value, value.replace("_", " ").title())

def get_biodex_throwing_context_options():
    return [
        "pre_session",
        "arm_care",
        "after_1st_inning",
        "after_2nd_inning",
        "after_3rd_inning",
        "after_4th_inning",
        "after_5th_inning",
        "after_6th_inning",
        "after_7th_inning",
        "after_8th_inning",
        "after_9th_inning",
        "post_session",
    ]

def format_biodex_throwing_context_label(value):
    return {
        "pre_session": "Pre-Session / Before Pen",
        "arm_care": "Arm Care",
        "after_1st_inning": "After 1st Inning",
        "after_2nd_inning": "After 2nd Inning",
        "after_3rd_inning": "After 3rd Inning",
        "after_4th_inning": "After 4th Inning",
        "after_5th_inning": "After 5th Inning",
        "after_6th_inning": "After 6th Inning",
        "after_7th_inning": "After 7th Inning",
        "after_8th_inning": "After 8th Inning",
        "after_9th_inning": "After 9th Inning",
        "post_session": "Post-Session",
    }.get(value, value.replace("_", " ").title())

def table_has_column(cur, table_name, column_name):
    cur.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = %s
          AND column_name = %s
        LIMIT 1
        """,
        (table_name, column_name),
    )
    return cur.fetchone() is not None

def get_biodex_effective_speed(protocol_type, speed_deg_per_sec):
    if protocol_type == "reactive_eccentric":
        return None
    return int(speed_deg_per_sec) if speed_deg_per_sec is not None else None

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
    throwing_context=None,
):
    if table_has_column(cur, "biodex_tests", "throwing_context"):
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
                notes,
                throwing_context
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                throwing_context,
            ),
        )
    else:
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
    query = """
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
    """
    params = [
        int(athlete_id),
        protocol_type,
        movement,
        limb,
    ]
    if protocol_type != "reactive_eccentric":
        query += " AND bt.speed_deg_per_sec = %s"
        params.append(int(speed_deg_per_sec))

    query += """
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
    """
    cur.execute(query, tuple(params))
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

def fetch_biodex_tests_for_restore(cur):
    throwing_context_expr = (
        "COALESCE(bt.throwing_context, '') AS throwing_context"
        if table_has_column(cur, "biodex_tests", "throwing_context")
        else "'' AS throwing_context"
    )
    cur.execute(
        f"""
        SELECT
            bt.biodex_test_id,
            bt.test_name,
            bt.test_date,
            bt.protocol_type,
            bt.movement,
            bt.limb,
            bt.speed_deg_per_sec,
            bt.source_file_name,
            bt.notes,
            a.athlete_name,
            {throwing_context_expr},
            COUNT(DISTINCT pr.biodex_processing_run_id) AS processing_run_count
        FROM biodex_tests bt
        JOIN athletes a
            ON bt.athlete_id = a.athlete_id
        LEFT JOIN biodex_processing_runs pr
            ON bt.biodex_test_id = pr.biodex_test_id
        GROUP BY
            bt.biodex_test_id,
            bt.test_name,
            bt.test_date,
            bt.protocol_type,
            bt.movement,
            bt.limb,
            bt.speed_deg_per_sec,
            bt.source_file_name,
            bt.notes,
            a.athlete_name,
            throwing_context
        ORDER BY bt.test_date DESC NULLS LAST, bt.biodex_test_id DESC
        """
    )
    return pd.DataFrame(
        cur.fetchall(),
        columns=[
            "biodex_test_id",
            "test_name",
            "test_date",
            "protocol_type",
            "movement",
            "limb",
            "speed_deg_per_sec",
            "source_file_name",
            "notes",
            "athlete_name",
            "throwing_context",
            "processing_run_count",
        ],
    )

def ensure_biodex_manual_landmarks_table(cur):
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS biodex_manual_landmarks (
            biodex_manual_landmark_id BIGSERIAL PRIMARY KEY,
            biodex_test_id BIGINT NOT NULL REFERENCES biodex_tests(biodex_test_id) ON DELETE CASCADE,
            landmark_type TEXT NOT NULL,
            sample_index INTEGER,
            time_seconds DOUBLE PRECISION,
            position_deg DOUBLE PRECISION,
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        )
        """
    )
    cur.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_biodex_manual_landmarks_test_type
        ON biodex_manual_landmarks (biodex_test_id, landmark_type)
        """
    )

def fetch_biodex_manual_landmark(cur, biodex_test_id, landmark_type):
    ensure_biodex_manual_landmarks_table(cur)
    cur.execute(
        """
        SELECT
            biodex_manual_landmark_id,
            biodex_test_id,
            landmark_type,
            sample_index,
            time_seconds,
            position_deg,
            created_at
        FROM biodex_manual_landmarks
        WHERE biodex_test_id = %s
          AND landmark_type = %s
        LIMIT 1
        """,
        (int(biodex_test_id), landmark_type),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return {
        "biodex_manual_landmark_id": int(row[0]),
        "biodex_test_id": int(row[1]),
        "landmark_type": row[2],
        "sample_index": int(row[3]) if row[3] is not None else None,
        "time_seconds": float(row[4]) if row[4] is not None else None,
        "position_deg": float(row[5]) if row[5] is not None else None,
        "created_at": row[6],
    }

def upsert_biodex_manual_landmark(
    cur,
    conn,
    biodex_test_id,
    landmark_type,
    sample_index,
    time_seconds,
    position_deg,
):
    ensure_biodex_manual_landmarks_table(cur)
    cur.execute(
        """
        INSERT INTO biodex_manual_landmarks (
            biodex_test_id,
            landmark_type,
            sample_index,
            time_seconds,
            position_deg
        )
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (biodex_test_id, landmark_type)
        DO UPDATE SET
            sample_index = EXCLUDED.sample_index,
            time_seconds = EXCLUDED.time_seconds,
            position_deg = EXCLUDED.position_deg,
            created_at = NOW()
        RETURNING biodex_manual_landmark_id
        """,
        (
            int(biodex_test_id),
            landmark_type,
            int(sample_index) if sample_index is not None else None,
            float(time_seconds) if time_seconds is not None else None,
            float(position_deg) if position_deg is not None else None,
        ),
    )
    landmark_id = int(cur.fetchone()[0])
    conn.commit()
    return landmark_id

def delete_biodex_manual_landmark(cur, conn, biodex_test_id, landmark_type):
    ensure_biodex_manual_landmarks_table(cur)
    cur.execute(
        """
        DELETE FROM biodex_manual_landmarks
        WHERE biodex_test_id = %s
          AND landmark_type = %s
        """,
        (int(biodex_test_id), landmark_type),
    )
    deleted_count = cur.rowcount
    conn.commit()
    return int(deleted_count)

def build_biodex_preview_item_from_db(cur, biodex_test_id):
    restore_tests_df = fetch_biodex_tests_for_restore(cur)
    selected_rows = restore_tests_df.loc[
        restore_tests_df["biodex_test_id"] == int(biodex_test_id)
    ]
    if selected_rows.empty:
        raise ValueError("That Biodex test could not be found in the database.")

    test_row = selected_rows.iloc[0]
    raw_df = fetch_biodex_raw_time_series(cur, biodex_test_id=int(biodex_test_id))
    if raw_df.empty:
        raise ValueError("No raw time-series rows were found for that Biodex test.")

    preview_df = pd.DataFrame()
    preview_df["Time"] = pd.to_datetime(raw_df["time_raw"], errors="coerce")
    if raw_df["time_seconds"].notna().any():
        preview_df["Elapsed Seconds"] = pd.to_numeric(raw_df["time_seconds"], errors="coerce")
    elif preview_df["Time"].notna().any():
        preview_df["Elapsed Seconds"] = (
            preview_df["Time"] - preview_df["Time"].dropna().iloc[0]
        ).dt.total_seconds()
    else:
        preview_df["Elapsed Seconds"] = pd.to_numeric(raw_df["sample_index"], errors="coerce")

    preview_df["Angular_Velocity_Deg/Sec"] = pd.to_numeric(raw_df["angular_velocity_deg_sec"], errors="coerce")
    preview_df["Position_Deg"] = pd.to_numeric(raw_df["position_deg"], errors="coerce")
    preview_df["Torque_Nm"] = pd.to_numeric(raw_df["torque_nm"], errors="coerce")
    preview_df = preview_df.sort_values("Elapsed Seconds").reset_index(drop=True)

    numeric_columns = [
        col for col in ["Angular_Velocity_Deg/Sec", "Position_Deg", "Torque_Nm"]
        if col in preview_df.columns and preview_df[col].notna().any()
    ]
    manual_rom_end = fetch_biodex_manual_landmark(
        cur,
        biodex_test_id=int(test_row["biodex_test_id"]),
        landmark_type="posterior_cuff_rom_end",
    )

    return {
        "biodex_test_id": int(test_row["biodex_test_id"]),
        "name": str(test_row["source_file_name"]),
        "row_count": int(len(preview_df)),
        "df": preview_df,
        "numeric_columns": numeric_columns,
        "test_name": str(test_row["test_name"]),
        "movement": test_row["movement"],
        "protocol_type": test_row["protocol_type"],
        "test_date": test_row["test_date"],
        "throwing_context": test_row["throwing_context"],
        "manual_rom_end": manual_rom_end,
    }

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

def smooth_position_deg_signal(
    time_values,
    position_values,
    lowpass_cutoff_hz=4.0,
    secondary_window_length=31,
):
    """Create a low-pass filtered Position_Deg signal suitable for ROM inspection."""
    time_arr = np.asarray(time_values, dtype=float)
    arr = np.asarray(position_values, dtype=float)
    if len(arr) == 0:
        return arr, None, arr

    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return arr, None, arr

    clean_position = arr.copy()
    if not finite_mask.all():
        valid_idx = np.flatnonzero(finite_mask)
        clean_position[~finite_mask] = np.interp(
            np.flatnonzero(~finite_mask),
            valid_idx,
            arr[finite_mask],
        )

    finite_time_mask = np.isfinite(time_arr)
    if finite_time_mask.any():
        clean_time = time_arr.copy()
        if not finite_time_mask.all():
            valid_time_idx = np.flatnonzero(finite_time_mask)
            clean_time[~finite_time_mask] = np.interp(
                np.flatnonzero(~finite_time_mask),
                valid_time_idx,
                time_arr[finite_time_mask],
            )
    else:
        clean_time = np.arange(len(clean_position), dtype=float)

    dt = np.diff(clean_time)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    fs = float(1.0 / np.nanmedian(dt)) if dt.size else None

    if fs is not None and fs > (lowpass_cutoff_hz * 2.5) and len(clean_position) >= 9:
        nyquist = fs / 2.0
        normalized_cutoff = min(0.99, float(lowpass_cutoff_hz) / nyquist)
        if 0.0 < normalized_cutoff < 1.0:
            b, a = butter(4, normalized_cutoff, btype="low")
            smooth_position = filtfilt(b, a, clean_position)
        else:
            smooth_window = get_valid_savgol_window(51, len(clean_position), 3)
            smooth_position = savgol_filter(clean_position, window_length=smooth_window, polyorder=3) if smooth_window is not None else clean_position
    else:
        smooth_window = get_valid_savgol_window(51, len(clean_position), 3)
        smooth_position = savgol_filter(clean_position, window_length=smooth_window, polyorder=3) if smooth_window is not None else clean_position

    secondary_window = get_valid_savgol_window(int(secondary_window_length), len(smooth_position), 2)
    if secondary_window is not None:
        smooth_position = savgol_filter(smooth_position, window_length=secondary_window, polyorder=2)

    return smooth_position, fs, clean_position

def lowpass_butterworth_position_signal(
    time_values,
    position_values,
    lowpass_cutoff_hz=4.0,
    order=4,
):
    """Apply a Butterworth low-pass filter to Position_Deg without extra smoothing."""
    time_arr = np.asarray(time_values, dtype=float)
    arr = np.asarray(position_values, dtype=float)
    if len(arr) == 0:
        return arr, None, arr

    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return arr, None, arr

    clean_position = arr.copy()
    if not finite_mask.all():
        valid_idx = np.flatnonzero(finite_mask)
        clean_position[~finite_mask] = np.interp(
            np.flatnonzero(~finite_mask),
            valid_idx,
            arr[finite_mask],
        )

    finite_time_mask = np.isfinite(time_arr)
    if finite_time_mask.any():
        clean_time = time_arr.copy()
        if not finite_time_mask.all():
            valid_time_idx = np.flatnonzero(finite_time_mask)
            clean_time[~finite_time_mask] = np.interp(
                np.flatnonzero(~finite_time_mask),
                valid_time_idx,
                time_arr[finite_time_mask],
            )
    else:
        clean_time = np.arange(len(clean_position), dtype=float)

    dt = np.diff(clean_time)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    fs = float(1.0 / np.nanmedian(dt)) if dt.size else None
    if fs is None or fs <= 0 or len(clean_position) < 9:
        return clean_position, fs, clean_position

    nyquist = fs / 2.0
    normalized_cutoff = min(0.99, float(lowpass_cutoff_hz) / nyquist)
    if not (0.0 < normalized_cutoff < 1.0):
        return clean_position, fs, clean_position

    b, a = butter(int(order), normalized_cutoff, btype="low")
    filtered_position = filtfilt(b, a, clean_position)
    return filtered_position, fs, clean_position

def find_first_common_rom_band_entry(
    smooth_position_values,
    start_idx,
    fs,
    target_angle,
    angle_tolerance_deg=2.5,
    velocity_tolerance_deg_per_second=15.0,
    hold_time_seconds=0.08,
):
    """Find the first sustained entry into a shared end-ROM band."""
    values = np.asarray(smooth_position_values, dtype=float)
    if len(values) == 0:
        return None

    start_idx = int(max(0, start_idx))
    if start_idx >= len(values):
        return None

    if fs is not None and fs > 0:
        velocity = np.gradient(values, 1.0 / float(fs))
        hold_samples = max(2, int(round(float(fs) * float(hold_time_seconds))))
        velocity_tolerance = float(velocity_tolerance_deg_per_second)
    else:
        velocity = np.gradient(values)
        hold_samples = 2
        velocity_tolerance = max(0.35, float(angle_tolerance_deg) * 0.5)

    target_low = float(target_angle) - float(angle_tolerance_deg)
    target_high = float(target_angle) + float(angle_tolerance_deg)

    for idx in range(start_idx, len(values) - hold_samples + 1):
        value_window = values[idx:idx + hold_samples]
        velocity_window = velocity[idx:idx + hold_samples]
        in_band = np.all((value_window >= target_low) & (value_window <= target_high))
        low_velocity = np.nanmean(np.abs(velocity_window)) <= velocity_tolerance
        if in_band and low_velocity:
            return int(idx)

    for idx in range(start_idx, len(values)):
        if target_low <= values[idx] <= target_high:
            return int(idx)

    return None

def find_first_position_ascent_threshold(
    smooth_position_values,
    start_idx,
    target_angle,
):
    """Find the first time a smoothed ROM curve reaches a target angle on ascent."""
    values = np.asarray(smooth_position_values, dtype=float)
    if len(values) == 0:
        return None

    start_idx = int(max(0, start_idx))
    if start_idx >= len(values):
        return None

    for idx in range(start_idx, len(values)):
        if not np.isfinite(values[idx]):
            continue
        if values[idx] >= float(target_angle):
            return int(idx)

    return None

def extract_position_fraction_aligned_curves(
    preview_items,
    rom_fraction=0.50,
    time_col="Elapsed Seconds",
    position_col="Position_Deg",
    n_points=201,
    lowpass_cutoff_hz=1.0,
):
    if not preview_items:
        return pd.DataFrame(), pd.DataFrame(), []

    aligned_reps = []
    for rep_number, item in enumerate(preview_items, start=1):
        rep_df = item["df"].copy()
        if time_col not in rep_df.columns or position_col not in rep_df.columns:
            continue

        rep_df[time_col] = pd.to_numeric(rep_df[time_col], errors="coerce")
        rep_df[position_col] = pd.to_numeric(rep_df[position_col], errors="coerce")
        rep_df = rep_df.dropna(subset=[time_col, position_col]).reset_index(drop=True)
        if len(rep_df) < 7:
            continue

        time_values = rep_df[time_col].to_numpy(dtype=float)
        position_values = rep_df[position_col].to_numpy(dtype=float)
        smooth_position, _fs, _clean_position = smooth_position_deg_signal(
            time_values,
            position_values,
            lowpass_cutoff_hz=float(lowpass_cutoff_hz),
        )
        position_bounds = detect_position_deg_rep_bounds(
            time_values,
            position_values,
        )
        start_idx = int(position_bounds["start_idx"])
        final_plateau_value = float(position_bounds["final_plateau_value"])
        baseline_value = float(np.nanmedian(smooth_position[:max(start_idx + 1, 1)]))
        target_angle = baseline_value + ((final_plateau_value - baseline_value) * float(rom_fraction))
        anchor_idx = find_first_position_ascent_threshold(
            smooth_position,
            start_idx,
            target_angle,
        )
        if anchor_idx is None:
            continue

        aligned_reps.append({
            "rep_number": rep_number,
            "file_name": item["name"],
            "aligned_time": time_values - float(time_values[anchor_idx]),
            "position_values": np.asarray(smooth_position, dtype=float),
            "anchor_idx": int(anchor_idx),
            "anchor_time": float(time_values[anchor_idx]),
            "anchor_position_deg": float(smooth_position[anchor_idx]),
            "anchor_label": f"{int(round(float(rom_fraction) * 100.0))}% ROM",
        })

    if not aligned_reps:
        return pd.DataFrame(), pd.DataFrame(), []

    common_start = max(float(rep["aligned_time"][0]) for rep in aligned_reps)
    common_end = min(float(rep["aligned_time"][-1]) for rep in aligned_reps)
    if common_end <= common_start:
        return pd.DataFrame(), pd.DataFrame(), aligned_reps

    common_axis = np.linspace(common_start, common_end, int(n_points))
    rep_rows = []
    interpolated_curves = []
    for rep in aligned_reps:
        interp_position = np.interp(common_axis, rep["aligned_time"], rep["position_values"])
        interpolated_curves.append(interp_position)
        rep_rows.append(pd.DataFrame({
            "rep_number": rep["rep_number"],
            "file_name": rep["file_name"],
            "alignment_x": common_axis,
            "position_deg": interp_position,
        }))

    reps_long_df = pd.concat(rep_rows, ignore_index=True)
    curves_arr = np.vstack(interpolated_curves)
    mean_df = pd.DataFrame({
        "alignment_x": common_axis,
        "mean_position_deg": np.nanmean(curves_arr, axis=0),
        "std_position_deg": np.nanstd(curves_arr, axis=0),
    })
    mean_df["upper_band"] = mean_df["mean_position_deg"] + mean_df["std_position_deg"]
    mean_df["lower_band"] = mean_df["mean_position_deg"] - mean_df["std_position_deg"]
    mean_df.attrs["x_axis_title"] = "Aligned Time (s)"
    mean_df.attrs["anchor_x"] = 0.0
    mean_df.attrs["anchor_label"] = f"{int(round(float(rom_fraction) * 100.0))}% ROM"

    return reps_long_df, mean_df, aligned_reps

def extract_torque_fraction_window_curves(
    preview_items,
    start_fraction=0.05,
    end_fraction=0.98,
    start_mode="fraction",
    value_col="Torque_Nm",
    time_col="Elapsed Seconds",
    position_col="Position_Deg",
    n_points=201,
):
    if not preview_items:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    percent_axis = np.linspace(0.0, 100.0, int(n_points))
    torque_rep_rows = []
    torque_curves = []
    position_rep_rows = []
    position_curves = []
    aligned_metadata = []

    for rep_number, item in enumerate(preview_items, start=1):
        rep_df = item["df"].copy()
        if value_col not in rep_df.columns or time_col not in rep_df.columns:
            continue

        rep_df[time_col] = pd.to_numeric(rep_df[time_col], errors="coerce")
        rep_df[value_col] = pd.to_numeric(rep_df[value_col], errors="coerce")
        if position_col in rep_df.columns:
            rep_df[position_col] = pd.to_numeric(rep_df[position_col], errors="coerce")
        rep_df = rep_df.dropna(subset=[time_col, value_col]).reset_index(drop=True)
        if len(rep_df) < 7:
            continue

        time_values = rep_df[time_col].to_numpy(dtype=float)
        torque_values = rep_df[value_col].to_numpy(dtype=float)
        smooth_torque = smooth_biodex_display_curve(torque_values, window_length=31, polyorder=3)
        if smooth_torque is None:
            smooth_torque = torque_values

        peak_idx = int(np.nanargmax(smooth_torque))
        baseline_window = max(3, min(len(smooth_torque), max(5, peak_idx)))
        baseline_value = float(np.nanmedian(smooth_torque[:baseline_window]))
        peak_value = float(smooth_torque[peak_idx])
        amplitude = peak_value - baseline_value
        if not np.isfinite(amplitude) or amplitude <= 0:
            continue

        start_threshold = baseline_value + (float(start_fraction) * amplitude)
        end_threshold = baseline_value + (float(end_fraction) * amplitude)

        start_idx = None
        end_idx = None
        if start_mode == "zero_torque_rise":
            start_idx = peak_idx
            for idx in range(peak_idx, 0, -1):
                prev_val = float(smooth_torque[idx - 1])
                curr_val = float(smooth_torque[idx])
                if prev_val <= 0.0 < curr_val:
                    start_idx = int(idx)
                    break
        else:
            for idx in range(0, peak_idx + 1):
                if np.isfinite(smooth_torque[idx]) and smooth_torque[idx] >= start_threshold:
                    start_idx = int(idx)
                    break
        if start_idx is None:
            continue
        for idx in range(start_idx, peak_idx + 1):
            if np.isfinite(smooth_torque[idx]) and smooth_torque[idx] >= end_threshold:
                end_idx = int(idx)
                break
        if end_idx is None or end_idx <= start_idx:
            continue

        torque_window = torque_values[start_idx:end_idx + 1]
        interp_torque = np.interp(
            percent_axis,
            np.linspace(0.0, 100.0, len(torque_window)),
            torque_window,
        )
        torque_curves.append(interp_torque)
        torque_rep_rows.append(pd.DataFrame({
            "rep_number": rep_number,
            "file_name": item["name"],
            "alignment_x": percent_axis,
            "torque_nm": interp_torque,
        }))

        if position_col in rep_df.columns and rep_df[position_col].notna().any():
            position_values = rep_df[position_col].to_numpy(dtype=float)
            position_window = position_values[start_idx:end_idx + 1]
            interp_position = np.interp(
                percent_axis,
                np.linspace(0.0, 100.0, len(position_window)),
                position_window,
            )
            position_curves.append(interp_position)
            position_rep_rows.append(pd.DataFrame({
                "rep_number": rep_number,
                "file_name": item["name"],
                "alignment_x": percent_axis,
                "position_deg": interp_position,
            }))

        aligned_metadata.append({
            "rep_number": rep_number,
            "file_name": item["name"],
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "start_time": float(time_values[start_idx]),
            "end_time": float(time_values[end_idx]),
            "start_threshold": float(start_threshold),
            "end_threshold": float(end_threshold),
            "peak_positive_torque": float(peak_value),
            "start_mode": start_mode,
        })

    if not torque_curves:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    torque_reps_long_df = pd.concat(torque_rep_rows, ignore_index=True)
    torque_mean_df = pd.DataFrame({
        "alignment_x": percent_axis,
        "mean_torque_nm": np.nanmean(np.vstack(torque_curves), axis=0),
        "std_torque_nm": np.nanstd(np.vstack(torque_curves), axis=0),
    })
    torque_mean_df["upper_band"] = torque_mean_df["mean_torque_nm"] + torque_mean_df["std_torque_nm"]
    torque_mean_df["lower_band"] = torque_mean_df["mean_torque_nm"] - torque_mean_df["std_torque_nm"]
    if start_mode == "zero_torque_rise":
        torque_mean_df.attrs["x_axis_title"] = f"0 Torque Rise to {int(round(end_fraction * 100))}% Peak Positive Torque (%)"
        torque_mean_df.attrs["anchor_label"] = "0 Torque Rise"
    else:
        torque_mean_df.attrs["x_axis_title"] = f"{int(round(start_fraction * 100))}% to {int(round(end_fraction * 100))}% Peak Positive Torque (%)"
        torque_mean_df.attrs["anchor_label"] = f"{int(round(start_fraction * 100))}% Torque"
    torque_mean_df.attrs["anchor_x"] = 0.0
    torque_mean_df.attrs["secondary_anchor_x"] = 100.0
    torque_mean_df.attrs["secondary_anchor_label"] = f"{int(round(end_fraction * 100))}% Torque"

    if position_curves:
        position_reps_long_df = pd.concat(position_rep_rows, ignore_index=True)
        position_mean_df = pd.DataFrame({
            "alignment_x": percent_axis,
            "mean_position_deg": np.nanmean(np.vstack(position_curves), axis=0),
            "std_position_deg": np.nanstd(np.vstack(position_curves), axis=0),
        })
        position_mean_df["upper_band"] = position_mean_df["mean_position_deg"] + position_mean_df["std_position_deg"]
        position_mean_df["lower_band"] = position_mean_df["mean_position_deg"] - position_mean_df["std_position_deg"]
        position_mean_df.attrs = torque_mean_df.attrs.copy()
    else:
        position_reps_long_df = pd.DataFrame()
        position_mean_df = pd.DataFrame()

    return torque_reps_long_df, torque_mean_df, position_reps_long_df, position_mean_df, aligned_metadata

def detect_position_deg_rep_bounds(
    time_values,
    position_values,
    plateau_window_fraction=0.20,
    rom_fraction=0.90,
    angle_tolerance_deg=3.0,
    velocity_tolerance_deg_per_second=15.0,
    hold_time_seconds=0.20,
    lowpass_cutoff_hz=6.0,
):
    """
    Detect the active Position_Deg movement window.

    End of rep is defined as the first point where the smoothed position reaches
    the final ROM plateau and remains stable, instead of the later noisy oscillation
    around that same end-range angle.
    """
    time_arr = np.asarray(time_values, dtype=float)
    arr = np.asarray(position_values, dtype=float)
    if len(arr) < 7:
        return {
            "clean_position": arr,
            "smooth_position": arr,
            "start_idx": 0,
            "end_idx": max(0, len(arr) - 1),
            "plateau_idx": None,
            "final_plateau_value": float(arr[-1]) if len(arr) else np.nan,
            "plateau_band_low": np.nan,
            "plateau_band_high": np.nan,
        }

    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return {
            "clean_position": arr,
            "smooth_position": arr,
            "start_idx": 0,
            "end_idx": max(0, len(arr) - 1),
            "plateau_idx": None,
            "final_plateau_value": np.nan,
            "plateau_band_low": np.nan,
            "plateau_band_high": np.nan,
        }

    clean_position = arr.copy()
    if not finite_mask.all():
        valid_idx = np.flatnonzero(finite_mask)
        clean_position[~finite_mask] = np.interp(
            np.flatnonzero(~finite_mask),
            valid_idx,
            arr[finite_mask],
        )

    smooth_position, fs, clean_position = smooth_position_deg_signal(
        time_arr,
        clean_position,
        lowpass_cutoff_hz=lowpass_cutoff_hz,
    )

    position_span = float(np.nanmax(smooth_position) - np.nanmin(smooth_position))
    if position_span <= 0:
        return {
            "clean_position": clean_position,
            "smooth_position": smooth_position,
            "start_idx": 0,
            "end_idx": max(0, len(clean_position) - 1),
            "plateau_idx": None,
            "final_plateau_value": float(np.nanmedian(smooth_position)),
            "plateau_band_low": float(np.nanmedian(smooth_position)),
            "plateau_band_high": float(np.nanmedian(smooth_position)),
        }

    # Step 2: estimate the starting baseline from the first quiet portion.
    baseline_window = max(5, min(len(smooth_position) // 4, 50))
    baseline_value = float(np.nanmedian(smooth_position[:baseline_window]))

    # Step 3: estimate the final ROM plateau from the last portion of the file.
    # Using the median makes this robust to end-range oscillations/spikes.
    plateau_window = max(5, int(len(smooth_position) * float(plateau_window_fraction)))
    plateau_window = min(plateau_window, len(smooth_position))
    final_plateau_value = float(np.nanmedian(smooth_position[-plateau_window:]))

    movement_direction = 1.0 if final_plateau_value >= baseline_value else -1.0
    signed_position = movement_direction * smooth_position
    signed_baseline = movement_direction * baseline_value
    signed_plateau = movement_direction * final_plateau_value
    signed_span = max(1e-9, signed_plateau - signed_baseline)

    # Step 4: find movement start as the first sustained departure from baseline.
    start_threshold = signed_baseline + max(8.0, signed_span * 0.10)
    slope = np.gradient(signed_position)
    positive_slope_threshold = max(0.25, abs(signed_span) * 0.002)
    start_hold_samples = max(3, min(8, len(smooth_position) // 20))

    start_idx = 0
    for idx in range(0, len(signed_position) - start_hold_samples + 1):
        value_window = signed_position[idx:idx + start_hold_samples]
        slope_window = slope[idx:idx + start_hold_samples]
        if np.all(value_window >= start_threshold) and np.nanmean(slope_window) > positive_slope_threshold:
            start_idx = idx
            break

    # Step 5: search for the first sustained end-range plateau.
    # This ignores torque after the player has already reached max ROM.
    near_plateau_threshold = signed_baseline + (signed_span * float(rom_fraction))
    if fs is not None and fs > 0:
        velocity = np.gradient(smooth_position, 1.0 / fs) * movement_direction
        hold_samples = max(3, int(round(fs * float(hold_time_seconds))))
        velocity_tolerance = float(velocity_tolerance_deg_per_second)
    else:
        velocity = np.gradient(signed_position)
        hold_samples = max(3, int(round(len(smooth_position) * 0.02)))
        velocity_tolerance = max(0.35, abs(signed_span) * 0.003)

    plateau_idx = None
    for idx in range(start_idx, len(signed_position) - hold_samples + 1):
        value_window = signed_position[idx:idx + hold_samples]
        velocity_window = velocity[idx:idx + hold_samples]

        is_near_final_rom = np.all(value_window >= near_plateau_threshold)
        is_inside_plateau_band = np.all(np.abs(value_window - signed_plateau) <= float(angle_tolerance_deg))
        is_stable = np.nanmean(np.abs(velocity_window)) <= float(velocity_tolerance)

        if is_near_final_rom and is_inside_plateau_band and is_stable:
            plateau_idx = idx
            break

    # Fallback: if the stability rule is too strict, use the first time the curve
    # reaches 90% of the final ROM. This is still better than going deep into the
    # noisy post-movement torque/position oscillation.
    if plateau_idx is None:
        candidates = np.flatnonzero(signed_position[start_idx:] >= near_plateau_threshold)
        if candidates.size > 0:
            plateau_idx = int(start_idx + candidates[0])
        else:
            plateau_idx = int(np.nanargmax(signed_position))

    end_idx = int(plateau_idx)

    return {
        "clean_position": clean_position,
        "smooth_position": smooth_position,
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "plateau_idx": int(plateau_idx) if plateau_idx is not None else None,
        "final_plateau_value": float(final_plateau_value),
        "plateau_band_low": float(final_plateau_value - float(angle_tolerance_deg)),
        "plateau_band_high": float(final_plateau_value + float(angle_tolerance_deg)),
        "fs": fs,
    }

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

def detect_posterior_cuff_reactive_eccentric_reps(
    df,
    value_col="Torque_Nm",
    threshold=20.0,
    min_samples=15,
    buffer_samples=20,
):
    rep_windows = detect_biodex_reps(
        df,
        value_col=value_col,
        threshold=threshold,
        min_samples=min_samples,
        buffer_samples=buffer_samples,
    )
    if not rep_windows or df.empty or value_col not in df.columns:
        return rep_windows

    signal = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    if len(signal) == 0:
        return rep_windows

    finite_mask = np.isfinite(signal)
    if not finite_mask.any():
        return rep_windows

    clean_signal = signal.copy()
    if not finite_mask.all():
        valid_idx = np.flatnonzero(finite_mask)
        clean_signal[~finite_mask] = np.interp(
            np.flatnonzero(~finite_mask),
            valid_idx,
            signal[finite_mask],
        )

    onset_window = max(5, min(len(clean_signal), int(min_samples) * 2 + 1))
    if onset_window % 2 == 0:
        onset_window = max(1, onset_window - 1)

    smooth_signal = (
        pd.Series(clean_signal)
        .rolling(window=onset_window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )

    n_rows = len(df)
    search_span = max(int(buffer_samples) * 4, int(min_samples) * 4, 20)
    sustain_samples = max(3, int(min_samples) // 3)
    negative_onset_threshold = -max(3.0, float(threshold) * 0.25)
    pre_dip_padding = max(1, int(buffer_samples) // 4)
    adjusted_windows = []

    for start_idx, end_idx in rep_windows:
        search_start = max(0, int(start_idx) - search_span)
        search_end = min(n_rows - 1, int(start_idx))
        local_signal = smooth_signal[search_start:search_end + 1]
        negative_mask = local_signal <= negative_onset_threshold
        negative_regions = _build_contiguous_regions(np.flatnonzero(negative_mask).tolist())

        adjusted_start = int(start_idx)
        if negative_regions:
            eligible_regions = [
                (region_start, region_end)
                for region_start, region_end in negative_regions
                if (region_end - region_start + 1) >= sustain_samples
            ]
            if eligible_regions:
                region_start, _region_end = eligible_regions[-1]
                adjusted_start = max(0, search_start + int(region_start) - pre_dip_padding)

        adjusted_windows.append((adjusted_start, int(end_idx)))

    return adjusted_windows

def detect_shoulder_er_ir_speed_reps(
    df,
    position_col="Position_Deg",
    time_col="Elapsed Seconds",
    lowpass_cutoff_hz=1.0,
    min_samples=15,
    buffer_samples=0,
    drop_fraction=0.10,
):
    if df.empty or position_col not in df.columns:
        return [], None

    position_values = pd.to_numeric(df[position_col], errors="coerce").to_numpy(dtype=float)
    if len(position_values) < 15 or np.all(np.isnan(position_values)):
        return [], None

    if time_col in df.columns:
        time_values = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    else:
        time_values = np.arange(len(position_values), dtype=float)

    smooth_position, fs, clean_position = lowpass_butterworth_position_signal(
        time_values,
        position_values,
        lowpass_cutoff_hz=float(lowpass_cutoff_hz),
    )
    smooth_position = np.asarray(smooth_position, dtype=float)
    if len(smooth_position) == 0 or not np.isfinite(smooth_position).any():
        return [], None

    plateau_value = float(np.nanmedian(smooth_position))
    low_reference = float(np.nanpercentile(smooth_position, 5))
    high_reference = float(np.nanpercentile(smooth_position, 95))
    downward_excursion = max(0.0, plateau_value - low_reference)
    upward_excursion = max(0.0, high_reference - plateau_value)

    excursion_direction = "up" if upward_excursion >= downward_excursion else "down"
    excursion_amplitude = max(1.0, upward_excursion if excursion_direction == "up" else downward_excursion)
    threshold_depth = float(drop_fraction)

    if excursion_direction == "up":
        active_threshold = plateau_value + (excursion_amplitude * threshold_depth)
        return_threshold = plateau_value + (excursion_amplitude * 0.04)
        active_idx = np.flatnonzero(smooth_position >= active_threshold)
    else:
        active_threshold = plateau_value - (excursion_amplitude * threshold_depth)
        return_threshold = plateau_value - (excursion_amplitude * 0.04)
        active_idx = np.flatnonzero(smooth_position <= active_threshold)

    if active_idx.size == 0:
        return [], {
            "smooth_position": smooth_position,
            "active_threshold": float(active_threshold),
            "return_threshold": float(return_threshold),
            "plateau_value": float(plateau_value),
            "excursion_direction": excursion_direction,
            "excursion_amplitude": float(excursion_amplitude),
            "fs": fs,
        }

    # A second threshold marks a "true return" toward the dominant plateau. We
    # only close a rep once the filtered signal gets back near that plateau and
    # stays there briefly, which keeps multi-dip / multi-spike clusters together
    # regardless of whether the motion is flipped upward or downward.
    if fs is not None and np.isfinite(fs) and fs > 0:
        plateau_hold_samples = max(3, int(round(float(fs) * 0.08)))
    else:
        plateau_hold_samples = max(3, int(min_samples) // 2)

    raw_regions = _build_contiguous_regions(active_idx.tolist())
    rep_windows = []
    region_cursor = 0
    n_samples = len(smooth_position)

    while region_cursor < len(raw_regions):
        region_start, region_end = raw_regions[region_cursor]
        cluster_start = int(region_start)
        cluster_end = int(region_end)

        search_idx = int(cluster_end) + 1
        while search_idx < n_samples:
            plateau_window_end = min(n_samples, search_idx + plateau_hold_samples)
            if plateau_window_end - search_idx < plateau_hold_samples:
                cluster_end = n_samples - 1
                break

            plateau_window = smooth_position[search_idx:plateau_window_end]
            if (
                np.all(plateau_window <= return_threshold)
                if excursion_direction == "up"
                else np.all(plateau_window >= return_threshold)
            ):
                cluster_end = int(search_idx)
                break

            next_region_idx = region_cursor + 1
            if next_region_idx < len(raw_regions):
                next_start, next_end = raw_regions[next_region_idx]
                if int(next_start) < plateau_window_end:
                    cluster_end = int(next_end)
                    region_cursor = next_region_idx
                    search_idx = int(next_end) + 1
                    continue
            search_idx += 1

        if (cluster_end - cluster_start + 1) >= int(min_samples):
            start_idx = int(cluster_start)
            end_idx = int(cluster_end)

            for idx in range(int(cluster_start), 0, -1):
                prev_val = float(smooth_position[idx - 1])
                curr_val = float(smooth_position[idx])
                if excursion_direction == "up":
                    if prev_val < active_threshold <= curr_val:
                        start_idx = int(idx - 1)
                        break
                else:
                    if prev_val > active_threshold >= curr_val:
                        start_idx = int(idx - 1)
                        break

            buffered_start = max(0, int(start_idx) - int(buffer_samples))
            buffered_end = min(n_samples - 1, int(end_idx) + int(buffer_samples))
            if buffered_end > buffered_start:
                rep_windows.append((buffered_start, buffered_end))

        region_cursor += 1

    return rep_windows, {
        "smooth_position": smooth_position,
        "active_threshold": float(active_threshold),
        "return_threshold": float(return_threshold),
        "plateau_value": float(plateau_value),
        "excursion_direction": excursion_direction,
        "excursion_amplitude": float(excursion_amplitude),
        "fs": fs,
    }

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

def detect_shoulder_er_ir_speed_rep_landmarks(rep_df, value_col="Torque_Nm", prominence_ratio=0.12):
    if rep_df.empty or value_col not in rep_df.columns:
        return None

    torque_values = pd.to_numeric(rep_df[value_col], errors="coerce").to_numpy(dtype=float)
    if len(torque_values) < 11 or np.all(np.isnan(torque_values)):
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

    min_distance = max(1, len(smooth_values) // 18)
    prominence = max(1.0, amplitude_span * float(prominence_ratio))

    pos_peaks, pos_props = find_peaks(smooth_values, prominence=prominence, distance=min_distance)
    neg_peaks, neg_props = find_peaks(-smooth_values, prominence=prominence, distance=min_distance)

    if len(pos_peaks) < 3 or len(neg_peaks) < 3:
        return None

    top_pos_idx = np.argsort(pos_props["prominences"])[-3:]
    top_neg_idx = np.argsort(neg_props["prominences"])[-3:]

    landmark_pairs = []
    for idx in top_pos_idx:
        landmark_pairs.append((int(pos_peaks[idx]), "pos"))
    for idx in top_neg_idx:
        landmark_pairs.append((int(neg_peaks[idx]), "neg"))

    landmark_pairs = sorted(landmark_pairs, key=lambda item: item[0])
    landmark_indices = [idx for idx, _kind in landmark_pairs]
    if any(b <= a for a, b in zip(landmark_indices, landmark_indices[1:])):
        return None

    return {
        "indices": landmark_indices,
        "kinds": [kind for _idx, kind in landmark_pairs],
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

    landmark_counts = {}
    landmark_labels = []
    for kind in aligned_rep_metadata[0]["landmark_kinds"]:
        landmark_counts[kind] = landmark_counts.get(kind, 0) + 1
        landmark_labels.append(f"{kind.upper()}{landmark_counts[kind]}")
    mean_df.attrs["landmark_boundary_pct"] = boundary_pct[1:-1].tolist()
    mean_df.attrs["landmark_labels"] = landmark_labels

    return reps_long_df, mean_df, aligned_rep_metadata

def extract_position_window_normalized_biodex_reps(
    df,
    rep_windows,
    time_col="Elapsed Seconds",
    value_col="Torque_Nm",
    n_points=101,
):
    if df.empty or not rep_windows:
        return pd.DataFrame(), pd.DataFrame(), []

    percent_axis = np.linspace(0.0, 100.0, int(n_points))
    rep_rows = []
    normalized_curves = []
    aligned_rep_metadata = []

    for rep_number, (start_idx, end_idx) in enumerate(rep_windows, start=1):
        rep_df = df.iloc[int(start_idx):int(end_idx) + 1].copy()
        if rep_df.empty:
            continue

        rep_df[time_col] = pd.to_numeric(rep_df[time_col], errors="coerce")
        rep_df[value_col] = pd.to_numeric(rep_df[value_col], errors="coerce")
        rep_df = rep_df.dropna(subset=[time_col, value_col]).reset_index(drop=True)
        if len(rep_df) < 5:
            continue

        torque_values = rep_df[value_col].to_numpy(dtype=float)
        rep_pct = np.linspace(0.0, 100.0, len(rep_df))
        interp_torque = np.interp(percent_axis, rep_pct, torque_values)
        normalized_curves.append(interp_torque)
        rep_rows.append(pd.DataFrame({
            "rep_number": rep_number,
            "movement_pct": percent_axis,
            "torque_nm": interp_torque,
        }))
        aligned_rep_metadata.append({
            "rep_number": rep_number,
            "start_time": float(rep_df[time_col].iloc[0]),
            "end_time": float(rep_df[time_col].iloc[-1]),
        })

    if not normalized_curves:
        return pd.DataFrame(), pd.DataFrame(), []

    curves_arr = np.vstack(normalized_curves)
    reps_long_df = pd.concat(rep_rows, ignore_index=True)
    mean_df = pd.DataFrame({
        "movement_pct": percent_axis,
        "mean_torque_nm": np.nanmean(curves_arr, axis=0),
        "std_torque_nm": np.nanstd(curves_arr, axis=0),
    })
    mean_df["upper_band"] = mean_df["mean_torque_nm"] + mean_df["std_torque_nm"]
    mean_df["lower_band"] = mean_df["mean_torque_nm"] - mean_df["std_torque_nm"]
    mean_df.attrs["landmark_boundary_pct"] = []
    mean_df.attrs["landmark_labels"] = []
    mean_df.attrs["x_axis_title"] = "Rep Window (%)"
    mean_df.attrs["title"] = "Position Start -> End Normalized Torque Comparison"
    return reps_long_df, mean_df, aligned_rep_metadata

def extract_position_window_subcycle_biodex_reps(
    df,
    rep_windows,
    time_col="Elapsed Seconds",
    value_col="Torque_Nm",
    n_points=101,
    n_subcycles=3,
):
    if df.empty or not rep_windows or int(n_subcycles) < 1:
        return pd.DataFrame(), pd.DataFrame()

    percent_axis = np.linspace(0.0, 100.0, int(n_points))
    subcycle_rows = []
    mean_rows = []

    for cycle_number in range(1, int(n_subcycles) + 1):
        cycle_curves = []
        cycle_rep_rows = []
        cycle_start_pct = ((cycle_number - 1) / float(n_subcycles)) * 100.0
        cycle_end_pct = (cycle_number / float(n_subcycles)) * 100.0

        for rep_number, (start_idx, end_idx) in enumerate(rep_windows, start=1):
            rep_df = df.iloc[int(start_idx):int(end_idx) + 1].copy()
            if rep_df.empty:
                continue

            rep_df[time_col] = pd.to_numeric(rep_df[time_col], errors="coerce")
            rep_df[value_col] = pd.to_numeric(rep_df[value_col], errors="coerce")
            rep_df = rep_df.dropna(subset=[time_col, value_col]).reset_index(drop=True)
            if len(rep_df) < 7:
                continue

            torque_values = rep_df[value_col].to_numpy(dtype=float)
            rep_pct = np.linspace(0.0, 100.0, len(rep_df))
            cycle_mask = (rep_pct >= cycle_start_pct) & (rep_pct <= cycle_end_pct)
            if np.count_nonzero(cycle_mask) < 3:
                continue

            cycle_pct = rep_pct[cycle_mask]
            cycle_torque = torque_values[cycle_mask]
            local_pct = np.interp(cycle_pct, [cycle_start_pct, cycle_end_pct], [0.0, 100.0])
            interp_torque = np.interp(percent_axis, local_pct, cycle_torque)
            cycle_curves.append(interp_torque)
            cycle_rep_rows.append(pd.DataFrame({
                "rep_number": rep_number,
                "cycle_number": cycle_number,
                "cycle_pct": percent_axis,
                "torque_nm": interp_torque,
            }))

        if cycle_curves:
            cycle_arr = np.vstack(cycle_curves)
            subcycle_rows.extend(cycle_rep_rows)
            mean_rows.append(pd.DataFrame({
                "cycle_number": cycle_number,
                "cycle_pct": percent_axis,
                "mean_torque_nm": np.nanmean(cycle_arr, axis=0),
                "std_torque_nm": np.nanstd(cycle_arr, axis=0),
            }))

    if not subcycle_rows or not mean_rows:
        return pd.DataFrame(), pd.DataFrame()

    subcycles_long_df = pd.concat(subcycle_rows, ignore_index=True)
    subcycle_mean_df = pd.concat(mean_rows, ignore_index=True)
    subcycle_mean_df["upper_band"] = subcycle_mean_df["mean_torque_nm"] + subcycle_mean_df["std_torque_nm"]
    subcycle_mean_df["lower_band"] = subcycle_mean_df["mean_torque_nm"] - subcycle_mean_df["std_torque_nm"]
    return subcycles_long_df, subcycle_mean_df

def extract_shoulder_er_ir_speed_landmark_aligned_biodex_reps(
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
        if len(rep_df) < 11:
            continue

        landmark_info = detect_shoulder_er_ir_speed_rep_landmarks(
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

    landmark_counts = {}
    landmark_labels = []
    for kind in aligned_rep_metadata[0]["landmark_kinds"]:
        landmark_counts[kind] = landmark_counts.get(kind, 0) + 1
        landmark_labels.append(f"{kind.upper()}{landmark_counts[kind]}")

    mean_df.attrs["landmark_boundary_pct"] = boundary_pct[1:-1].tolist()
    mean_df.attrs["landmark_labels"] = landmark_labels

    return reps_long_df, mean_df, aligned_rep_metadata

def extract_single_rep_file_aligned_curves(
    preview_items,
    anchor_mode="peak_negative_torque",
    value_col="Torque_Nm",
    time_col="Elapsed Seconds",
    n_points=201,
    x_axis_mode="raw_time",
    common_rom_end_tolerance_deg=2.5,
    common_rom_end_hold_time_seconds=0.08,
    rom_display_lowpass_cutoff_hz=None,
):
    if not preview_items:
        return pd.DataFrame(), pd.DataFrame(), [], pd.DataFrame(), pd.DataFrame()

    aligned_reps = []
    for rep_number, item in enumerate(preview_items, start=1):
        rep_df = item["df"].copy()
        if value_col not in rep_df.columns or time_col not in rep_df.columns:
            continue

        rep_df[time_col] = pd.to_numeric(rep_df[time_col], errors="coerce")
        rep_df[value_col] = pd.to_numeric(rep_df[value_col], errors="coerce")
        if "Position_Deg" in rep_df.columns:
            rep_df["Position_Deg"] = pd.to_numeric(rep_df["Position_Deg"], errors="coerce")
        rep_df = rep_df.dropna(subset=[time_col, value_col]).reset_index(drop=True)
        if len(rep_df) < 5:
            continue

        time_values = rep_df[time_col].to_numpy(dtype=float)
        torque_values = rep_df[value_col].to_numpy(dtype=float)
        position_values = (
            rep_df["Position_Deg"].to_numpy(dtype=float)
            if "Position_Deg" in rep_df.columns and rep_df["Position_Deg"].notna().any()
            else None
        )
        if position_values is not None:
            finite_position_mask = np.isfinite(position_values)
            if not finite_position_mask.any():
                position_values = None
            elif not finite_position_mask.all():
                valid_idx = np.flatnonzero(finite_position_mask)
                cleaned_position = position_values.copy()
                cleaned_position[~finite_position_mask] = np.interp(
                    np.flatnonzero(~finite_position_mask),
                    valid_idx,
                    position_values[finite_position_mask],
                )
                position_values = cleaned_position

        rom_display_position_values = position_values
        if position_values is not None and rom_display_lowpass_cutoff_hz is not None:
            filtered_position_values, _filtered_fs, _clean_position = smooth_position_deg_signal(
                time_values,
                position_values,
                lowpass_cutoff_hz=float(rom_display_lowpass_cutoff_hz),
            )
            if filtered_position_values is not None:
                rom_display_position_values = np.asarray(filtered_position_values, dtype=float)

        position_rep_bounds = (
            detect_position_deg_rep_bounds(time_values, position_values)
            if position_values is not None
            else None
        )
        filtered_position_rep_bounds = (
            detect_position_deg_rep_bounds(
                time_values,
                position_values,
                lowpass_cutoff_hz=1.0,
            )
            if position_values is not None
            else None
        )
        position_start_idx = (
            int(position_rep_bounds["start_idx"])
            if position_rep_bounds is not None
            else 0
        )
        position_end_idx = (
            int(position_rep_bounds["end_idx"])
            if position_rep_bounds is not None
            else max(0, len(torque_values) - 1)
        )
        if position_end_idx <= position_start_idx:
            position_start_idx = 0
            position_end_idx = max(0, len(torque_values) - 1)

        filtered_position_start_idx = (
            int(filtered_position_rep_bounds["start_idx"])
            if filtered_position_rep_bounds is not None
            else position_start_idx
        )
        filtered_position_end_idx = (
            int(filtered_position_rep_bounds["end_idx"])
            if filtered_position_rep_bounds is not None
            else position_end_idx
        )
        if filtered_position_end_idx <= filtered_position_start_idx:
            filtered_position_start_idx = position_start_idx
            filtered_position_end_idx = position_end_idx

        manual_rom_end = item.get("manual_rom_end") or {}
        manual_position_end_idx = None
        manual_position_end_time = None
        if manual_rom_end:
            manual_time_seconds = manual_rom_end.get("time_seconds")
            if manual_time_seconds is not None and np.isfinite(manual_time_seconds):
                manual_position_end_idx = int(np.argmin(np.abs(time_values - float(manual_time_seconds))))
            else:
                manual_sample_index = manual_rom_end.get("sample_index")
                if manual_sample_index is not None:
                    manual_position_end_idx = min(
                        max(int(manual_sample_index) - 1, 0),
                        max(len(torque_values) - 1, 0),
                    )
            if manual_position_end_idx is not None:
                if manual_position_end_idx <= position_start_idx:
                    manual_position_end_idx = None
                else:
                    position_end_idx = int(manual_position_end_idx)
                    filtered_position_end_idx = int(manual_position_end_idx)
                    manual_position_end_time = float(time_values[manual_position_end_idx])

        peak_pos_idx = int(np.argmax(torque_values))
        zero_torque_rise_idx = peak_pos_idx
        for idx in range(peak_pos_idx, 0, -1):
            prev_val = float(torque_values[idx - 1])
            curr_val = float(torque_values[idx])
            if prev_val <= 0.0 < curr_val:
                zero_torque_rise_idx = idx
                break
        zero_torque_fall_idx = peak_pos_idx
        for idx in range(peak_pos_idx + 1, len(torque_values)):
            prev_val = float(torque_values[idx - 1])
            curr_val = float(torque_values[idx])
            if prev_val >= 0.0 > curr_val:
                zero_torque_fall_idx = idx
                break

        rom_plateau_idx = None
        if position_values is not None and len(position_values) >= 7:
            smooth_position_window = min(len(position_values), 11)
            if smooth_position_window % 2 == 0:
                smooth_position_window = max(1, smooth_position_window - 1)
            if smooth_position_window >= 5:
                smooth_position = savgol_filter(position_values, window_length=smooth_position_window, polyorder=min(3, smooth_position_window - 2))
            else:
                smooth_position = position_values

            peak_position_idx = int(np.argmax(smooth_position))
            peak_position_value = float(smooth_position[peak_position_idx])
            plateau_threshold = peak_position_value * 0.95
            plateau_tolerance = max(3.0, abs(peak_position_value) * 0.03)
            sustain_needed = 3

            for idx in range(peak_position_idx, -1, -1):
                window_end = min(len(smooth_position), idx + sustain_needed)
                if window_end - idx < sustain_needed:
                    continue
                within_plateau = smooth_position[idx:window_end] >= plateau_threshold
                stable_band = np.abs(smooth_position[idx:window_end] - peak_position_value) <= plateau_tolerance
                if np.all(within_plateau) and np.all(stable_band):
                    rom_plateau_idx = idx
                elif rom_plateau_idx is not None:
                    break

        if anchor_mode == "peak_positive_torque":
            anchor_idx = int(np.argmax(torque_values))
            anchor_label = "Peak Positive Torque"
        elif anchor_mode == "zero_torque_rise":
            anchor_idx = zero_torque_rise_idx
            anchor_label = "0 Torque Rise"
        elif anchor_mode == "positive_rise_onset":
            peak_pos_idx = int(np.argmax(torque_values))
            baseline_window = max(3, min(len(torque_values), 15))
            baseline_value = float(np.nanmedian(torque_values[:baseline_window]))
            onset_threshold = baseline_value + max(2.0, np.nanstd(torque_values[:baseline_window]) * 1.5)
            anchor_idx = peak_pos_idx
            sustain_needed = 3
            for idx in range(peak_pos_idx, -1, -1):
                window_end = min(len(torque_values), idx + sustain_needed)
                if np.all(torque_values[idx:window_end] >= onset_threshold):
                    anchor_idx = idx
                elif anchor_idx != peak_pos_idx:
                    break
            anchor_label = "Positive Rise Onset"
        elif anchor_mode == "negative_torque_onset":
            peak_neg_idx = int(np.argmin(torque_values))
            baseline_window = max(3, min(len(torque_values), 15))
            baseline_value = float(np.nanmedian(torque_values[:baseline_window]))
            onset_threshold = baseline_value - max(2.0, np.nanstd(torque_values[:baseline_window]) * 1.5)
            anchor_idx = peak_neg_idx
            sustain_needed = 3
            for idx in range(peak_neg_idx, -1, -1):
                window_end = min(len(torque_values), idx + sustain_needed)
                if np.all(torque_values[idx:window_end] <= onset_threshold):
                    anchor_idx = idx
                elif anchor_idx != peak_neg_idx:
                    break
            anchor_label = "Negative Torque Onset"
        else:
            anchor_idx = int(np.argmin(torque_values))
            anchor_label = "Peak Negative Torque"

        anchor_time = float(time_values[anchor_idx])
        zero_torque_rise_time = float(time_values[zero_torque_rise_idx])
        peak_positive_time = float(time_values[peak_pos_idx])
        zero_to_peak_duration = peak_positive_time - zero_torque_rise_time
        aligned_reps.append({
            "rep_number": rep_number,
            "file_name": item["name"],
            "aligned_time": time_values - anchor_time,
            "sample_pct": np.linspace(0.0, 100.0, len(rep_df)),
            "torque_values": torque_values,
            "position_values": position_values,
            "rom_display_position_values": rom_display_position_values,
            "smooth_position_values": (
                np.asarray(position_rep_bounds["smooth_position"], dtype=float)
                if position_rep_bounds is not None
                else None
            ),
            "position_fs": (
                float(position_rep_bounds["fs"])
                if position_rep_bounds is not None and position_rep_bounds.get("fs") is not None
                else None
            ),
            "final_plateau_value": (
                float(position_rep_bounds["final_plateau_value"])
                if position_rep_bounds is not None and position_rep_bounds.get("final_plateau_value") is not None
                else None
            ),
            "anchor_idx": anchor_idx,
            "zero_torque_rise_idx": zero_torque_rise_idx,
            "peak_positive_idx": peak_pos_idx,
            "zero_torque_fall_idx": zero_torque_fall_idx,
            "rom_plateau_idx": rom_plateau_idx,
            "position_start_idx": position_start_idx,
            "position_end_idx": position_end_idx,
            "filtered_position_start_idx": filtered_position_start_idx,
            "filtered_position_end_idx": filtered_position_end_idx,
            "manual_position_end_idx": manual_position_end_idx,
            "manual_position_end_time": manual_position_end_time,
            "anchor_time": anchor_time,
            "anchor_torque": float(torque_values[anchor_idx]),
            "anchor_label": anchor_label,
            "zero_torque_rise_time": zero_torque_rise_time,
            "peak_positive_time": peak_positive_time,
            "zero_to_peak_duration_s": float(zero_to_peak_duration),
        })

    if not aligned_reps:
        return pd.DataFrame(), pd.DataFrame(), [], pd.DataFrame(), pd.DataFrame()

    rep_rows = []
    interpolated_curves = []
    rom_rep_rows = []
    interpolated_rom_curves = []
    if x_axis_mode == "zero_to_position_136_ascent_normalized":
        percent_axis = np.linspace(0.0, 100.0, int(n_points))
        target_angle = 136.0
        valid_segment_reps = []
        for rep in aligned_reps:
            zero_idx = int(rep["zero_torque_rise_idx"])
            smooth_position_values = rep.get("smooth_position_values")
            position_end_idx = find_first_position_ascent_threshold(
                smooth_position_values,
                zero_idx,
                target_angle,
            )
            if position_end_idx is None or position_end_idx <= zero_idx:
                continue
            valid_segment_reps.append((rep, zero_idx, position_end_idx))

        if not valid_segment_reps:
            return pd.DataFrame(), pd.DataFrame(), aligned_reps, pd.DataFrame(), pd.DataFrame()

        for rep, zero_idx, position_end_idx in valid_segment_reps:
            torque_window = rep["torque_values"][zero_idx:position_end_idx + 1]
            interp_torque = np.interp(
                percent_axis,
                np.linspace(0.0, 100.0, len(torque_window)),
                torque_window,
            )
            interpolated_curves.append(interp_torque)
            rep_rows.append(pd.DataFrame({
                "rep_number": rep["rep_number"],
                "file_name": rep["file_name"],
                "alignment_x": percent_axis,
                "torque_nm": interp_torque,
            }))
            display_position_values = rep.get("rom_display_position_values")
            if display_position_values is not None:
                position_window = display_position_values[zero_idx:position_end_idx + 1]
                interp_position = np.interp(
                    percent_axis,
                    np.linspace(0.0, 100.0, len(position_window)),
                    position_window,
                )
                interpolated_rom_curves.append(interp_position)
                rom_rep_rows.append(pd.DataFrame({
                    "rep_number": rep["rep_number"],
                    "file_name": rep["file_name"],
                    "alignment_x": percent_axis,
                    "position_deg": interp_position,
                }))

        mean_df = pd.DataFrame({
            "alignment_x": percent_axis,
            "mean_torque_nm": np.nanmean(np.vstack(interpolated_curves), axis=0),
            "std_torque_nm": np.nanstd(np.vstack(interpolated_curves), axis=0),
        })
        mean_df.attrs["x_axis_mode"] = "zero_to_position_136_ascent_normalized"
        mean_df.attrs["x_axis_title"] = "0 Torque Rise to 136° Ascent (%)"
        mean_df.attrs["anchor_x"] = 0.0
        mean_df.attrs["anchor_label"] = "0 Torque Rise"
        mean_df.attrs["secondary_anchor_x"] = 100.0
        mean_df.attrs["secondary_anchor_label"] = "136° On Ascent"
    elif x_axis_mode == "zero_to_common_smoothed_rom_end_normalized":
        percent_axis = np.linspace(0.0, 100.0, int(n_points))
        common_plateau_values = [
            float(rep["final_plateau_value"])
            for rep in aligned_reps
            if rep.get("final_plateau_value") is not None and np.isfinite(rep["final_plateau_value"])
        ]
        if not common_plateau_values:
            return pd.DataFrame(), pd.DataFrame(), aligned_reps, pd.DataFrame(), pd.DataFrame()

        common_plateau_angle = float(np.nanmedian(common_plateau_values))
        common_angle_tolerance = float(common_rom_end_tolerance_deg)
        valid_segment_reps = []
        for rep in aligned_reps:
            zero_idx = int(rep["zero_torque_rise_idx"])
            manual_position_end_idx = rep.get("manual_position_end_idx")
            if manual_position_end_idx is not None:
                position_end_idx = int(manual_position_end_idx)
            else:
                smooth_position_values = rep.get("smooth_position_values")
                position_end_idx = find_first_common_rom_band_entry(
                    smooth_position_values,
                    zero_idx,
                    rep.get("position_fs"),
                    common_plateau_angle,
                    angle_tolerance_deg=common_angle_tolerance,
                    velocity_tolerance_deg_per_second=15.0,
                    hold_time_seconds=float(common_rom_end_hold_time_seconds),
                )
            if position_end_idx is None:
                position_end_idx = int(rep["position_end_idx"])
            if position_end_idx <= zero_idx:
                continue
            valid_segment_reps.append((rep, zero_idx, position_end_idx))

        if not valid_segment_reps:
            return pd.DataFrame(), pd.DataFrame(), aligned_reps, pd.DataFrame(), pd.DataFrame()

        for rep, zero_idx, position_end_idx in valid_segment_reps:
            torque_window = rep["torque_values"][zero_idx:position_end_idx + 1]
            interp_torque = np.interp(
                percent_axis,
                np.linspace(0.0, 100.0, len(torque_window)),
                torque_window,
            )
            interpolated_curves.append(interp_torque)
            rep_rows.append(pd.DataFrame({
                "rep_number": rep["rep_number"],
                "file_name": rep["file_name"],
                "alignment_x": percent_axis,
                "torque_nm": interp_torque,
            }))
            display_position_values = rep.get("rom_display_position_values")
            if display_position_values is not None:
                position_window = display_position_values[zero_idx:position_end_idx + 1]
                interp_position = np.interp(
                    percent_axis,
                    np.linspace(0.0, 100.0, len(position_window)),
                    position_window,
                )
                interpolated_rom_curves.append(interp_position)
                rom_rep_rows.append(pd.DataFrame({
                    "rep_number": rep["rep_number"],
                    "file_name": rep["file_name"],
                    "alignment_x": percent_axis,
                    "position_deg": interp_position,
                }))

        mean_df = pd.DataFrame({
            "alignment_x": percent_axis,
            "mean_torque_nm": np.nanmean(np.vstack(interpolated_curves), axis=0),
            "std_torque_nm": np.nanstd(np.vstack(interpolated_curves), axis=0),
        })
        mean_df.attrs["x_axis_mode"] = "zero_to_common_smoothed_rom_end_normalized"
        mean_df.attrs["x_axis_title"] = "0 Torque Rise to Stabilized Peak ROM End (%)"
        mean_df.attrs["anchor_x"] = 0.0
        mean_df.attrs["anchor_label"] = "0 Torque Rise"
        mean_df.attrs["secondary_anchor_x"] = 100.0
        mean_df.attrs["secondary_anchor_label"] = "Stabilized Peak ROM End"
        mean_df.attrs["common_plateau_angle"] = common_plateau_angle
    elif x_axis_mode == "filtered_position_window_normalized":
        percent_axis = np.linspace(0.0, 100.0, int(n_points))
        segment_fraction_rows = []
        valid_segment_reps = []

        for rep in aligned_reps:
            start_pct = (float(rep["filtered_position_start_idx"]) / max(1.0, float(len(rep["torque_values"]) - 1))) * 100.0
            end_pct = (float(rep["filtered_position_end_idx"]) / max(1.0, float(len(rep["torque_values"]) - 1))) * 100.0
            if not (0.0 <= start_pct < end_pct <= 100.0):
                continue
            segment_fraction_rows.append(np.array([
                start_pct,
                end_pct - start_pct,
                100.0 - end_pct,
            ], dtype=float) / 100.0)
            valid_segment_reps.append((rep, start_pct, end_pct))

        if not valid_segment_reps:
            return pd.DataFrame(), pd.DataFrame(), aligned_reps, pd.DataFrame(), pd.DataFrame()

        median_segment_fractions = np.nanmedian(np.vstack(segment_fraction_rows), axis=0)
        median_segment_fractions = median_segment_fractions / median_segment_fractions.sum()
        start_target_pct = float(median_segment_fractions[0] * 100.0)
        end_target_pct = float((median_segment_fractions[0] + median_segment_fractions[1]) * 100.0)

        for rep, start_pct, end_pct in valid_segment_reps:
            mapped_pct = np.interp(
                rep["sample_pct"],
                [0.0, start_pct, end_pct, 100.0],
                [0.0, start_target_pct, end_target_pct, 100.0],
            )
            interp_torque = np.interp(percent_axis, mapped_pct, rep["torque_values"])
            interpolated_curves.append(interp_torque)
            rep_rows.append(pd.DataFrame({
                "rep_number": rep["rep_number"],
                "file_name": rep["file_name"],
                "alignment_x": percent_axis,
                "torque_nm": interp_torque,
            }))
            display_position_values = rep.get("rom_display_position_values")
            if display_position_values is not None:
                interp_position = np.interp(percent_axis, mapped_pct, display_position_values)
                interpolated_rom_curves.append(interp_position)
                rom_rep_rows.append(pd.DataFrame({
                    "rep_number": rep["rep_number"],
                    "file_name": rep["file_name"],
                    "alignment_x": percent_axis,
                    "position_deg": interp_position,
                }))

        mean_df = pd.DataFrame({
            "alignment_x": percent_axis,
            "mean_torque_nm": np.nanmean(np.vstack(interpolated_curves), axis=0),
            "std_torque_nm": np.nanstd(np.vstack(interpolated_curves), axis=0),
        })
        mean_df.attrs["x_axis_mode"] = "filtered_position_window_normalized"
        mean_df.attrs["x_axis_title"] = "Filtered ROM Window (%)"
        mean_df.attrs["anchor_x"] = start_target_pct
        mean_df.attrs["anchor_label"] = "Filtered ROM Start"
        mean_df.attrs["secondary_anchor_x"] = end_target_pct
        mean_df.attrs["secondary_anchor_label"] = "Filtered ROM End"
    elif x_axis_mode == "position_window_normalized":
        percent_axis = np.linspace(0.0, 100.0, int(n_points))
        segment_fraction_rows = []
        valid_segment_reps = []

        for rep in aligned_reps:
            start_pct = (float(rep["position_start_idx"]) / max(1.0, float(len(rep["torque_values"]) - 1))) * 100.0
            end_pct = (float(rep["position_end_idx"]) / max(1.0, float(len(rep["torque_values"]) - 1))) * 100.0
            if not (0.0 <= start_pct < end_pct <= 100.0):
                continue
            segment_fraction_rows.append(np.array([
                start_pct,
                end_pct - start_pct,
                100.0 - end_pct,
            ], dtype=float) / 100.0)
            valid_segment_reps.append((rep, start_pct, end_pct))

        if not valid_segment_reps:
            return pd.DataFrame(), pd.DataFrame(), aligned_reps, pd.DataFrame(), pd.DataFrame()

        median_segment_fractions = np.nanmedian(np.vstack(segment_fraction_rows), axis=0)
        median_segment_fractions = median_segment_fractions / median_segment_fractions.sum()
        start_target_pct = float(median_segment_fractions[0] * 100.0)
        end_target_pct = float((median_segment_fractions[0] + median_segment_fractions[1]) * 100.0)

        for rep, start_pct, end_pct in valid_segment_reps:
            mapped_pct = np.interp(
                rep["sample_pct"],
                [0.0, start_pct, end_pct, 100.0],
                [0.0, start_target_pct, end_target_pct, 100.0],
            )
            interp_torque = np.interp(percent_axis, mapped_pct, rep["torque_values"])
            interpolated_curves.append(interp_torque)
            rep_rows.append(pd.DataFrame({
                "rep_number": rep["rep_number"],
                "file_name": rep["file_name"],
                "alignment_x": percent_axis,
                "torque_nm": interp_torque,
            }))
            display_position_values = rep.get("rom_display_position_values")
            if display_position_values is not None:
                interp_position = np.interp(percent_axis, mapped_pct, display_position_values)
                interpolated_rom_curves.append(interp_position)
                rom_rep_rows.append(pd.DataFrame({
                    "rep_number": rep["rep_number"],
                    "file_name": rep["file_name"],
                    "alignment_x": percent_axis,
                    "position_deg": interp_position,
                }))

        mean_df = pd.DataFrame({
            "alignment_x": percent_axis,
            "mean_torque_nm": np.nanmean(np.vstack(interpolated_curves), axis=0),
            "std_torque_nm": np.nanstd(np.vstack(interpolated_curves), axis=0),
        })
        mean_df.attrs["x_axis_mode"] = "position_window_normalized"
        mean_df.attrs["x_axis_title"] = "Normalized Rep Duration (%)"
        mean_df.attrs["anchor_x"] = start_target_pct
        mean_df.attrs["anchor_label"] = "Position Start"
        mean_df.attrs["secondary_anchor_x"] = end_target_pct
        mean_df.attrs["secondary_anchor_label"] = "Position End"
    elif x_axis_mode == "zero_to_peak_normalized":
        percent_axis = np.linspace(0.0, 100.0, int(n_points))
        segment_fraction_rows = []
        valid_segment_reps = []

        for rep in aligned_reps:
            zero_pct = (float(rep["zero_torque_rise_idx"]) / max(1.0, float(len(rep["torque_values"]) - 1))) * 100.0
            peak_pct = (float(rep["peak_positive_idx"]) / max(1.0, float(len(rep["torque_values"]) - 1))) * 100.0
            zero_fall_pct = (float(rep["zero_torque_fall_idx"]) / max(1.0, float(len(rep["torque_values"]) - 1))) * 100.0
            if not (0.0 < zero_pct < peak_pct < zero_fall_pct <= 100.0):
                continue
            segment_fraction_rows.append(np.array([
                zero_pct,
                peak_pct - zero_pct,
                zero_fall_pct - peak_pct,
                100.0 - zero_fall_pct,
            ], dtype=float) / 100.0)
            valid_segment_reps.append((rep, zero_pct, peak_pct, zero_fall_pct))

        if not valid_segment_reps:
            return pd.DataFrame(), pd.DataFrame(), aligned_reps, pd.DataFrame(), pd.DataFrame()

        median_segment_fractions = np.nanmedian(np.vstack(segment_fraction_rows), axis=0)
        median_segment_fractions = median_segment_fractions / median_segment_fractions.sum()
        zero_target_pct = float(median_segment_fractions[0] * 100.0)
        peak_target_pct = float((median_segment_fractions[0] + median_segment_fractions[1]) * 100.0)
        zero_fall_target_pct = float((median_segment_fractions[0] + median_segment_fractions[1] + median_segment_fractions[2]) * 100.0)

        for rep, zero_pct, peak_pct, zero_fall_pct in valid_segment_reps:
            mapped_pct = np.interp(
                rep["sample_pct"],
                [0.0, zero_pct, peak_pct, zero_fall_pct, 100.0],
                [0.0, zero_target_pct, peak_target_pct, zero_fall_target_pct, 100.0],
            )
            interp_torque = np.interp(percent_axis, mapped_pct, rep["torque_values"])
            interpolated_curves.append(interp_torque)
            rep_rows.append(pd.DataFrame({
                "rep_number": rep["rep_number"],
                "file_name": rep["file_name"],
                "alignment_x": percent_axis,
                "torque_nm": interp_torque,
            }))
            display_position_values = rep.get("rom_display_position_values")
            if display_position_values is not None:
                interp_position = np.interp(percent_axis, mapped_pct, display_position_values)
                interpolated_rom_curves.append(interp_position)
                rom_rep_rows.append(pd.DataFrame({
                    "rep_number": rep["rep_number"],
                    "file_name": rep["file_name"],
                    "alignment_x": percent_axis,
                    "position_deg": interp_position,
                }))

        mean_df = pd.DataFrame({
            "alignment_x": percent_axis,
            "mean_torque_nm": np.nanmean(np.vstack(interpolated_curves), axis=0),
            "std_torque_nm": np.nanstd(np.vstack(interpolated_curves), axis=0),
        })
        mean_df.attrs["x_axis_mode"] = "zero_to_peak_normalized"
        mean_df.attrs["x_axis_title"] = "Normalized Rep Duration (%)"
        mean_df.attrs["anchor_x"] = zero_target_pct
        mean_df.attrs["secondary_anchor_x"] = peak_target_pct
        mean_df.attrs["secondary_anchor_label"] = "Peak Positive Torque"
        mean_df.attrs["tertiary_anchor_x"] = zero_fall_target_pct
        mean_df.attrs["tertiary_anchor_label"] = "0 Torque Fall"
        rom_plateau_pcts = [
            (
                float(rep["rom_plateau_idx"]) / max(1.0, float(len(rep["position_values"]) - 1))
            ) * 100.0
            for rep, _zero_pct, _peak_pct, _zero_fall_pct in valid_segment_reps
            if rep["position_values"] is not None and rep["rom_plateau_idx"] is not None
        ]
        if rom_plateau_pcts:
            rom_plateau_target_pct = float(np.nanmedian(rom_plateau_pcts))
            mean_df.attrs["quaternary_anchor_x"] = rom_plateau_target_pct
            mean_df.attrs["quaternary_anchor_label"] = "ROM Plateau"
    elif x_axis_mode == "normalized_duration":
        percent_axis = np.linspace(0.0, 100.0, int(n_points))
        anchor_pcts = [
            (float(rep["anchor_idx"]) / max(1.0, float(len(rep["torque_values"]) - 1))) * 100.0
            for rep in aligned_reps
        ]
        anchor_pct_target = float(np.nanmedian(anchor_pcts))

        for rep in aligned_reps:
            anchor_pct = (float(rep["anchor_idx"]) / max(1.0, float(len(rep["torque_values"]) - 1))) * 100.0
            mapped_pct = np.interp(
                rep["sample_pct"],
                [0.0, anchor_pct, 100.0],
                [0.0, anchor_pct_target, 100.0],
            )
            interp_torque = np.interp(percent_axis, mapped_pct, rep["torque_values"])
            interpolated_curves.append(interp_torque)
            rep_rows.append(pd.DataFrame({
                "rep_number": rep["rep_number"],
                "file_name": rep["file_name"],
                "alignment_x": percent_axis,
                "torque_nm": interp_torque,
            }))
            display_position_values = rep.get("rom_display_position_values")
            if display_position_values is not None:
                interp_position = np.interp(percent_axis, mapped_pct, display_position_values)
                interpolated_rom_curves.append(interp_position)
                rom_rep_rows.append(pd.DataFrame({
                    "rep_number": rep["rep_number"],
                    "file_name": rep["file_name"],
                    "alignment_x": percent_axis,
                    "position_deg": interp_position,
                }))

        mean_df = pd.DataFrame({
            "alignment_x": percent_axis,
            "mean_torque_nm": np.nanmean(np.vstack(interpolated_curves), axis=0),
            "std_torque_nm": np.nanstd(np.vstack(interpolated_curves), axis=0),
        })
        mean_df.attrs["x_axis_mode"] = "normalized_duration"
        mean_df.attrs["x_axis_title"] = "Normalized Rep Duration (%)"
        mean_df.attrs["anchor_x"] = anchor_pct_target
    else:
        common_start = max(float(rep["aligned_time"][0]) for rep in aligned_reps)
        common_end = min(float(rep["aligned_time"][-1]) for rep in aligned_reps)
        if common_end <= common_start:
            return pd.DataFrame(), pd.DataFrame(), aligned_reps, pd.DataFrame(), pd.DataFrame()

        common_axis = np.linspace(common_start, common_end, int(n_points))
        for rep in aligned_reps:
            interp_torque = np.interp(common_axis, rep["aligned_time"], rep["torque_values"])
            interpolated_curves.append(interp_torque)
            rep_rows.append(pd.DataFrame({
                "rep_number": rep["rep_number"],
                "file_name": rep["file_name"],
                "alignment_x": common_axis,
                "torque_nm": interp_torque,
            }))
            display_position_values = rep.get("rom_display_position_values")
            if display_position_values is not None:
                interp_position = np.interp(common_axis, rep["aligned_time"], display_position_values)
                interpolated_rom_curves.append(interp_position)
                rom_rep_rows.append(pd.DataFrame({
                    "rep_number": rep["rep_number"],
                    "file_name": rep["file_name"],
                    "alignment_x": common_axis,
                    "position_deg": interp_position,
                }))

        mean_df = pd.DataFrame({
            "alignment_x": common_axis,
            "mean_torque_nm": np.nanmean(np.vstack(interpolated_curves), axis=0),
            "std_torque_nm": np.nanstd(np.vstack(interpolated_curves), axis=0),
        })
        mean_df.attrs["x_axis_mode"] = "raw_time"
        mean_df.attrs["x_axis_title"] = "Aligned Time (s)"
        mean_df.attrs["anchor_x"] = 0.0

    curves_arr = np.vstack(interpolated_curves)
    reps_long_df = pd.concat(rep_rows, ignore_index=True)
    mean_df["upper_band"] = mean_df["mean_torque_nm"] + mean_df["std_torque_nm"]
    mean_df["lower_band"] = mean_df["mean_torque_nm"] - mean_df["std_torque_nm"]
    mean_df.attrs["anchor_label"] = aligned_reps[0]["anchor_label"]

    rom_reps_long_df = (
        pd.concat(rom_rep_rows, ignore_index=True)
        if rom_rep_rows
        else pd.DataFrame()
    )
    if interpolated_rom_curves:
        rom_mean_df = pd.DataFrame({
            "alignment_x": mean_df["alignment_x"],
            "mean_position_deg": np.nanmean(np.vstack(interpolated_rom_curves), axis=0),
            "std_position_deg": np.nanstd(np.vstack(interpolated_rom_curves), axis=0),
        })
        rom_mean_df["upper_band"] = rom_mean_df["mean_position_deg"] + rom_mean_df["std_position_deg"]
        rom_mean_df["lower_band"] = rom_mean_df["mean_position_deg"] - rom_mean_df["std_position_deg"]
        rom_mean_df.attrs = mean_df.attrs.copy()
    else:
        rom_mean_df = pd.DataFrame()

    return reps_long_df, mean_df, aligned_reps, rom_reps_long_df, rom_mean_df

def detect_d2_biodex_reps(df, position_col="Position_Deg", buffer_samples=20):
    if df.empty or position_col not in df.columns:
        return []

    position_values = pd.to_numeric(df[position_col], errors="coerce").to_numpy(dtype=float)
    if len(position_values) < 15 or np.all(np.isnan(position_values)):
        return []

    finite_mask = np.isfinite(position_values)
    if not finite_mask.any():
        return []

    clean_position = position_values.copy()
    if not finite_mask.all():
        valid_idx = np.flatnonzero(finite_mask)
        clean_position[~finite_mask] = np.interp(
            np.flatnonzero(~finite_mask),
            valid_idx,
            position_values[finite_mask],
        )

    smooth_window = get_valid_savgol_window(21, len(clean_position), 3)
    if smooth_window is not None:
        smooth_position = savgol_filter(clean_position, window_length=smooth_window, polyorder=3)
    else:
        smooth_position = clean_position

    position_span = float(np.nanmax(smooth_position) - np.nanmin(smooth_position))
    if position_span <= 0:
        return []

    low_position = float(np.nanpercentile(smooth_position, 10))
    high_position = float(np.nanpercentile(smooth_position, 90))
    low_return_threshold = low_position + ((high_position - low_position) * 0.08)

    prominence = max(5.0, position_span * 0.25)
    min_distance = max(20, len(smooth_position) // 8)
    max_indices, _ = find_peaks(smooth_position, prominence=prominence, distance=min_distance)

    rep_windows = []
    n_rows = len(df)
    for max_idx in max_indices:
        if smooth_position[max_idx] < high_position:
            continue

        previous_low_idx = np.flatnonzero(smooth_position[:max_idx] <= low_return_threshold)
        next_low_idx = np.flatnonzero(smooth_position[max_idx:] <= low_return_threshold)
        if len(previous_low_idx) == 0 or len(next_low_idx) == 0:
            continue

        start_idx = int(previous_low_idx[-1])
        end_idx = int(max_idx + next_low_idx[0])
        if end_idx <= start_idx:
            continue

        buffered_start = max(0, start_idx - int(buffer_samples))
        buffered_end = min(n_rows - 1, end_idx + int(buffer_samples))
        if rep_windows and buffered_start <= rep_windows[-1][1]:
            continue
        rep_windows.append((buffered_start, buffered_end))

    return rep_windows

def detect_d2_rep_landmarks(rep_df, position_col="Position_Deg", value_col="Torque_Nm"):
    if rep_df.empty or position_col not in rep_df.columns or value_col not in rep_df.columns:
        return None

    position_values = pd.to_numeric(rep_df[position_col], errors="coerce").to_numpy(dtype=float)
    torque_values = pd.to_numeric(rep_df[value_col], errors="coerce").to_numpy(dtype=float)
    if len(rep_df) < 7 or np.all(np.isnan(position_values)) or np.all(np.isnan(torque_values)):
        return None

    position_values = pd.Series(position_values).interpolate(limit_direction="both").to_numpy(dtype=float)
    torque_values = pd.Series(torque_values).interpolate(limit_direction="both").to_numpy(dtype=float)

    position_window = get_valid_savgol_window(21, len(position_values), 3)
    if position_window is not None:
        smooth_position = savgol_filter(position_values, window_length=position_window, polyorder=3)
    else:
        smooth_position = position_values

    reversal_idx = int(np.nanargmax(smooth_position))
    if reversal_idx <= 2 or reversal_idx >= len(rep_df) - 3:
        return None

    smooth_window = get_valid_savgol_window(11, len(torque_values), 3)
    if smooth_window is not None:
        smooth_torque = savgol_filter(torque_values, window_length=smooth_window, polyorder=3)
    else:
        smooth_torque = torque_values

    torque_span = float(np.nanmax(smooth_torque) - np.nanmin(smooth_torque))
    if torque_span <= 0:
        return None

    min_distance = max(1, len(smooth_torque) // 12)
    prominence = max(5.0, torque_span * 0.10)
    pos_peaks, pos_props = find_peaks(smooth_torque, prominence=prominence, distance=min_distance)
    neg_peaks, neg_props = find_peaks(-smooth_torque, prominence=prominence, distance=min_distance)

    neg_phase_peak_mask = neg_peaks < reversal_idx
    pos_phase_peak_mask = pos_peaks > reversal_idx
    neg_phase_peaks = neg_peaks[neg_phase_peak_mask]
    pos_phase_peaks = pos_peaks[pos_phase_peak_mask]
    neg_phase_prominences = neg_props["prominences"][neg_phase_peak_mask]
    pos_phase_prominences = pos_props["prominences"][pos_phase_peak_mask]

    if len(pos_phase_peaks) < 2 or len(neg_phase_peaks) < 2:
        return None

    top_neg_indices = np.sort(neg_phase_peaks[np.argsort(neg_phase_prominences)[-2:]])
    top_pos_indices = np.sort(pos_phase_peaks[np.argsort(pos_phase_prominences)[-2:]])

    landmark_indices = [
        int(top_neg_indices[0]),
        int(top_neg_indices[1]),
        int(top_pos_indices[0]),
        int(top_pos_indices[1]),
    ]
    if any(b <= a for a, b in zip(landmark_indices, landmark_indices[1:])):
        return None

    return {
        "indices": landmark_indices,
        "kinds": ["neg", "neg", "pos", "pos"],
    }

def extract_d2_landmark_aligned_biodex_reps(
    df,
    rep_windows,
    time_col="Elapsed Seconds",
    position_col="Position_Deg",
    value_col="Torque_Nm",
    n_points=101,
):
    if df.empty or not rep_windows:
        return pd.DataFrame(), pd.DataFrame(), []

    percent_axis = np.linspace(0.0, 100.0, int(n_points))
    valid_reps = []
    phase_fraction_rows = []

    for rep_number, (start_idx, end_idx) in enumerate(rep_windows, start=1):
        rep_df = df.iloc[int(start_idx):int(end_idx) + 1].copy()
        rep_df[time_col] = pd.to_numeric(rep_df[time_col], errors="coerce")
        rep_df[position_col] = pd.to_numeric(rep_df[position_col], errors="coerce")
        rep_df[value_col] = pd.to_numeric(rep_df[value_col], errors="coerce")
        rep_df = rep_df.dropna(subset=[time_col, position_col, value_col]).reset_index(drop=True)
        if len(rep_df) < 7:
            continue

        landmark_info = detect_d2_rep_landmarks(
            rep_df,
            position_col=position_col,
            value_col=value_col,
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
    landmark_counts = {}
    landmark_labels = []
    for kind in aligned_rep_metadata[0]["landmark_kinds"]:
        landmark_counts[kind] = landmark_counts.get(kind, 0) + 1
        landmark_labels.append(f"{kind.upper()}{landmark_counts[kind]}")

    mean_df.attrs["landmark_boundary_pct"] = boundary_pct[1:-1].tolist()
    mean_df.attrs["landmark_labels"] = landmark_labels

    return reps_long_df, mean_df, aligned_rep_metadata

def detect_d2_speed_rep_landmarks(rep_df, value_col="Torque_Nm", prominence_ratio=0.10):
    if rep_df.empty or value_col not in rep_df.columns:
        return None

    torque_values = pd.to_numeric(rep_df[value_col], errors="coerce").to_numpy(dtype=float)
    if len(rep_df) < 9 or np.all(np.isnan(torque_values)):
        return None

    torque_values = pd.Series(torque_values).interpolate(limit_direction="both").to_numpy(dtype=float)
    smooth_window = get_valid_savgol_window(9, len(torque_values), 3)
    if smooth_window is not None:
        smooth_torque = savgol_filter(torque_values, window_length=smooth_window, polyorder=3)
    else:
        smooth_torque = torque_values

    torque_span = float(np.nanmax(smooth_torque) - np.nanmin(smooth_torque))
    if torque_span <= 0:
        return None

    min_distance = max(1, len(smooth_torque) // 18)
    prominence = max(5.0, torque_span * float(prominence_ratio))
    pos_peaks, pos_props = find_peaks(smooth_torque, prominence=prominence, distance=min_distance)
    neg_peaks, neg_props = find_peaks(-smooth_torque, prominence=prominence, distance=min_distance)
    if len(pos_peaks) < 3 or len(neg_peaks) < 3:
        return None

    pos_height_threshold = float(np.nanmax(smooth_torque)) * 0.35
    neg_height_threshold = float(np.nanmin(smooth_torque)) * 0.35
    pos_candidates = [int(idx) for idx in pos_peaks if smooth_torque[int(idx)] >= pos_height_threshold]
    neg_candidates = [int(idx) for idx in neg_peaks if smooth_torque[int(idx)] <= neg_height_threshold]
    if len(pos_candidates) < 3:
        top_pos = np.argsort(pos_props["prominences"])[-3:]
        pos_candidates = sorted(int(pos_peaks[idx]) for idx in top_pos)
    else:
        pos_candidates = sorted(pos_candidates[:3])
    if len(neg_candidates) < 3:
        top_neg = np.argsort(neg_props["prominences"])[-3:]
        neg_candidates = sorted(int(neg_peaks[idx]) for idx in top_neg)
    else:
        neg_candidates = sorted(neg_candidates[:3])

    landmark_pairs = None
    expected_sequence = ["pos", "neg", "pos", "neg", "pos", "neg"]
    for start_pos_idx in pos_candidates:
        current_idx = int(start_pos_idx)
        current_pairs = [(current_idx, "pos")]
        valid_sequence = True

        for expected_kind in expected_sequence[1:]:
            candidate_pool = neg_candidates if expected_kind == "neg" else pos_candidates
            next_candidates = [int(idx) for idx in candidate_pool if int(idx) > current_idx]
            if not next_candidates:
                valid_sequence = False
                break

            current_idx = int(next_candidates[0])
            current_pairs.append((current_idx, expected_kind))

        if valid_sequence:
            landmark_pairs = current_pairs
            break

    if landmark_pairs is None:
        return None

    landmark_indices = [idx for idx, _kind in landmark_pairs]
    landmark_kinds = [kind for _idx, kind in landmark_pairs]

    return {
        "indices": landmark_indices,
        "kinds": landmark_kinds,
        "smooth_values": smooth_torque,
    }

def detect_d2_speed_biodex_reps(
    df,
    value_col="Torque_Nm",
    threshold=20.0,
    min_samples=15,
    tail_buffer_samples=10,
    baseline_hold_samples=6,
):
    if df.empty or value_col not in df.columns:
        return []

    torque_values = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    if len(torque_values) < 9 or np.all(np.isnan(torque_values)):
        return []

    torque_values = pd.Series(torque_values).interpolate(limit_direction="both").to_numpy(dtype=float)
    smooth_window = get_valid_savgol_window(9, len(torque_values), 3)
    if smooth_window is not None:
        smooth_torque = savgol_filter(torque_values, window_length=smooth_window, polyorder=3)
    else:
        smooth_torque = torque_values

    abs_envelope = pd.Series(np.abs(smooth_torque)).rolling(window=9, center=True, min_periods=1).max().to_numpy(dtype=float)
    active_idx = np.flatnonzero(abs_envelope >= float(threshold))
    if active_idx.size == 0:
        return []

    regions = _build_contiguous_regions(active_idx.tolist())
    regions = _merge_close_regions(regions, max(int(min_samples) * 2, 12))
    rep_windows = []
    n_rows = len(df)
    baseline_threshold = max(5.0, float(threshold) * 0.4)
    positive_onset_threshold = max(8.0, float(threshold) * 0.5)

    for region_start, region_end in regions:
        region_df = df.iloc[int(region_start):int(region_end) + 1].copy().reset_index(drop=True)
        landmark_info = detect_d2_speed_rep_landmarks(region_df, value_col=value_col)
        if landmark_info is None:
            continue

        landmark_indices = landmark_info["indices"]
        smooth_region = landmark_info["smooth_values"]
        first_peak_idx = int(landmark_indices[0])
        last_peak_idx = int(landmark_indices[-1])

        start_idx_local = 0
        for idx in range(first_peak_idx, -1, -1):
            if smooth_region[idx] <= baseline_threshold:
                start_idx_local = min(len(smooth_region) - 1, idx + 1)
                break
        for idx in range(start_idx_local, first_peak_idx + 1):
            if smooth_region[idx] >= positive_onset_threshold:
                start_idx_local = idx
                break

        end_idx_local = len(smooth_region) - 1
        hold_count = 0
        for idx in range(last_peak_idx, len(smooth_region)):
            if abs(smooth_region[idx]) <= baseline_threshold:
                hold_count += 1
            else:
                hold_count = 0

            if hold_count >= int(baseline_hold_samples):
                end_idx_local = min(
                    len(smooth_region) - 1,
                    idx + int(tail_buffer_samples),
                )
                break

        start_idx = max(0, int(region_start) + int(start_idx_local))
        end_idx = min(n_rows - 1, int(region_start) + int(end_idx_local))
        if end_idx <= start_idx:
            continue
        rep_windows.append((start_idx, end_idx))

    return rep_windows

def extract_d2_speed_landmark_aligned_biodex_reps(
    df,
    rep_windows,
    time_col="Elapsed Seconds",
    value_col="Torque_Nm",
    n_points=101,
    prominence_ratio=0.10,
):
    if df.empty or not rep_windows:
        return pd.DataFrame(), pd.DataFrame(), []

    percent_axis = np.linspace(0.0, 100.0, int(n_points))
    valid_reps = []
    phase_fraction_rows = []

    for rep_number, (start_idx, end_idx) in enumerate(rep_windows, start=1):
        rep_df = df.iloc[int(start_idx):int(end_idx) + 1].copy()
        rep_df[time_col] = pd.to_numeric(rep_df[time_col], errors="coerce")
        rep_df[value_col] = pd.to_numeric(rep_df[value_col], errors="coerce")
        rep_df = rep_df.dropna(subset=[time_col, value_col]).reset_index(drop=True)
        if len(rep_df) < 9:
            continue

        landmark_info = detect_d2_speed_rep_landmarks(
            rep_df,
            value_col=value_col,
            prominence_ratio=prominence_ratio,
        )
        if landmark_info is None:
            continue

        trim_start = int(landmark_info["indices"][0])
        trim_end = int(landmark_info["indices"][-1])
        if trim_end <= trim_start:
            continue

        rep_df = rep_df.iloc[trim_start:trim_end + 1].reset_index(drop=True)
        trimmed_landmark_indices = [int(idx - trim_start) for idx in landmark_info["indices"]]
        boundary_idx = trimmed_landmark_indices
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
            "landmark_indices": trimmed_landmark_indices,
            "landmark_kinds": landmark_info["kinds"],
        })

    if not valid_reps:
        return pd.DataFrame(), pd.DataFrame(), []

    median_phase_fractions = np.nanmedian(np.vstack(phase_fraction_rows), axis=0)
    median_phase_fractions = median_phase_fractions / median_phase_fractions.sum()
    boundary_pct = np.concatenate(([0.0], np.cumsum(median_phase_fractions[:-1]) * 100.0, [100.0]))

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

    landmark_counts = {}
    landmark_labels = []
    for kind in aligned_rep_metadata[0]["landmark_kinds"]:
        landmark_counts[kind] = landmark_counts.get(kind, 0) + 1
        landmark_labels.append(f"{kind.upper()}{landmark_counts[kind]}")

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

# --------------------------------------------------
# Terra sidebar + kinematic sequence support
# --------------------------------------------------
KINEMATIC_FPS = 250
MS_PER_FRAME = 1000 / KINEMATIC_FPS


def to_rgba(color, alpha=0.35):
    named_colors = {
        "blue": (31, 119, 180),
        "orange": (255, 127, 14),
        "green": (44, 160, 44),
        "red": (214, 39, 40),
        "purple": (148, 103, 189),
        "brown": (140, 86, 75),
        "pink": (227, 119, 194),
        "gray": (127, 127, 127),
        "olive": (188, 189, 34),
        "teal": (23, 190, 207),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "deeppink": (255, 20, 147),
        "dodgerblue": (30, 144, 255),
        "crimson": (220, 20, 60),
        "darkorange": (255, 140, 0),
        "charcoal": (55, 65, 81),
        "darkblue": (0, 0, 139),
        "darkred": (139, 0, 0),
        "darkgreen": (0, 100, 0),
        "navy": (0, 0, 128),
        "limegreen": (50, 205, 50),
    }

    if isinstance(color, str):
        color = color.lower()
        if color.startswith("#") and len(color) == 7:
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            return f"rgba({r},{g},{b},{alpha})"
        if color in named_colors:
            r, g, b = named_colors[color]
            return f"rgba({r},{g},{b},{alpha})"
    return f"rgba(150,150,150,{alpha})"


def rel_frame_to_ms(rel_frame):
    return int(round(rel_frame * MS_PER_FRAME))


def ms_to_rel_frame(milliseconds):
    return int(round(milliseconds / MS_PER_FRAME))


SEGMENT_DISPLAY_NAMES = {
    "Pelvis": "Pelvis Rotation",
    "Torso": "Torso Rotation",
    "Elbow": "Elbow Extension",
    "Shoulder": "Shoulder Internal Rotation",
    "Shoulder IR": "Shoulder Internal Rotation",
}


def segment_display_name(label):
    return SEGMENT_DISPLAY_NAMES.get(label, label)


def add_event_iqr_band(fig, event_frames, color, show_band, opacity=0.10):
    if not show_band or not event_frames:
        return

    event_q1_frame = int(np.percentile(event_frames, 25))
    event_q3_frame = int(np.percentile(event_frames, 75))
    event_start_ms = rel_frame_to_ms(event_q1_frame)
    event_end_ms = rel_frame_to_ms(event_q3_frame)

    if event_start_ms == event_end_ms:
        return

    fig.add_vrect(
        x0=event_start_ms,
        x1=event_end_ms,
        fillcolor=color,
        opacity=opacity,
        layer="below",
        line_width=0,
    )

def get_all_pitchers():
    """
    Returns all athlete names from the athletes table.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT athlete_name
                FROM athletes
                WHERE athlete_name IS NOT NULL
                ORDER BY athlete_name
            """)
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


# --------------------------------------------------
# Velocity Bounds Helper
# --------------------------------------------------
@st.cache_data(ttl=300)
def get_velocity_bounds(athlete_name, selected_dates):
    """
    Returns (min_velocity, max_velocity) for the selected pitcher and dates.
    Assumes pitch velocity is stored as `pitch_velo` on the takes table.
    """
    if athlete_name is None:
        return None, None

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if not selected_dates or "All Dates" in selected_dates:
                cur.execute("""
                    SELECT MIN(t.pitch_velo), MAX(t.pitch_velo)
                    FROM takes t
                    JOIN athletes a ON a.athlete_id = t.athlete_id
                    WHERE a.athlete_name = %s
                      AND t.pitch_velo IS NOT NULL
                """, (athlete_name,))
            else:
                placeholders = ",".join(["%s"] * len(selected_dates))
                cur.execute(f"""
                    SELECT MIN(t.pitch_velo), MAX(t.pitch_velo)
                    FROM takes t
                    JOIN athletes a ON a.athlete_id = t.athlete_id
                    WHERE a.athlete_name = %s
                      AND t.take_date IN ({placeholders})
                      AND t.pitch_velo IS NOT NULL
                """, (athlete_name, *selected_dates))

            row = cur.fetchone()
            return row if row else (None, None)
    finally:
        conn.close()

@st.cache_data(ttl=300, show_spinner=False)
def get_control_group_take_pool(handedness_filter):
    """
    Returns control-group candidates from all takes in the database, optionally
    filtered by pitcher handedness.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            params = []
            handedness_clause = ""
            if handedness_filter in ("R", "L"):
                handedness_clause = "AND a.handedness = %s"
                params.append(handedness_filter)

            cur.execute(f"""
                WITH ids AS (
                    SELECT
                        (SELECT category_id FROM categories WHERE category_name = 'KINETIC_KINEMATIC_CGVel') AS cat_kk_cgvel,
                        (SELECT category_id FROM categories WHERE category_name = 'KINETIC_KINEMATIC_ProxEndPos') AS cat_kk_prox_pos,
                        (SELECT category_id FROM categories WHERE category_name = 'KINETIC_KINEMATIC_DistEndPos') AS cat_kk_dist_endpos,
                        (SELECT segment_id FROM segments WHERE segment_name = 'LHA') AS seg_hand_l,
                        (SELECT segment_id FROM segments WHERE segment_name = 'RHA') AS seg_hand_r,
                        (SELECT segment_id FROM segments WHERE segment_name = 'LAR') AS seg_arm_l,
                        (SELECT segment_id FROM segments WHERE segment_name = 'RAR') AS seg_arm_r
                ),
                candidate_takes AS (
                    SELECT
                        t.take_id,
                        t.pitch_velo,
                        a.athlete_name,
                        a.handedness,
                        CASE WHEN a.handedness = 'L' THEN i.seg_hand_l ELSE i.seg_hand_r END AS seg_hand_dom,
                        CASE WHEN a.handedness = 'L' THEN i.seg_arm_l ELSE i.seg_arm_r END AS seg_arm_dom,
                        i.seg_hand_l,
                        i.seg_hand_r,
                        i.cat_kk_cgvel,
                        i.cat_kk_prox_pos,
                        i.cat_kk_dist_endpos
                    FROM takes t
                    JOIN athletes a ON a.athlete_id = t.athlete_id
                    CROSS JOIN ids i
                    WHERE t.pitch_velo IS NOT NULL
                      AND t.throw_type = 'Mound'
                      AND a.handedness IN ('R', 'L')
                      {handedness_clause}
                      AND EXISTS (
                          SELECT 1
                          FROM time_series_data d
                          WHERE d.take_id = t.take_id
                      )
                ),
                hand_vel AS (
                    SELECT
                        t.take_id,
                        t.pitch_velo,
                        t.handedness,
                        d.frame,
                        d.x_data,
                        LAG(d.x_data) OVER (
                            PARTITION BY t.take_id
                            ORDER BY d.frame
                        ) AS prev_x
                    FROM time_series_data d
                    JOIN candidate_takes t ON t.take_id = d.take_id
                    WHERE d.category_id = t.cat_kk_cgvel
                      AND d.segment_id IN (t.seg_hand_l, t.seg_hand_r)
                      AND d.x_data IS NOT NULL
                ),
                cross_15 AS (
                    SELECT DISTINCT ON (take_id)
                        take_id,
                        frame AS cross_frame
                    FROM hand_vel
                    WHERE x_data >= 15
                      AND (prev_x < 15 OR prev_x IS NULL)
                    ORDER BY take_id, frame
                ),
                positive_phase AS (
                    SELECT
                        h.take_id,
                        h.frame,
                        h.x_data,
                        LAG(h.x_data, 5) OVER (
                            PARTITION BY h.take_id
                            ORDER BY h.frame
                        ) AS prev_5_x,
                        LEAD(h.x_data) OVER (
                            PARTITION BY h.take_id
                            ORDER BY h.frame
                        ) AS next_x
                    FROM hand_vel h
                    JOIN cross_15 c ON c.take_id = h.take_id
                    WHERE h.frame >= c.cross_frame
                      AND h.x_data > 0
                ),
                per_take_br AS (
                    SELECT DISTINCT ON (p.take_id)
                        p.take_id,
                        p.frame AS br_frame
                    FROM positive_phase p
                    WHERE p.next_x IS NOT NULL
                      AND p.prev_5_x IS NOT NULL
                      AND p.x_data > p.next_x
                      AND p.x_data > p.prev_5_x
                    ORDER BY p.take_id, p.frame ASC
                ),
                per_take_arm_points AS (
                    SELECT
                        br.take_id,
                        MAX(CASE WHEN d_arm.segment_id = t.seg_arm_dom THEN d_arm.x_data END) AS x_arm,
                        MAX(CASE WHEN d_arm.segment_id = t.seg_arm_dom THEN d_arm.y_data END) AS y_arm,
                        MAX(CASE WHEN d_arm.segment_id = t.seg_arm_dom THEN d_arm.z_data END) AS z_arm,
                        MAX(CASE WHEN d_hand.segment_id = t.seg_hand_dom THEN d_hand.x_data END) AS x_hand,
                        MAX(CASE WHEN d_hand.segment_id = t.seg_hand_dom THEN d_hand.y_data END) AS y_hand,
                        MAX(CASE WHEN d_hand.segment_id = t.seg_hand_dom THEN d_hand.z_data END) AS z_hand
                    FROM per_take_br br
                    JOIN candidate_takes t ON t.take_id = br.take_id
                    LEFT JOIN time_series_data d_arm
                        ON d_arm.take_id = br.take_id
                       AND d_arm.frame = br.br_frame
                       AND d_arm.category_id = t.cat_kk_prox_pos
                       AND d_arm.segment_id = t.seg_arm_dom
                    LEFT JOIN time_series_data d_hand
                        ON d_hand.take_id = br.take_id
                       AND d_hand.frame = br.br_frame
                       AND d_hand.category_id = t.cat_kk_dist_endpos
                       AND d_hand.segment_id = t.seg_hand_dom
                    GROUP BY br.take_id
                ),
                per_take_arm_angle AS (
                    SELECT
                        p.take_id,
                        CASE
                            WHEN p.x_arm IS NULL OR p.x_hand IS NULL THEN NULL
                            ELSE DEGREES(
                                ATAN2(
                                    (p.z_hand - p.z_arm),
                                    NULLIF(SQRT(
                                        POWER(p.x_hand - p.x_arm, 2) +
                                        POWER(p.y_hand - p.y_arm, 2)
                                    ), 0)
                                )
                            )
                        END AS arm_slot_deg
                    FROM per_take_arm_points p
                )
                SELECT
                    t.take_id,
                    t.pitch_velo,
                    t.athlete_name,
                    t.handedness,
                    a.arm_slot_deg
                FROM candidate_takes t
                LEFT JOIN per_take_arm_angle a ON a.take_id = t.take_id
                ORDER BY t.take_id
            """, tuple(params))
            return cur.fetchall()
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_session_dates_for_pitcher(athlete_name):
    """
    Returns distinct session dates (take_date) for a given pitcher.
    """
    if athlete_name is None:
        return []

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT t.take_date
                FROM takes t
                JOIN athletes a ON a.athlete_id = t.athlete_id
                WHERE a.athlete_name = %s
                ORDER BY t.take_date
            """, (athlete_name,))
            return [row[0].strftime("%Y-%m-%d") for row in cur.fetchall()]
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_pitcher_handedness(athlete_name):
    """
    Returns handedness ('R' or 'L') for a given pitcher.
    """
    if athlete_name is None:
        return None

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT handedness
                FROM athletes
                WHERE athlete_name = %s
                LIMIT 1
            """, (athlete_name,))
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_pelvis_angular_velocity(take_ids):
    """
    Returns pelvis angular velocity (z_data) over frames for given take_ids.
    """
    if not take_ids:
        return {}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = 'PELVIS_ANGULAR_VELOCITY'
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, tuple(take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, z in rows:
                data.setdefault(take_id, {"frame": [], "z": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["z"].append(z)

            return data
    finally:
        conn.close()

# --------------------------------------------------
# Torso Angular Velocity (Z) helper
# --------------------------------------------------

@st.cache_data(ttl=300)
def get_torso_angular_velocity(take_ids):
    """
    Returns torso angular velocity (z_data) over frames for given take_ids.
    Category: ORIGINAL
    Segment: TORSO_ANGULAR_VELOCITY
    """
    if not take_ids:
        return {}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = 'TORSO_ANGULAR_VELOCITY'
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, tuple(take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, z in rows:
                data.setdefault(take_id, {"frame": [], "z": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["z"].append(z)

            return data
    finally:
        conn.close()

# --------------------------------------------------
# Torso-Pelvis Angular Velocity (Z) helper
# --------------------------------------------------
@st.cache_data(ttl=300)
def get_torso_pelvis_angular_velocity(take_ids):
    """
    Returns torso-pelvis angular velocity (z_data) over frames for given take_ids.
    Category: ORIGINAL
    Segment: TORSO_PELVIS_ANGULAR_VELOCITY
    """
    if not take_ids:
        return {}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = 'TORSO_PELVIS_ANGULAR_VELOCITY'
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, tuple(take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, z in rows:
                data.setdefault(take_id, {"frame": [], "z": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["z"].append(z)

            return data
    finally:
        conn.close()

# --------------------------------------------------
# Elbow Angular Velocity (X) helper
# --------------------------------------------------
@st.cache_data(ttl=300)
def get_elbow_angular_velocity(take_ids, handedness):
    """
    Returns elbow angular velocity (x_data) over frames for given take_ids.

    Category: ORIGINAL
    Segments:
      RHP → RT_ELBOW_ANGULAR_VELOCITY
      LHP → LT_ELBOW_ANGULAR_VELOCITY
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = (
        "RT_ELBOW_ANGULAR_VELOCITY"
        if handedness == "R"
        else "LT_ELBOW_ANGULAR_VELOCITY"
    )

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "x": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["x"].append(x)

            return data
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_elbow_flexion_angle(take_ids, handedness):
    """
    Returns elbow flexion angle (x_data) for the throwing elbow.

    Category: ORIGINAL
    Segments:
      RHP → RT_ELBOW_ANGLE
      LHP → LT_ELBOW_ANGLE
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = (
        "RT_ELBOW_ANGLE"
        if handedness == "R"
        else "LT_ELBOW_ANGLE"
    )

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(x)

            return data
    finally:
        conn.close()

# --- SHOULDER EXTERNAL ROTATION ANGLE helper ---
@st.cache_data(ttl=300)
def get_shoulder_er_angle(take_ids, handedness):
    """
    Returns shoulder external rotation angle (z_data) for the throwing shoulder.

    Category: ORIGINAL
    Segments:
      RHP → RT_SHOULDER_ANGLE
      LHP → LT_SHOULDER_ANGLE
    """
    if not take_ids:
        return {}

    segment = "RT_SHOULDER" if handedness == "R" else "LT_SHOULDER"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                    SELECT
                        ts.take_id,
                        ts.frame,
                        ts.z_data
                    FROM time_series_data ts
                    JOIN categories c ON ts.category_id = c.category_id
                    JOIN segments s ON ts.segment_id = s.segment_id
                    WHERE c.category_name = 'JOINT_ANGLES'
                      AND s.segment_name = %s
                      AND ts.take_id IN ({placeholders})
                      AND ts.z_data IS NOT NULL
                    ORDER BY ts.take_id, ts.frame
                """, (segment, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, z in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(z)

            return data
    finally:
        conn.close()

# --- SHOULDER ABDUCTION ANGLE helper ---

@st.cache_data(ttl=300)
def get_shoulder_abduction_angle(take_ids, handedness):
    """
    Returns shoulder abduction angle (y_data) for the throwing shoulder.

    Category: JOINT_ANGLES
    Segments:
      RHP → RT_SHOULDER
      LHP → LT_SHOULDER
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = (
        "RT_SHOULDER"
        if handedness == "R"
        else "LT_SHOULDER"
    )

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.y_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'JOINT_ANGLES'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, y in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(y)

            return data
    finally:
        conn.close()

# --- SHOULDER HORIZONTAL ABDUCTION ANGLE helper ---

@st.cache_data(ttl=300)
def get_front_knee_flexion_angle(take_ids, handedness):
    """
    Returns front (lead) knee flexion angle (x_data).

    Category: ORIGINAL
    Segments:
      RHP → LT_KNEE_ANGLE
      LHP → RT_KNEE_ANGLE
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = (
        "LT_KNEE_ANGLE"
        if handedness == "R"
        else "RT_KNEE_ANGLE"
    )

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(x)

            return data
    finally:
        conn.close()

# --- FRONT KNEE EXTENSION VELOCITY helper ---
@st.cache_data(ttl=300)
def get_front_knee_extension_velocity(take_ids, handedness):
    """
    Returns front (lead) knee angular velocity (x_data).

    Category: ORIGINAL
    Segments:
      RHP → LT_KNEE_ANGULAR_VELOCITY
      LHP → RT_KNEE_ANGULAR_VELOCITY
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = (
        "LT_KNEE_ANGULAR_VELOCITY"
        if handedness == "R"
        else "RT_KNEE_ANGULAR_VELOCITY"
    )

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(x)

            return data
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_shoulder_horizontal_abduction_angle(take_ids, handedness):
    """
    Returns shoulder horizontal abduction angle (x_data) for the throwing shoulder.

    Category: JOINT_ANGLES
    Segments:
      RHP → RT_SHOULDER
      LHP → LT_SHOULDER
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = (
        "RT_SHOULDER"
        if handedness == "R"
        else "LT_SHOULDER"
    )

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'JOINT_ANGLES'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(x)

            return data
    finally:
        conn.close()

# --- TORSO ANGLE COMPONENTS helper ---
@st.cache_data(ttl=300)
def get_torso_angle_components(take_ids):
    """
    Returns torso angle components for each take.

    Category: ORIGINAL
    Segment: TORSO_ANGLE

    Components:
      x_data → Forward Trunk Tilt
      y_data → Lateral Trunk Tilt
      z_data → Trunk Angle
    """
    if not take_ids:
        return {}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data,
                    ts.y_data,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = 'TORSO_ANGLE'
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, tuple(take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, x, y, z in rows:
                data.setdefault(take_id, {
                    "frame": [],
                    "x": [],
                    "y": [],
                    "z": []
                })
                data[take_id]["frame"].append(frame)
                data[take_id]["x"].append(x)
                data[take_id]["y"].append(y)
                data[take_id]["z"].append(z)

            return data
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_pelvis_angle(take_ids):
    """
    Returns pelvis angle (z_data).

    Category: ORIGINAL
    Segment: PELVIS_ANGLE
    """
    if not take_ids:
        return {}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = 'PELVIS_ANGLE'
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, tuple(take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, z in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(z)

            return data
    finally:
        conn.close()


@st.cache_data(ttl=300)
def get_pelvic_lateral_tilt(take_ids):
    """
    Returns pelvic lateral tilt from pelvis angle (y_data).

    Category: ORIGINAL
    Segment: PELVIS_ANGLE
    """
    if not take_ids:
        return {}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.y_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = 'PELVIS_ANGLE'
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, tuple(take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, y in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(y)

            return data
    finally:
        conn.close()


@st.cache_data(ttl=300)
def get_hip_shoulder_separation(take_ids):
    """
    Returns hip–shoulder separation angle (z_data).

    Category: ORIGINAL
    Segment: TORSO_PELVIS_ANGLE
    """
    if not take_ids:
        return {}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = 'TORSO_PELVIS_ANGLE'
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, tuple(take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, z in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(z)

            return data
    finally:
        conn.close()


@st.cache_data(ttl=300)
def get_shoulder_ir_velocity(take_ids, handedness):
    """
    Returns shoulder internal rotation angular velocity (x_data).

    Category: ORIGINAL
    Segments:
      RHP → RT_SHOULDER_ANGULAR_VELOCITY
      LHP → LT_SHOULDER_ANGULAR_VELOCITY

    L-handed pitchers will be sign-normalized later.
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = (
        "RT_SHOULDER_ANGULAR_VELOCITY"
        if handedness == "R"
        else "LT_SHOULDER_ANGULAR_VELOCITY"
    )

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s   ON ts.segment_id  = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "x": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["x"].append(x)

            return data
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_arm_proximal_energy_transfer(take_ids, handedness):
    """
    Arm proximal energy transfer (power flowing into the arm).

    Category: SEGMENT_POWERS
    Segments:
      RHP → RAR_PROX
      LHP → LAR_PROX
    Component: x_data
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "RAR_PROX" if handedness == "R" else "LAR_PROX"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'SEGMENT_POWERS'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.x_data IS NOT NULL
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(x)

            return data
    finally:
        conn.close()

# --- DISTAL ARM SEGMENT POWER loader ---
@st.cache_data(ttl=300)
def get_distal_arm_segment_power(take_ids, handedness):
    """
    Returns distal throwing arm segment power (Watts).

    Category: SEGMENT_POWERS
    Segments:
      RHP → RTA_DIST_R
      LHP → RTA_DIST_L
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "RTA_DIST_R" if handedness == "R" else "RTA_DIST_L"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'SEGMENT_POWERS'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.x_data IS NOT NULL
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(x)

            return data
    finally:
        conn.close()


@st.cache_data(ttl=300)
def get_glove_side_trunk_shoulder_energy_flow(take_ids, handedness):
    """
    Returns glove-side distal arm/trunk-shoulder energy flow (Watts).

    Category: SEGMENT_POWERS
    Segments:
      RHP -> RTA_DIST_L
      LHP -> RTA_DIST_R
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "RTA_DIST_L" if handedness == "R" else "RTA_DIST_R"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'SEGMENT_POWERS'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.x_data IS NOT NULL
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(x)

            return data
    finally:
        conn.close()


@st.cache_data(ttl=300)
def get_glove_arm_energy_flow(take_ids, handedness):
    """
    Returns glove-side proximal arm energy flow (Watts).

    Category: SEGMENT_POWERS
    Segments:
      RHP -> LAR_PROX
      LHP -> RAR_PROX
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "LAR_PROX" if handedness == "R" else "RAR_PROX"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'SEGMENT_POWERS'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.x_data IS NOT NULL
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(x)

            return data
    finally:
        conn.close()


@st.cache_data(ttl=300)
def get_trunk_shoulder_rot_energy_flow(take_ids, handedness):
    """
    Trunk–Shoulder rotational energy flow.

    Category: JCS_STP_ROT
    Segments:
      RHP → RTA_RAR
      LHP → RTA_LAR
    Component: x_data
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "RTA_RAR" if handedness == "R" else "RTA_LAR"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s   ON ts.segment_id  = s.segment_id
                WHERE c.category_name = 'JCS_STP_ROT'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.x_data IS NOT NULL
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(x)

            return data
    finally:
        conn.close()


# --- Trunk–Shoulder Elevation/Depression Energy Flow loader ---

@st.cache_data(ttl=300)
def get_trunk_shoulder_elev_energy_flow(take_ids, handedness):
    """
    Trunk–Shoulder elevation/depression energy flow.

    Category: JCS_STP_ELEV
    Segments:
      RHP → RTA_RAR
      LHP → RTA_LAR
    Component: x_data
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "RTA_RAR" if handedness == "R" else "RTA_LAR"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s   ON ts.segment_id  = s.segment_id
                WHERE c.category_name = 'JCS_STP_ELEV'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.x_data IS NOT NULL
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(x)

            return data
    finally:
        conn.close()


# --- Trunk–Shoulder Horizontal Abduction/Adduction Energy Flow loader ---

@st.cache_data(ttl=300)
def get_trunk_shoulder_horizabd_energy_flow(take_ids, handedness):
    """
    Trunk–Shoulder horizontal abduction/adduction energy flow.

    Category: JCS_STP_HORIZABD
    Segments:
      RHP → RTA_RAR
      LHP → RTA_LAR
    Component: x_data
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "RTA_RAR" if handedness == "R" else "RTA_LAR"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s   ON ts.segment_id  = s.segment_id
                WHERE c.category_name = 'JCS_STP_HORIZABD'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.x_data IS NOT NULL
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(x)

            return data
    finally:
        conn.close()


# --- Arm Rotational Energy Flow loader ---
@st.cache_data(ttl=300)
def get_arm_rot_energy_flow(take_ids, handedness):
    """
    Arm rotational energy flow.

    Category: JCS_STP_ROT
    Segments:
      RHP → RTA_RAR
      LHP → RTA_LAR
    Component: x_data
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "RAR" if handedness == "R" else "LAR"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT ts.take_id, ts.frame, ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s   ON ts.segment_id  = s.segment_id
                WHERE c.category_name = 'JCS_STP_ROT'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.x_data IS NOT NULL
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()
            data = {}
            for tid, frame, x in rows:
                data.setdefault(tid, {"frame": [], "value": []})
                data[tid]["frame"].append(frame)
                data[tid]["value"].append(x)
            return data
    finally:
        conn.close()


# --- Arm Elevation/Depression Energy Flow loader ---
@st.cache_data(ttl=300)
def get_arm_elev_energy_flow(take_ids, handedness):
    """
    Arm elevation/depression energy flow.

    Category: JCS_STP_ELEV
    Segments:
      RHP → RTA_RAR
      LHP → RTA_LAR
    Component: x_data
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "RAR" if handedness == "R" else "LAR"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT ts.take_id, ts.frame, ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s   ON ts.segment_id  = s.segment_id
                WHERE c.category_name = 'JCS_STP_ELEV'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.x_data IS NOT NULL
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()
            data = {}
            for tid, frame, x in rows:
                data.setdefault(tid, {"frame": [], "value": []})
                data[tid]["frame"].append(frame)
                data[tid]["value"].append(x)
            return data
    finally:
        conn.close()


# --- Arm Horizontal Abduction/Adduction Energy Flow loader ---
@st.cache_data(ttl=300)
def get_arm_horizabd_energy_flow(take_ids, handedness):
    """
    Arm horizontal abduction/adduction energy flow.

    Category: JCS_STP_HORIZABD
    Segments:
      RHP → RTA_RAR
      LHP → RTA_LAR
    Component: x_data
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "RAR" if handedness == "R" else "LAR"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT ts.take_id, ts.frame, ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s   ON ts.segment_id  = s.segment_id
                WHERE c.category_name = 'JCS_STP_HORIZABD'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.x_data IS NOT NULL
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()
            data = {}
            for tid, frame, x in rows:
                data.setdefault(tid, {"frame": [], "value": []})
                data[tid]["frame"].append(frame)
                data[tid]["value"].append(x)
            return data
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_energy_flow_from_segment(take_ids, segment_name, component="x"):
    """
    Generic energy-flow loader by segment name and component.
    """
    if not take_ids or not segment_name:
        return {}
    component_col = {
        "x": "ts.x_data",
        "y": "ts.y_data",
        "z": "ts.z_data",
    }.get(component, "ts.x_data")

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    {component_col}
                FROM time_series_data ts
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND {component_col} IS NOT NULL
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()
            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(x)
            return data
    finally:
        conn.close()

NEW_TRUNK_PELVIS_ENERGY_METRIC_MAP = {
    "RPV_DIST_STP_FLEX": ("RPV_DIST", "JCS_STP_FLEX"),
    "RPV_DIST_STP_SIDE": ("RPV_DIST", "JCS_STP_SIDE"),
    "RPV_DIST_STP_ROT": ("RPV_DIST", "JCS_STP_ROT"),
    "RTA_PROX_STP_FLEX": ("RTA_PROX", "JCS_STP_FLEX"),
    "RTA_PROX_STP_SIDE": ("RTA_PROX", "JCS_STP_SIDE"),
    "RTA_PROX_STP_ROT": ("RTA_PROX", "JCS_STP_ROT"),
    "RTA_PROX_STP_X": ("RTA_PROX", "JCS_STP_X"),
    "RTA_PROX_STP_Y": ("RTA_PROX", "JCS_STP_Y"),
    "RTA_PROX_STP_Z": ("RTA_PROX", "JCS_STP_Z"),
}

NEW_TRUNK_PELVIS_ENERGY_METRICS = list(NEW_TRUNK_PELVIS_ENERGY_METRIC_MAP.keys())

NEW_TRUNK_PELVIS_ENERGY_COLOR_MAP = {
    "RPV_DIST_STP_FLEX": "#0F766E",
    "RPV_DIST_STP_SIDE": "#1D4ED8",
    "RPV_DIST_STP_ROT": "#A16207",
    "RTA_PROX_STP_FLEX": "#059669",
    "RTA_PROX_STP_SIDE": "#2563EB",
    "RTA_PROX_STP_ROT": "#CA8A04",
    "RTA_PROX_STP_X": "#BE123C",
    "RTA_PROX_STP_Y": "#6D28D9",
    "RTA_PROX_STP_Z": "#7C3AED",
}

@st.cache_data(ttl=300)
def get_energy_flow_from_category_segment(take_ids, category_name, segment_name, component="x"):
    """
    Generic energy-flow loader by category, segment name, and component.
    """
    if not take_ids or not category_name or not segment_name:
        return {}
    component_col = {
        "x": "ts.x_data",
        "y": "ts.y_data",
        "z": "ts.z_data",
    }.get(component, "ts.x_data")

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    {component_col}
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = %s
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND {component_col} IS NOT NULL
                ORDER BY ts.take_id, ts.frame
            """, (category_name, segment_name, *take_ids))

            rows = cur.fetchall()
            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(x)
            return data
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_hand_cg_velocity(take_ids, handedness):
    """
    Returns CG velocity (x_data) for the throwing hand based on handedness.
    Category: KINETIC_KINEMATIC_CGVel
    Segments: RHA / LHA
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "RHA" if handedness == "R" else "LHA"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'KINETIC_KINEMATIC_CGVel'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "x": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["x"].append(x)

            return data
    finally:
        conn.close()


@st.cache_data(ttl=300)
def get_hand_speed(take_ids, handedness):
    """
    Returns hand speed magnitude from CG velocity components:
      speed = sqrt(x^2 + y^2 + z^2)
    Category: KINETIC_KINEMATIC_CGVel
    Segments: RHA / LHA
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "RHA" if handedness == "R" else "LHA"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data,
                    ts.y_data,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'KINETIC_KINEMATIC_CGVel'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()
            data = {}
            for take_id, frame, x, y, z in rows:
                if x is None or y is None or z is None:
                    continue

                speed = float(np.sqrt(x**2 + y**2 + z**2))
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(speed)

            return data
    finally:
        conn.close()


@st.cache_data(ttl=300)
def get_center_of_mass_velocity_x(take_ids):
    """
    Returns Center of Mass velocity in the x direction.

    Category: PROCESSED
    Segment: CenterOfMass_VELO
    """
    if not take_ids:
        return {}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'PROCESSED'
                  AND s.segment_name = 'CenterOfMass_VELO'
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, tuple(take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(x)

            return data
    finally:
        conn.close()


@st.cache_data(ttl=300)
def get_shoulder_er_angles(take_ids, handedness):
    """
    Returns shoulder joint angle z_data for MER detection.
    Category: JOINT_ANGLES
    Segments:
      - RT_SHOULDER for R-handed
      - LT_SHOULDER for L-handed
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "RT_SHOULDER" if handedness == "R" else "LT_SHOULDER"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'JOINT_ANGLES'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, z in rows:
                data.setdefault(take_id, {"frame": [], "z": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["z"].append(z)

            return data
    finally:
        conn.close()


@st.cache_data(ttl=300)
def get_forearm_pron_sup_angle(take_ids, handedness):
    """
    Forearm Pronation / Supination angle.

    Category: ORIGINAL
    Segments:
      RHP → RT_ELBOW_ANGLE
      LHP → LT_ELBOW_ANGLE
    Component: z_data
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "RT_ELBOW_ANGLE" if handedness == "R" else "LT_ELBOW_ANGLE"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s   ON ts.segment_id  = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.z_data IS NOT NULL
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, z in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(z)

            return data
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_peak_glove_knee_pre_br(take_ids, handedness, br_frames):
    """
    Peak glove-side knee height (Z position) prior to Ball Release.

    Category: KINETIC_KINEMATIC_ProxEndPos
    Segments:
      RHP → LSK
      LHP → RSK
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "LSK" if handedness == "R" else "RSK"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s   ON ts.segment_id  = s.segment_id
                WHERE c.category_name = 'KINETIC_KINEMATIC_ProxEndPos'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.z_data IS NOT NULL
                ORDER BY ts.take_id, ts.z_data DESC, ts.frame ASC
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            out = {}
            for take_id, frame, z in rows:
                br_frame = br_frames.get(take_id)
                if br_frame is None:
                    continue
                if frame < br_frame and take_id not in out:
                    out[take_id] = int(frame)

            return out
    finally:
        conn.close()


# --------------------------------------------------
# Foot Plant event helper
# --------------------------------------------------
@st.cache_data(ttl=300)
def get_terra_window_foot_plant_frame(
    take_ids,
    handedness,
    knee_peak_frames,
    br_frames
):
    """
    Estimate Foot Plant as the LAST frame where lead ankle Z velocity is negative
    between peak knee height and ball release.

    Category: KINETIC_KINEMATIC_DistEndVel
    Segments:
      RHP → LFT
      LHP → RFT
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "LFT" if handedness == "R" else "RFT"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s   ON ts.segment_id  = s.segment_id
                WHERE c.category_name = 'KINETIC_KINEMATIC_DistEndVel'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.z_data IS NOT NULL
                ORDER BY ts.take_id, ts.frame ASC
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            out = {}
            for take_id, frame, z in rows:
                knee_frame = knee_peak_frames.get(take_id)
                br_frame   = br_frames.get(take_id)

                if knee_frame is None or br_frame is None:
                    continue

                # constrain search window: knee peak → ball release
                if frame < knee_frame or frame > br_frame:
                    continue

                # last downward ankle velocity frame
                if z < 0:
                    out[take_id] = int(frame)

            return out
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_peak_ankle_prox_x_velocity(
    take_ids,
    handedness
):
    """
    Peak lead ankle proximal X velocity.

    Category: KINETIC_KINEMATIC_ProxEndVel
    Segments:
      RHP → LFT
      LHP → RFT
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "LFT" if handedness == "R" else "RFT"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s   ON ts.segment_id  = s.segment_id
                WHERE c.category_name = 'KINETIC_KINEMATIC_ProxEndVel'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.x_data IS NOT NULL
                ORDER BY ts.take_id, ts.x_data DESC
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            out = {}
            for take_id, frame, x in rows:
                if take_id not in out:
                    out[take_id] = int(frame)

            return out
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_terra_ankle_min_frame(
    take_ids,
    handedness,
    ankle_prox_x_peak_frames,
    shoulder_er_max_frames
):
    """
    Deepest lead ankle distal Z-velocity dip between ankle prox-X peak and max shoulder ER.

    Category: KINETIC_KINEMATIC_DistEndVel
    Segments:
      RHP → LFT
      LHP → RFT
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "LFT" if handedness == "R" else "RFT"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s   ON ts.segment_id  = s.segment_id
                WHERE c.category_name = 'KINETIC_KINEMATIC_DistEndVel'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.z_data IS NOT NULL
                ORDER BY ts.take_id, ts.z_data ASC, ts.frame ASC
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            out = {}
            for take_id, frame, _z in rows:
                px_frame = ankle_prox_x_peak_frames.get(take_id)
                er_frame = shoulder_er_max_frames.get(take_id)

                if px_frame is None or er_frame is None:
                    continue

                if frame < px_frame or frame > er_frame:
                    continue

                if take_id not in out:
                    out[take_id] = int(frame)

            return out
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_foot_plant_frame_zero_cross(
    take_ids,
    handedness,
    ankle_min_frames,
    shoulder_er_max_frames
):
    """
    Refined Foot Plant using zero-cross logic.

    Search window:
      lead ankle distal Z minimum → max shoulder ER

    Rule:
      first frame where ankle Z velocity >= -0.05
      foot plant frame = frame - 1

    Category: KINETIC_KINEMATIC_DistEndVel
    Segments:
      RHP → LFT
      LHP → RFT
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "LFT" if handedness == "R" else "RFT"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s   ON ts.segment_id  = s.segment_id
                WHERE c.category_name = 'KINETIC_KINEMATIC_DistEndVel'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.z_data IS NOT NULL
                ORDER BY ts.take_id, ts.frame ASC
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            out = {}
            for take_id, frame, z in rows:
                ankle_min_frame = ankle_min_frames.get(take_id)
                er_frame = shoulder_er_max_frames.get(take_id)

                if ankle_min_frame is None or er_frame is None:
                    continue

                # refined biomechanical bounds
                if frame < ankle_min_frame or frame > er_frame:
                    continue

                # zero-cross detection
                if z >= -0.05 and take_id not in out:
                    out[take_id] = int(frame - 1)

            return out
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_lead_heel_contact_frame(
    take_ids,
    handedness,
    start_frames,
    end_frames,
    anchor_frames,
    contact_ratio=0.15,
    absolute_floor_buffer=0.03,
    min_consecutive_frames=3,
    pre_anchor_frames=4,
    post_anchor_frames=6,
    flattening_tolerance=0.01
):
    """
    Estimate lead-foot contact timing from heel height using a take-specific near-floor threshold.

    Category: LANDMARK_ORIGINAL
    Segments:
      RHP → L_HEEL
      LHP → R_HEEL
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    heel_segment = "L_HEEL" if handedness == "R" else "R_HEEL"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s   ON ts.segment_id  = s.segment_id
                WHERE c.category_name = 'LANDMARK_ORIGINAL'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.z_data IS NOT NULL
                ORDER BY ts.take_id, ts.frame ASC
            """, (heel_segment, *take_ids))

            rows = cur.fetchall()

            rows_by_take = {}
            for take_id, frame, z in rows:
                rows_by_take.setdefault(take_id, []).append((int(frame), float(z)))

            out = {}
            for take_id, take_rows in rows_by_take.items():
                start_frame = start_frames.get(take_id)
                end_frame = end_frames.get(take_id)
                anchor_frame = anchor_frames.get(take_id)

                if start_frame is None or end_frame is None or start_frame > end_frame:
                    continue
                if anchor_frame is None:
                    continue

                full_window_rows = [
                    (frame, z)
                    for frame, z in take_rows
                    if start_frame <= frame <= end_frame
                ]
                if not full_window_rows:
                    continue

                heel_values = [z for _, z in full_window_rows]
                heel_floor = min(heel_values)
                heel_ceil = max(heel_values)
                heel_range = heel_ceil - heel_floor
                relative_threshold = (
                    heel_floor
                    if heel_range <= 1e-9 else
                    heel_floor + contact_ratio * heel_range
                )
                absolute_threshold = heel_floor + absolute_floor_buffer
                heel_threshold = min(relative_threshold, absolute_threshold)

                search_start = max(start_frame, int(anchor_frame) - pre_anchor_frames)
                search_end = min(end_frame, int(anchor_frame) + post_anchor_frames)
                search_rows = [
                    (frame, z)
                    for frame, z in full_window_rows
                    if search_start <= frame <= search_end
                ]
                if len(search_rows) < min_consecutive_frames:
                    continue

                for i in range(0, len(search_rows) - min_consecutive_frames + 1):
                    block = search_rows[i:i + min_consecutive_frames]
                    block_values = [z for _, z in block]
                    if not all(z <= heel_threshold for z in block_values):
                        continue

                    # Contact should look settled, not like a single-frame downward spike.
                    block_diffs = [
                        block_values[j + 1] - block_values[j]
                        for j in range(len(block_values) - 1)
                    ]
                    if any(diff < -flattening_tolerance for diff in block_diffs):
                        continue

                    out[take_id] = int(block[0][0])
                    break

            return out
    finally:
        conn.close()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
# --- Sidebar Logo ---
sidebar_logo_path = Path(__file__).parent / "assets" / "logo.png"
if sidebar_logo_path.exists():
    st.sidebar.image(str(sidebar_logo_path), use_container_width=True)


st.sidebar.markdown("### Dashboard Controls")
st.markdown(
    """
    <style>
    div[data-testid="stSidebar"] .stButton > button {
        font-size: 1.17em;
        font-weight: 600;
        background-color: #C62828;
        color: #FFFFFF;
        border: 1px solid #C62828;
    }
    div[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #B71C1C;
        color: #FFFFFF;
        border: 1px solid #B71C1C;
    }
    /* Make selected multiselect tags readable in the sidebar */
    div[data-testid="stSidebar"] div[data-baseweb="tag"] {
        max-width: 100% !important;
        height: auto !important;
        white-space: normal !important;
    }
    div[data-testid="stSidebar"] div[data-baseweb="tag"] > span {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        line-height: 1.25 !important;
    }
    /* Make selected multiselect tags readable in the main page as well */
    div[data-testid="stMultiSelect"] div[data-baseweb="tag"] {
        max-width: 100% !important;
        height: auto !important;
        white-space: normal !important;
    }
    div[data-testid="stMultiSelect"] div[data-baseweb="tag"] > span {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        line-height: 1.25 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -------------------------------
# Initialize session state for excluded takes
# -------------------------------
if "excluded_take_ids" not in st.session_state:
    st.session_state["excluded_take_ids"] = []
if "create_groups_mode" not in st.session_state:
    st.session_state["create_groups_mode"] = False
if "show_control_group_velocity" not in st.session_state:
    st.session_state["show_control_group_velocity"] = False
if "control_group_take_ids" not in st.session_state:
    st.session_state["control_group_take_ids"] = []
if "control_group_handedness" not in st.session_state:
    st.session_state["control_group_handedness"] = "Both"
if "control_group_arm_slot_ids" not in st.session_state:
    st.session_state["control_group_arm_slot_ids"] = []
if "control_group_pitchers" not in st.session_state:
    st.session_state["control_group_pitchers"] = []
if "control_group_velocity_range" not in st.session_state:
    st.session_state["control_group_velocity_range"] = (50.0, 100.0)
control_group_arm_slot_categories = [
    ("Over The Top", 50, 90),
    ("High 3/4", 30, 49),
    ("Low 3/4", 10, 29),
    ("Sidearm", -5, 9),
    ("Submarine", -90, -6),
]

for category_label, _, _ in control_group_arm_slot_categories:
    category_key = f"control_group_arm_slot_category_{category_label.lower().replace(' ', '_').replace('/', '')}"
    if category_key not in st.session_state:
        st.session_state[category_key] = True
if "control_group_status_message" not in st.session_state:
    st.session_state["control_group_status_message"] = ""

COMPENSATION_ENERGY_OPTIONS = [
    "Torso Power",
    "STP Elevation",
    "STP Horizontal Abduction",
    "STP Rotational",
    "STP Rotational into Layback",
    "STP Rotational into Ball",
]

group_mode_enabled = st.session_state.get("create_groups_mode", False)
if "group_count" not in st.session_state:
    st.session_state["group_count"] = 1

pitcher_names = get_all_pitchers()
group_configs = []
selected_pitchers = []
pitcher_filters = {}

def build_pitcher_filters_for_group(selected_group_pitchers, group_index, show_group_prefix):
    group_pitcher_filters = {}
    multi_pitcher_group = len(selected_group_pitchers) > 1

    for i, pitcher in enumerate(selected_group_pitchers):
        label_suffix = f" - {pitcher}" if multi_pitcher_group else ""
        if multi_pitcher_group:
            st.sidebar.markdown(f"**{pitcher} Filters**")

        session_dates = get_session_dates_for_pitcher(pitcher)
        if session_dates:
            session_dates_with_all = ["All Dates"] + session_dates
            session_dates_label = (
                f"Group {group_index} Session Dates{label_suffix}"
                if show_group_prefix else
                f"Session Dates{label_suffix}"
            )
            selected_dates_i = st.sidebar.multiselect(
                session_dates_label,
                options=session_dates_with_all,
                default=["All Dates"],
                key=f"group{group_index}_select_session_dates_{i}"
            )
        else:
            st.sidebar.info(f"No session dates found for {pitcher}.")
            selected_dates_i = []

        throw_type_label = (
            f"Group {group_index} Throw Type{label_suffix}"
            if show_group_prefix else
            f"Throw Type{label_suffix}"
        )
        throw_types_i = st.sidebar.multiselect(
            throw_type_label,
            options=["Mound", "Pulldown"],
            default=["Mound"],
            key=f"group{group_index}_throw_types_{i}"
        )
        if not throw_types_i:
            throw_types_i = ["Mound"]

        if (
            not group_mode_enabled
            and group_index == 0
            and i == 0
        ):
            st.sidebar.multiselect(
                "Energy Flow Type",
                COMPENSATION_ENERGY_OPTIONS,
                default=["Torso Power"],
                key="tab1_energy_plot_options",
            )

        vel_min_i, vel_max_i = get_velocity_bounds(pitcher, selected_dates_i)
        if vel_min_i is not None and vel_max_i is not None:
            velocity_label = (
                f"Group {group_index} Velocity Range{label_suffix} (mph)"
                if show_group_prefix else
                f"Velocity Range{label_suffix} (mph)"
            )
            vel_min_float = float(vel_min_i)
            vel_max_float = float(vel_max_i)
            if vel_min_float == vel_max_float:
                velocity_min_i = vel_min_float
                velocity_max_i = vel_max_float
                st.sidebar.caption(f"{velocity_label}: {vel_min_float:.1f}")
            else:
                velocity_range_i = st.sidebar.slider(
                    velocity_label,
                    min_value=vel_min_float,
                    max_value=vel_max_float,
                    value=(vel_min_float, vel_max_float),
                    step=0.5,
                    key=f"group{group_index}_velocity_range_{i}"
                )
                velocity_min_i, velocity_max_i = velocity_range_i
        else:
            velocity_min_i, velocity_max_i = None, None
            st.sidebar.info(f"Velocity data not available for {pitcher}.")

        group_pitcher_filters[pitcher] = {
            "selected_dates": selected_dates_i,
            "throw_types": throw_types_i,
            "velocity_min": velocity_min_i,
            "velocity_max": velocity_max_i,
        }

    return group_pitcher_filters

def get_filtered_takes_for_pitcher(pitcher, cfg):
    selected_dates_i = cfg.get("selected_dates", [])
    throw_types_i = cfg.get("throw_types", [])
    velocity_min_i = cfg.get("velocity_min")
    velocity_max_i = cfg.get("velocity_max")

    if velocity_min_i is None or velocity_max_i is None:
        return []

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if "All Dates" in selected_dates_i or not selected_dates_i:
                cur.execute("""
                    SELECT t.take_id, t.pitch_velo, t.take_date, a.athlete_name
                    FROM takes t
                    JOIN athletes a ON a.athlete_id = t.athlete_id
                    WHERE a.athlete_name = %s
                      AND t.throw_type = ANY(%s)
                      AND t.pitch_velo BETWEEN %s AND %s
                    ORDER BY a.athlete_name, t.take_date, t.take_id
                """, (pitcher, throw_types_i, velocity_min_i, velocity_max_i))
            else:
                placeholders = ",".join(["%s"] * len(selected_dates_i))
                cur.execute(f"""
                    SELECT t.take_id, t.pitch_velo, t.take_date, a.athlete_name
                    FROM takes t
                    JOIN athletes a ON a.athlete_id = t.athlete_id
                    WHERE a.athlete_name = %s
                      AND t.throw_type = ANY(%s)
                      AND t.take_date IN ({placeholders})
                      AND t.pitch_velo BETWEEN %s AND %s
                    ORDER BY a.athlete_name, t.take_date, t.take_id
                """, (pitcher, throw_types_i, *selected_dates_i, velocity_min_i, velocity_max_i))
            return cur.fetchall()
    finally:
        conn.close()

def build_take_options_for_group(group_pitcher_filters):
    from collections import defaultdict

    takes_by_id = {}
    for pitcher, cfg in group_pitcher_filters.items():
        for take_id, velo, date, pitcher_name in get_filtered_takes_for_pitcher(pitcher, cfg):
            takes_by_id[take_id] = (take_id, velo, date, pitcher_name)

    if not takes_by_id:
        return [], {}

    sorted_rows = sorted(
        takes_by_id.values(),
        key=lambda row: (row[3], row[2], row[0])
    )

    date_groups = defaultdict(list)
    for tid, velo, date, pitcher_name in sorted_rows:
        date_groups[(pitcher_name, date)].append((tid, velo))

    options = []
    label_to_take_id = {}
    for (pitcher_name, date), items in date_groups.items():
        for order, (tid, velo) in enumerate(items, start=1):
            velo_text = f"{velo:.1f}" if velo is not None else "N/A"
            label = f"{pitcher_name} | {date.strftime('%Y-%m-%d')} – Pitch {order} ({velo_text} mph)"
            options.append(label)
            label_to_take_id[label] = tid

    return options, label_to_take_id


def get_compensation_take_rows_from_sidebar():
    selected_rows = []
    excluded_ids = set(st.session_state.get("excluded_take_ids", []))

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for pitcher, cfg in pitcher_filters.items():
                selected_dates_i = cfg.get("selected_dates", [])
                throw_types_i = cfg.get("throw_types", [])
                velocity_min_i = cfg.get("velocity_min")
                velocity_max_i = cfg.get("velocity_max")

                if velocity_min_i is None or velocity_max_i is None:
                    continue

                if "All Dates" in selected_dates_i or not selected_dates_i:
                    cur.execute("""
                        SELECT
                            t.take_id,
                            t.file_name,
                            t.pitch_velo,
                            t.take_date,
                            COALESCE(t.throw_type, 'Mound') AS throw_type,
                            a.athlete_name,
                            a.handedness
                        FROM takes t
                        JOIN athletes a ON t.athlete_id = a.athlete_id
                        WHERE a.athlete_name = %s
                          AND COALESCE(t.throw_type, 'Mound') = ANY(%s)
                          AND t.pitch_velo BETWEEN %s AND %s
                        ORDER BY a.athlete_name, t.take_date, t.file_name
                    """, (pitcher, throw_types_i, velocity_min_i, velocity_max_i))
                else:
                    placeholders = ",".join(["%s"] * len(selected_dates_i))
                    cur.execute(f"""
                        SELECT
                            t.take_id,
                            t.file_name,
                            t.pitch_velo,
                            t.take_date,
                            COALESCE(t.throw_type, 'Mound') AS throw_type,
                            a.athlete_name,
                            a.handedness
                        FROM takes t
                        JOIN athletes a ON t.athlete_id = a.athlete_id
                        WHERE a.athlete_name = %s
                          AND t.take_date IN ({placeholders})
                          AND COALESCE(t.throw_type, 'Mound') = ANY(%s)
                          AND t.pitch_velo BETWEEN %s AND %s
                        ORDER BY a.athlete_name, t.take_date, t.file_name
                    """, (pitcher, *selected_dates_i, throw_types_i, velocity_min_i, velocity_max_i))

                selected_rows.extend(cur.fetchall())
    finally:
        conn.close()

    if group_mode_enabled:
        if selected_take_ids_union:
            selected_ids = set(selected_take_ids_union)
            selected_rows = [row for row in selected_rows if int(row[0]) in selected_ids]
        else:
            selected_rows = []
    else:
        selected_rows = [row for row in selected_rows if int(row[0]) not in excluded_ids]

    return selected_rows


def exit_group_mode():
    st.session_state["create_groups_mode"] = False
    st.session_state["group_count"] = 1

    keys_to_clear = [
        key for key in st.session_state.keys()
        if key.startswith("group") or key.startswith("create_groups_mode")
    ]
    for key in keys_to_clear:
        if key in {"create_groups_mode", "group_count"}:
            continue
        del st.session_state[key]

    st.rerun()


def remove_control_group():
    st.session_state["show_control_group_velocity"] = False
    st.session_state["control_group_take_ids"] = []
    st.session_state["control_group_arm_slot_ids"] = []
    st.session_state["control_group_pitchers"] = []
    st.session_state["control_group_handedness"] = "Both"
    st.session_state["control_group_status_message"] = ""
    for key in ["control_group_velocity_range"]:
        if key in st.session_state:
            del st.session_state[key]

    st.rerun()


def render_control_group_arm_slot_category_checkboxes():
    st.markdown("Arm Slot Categories")
    checkbox_cols = st.columns(2)
    for idx, (category_label, _, _) in enumerate(control_group_arm_slot_categories):
        category_key = f"control_group_arm_slot_category_{category_label.lower().replace(' ', '_').replace('/', '')}"
        with checkbox_cols[idx % 2]:
            st.checkbox(
                category_label,
                key=category_key,
            )


def arm_slot_matches_control_group_categories(arm_slot_deg):
    if arm_slot_deg is None:
        return False

    selected_categories = [
        (min_slot, max_slot)
        for category_label, min_slot, max_slot in control_group_arm_slot_categories
        if st.session_state.get(
            f"control_group_arm_slot_category_{category_label.lower().replace(' ', '_').replace('/', '')}",
            False
        )
    ]

    if not selected_categories:
        return False

    arm_slot_value = float(arm_slot_deg)
    return any(min_slot <= arm_slot_value <= max_slot for min_slot, max_slot in selected_categories)

if not pitcher_names:
    st.sidebar.warning("No pitchers found in the database.")
else:
    if group_mode_enabled:
        group_count = max(1, int(st.session_state.get("group_count", 1)))
        st.session_state["group_count"] = group_count

        for group_idx in range(1, group_count + 1):
            st.sidebar.markdown(f"**Group {group_idx}**")
            selected_group_pitchers = st.sidebar.multiselect(
                f"Select Group {group_idx} Pitchers",
                options=pitcher_names,
                default=[pitcher_names[0]] if group_idx == 1 and pitcher_names else [],
                key=f"group{group_idx}_select_pitchers"
            )
            group_pitcher_filters = build_pitcher_filters_for_group(
                selected_group_pitchers,
                group_idx,
                show_group_prefix=True
            )
            group_take_options, group_label_to_take_id = build_take_options_for_group(group_pitcher_filters)
            selected_group_take_labels = st.sidebar.multiselect(
                f"Group {group_idx} Selected Takes",
                options=group_take_options,
                default=[],
                key=f"group{group_idx}_selected_takes"
            )
            selected_group_take_ids = [
                group_label_to_take_id[label]
                for label in selected_group_take_labels
                if label in group_label_to_take_id
            ]
            group_configs.append({
                "group_index": group_idx,
                "selected_pitchers": selected_group_pitchers,
                "pitcher_filters": group_pitcher_filters,
                "selected_take_ids": selected_group_take_ids,
            })

            if group_idx < group_count:
                st.sidebar.markdown("---")

        st.sidebar.markdown("---")
        if st.sidebar.button("Create Another Group", key="create_another_group_btn", use_container_width=True):
            st.session_state["group_count"] = group_count + 1
            st.rerun()
        if st.sidebar.button("Exit Group Mode", key="exit_group_mode_btn", use_container_width=True):
            exit_group_mode()
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Control Group**")
        if not st.session_state.get("show_control_group_velocity"):
            if st.sidebar.button(
                "Create Control Group",
                key="create_control_group_btn_group_mode",
                use_container_width=True
            ):
                st.session_state["show_control_group_velocity"] = True
                st.rerun()

        # Merge group filters into the existing downstream data model.
        for group_cfg in group_configs:
            for pitcher in group_cfg["selected_pitchers"]:
                if pitcher not in selected_pitchers:
                    selected_pitchers.append(pitcher)

            for pitcher, cfg in group_cfg["pitcher_filters"].items():
                if pitcher not in pitcher_filters:
                    pitcher_filters[pitcher] = {
                        "selected_dates": list(cfg["selected_dates"]),
                        "throw_types": list(cfg["throw_types"]),
                        "velocity_min": cfg["velocity_min"],
                        "velocity_max": cfg["velocity_max"],
                    }
                    continue

                existing = pitcher_filters[pitcher]
                existing_dates = existing.get("selected_dates", [])
                new_dates = cfg.get("selected_dates", [])
                if "All Dates" in existing_dates or "All Dates" in new_dates:
                    existing["selected_dates"] = ["All Dates"]
                else:
                    existing["selected_dates"] = sorted(set(existing_dates + new_dates))

                existing["throw_types"] = sorted(set(existing.get("throw_types", []) + cfg.get("throw_types", [])))

                vmins = [v for v in [existing.get("velocity_min"), cfg.get("velocity_min")] if v is not None]
                vmaxs = [v for v in [existing.get("velocity_max"), cfg.get("velocity_max")] if v is not None]
                existing["velocity_min"] = min(vmins) if vmins else None
                existing["velocity_max"] = max(vmaxs) if vmaxs else None
    else:
        selected_pitchers = st.sidebar.multiselect(
            "Select Pitcher(s)",
            options=pitcher_names,
            default=[pitcher_names[0]] if pitcher_names else [],
            key="select_pitchers"
        )
        pitcher_filters = build_pitcher_filters_for_group(
            selected_pitchers,
            group_index=0,
            show_group_prefix=False
        )
        group_configs = [{
            "group_index": 1,
            "selected_pitchers": selected_pitchers,
            "pitcher_filters": pitcher_filters,
            "selected_take_ids": [],
        }]

selected_take_ids_union = set()
if group_mode_enabled:
    for group_cfg in group_configs:
        selected_take_ids_union.update(group_cfg.get("selected_take_ids", []))

group_palette = [
    "#1F77B4", "#D62728", "#2CA02C", "#FF7F0E", "#9467BD",
    "#17BECF", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22"
]

def get_group_display_label(group_cfg):
    group_idx = group_cfg["group_index"]
    return f"Group {group_idx}"

def is_control_group_label(label):
    return label == "Control Group"

group_color_map = {}
take_group_map = {}
if group_mode_enabled:
    for idx, group_cfg in enumerate(group_configs):
        group_label = get_group_display_label(group_cfg)
        group_color_map[group_label] = group_palette[idx % len(group_palette)]
        for tid in group_cfg.get("selected_take_ids", []):
            if tid not in take_group_map:
                take_group_map[tid] = group_label

all_throw_types = sorted({
    t
    for cfg in pitcher_filters.values()
    for t in cfg["throw_types"]
})
multi_pitcher_mode = len(selected_pitchers) > 1
mound_only_sidebar = bool(pitcher_filters) and all(
    set(cfg["throw_types"]) == {"Mound"}
    for cfg in pitcher_filters.values()
)

def render_group_selection_summary():
    if not group_mode_enabled:
        return
    if not group_configs:
        st.caption("Group mode active. No groups configured.")
        return

    for group_cfg in group_configs:
        group_idx = group_cfg["group_index"]
        group_pitchers = group_cfg["selected_pitchers"]
        group_pitcher_filters = group_cfg["pitcher_filters"]
        group_label = get_group_display_label(group_cfg)

        if not group_pitchers:
            st.caption(f"Group {group_idx} | No pitchers selected.")
            continue

        throw_types = sorted({
            throw_type
            for cfg in group_pitcher_filters.values()
            for throw_type in cfg.get("throw_types", [])
        })
        throw_types_label = ", ".join(throw_types) if throw_types else "None"

        per_pitcher_ranges = []
        for pitcher in group_pitchers:
            cfg = group_pitcher_filters.get(pitcher, {})
            vmin = cfg.get("velocity_min")
            vmax = cfg.get("velocity_max")
            if vmin is None or vmax is None:
                continue
            per_pitcher_ranges.append(f"{pitcher}: {vmin:.1f}-{vmax:.1f}")

        velocity_label = "; ".join(per_pitcher_ranges) if per_pitcher_ranges else "N/A"
        st.caption(
            f"{group_label} | "
            f"{', '.join(group_pitchers)} | "
            f"Throw Type: {throw_types_label} | "
            f"Velocity Range (mph): {velocity_label}"
        )

def aggregate_curves(curves_dict, stat="Median"):
    """
    curves_dict: { take_id: { "frame": [...], "value": [...] } }
    Returns aggregated_x, aggregated_y, iqr_low, iqr_high
    """
    all_frames = sorted(set(
        f for d in curves_dict.values() for f in d["frame"]
    ))

    agg_y = []
    iqr_low = []
    iqr_high = []

    for f in all_frames:
        vals = [
            d["value"][i]
            for d in curves_dict.values()
            for i, fr in enumerate(d["frame"])
            if fr == f
        ]

        if not vals:
            continue

        if stat == "Mean":
            agg_y.append(np.mean(vals))
        else:
            agg_y.append(np.median(vals))

        iqr_low.append(np.percentile(vals, 25))
        iqr_high.append(np.percentile(vals, 75))

    return all_frames, agg_y, iqr_low, iqr_high

def build_shared_dashboard_state():
    pitcher_handedness = {
        p: get_pitcher_handedness(p)
        for p in selected_pitchers
    }

    shared_take_ids = []
    shared_take_pitcher_map = {}
    primary_take_ids = []
    control_take_ids = []

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for pitcher, cfg in pitcher_filters.items():
                selected_dates_i = cfg["selected_dates"]
                throw_types_i = cfg["throw_types"]
                velocity_min_i = cfg["velocity_min"]
                velocity_max_i = cfg["velocity_max"]

                if velocity_min_i is None or velocity_max_i is None:
                    continue

                if "All Dates" in selected_dates_i or not selected_dates_i:
                    cur.execute("""
                        SELECT t.take_id
                        FROM takes t
                        JOIN athletes a ON a.athlete_id = t.athlete_id
                        WHERE a.athlete_name = %s
                          AND t.throw_type = ANY(%s)
                          AND t.pitch_velo BETWEEN %s AND %s
                    """, (pitcher, throw_types_i, velocity_min_i, velocity_max_i))
                else:
                    placeholders = ",".join(["%s"] * len(selected_dates_i))
                    cur.execute(f"""
                        SELECT t.take_id
                        FROM takes t
                        JOIN athletes a ON a.athlete_id = t.athlete_id
                        WHERE a.athlete_name = %s
                          AND t.throw_type = ANY(%s)
                          AND t.take_date IN ({placeholders})
                          AND t.pitch_velo BETWEEN %s AND %s
                    """, (pitcher, throw_types_i, *selected_dates_i, velocity_min_i, velocity_max_i))

                for (take_id,) in cur.fetchall():
                    if take_id not in shared_take_pitcher_map:
                        shared_take_pitcher_map[take_id] = pitcher
                        shared_take_ids.append(take_id)
    finally:
        conn.close()

    if group_mode_enabled:
        if selected_take_ids_union:
            shared_take_ids = [tid for tid in shared_take_ids if tid in selected_take_ids_union]
        else:
            shared_take_ids = []

    shared_take_handedness = {
        tid: pitcher_handedness.get(shared_take_pitcher_map.get(tid))
        for tid in shared_take_ids
    }
    shared_take_ids = [tid for tid in shared_take_ids if shared_take_handedness.get(tid) in ("R", "L")]

    shared_take_order = {}
    shared_take_velocity = {}
    shared_take_date_map = {}
    shared_take_pitcher_name_map = {}

    def merge_control_group_takes_into_shared_state():
        nonlocal shared_take_ids
        nonlocal shared_take_handedness
        nonlocal shared_take_order
        nonlocal shared_take_velocity
        nonlocal shared_take_date_map
        nonlocal shared_take_pitcher_name_map
        nonlocal primary_take_ids
        nonlocal control_take_ids

        primary_take_ids = list(shared_take_ids)
        control_take_ids = [
            tid for tid in st.session_state.get("control_group_take_ids", [])
            if tid not in primary_take_ids
        ]
        combined_take_ids = primary_take_ids + control_take_ids

        if not control_take_ids or not combined_take_ids:
            return

        conn = get_connection()
        try:
            with conn.cursor() as cur:
                placeholders = ",".join(["%s"] * len(combined_take_ids))
                cur.execute(f"""
                    SELECT t.take_id, t.pitch_velo, t.take_date, a.athlete_name, a.handedness
                    FROM takes t
                    JOIN athletes a ON a.athlete_id = t.athlete_id
                    WHERE t.take_id IN ({placeholders})
                    ORDER BY a.athlete_name, t.take_date, t.take_id
                """, tuple(combined_take_ids))
                combined_rows = cur.fetchall()
        finally:
            conn.close()

        from collections import defaultdict

        shared_take_ids = []
        shared_take_handedness = {}
        shared_take_order = {}
        shared_take_velocity = {}
        shared_take_date_map = {}
        shared_take_pitcher_name_map = {}

        combined_date_groups = defaultdict(list)
        for tid, velo, date, pitcher, handedness in combined_rows:
            if handedness not in ("R", "L"):
                continue
            shared_take_ids.append(tid)
            shared_take_handedness[tid] = handedness
            shared_take_velocity[tid] = velo
            shared_take_date_map[tid] = date.strftime("%Y-%m-%d")
            shared_take_pitcher_name_map[tid] = pitcher
            combined_date_groups[(pitcher, date)].append((tid, velo))

        for (pitcher, date), items in combined_date_groups.items():
            for i, (tid, velo) in enumerate(items, start=1):
                shared_take_order[tid] = i

    if shared_take_ids:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                placeholders = ",".join(["%s"] * len(shared_take_ids))
                cur.execute(f"""
                    SELECT t.take_id, t.pitch_velo, t.take_date, a.athlete_name
                    FROM takes t
                    JOIN athletes a ON a.athlete_id = t.athlete_id
                    WHERE t.take_id IN ({placeholders})
                    ORDER BY a.athlete_name, t.take_date, t.take_id
                """, tuple(shared_take_ids))
                rows = cur.fetchall()
        finally:
            conn.close()

        from collections import defaultdict

        date_groups = defaultdict(list)
        for tid, velo, date, pitcher in rows:
            date_groups[(pitcher, date)].append((tid, velo))

        for (pitcher, date), items in date_groups.items():
            for i, (tid, velo) in enumerate(items, start=1):
                shared_take_order[tid] = i
                shared_take_velocity[tid] = velo
                shared_take_date_map[tid] = date.strftime("%Y-%m-%d")
                shared_take_pitcher_name_map[tid] = pitcher

        if not group_mode_enabled:
            take_options = [
                (
                    f"{shared_take_pitcher_name_map[tid]} | {shared_take_date_map[tid]} - "
                    f"Pitch {shared_take_order[tid]} ({shared_take_velocity[tid]:.1f} mph)"
                )
                for tid in shared_take_ids
            ]

            label_to_take_id = {
                (
                    f"{shared_take_pitcher_name_map[tid]} | {shared_take_date_map[tid]} - "
                    f"Pitch {shared_take_order[tid]} ({shared_take_velocity[tid]:.1f} mph)"
                ): tid
                for tid in shared_take_ids
            }

            excluded_labels = st.sidebar.multiselect(
                "Exclude Takes",
                options=take_options,
                default=[
                    label for label, tid in label_to_take_id.items()
                    if tid in st.session_state["excluded_take_ids"]
                ],
                key="exclude_takes"
            )

            st.session_state["excluded_take_ids"] = [
                label_to_take_id[label] for label in excluded_labels
            ]
            if st.sidebar.button("Create Custom Groups", key="create_groups_mode_btn", use_container_width=True):
                st.session_state["create_groups_mode"] = True
                st.rerun()
            if not st.session_state.get("show_control_group_velocity"):
                if st.sidebar.button(
                    "Create Control Group",
                    key="create_control_group_btn",
                    use_container_width=True
                ):
                    st.session_state["show_control_group_velocity"] = True
                    st.rerun()
            shared_take_ids = [
                tid for tid in shared_take_ids
                if tid not in st.session_state["excluded_take_ids"]
            ]
            shared_take_handedness = {
                tid: shared_take_handedness[tid]
                for tid in shared_take_ids
                if tid in shared_take_handedness
            }
            shared_take_order = {
                tid: shared_take_order[tid]
                for tid in shared_take_ids
                if tid in shared_take_order
            }
            shared_take_velocity = {
                tid: shared_take_velocity[tid]
                for tid in shared_take_ids
                if tid in shared_take_velocity
            }
            shared_take_date_map = {
                tid: shared_take_date_map[tid]
                for tid in shared_take_ids
                if tid in shared_take_date_map
            }
            shared_take_pitcher_name_map = {
                tid: shared_take_pitcher_name_map[tid]
                for tid in shared_take_ids
                if tid in shared_take_pitcher_name_map
            }
            if st.session_state.get("show_control_group_velocity"):
                st.sidebar.markdown("**Control Group**")
                if st.sidebar.button(
                    "Remove Control Group",
                    key="remove_control_group_btn",
                    use_container_width=True
                ):
                    remove_control_group()
                with st.sidebar.form("control_group_filters_form"):
                    st.multiselect(
                        "Pitchers",
                        options=pitcher_names,
                        key="control_group_pitchers"
                    )
                    st.radio(
                        "Handedness",
                        options=["Both", "Left", "Right"],
                        key="control_group_handedness",
                        horizontal=True
                    )
                    st.slider(
                        "Velocity Range (mph)",
                        min_value=50.0,
                        max_value=100.0,
                        value=st.session_state.get("control_group_velocity_range", (50.0, 100.0)),
                        step=0.5,
                        key="control_group_velocity_range"
                    )
                    render_control_group_arm_slot_category_checkboxes()
                    generate_control_group = st.form_submit_button(
                        "Generate Control Group",
                        use_container_width=True
                    )

                if generate_control_group:
                    handedness_filter = st.session_state.get("control_group_handedness", "Both")
                    pool_handedness = (
                        "L" if handedness_filter == "Left"
                        else "R" if handedness_filter == "Right"
                        else None
                    )
                    all_control_group_pool = get_control_group_take_pool(pool_handedness)
                    selected_pitcher_set = set(st.session_state.get("control_group_pitchers", []))
                    selected_velocity_range = st.session_state.get("control_group_velocity_range", (50.0, 100.0))

                    final_candidate_control_take_ids = []
                    for take_id, pitch_velo, athlete_name, _, arm_slot_deg in all_control_group_pool:
                        if selected_pitcher_set and athlete_name not in selected_pitcher_set:
                            continue
                        if pitch_velo is None or not (selected_velocity_range[0] <= float(pitch_velo) <= selected_velocity_range[1]):
                            continue
                        if not arm_slot_matches_control_group_categories(arm_slot_deg):
                            continue
                        final_candidate_control_take_ids.append(take_id)

                    st.session_state["control_group_take_ids"] = list(final_candidate_control_take_ids)
                    st.session_state["control_group_arm_slot_ids"] = list(final_candidate_control_take_ids)
                    st.session_state["control_group_status_message"] = (
                        f"Total Pitches: {len(final_candidate_control_take_ids)}"
                        if final_candidate_control_take_ids else
                        "No control-group takes found for the selected filters."
                    )
                    st.rerun()

                if st.session_state.get("control_group_status_message"):
                    st.sidebar.caption(st.session_state["control_group_status_message"])
                if st.session_state.get("show_control_group_velocity"):
                    merge_control_group_takes_into_shared_state()
        elif group_mode_enabled and st.session_state.get("show_control_group_velocity"):
            if st.sidebar.button(
                "Remove Control Group",
                key="remove_control_group_btn_group_mode",
                use_container_width=True
            ):
                remove_control_group()
            with st.sidebar.form("control_group_filters_form_group_mode"):
                st.multiselect(
                    "Pitchers",
                    options=pitcher_names,
                    key="control_group_pitchers"
                )
                st.radio(
                    "Handedness",
                    options=["Both", "Left", "Right"],
                    key="control_group_handedness",
                    horizontal=True
                )
                st.slider(
                    "Velocity Range (mph)",
                    min_value=50.0,
                    max_value=100.0,
                    value=st.session_state.get("control_group_velocity_range", (50.0, 100.0)),
                    step=0.5,
                    key="control_group_velocity_range"
                )
                render_control_group_arm_slot_category_checkboxes()
                generate_control_group = st.form_submit_button(
                    "Generate Control Group",
                    use_container_width=True
                )

            if generate_control_group:
                handedness_filter = st.session_state.get("control_group_handedness", "Both")
                pool_handedness = (
                    "L" if handedness_filter == "Left"
                    else "R" if handedness_filter == "Right"
                    else None
                )
                all_control_group_pool = get_control_group_take_pool(pool_handedness)
                selected_pitcher_set = set(st.session_state.get("control_group_pitchers", []))
                selected_velocity_range = st.session_state.get("control_group_velocity_range", (50.0, 100.0))

                final_candidate_control_take_ids = []
                for take_id, pitch_velo, athlete_name, _, arm_slot_deg in all_control_group_pool:
                    if selected_pitcher_set and athlete_name not in selected_pitcher_set:
                        continue
                    if pitch_velo is None or not (selected_velocity_range[0] <= float(pitch_velo) <= selected_velocity_range[1]):
                        continue
                    if not arm_slot_matches_control_group_categories(arm_slot_deg):
                        continue
                    final_candidate_control_take_ids.append(take_id)

                st.session_state["control_group_take_ids"] = list(final_candidate_control_take_ids)
                st.session_state["control_group_arm_slot_ids"] = list(final_candidate_control_take_ids)
                st.session_state["control_group_status_message"] = (
                    f"Total Pitches: {len(final_candidate_control_take_ids)}"
                    if final_candidate_control_take_ids else
                    "No control-group takes found for the selected filters."
                )
                st.rerun()

            if st.session_state.get("control_group_status_message"):
                st.sidebar.caption(st.session_state["control_group_status_message"])

            merge_control_group_takes_into_shared_state()
        else:
            primary_take_ids = []
            control_take_ids = []
    else:
        primary_take_ids = []
        control_take_ids = []

    from collections import defaultdict

    shared_take_ids_by_handedness = defaultdict(list)
    for tid in shared_take_ids:
        hand = shared_take_handedness.get(tid)
        if hand in ("R", "L"):
            shared_take_ids_by_handedness[hand].append(tid)

    def load_by_handedness(loader_fn):
        merged = {}
        for hand, ids in shared_take_ids_by_handedness.items():
            if ids:
                merged.update(loader_fn(ids, hand))
        return merged

    shared_br_frames = {}
    shared_shoulder_er_max_frames = {}
    shared_knee_peak_frames = {}
    shared_foot_plant_zero_cross_frames = {}
    shared_knee_event_frames = []
    shared_fp_event_frames = []
    shared_mer_event_frames = []
    shared_window_start = -100

    if shared_take_ids:
        cg_data = load_by_handedness(get_hand_cg_velocity)
        shoulder_data = load_by_handedness(get_shoulder_er_angles)

        for take_id in shared_take_ids:
            if take_id in cg_data:
                cg_frames = cg_data[take_id]["frame"]
                cg_vals = cg_data[take_id]["x"]
                valid = [(i, v) for i, v in enumerate(cg_vals) if v is not None]
                if valid:
                    idx, _ = max(valid, key=lambda x: x[1])
                    shared_br_frames[take_id] = cg_frames[idx]

        for take_id, d in shoulder_data.items():
            frames = d["frame"]
            values = d["z"]
            valid = [(f, v) for f, v in zip(frames, values) if v is not None]
            if not valid:
                continue

            hand = shared_take_handedness.get(take_id)
            if hand == "R":
                er_frame, _ = min(valid, key=lambda x: x[1])
            else:
                er_frame, _ = max(valid, key=lambda x: x[1])
            shared_shoulder_er_max_frames[take_id] = er_frame

        ankle_prox_x_peak_frames = {}
        ankle_min_frames = {}
        heel_contact_frames = {}
        for hand, ids in shared_take_ids_by_handedness.items():
            if not ids:
                continue
            shared_knee_peak_frames.update(
                get_peak_glove_knee_pre_br(ids, hand, shared_br_frames)
            )
            hand_ankle_prox_x_peak_frames = get_peak_ankle_prox_x_velocity(ids, hand)
            ankle_prox_x_peak_frames.update(hand_ankle_prox_x_peak_frames)
            hand_ankle_min_frames = get_terra_ankle_min_frame(
                ids,
                hand,
                hand_ankle_prox_x_peak_frames,
                shared_shoulder_er_max_frames
            )
            ankle_min_frames.update(hand_ankle_min_frames)
            ankle_zero_cross_frames = get_foot_plant_frame_zero_cross(
                ids,
                hand,
                hand_ankle_min_frames,
                shared_shoulder_er_max_frames
            )
            heel_anchor_frames = {
                take_id: ankle_zero_cross_frames.get(take_id, hand_ankle_min_frames.get(take_id))
                for take_id in ids
            }
            heel_contact_frames.update(
                get_lead_heel_contact_frame(
                    ids,
                    hand,
                    hand_ankle_prox_x_peak_frames,
                    shared_shoulder_er_max_frames,
                    heel_anchor_frames
                )
            )

            for take_id in ids:
                ankle_fp_frame = ankle_zero_cross_frames.get(take_id)
                heel_fp_frame = heel_contact_frames.get(take_id)
                ankle_min_frame = hand_ankle_min_frames.get(take_id)
                prox_peak_frame = hand_ankle_prox_x_peak_frames.get(take_id)

                if ankle_fp_frame is not None and heel_fp_frame is not None:
                    shared_foot_plant_zero_cross_frames[take_id] = int(max(ankle_fp_frame, heel_fp_frame))
                elif ankle_fp_frame is not None:
                    shared_foot_plant_zero_cross_frames[take_id] = int(ankle_fp_frame)
                elif heel_fp_frame is not None:
                    shared_foot_plant_zero_cross_frames[take_id] = int(heel_fp_frame)
                elif ankle_min_frame is not None:
                    shared_foot_plant_zero_cross_frames[take_id] = int(ankle_min_frame)
                elif prox_peak_frame is not None:
                    shared_foot_plant_zero_cross_frames[take_id] = int(prox_peak_frame)

        for take_id, fp_frame in shared_foot_plant_zero_cross_frames.items():
            if take_id in shared_shoulder_er_max_frames:
                er_frame = shared_shoulder_er_max_frames[take_id]
                if fp_frame > er_frame:
                    shared_foot_plant_zero_cross_frames[take_id] = er_frame

        for take_id, fp_frame in shared_foot_plant_zero_cross_frames.items():
            if take_id in shared_br_frames:
                shared_fp_event_frames.append(fp_frame - shared_br_frames[take_id])
        for take_id, knee_frame in shared_knee_peak_frames.items():
            if take_id in shared_br_frames:
                shared_knee_event_frames.append(knee_frame - shared_br_frames[take_id])

        for take_id, er_frame in shared_shoulder_er_max_frames.items():
            if take_id in shared_br_frames:
                shared_mer_event_frames.append(er_frame - shared_br_frames[take_id])

        if shared_fp_event_frames:
            shared_window_start = int(np.median(shared_fp_event_frames)) - 50

    return {
        "take_ids": shared_take_ids,
        "primary_take_ids": primary_take_ids,
        "control_take_ids": control_take_ids,
        "take_order": shared_take_order,
        "take_velocity": shared_take_velocity,
        "take_date_map": shared_take_date_map,
        "take_pitcher_map": shared_take_pitcher_name_map,
        "take_handedness": shared_take_handedness,
        "take_ids_by_handedness": shared_take_ids_by_handedness,
        "br_frames": shared_br_frames,
        "foot_plant_zero_cross_frames": shared_foot_plant_zero_cross_frames,
        "shoulder_er_max_frames": shared_shoulder_er_max_frames,
        "knee_peak_frames": shared_knee_peak_frames,
        "fp_event_frames": shared_fp_event_frames,
        "knee_event_frames": shared_knee_event_frames,
        "mer_event_frames": shared_mer_event_frames,
        "window_start": shared_window_start,
    }






st.title("Biomechanics Viewer")

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab_kinematic, tab_joint, tab_energy, tab1, tab2, tab3, tab5, tab6 = st.tabs([
    "Kinematic Sequence",
    "Kinematics",
    "Energy Flow",
    "Compensation Analysis",
    "Session Comparison",
    "0-10 Report",
    "Biodex",
    "Biodex (Test)",
])

shoulder_er_max_frames = {}
knee_peak_frames = {}
fp_event_frames = []
knee_event_frames = []
mer_event_frames = []
window_start = -100
# Workaround for Streamlit tab reset on rerun:
# persist active tab in URL query param and re-select it after rerender.
components.html(
    """
    <script>
    const TAB_PARAM = "active_tab";
    const TAB_SYNC_VERSION = "4";
    let restoringTab = false;
    let hasInitialRestore = false;

    function getActiveTabFromUrl() {
      const url = new URL(parent.window.location.href);
      return url.searchParams.get(TAB_PARAM);
    }

    function setActiveTabInUrl(tabLabel, shouldReload = false) {
      const url = new URL(parent.window.location.href);
      if (url.searchParams.get(TAB_PARAM) === tabLabel) return;

      url.searchParams.set(TAB_PARAM, tabLabel);
      if (shouldReload) {
        parent.window.location.href = url.toString();
      } else {
        parent.window.history.replaceState({}, "", url.toString());
      }
    }

    function getTabButtons() {
      return Array.from(parent.document.querySelectorAll('button[role="tab"]'));
    }

    function getSelectedTabLabel() {
      const selected = getTabButtons().find(
        (button) => button.getAttribute("aria-selected") === "true"
      );
      return selected ? selected.textContent.trim() : "";
    }

    function setElementHidden(element, hidden) {
      if (!element) return;
      element.style.display = hidden ? "none" : "";
    }

    function toggleCompensationSidebarControls() {
      const sidebar = parent.document.querySelector('[data-testid="stSidebar"]');
      if (!sidebar) return;

      const showCompensationControls =
        getSelectedTabLabel() === "Compensation Analysis";
      const hideCompensationControls = !showCompensationControls;

      sidebar
        .querySelectorAll(".st-key-tab1_energy_plot_options")
        .forEach((element) => {
          setElementHidden(element, hideCompensationControls);
        });
    }

    function handleUserTabActivation(button) {
      if (restoringTab) return;
      setActiveTabInUrl(button.textContent.trim(), true);
    }

    function bindTabClicks() {
      const buttons = getTabButtons();
      buttons.forEach((button) => {
        if (button.dataset.terraTabBound === TAB_SYNC_VERSION) return;
        button.dataset.terraTabBound = TAB_SYNC_VERSION;
        button.addEventListener("pointerdown", () => {
          handleUserTabActivation(button);
        }, true);
        button.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            handleUserTabActivation(button);
          }
        }, true);
        button.addEventListener("click", () => {
          handleUserTabActivation(button);
        }, true);
      });
    }

    function restoreActiveTab() {
      if (hasInitialRestore) return;

      const desiredTab = getActiveTabFromUrl();
      if (!desiredTab) {
        hasInitialRestore = true;
        return;
      }

      const buttons = getTabButtons();
      const target = buttons.find(
        (button) => button.textContent.trim() === desiredTab
      );
      if (!target) {
        hasInitialRestore = true;
        return;
      }

      if (target.getAttribute("aria-selected") !== "true") {
        restoringTab = true;
        target.click();
        setTimeout(() => {
          restoringTab = false;
        }, 0);
      }
      hasInitialRestore = true;
    }

    function syncSelectedTabToUrl() {
      if (!hasInitialRestore || restoringTab) return;

      const selected = getTabButtons().find(
        (button) => button.getAttribute("aria-selected") === "true"
      );
      if (!selected) return;

      const selectedTab = selected.textContent.trim();
      const activeTab = getActiveTabFromUrl();

      if (activeTab && selectedTab !== activeTab) {
        setActiveTabInUrl(selectedTab, true);
      } else if (!activeTab) {
        setActiveTabInUrl(selectedTab);
      }
    }

    function syncTabs() {
      bindTabClicks();
      restoreActiveTab();
      toggleCompensationSidebarControls();
    }

    syncTabs();
    if (parent.window.__terraTabSyncInterval) {
      clearInterval(parent.window.__terraTabSyncInterval);
    }
    parent.window.__terraTabSyncInterval = setInterval(syncTabs, 250);
    </script>
    """,
    height=0,
)

shared_state = build_shared_dashboard_state()
take_ids = shared_state["take_ids"]
primary_take_ids = shared_state["primary_take_ids"]
control_take_ids = shared_state["control_take_ids"]
take_order = shared_state["take_order"]
take_velocity = shared_state["take_velocity"]
take_date_map = shared_state["take_date_map"]
br_frames = shared_state["br_frames"]
take_pitcher_map = shared_state["take_pitcher_map"]
take_handedness = shared_state["take_handedness"]
take_ids_by_handedness = shared_state["take_ids_by_handedness"]
foot_plant_zero_cross_frames = shared_state["foot_plant_zero_cross_frames"]
shoulder_er_max_frames = shared_state["shoulder_er_max_frames"]
knee_peak_frames = shared_state["knee_peak_frames"]
fp_event_frames = shared_state["fp_event_frames"]
knee_event_frames = shared_state["knee_event_frames"]
mer_event_frames = shared_state["mer_event_frames"]
window_start = shared_state["window_start"]
comparison_grouping_enabled = group_mode_enabled or bool(control_take_ids)

if control_take_ids:
    control_group_label = "Control Group"
    if not group_mode_enabled:
        selected_group_label = ", ".join(selected_pitchers) if selected_pitchers else "Selected Takes"
        group_color_map = {
            selected_group_label: group_palette[0],
            control_group_label: group_palette[1]
        }
        take_group_map = {
            **{tid: selected_group_label for tid in primary_take_ids},
            **{tid: control_group_label for tid in control_take_ids}
        }
    else:
        group_color_map[control_group_label] = group_palette[len(group_color_map) % len(group_palette)]
        for tid in control_take_ids:
            take_group_map[tid] = control_group_label

multi_pitcher_mode = len(set(take_pitcher_map.values())) > 1
group_mode_aggregate_across_pitchers = group_mode_enabled
show_group_pitcher_breakout = multi_pitcher_mode and not group_mode_aggregate_across_pitchers



with tab_kinematic:
    st.subheader("Kinematic Sequence")
    render_group_selection_summary()
    st.markdown(
        """
        <style>
        .ks-controls-label {
            font-size: 0.8rem;
            font-weight: 700;
            color: #6b7280;
            margin-bottom: 0.1rem;
        }

        div[data-testid="stSegmentedControl"] label,
        div[data-testid="stSegmentedControl"] div[role="radiogroup"] label,
        div[data-testid="stSegmentedControl"] div[role="radiogroup"] p,
        div[data-testid="stToggle"] label,
        div[data-testid="stToggle"] p {
            font-size: 1rem !important;
            font-weight: 400 !important;
        }

        .ks-toggle-label {
            margin-top: -0.1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    controls_col, options_col, spacer_col = st.columns([1.45, 1.75, 2.2])
    with controls_col:
        st.markdown('<div class="ks-controls-label">Display Mode</div>', unsafe_allow_html=True)
        display_mode = st.segmented_control(
            "Display Mode",
            ["Individual Throws", "Grouped"],
            default="Grouped",
            key="ks_display_mode",
            label_visibility="collapsed",
        )
    with options_col:
        st.markdown('<div class="ks-controls-label ks-toggle-label">Options</div>', unsafe_allow_html=True)
        event_toggle_col, signal_toggle_col = st.columns(2)
        with event_toggle_col:
            show_ks_fp_iqr_band = st.toggle(
                "Event Bands",
                value=False,
                key="ks_show_fp_iqr_band",
                help="Shows the middle 50% range for event timing across selected throws.",
            )
        with signal_toggle_col:
            show_ks_signal_iqr_band = st.toggle(
                "Signal Bands",
                value=True,
                key="ks_show_signal_iqr_band",
                help="Shows the middle 50% range of angular velocity around each grouped mean line.",
            )
    with spacer_col:
        st.markdown("")
    st.markdown(
        """
        <style>
        div[data-testid="stHorizontalBlock"] + div[data-testid="stVerticalBlock"] {
            margin-top: -0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if not take_ids:
        st.info("No takes found for this selection.")
    else:
        def load_by_handedness(loader_fn):
            merged = {}
            for hand, ids in take_ids_by_handedness.items():
                if ids:
                    merged.update(loader_fn(ids, hand))
            return merged

        data = get_pelvis_angular_velocity(take_ids)
        cg_data = load_by_handedness(get_hand_cg_velocity)
        torso_data = get_torso_angular_velocity(take_ids)
        elbow_data = load_by_handedness(get_elbow_angular_velocity)
        shoulder_ir_data = load_by_handedness(get_shoulder_ir_velocity)
        pre_fp_frames = ms_to_rel_frame(100)
        post_br_frames = ms_to_rel_frame(150)
        kinematic_window_start = (
            int(np.median(fp_event_frames)) - pre_fp_frames
            if fp_event_frames else -pre_fp_frames
        )
        kinematic_window_end = post_br_frames
        window_start_ms = rel_frame_to_ms(kinematic_window_start)
        window_end_ms = rel_frame_to_ms(kinematic_window_end)

        fig = go.Figure()
        grouped_pelvis = {}
        grouped_torso = {}
        grouped_elbow = {}
        grouped_shoulder_ir = {}

        # --- Date-based dash style map (Kinematic Sequence) ---
        unique_dates = sorted(set(take_date_map.values()))
        dash_styles = ["solid", "dash", "dot", "dashdot"]
        date_dash_map = {
            d: dash_styles[i % len(dash_styles)]
            for i, d in enumerate(unique_dates)
        }

        # Track legend entries to avoid duplicates (for condensed legend)
        legend_keys_added = set()

        for take_id, d in data.items():
            frames = d["frame"]
            values = d["z"]
            take_hand = take_handedness.get(take_id)
            take_group_label = take_group_map.get(take_id, "")
            control_group_take = is_control_group_label(take_group_label)
            hover_pitcher_name = "" if control_group_take else take_pitcher_map.get(take_id, "")

            # -----------------------------
            # Ball Release Detection (CGVel)
            # -----------------------------
            if take_id not in cg_data:
                continue

            cg_frames = cg_data[take_id]["frame"]
            cg_values = cg_data[take_id]["x"]

            valid_cg = [(i, v) for i, v in enumerate(cg_values) if v is not None]
            if not valid_cg:
                continue

            br_idx, _ = max(valid_cg, key=lambda x: x[1])
            br_frame = cg_frames[br_idx]

            # -----------------------------
            # Peak Glove-Side Knee Height
            # -----------------------------
            knee_rel_frame = None
            if take_id in knee_peak_frames:
                knee_rel_frame = knee_peak_frames[take_id] - br_frame

            # MER defined as max shoulder external rotation prior to ball release
            # -----------------------------
            mer_rel_frame = None
            if take_id in shoulder_er_max_frames:
                mer_rel_frame = shoulder_er_max_frames[take_id] - br_frame

            # -----------------------------
            # Normalize time to Ball Release
            # -----------------------------
            norm_frames = []
            norm_values = []

            for f, v in zip(frames, values):
                if v is None:
                    continue

                rel_frame = f - br_frame

                # Keep frames from 150 before median FP through +150 after BR
                if (
                    rel_frame >= kinematic_window_start
                    and rel_frame <= kinematic_window_end
                ):
                    norm_frames.append(rel_frame_to_ms(rel_frame))
                    # Handedness normalization for Pelvis AV (Kinematic Sequence only)
                    if take_hand == "L":
                        norm_values.append(-v)
                    else:
                        norm_values.append(v)

            grouped_pelvis[take_id] = {
                "frame": norm_frames,
                "value": norm_values
            }

            # -----------------------------
            # Normalize Torso Angular Velocity
            # -----------------------------
            if take_id in torso_data:
                torso_frames = torso_data[take_id]["frame"]
                torso_values = torso_data[take_id]["z"]

                norm_torso_frames = []
                norm_torso_values = []

                for f, v in zip(torso_frames, torso_values):
                    if v is None:
                        continue

                    rel_frame = f - br_frame
                    if (
                        rel_frame >= kinematic_window_start
                        and rel_frame <= kinematic_window_end
                    ):
                        norm_torso_frames.append(rel_frame_to_ms(rel_frame))
                        # Handedness normalization for Torso AV (Kinematic Sequence only)
                        if take_hand == "L":
                            norm_torso_values.append(-v)
                        else:
                            norm_torso_values.append(v)

                grouped_torso[take_id] = {
                    "frame": norm_torso_frames,
                    "value": norm_torso_values
                }

                if norm_torso_frames and display_mode == "Individual Throws":
                    legendgroup = "Control_Group_Torso" if control_group_take else f"Torso_{take_date_map[take_id]}"
                    pitcher_name = take_pitcher_map.get(take_id, "")
                    trace_name = (
                        f"Control Group | Torso – Pitch {take_order[take_id]} ({take_velocity[take_id]:.1f} MPH)"
                    ) if control_group_take else (
                        f"{take_group_label} | Torso – {take_date_map[take_id]} | "
                        f"Pitch {take_order[take_id]} ({take_velocity[take_id]:.1f} MPH)"
                    ) if comparison_grouping_enabled else None
                    # Actual data trace (no legend)
                    fig.add_trace(
                        go.Scatter(
                            x=norm_torso_frames,
                            y=norm_torso_values,
                            mode="lines",
                            line=dict(
                                color="orange",
                                dash=date_dash_map[take_date_map[take_id]]
                            ),
                            customdata=[[ "Torso", take_date_map[take_id], take_order[take_id], take_velocity[take_id], hover_pitcher_name ]] * len(norm_torso_frames),
                            hovertemplate=(
                                "%{customdata[0]} – %{customdata[1]} | "
                                "Pitch %{customdata[2]} (%{customdata[3]:.1f} MPH)"
                                + (" | %{customdata[4]}" if multi_pitcher_mode else "")
                                + "<br>Angular Velocity: %{y:.1f}°/s"
                                + "<br>Time: %{x:.0f} ms rel BR"
                                + "<extra></extra>"
                            ),
                            name=trace_name,
                            showlegend=False,
                            legendgroup=legendgroup,
                            legendgrouptitle_text=None
                        )
                    )
                    # Legend-only trace (once per Torso + Date)
                    legend_key = ("Control Group", "Torso") if control_group_take else None
                    if control_group_take and legend_key not in legend_keys_added:
                        fig.add_trace(
                            go.Scatter(
                                x=[None],
                                y=[None],
                                mode="lines",
                                line=dict(
                                    color="orange",
                                    dash=date_dash_map[take_date_map[take_id]],
                                    width=4
                                ),
                                name=(
                                    f"Control Group | Torso AV"
                                    if (comparison_grouping_enabled and control_group_take) else
                                    f"{take_group_label} | Torso AV | {take_date_map[take_id]} | {pitcher_name}"
                                    if (comparison_grouping_enabled and multi_pitcher_mode) else
                                    f"{take_group_label} | Torso AV | {take_date_map[take_id]}"
                                    if comparison_grouping_enabled else
                                    f"Torso AV | {take_date_map[take_id]} | {pitcher_name}"
                                    if multi_pitcher_mode else
                                    f"Torso AV | {take_date_map[take_id]}"
                                ),
                                showlegend=True,
                                legendgroup=legendgroup,
                                legendgrouptitle_text=None
                            )
                        )
                        legend_keys_added.add(legend_key)

            # -----------------------------
            # Normalize Elbow Angular Velocity (Extension)
            # -----------------------------
            if take_id in elbow_data:
                elbow_frames = elbow_data[take_id]["frame"]
                elbow_values = elbow_data[take_id]["x"]

                norm_elbow_frames = []
                norm_elbow_values = []

                for f, v in zip(elbow_frames, elbow_values):
                    if v is None:
                        continue

                    rel_frame = f - br_frame
                    if (
                        rel_frame >= kinematic_window_start
                        and rel_frame <= kinematic_window_end
                    ):
                        norm_elbow_frames.append(rel_frame_to_ms(rel_frame))
                        # Flip sign so elbow extension is positive on the plot
                        norm_elbow_values.append(-v)

                grouped_elbow[take_id] = {
                    "frame": norm_elbow_frames,
                    "value": norm_elbow_values
                }

                if norm_elbow_frames and display_mode == "Individual Throws":
                    legendgroup = "Control_Group_Elbow" if control_group_take else f"Elbow_{take_date_map[take_id]}"
                    pitcher_name = take_pitcher_map.get(take_id, "")
                    # Actual data trace (no legend)
                    fig.add_trace(
                        go.Scatter(
                            x=norm_elbow_frames,
                            y=norm_elbow_values,
                            mode="lines",
                            line=dict(
                                color="green",
                                dash=date_dash_map[take_date_map[take_id]]
                            ),
                            customdata=[[ "Elbow", take_date_map[take_id], take_order[take_id], take_velocity[take_id], hover_pitcher_name ]] * len(norm_elbow_frames),
                            hovertemplate=(
                                "%{customdata[0]} – %{customdata[1]} | "
                                "Pitch %{customdata[2]} (%{customdata[3]:.1f} MPH)"
                                + (" | %{customdata[4]}" if multi_pitcher_mode else "")
                                + "<br>Angular Velocity: %{y:.1f}°/s"
                                + "<br>Time: %{x:.0f} ms rel BR"
                                + "<extra></extra>"
                            ),
                            name=(
                                f"Control Group | Elbow – Pitch {take_order[take_id]} ({take_velocity[take_id]:.1f} MPH)"
                            ) if control_group_take else (
                                f"{take_group_label} | Elbow – {take_date_map[take_id]} | "
                                f"Pitch {take_order[take_id]} ({take_velocity[take_id]:.1f} MPH)"
                            ) if comparison_grouping_enabled else None,
                            showlegend=False,
                            legendgroup=legendgroup,
                            legendgrouptitle_text=None
                        )
                    )
                    # Legend-only trace (once per Elbow + Date)
                    legend_key = ("Control Group", "Elbow") if control_group_take else None
                    if control_group_take and legend_key not in legend_keys_added:
                        fig.add_trace(
                            go.Scatter(
                                x=[None],
                                y=[None],
                                mode="lines",
                                line=dict(
                                    color="green",
                                    dash=date_dash_map[take_date_map[take_id]],
                                    width=4
                                ),
                                name=(
                                    f"Control Group | Elbow AV"
                                    if (comparison_grouping_enabled and control_group_take) else
                                    f"{take_group_label} | Elbow AV | {take_date_map[take_id]} | {pitcher_name}"
                                    if (comparison_grouping_enabled and multi_pitcher_mode) else
                                    f"{take_group_label} | Elbow AV | {take_date_map[take_id]}"
                                    if comparison_grouping_enabled else
                                    f"Elbow AV | {take_date_map[take_id]} | {pitcher_name}"
                                    if multi_pitcher_mode else
                                    f"Elbow AV | {take_date_map[take_id]}"
                                ),
                                showlegend=True,
                                legendgroup=legendgroup,
                                legendgrouptitle_text=None
                            )
                        )
                        legend_keys_added.add(legend_key)

            # -----------------------------
            # Normalize Shoulder IR Angular Velocity
            # -----------------------------
            if take_id in shoulder_ir_data:
                sh_frames = shoulder_ir_data[take_id]["frame"]
                sh_values = shoulder_ir_data[take_id]["x"]

                norm_sh_frames = []
                norm_sh_values = []

                for f, v in zip(sh_frames, sh_values):
                    if v is None:
                        continue

                    rel_frame = f - br_frame
                    if (
                        rel_frame >= kinematic_window_start
                        and rel_frame <= kinematic_window_end
                    ):
                        norm_sh_frames.append(rel_frame_to_ms(rel_frame))
                        # Normalize so IR velocity is positive for both handedness
                        if take_hand == "L":
                            norm_sh_values.append(-v)
                        else:
                            norm_sh_values.append(v)

                grouped_shoulder_ir[take_id] = {
                    "frame": norm_sh_frames,
                    "value": norm_sh_values
                }

                if norm_sh_frames and display_mode == "Individual Throws":
                    legendgroup = "Control_Group_Shoulder_IR" if control_group_take else f"Shoulder IR_{take_date_map[take_id]}"
                    pitcher_name = take_pitcher_map.get(take_id, "")
                    # Actual data trace (no legend)
                    fig.add_trace(
                        go.Scatter(
                            x=norm_sh_frames,
                            y=norm_sh_values,
                            mode="lines",
                            line=dict(
                                color="red",
                                dash=date_dash_map[take_date_map[take_id]]
                            ),
                            customdata=[[ "Shoulder", take_date_map[take_id], take_order[take_id], take_velocity[take_id], hover_pitcher_name ]] * len(norm_sh_frames),
                            hovertemplate=(
                                "%{customdata[0]} – %{customdata[1]} | "
                                "Pitch %{customdata[2]} (%{customdata[3]:.1f} MPH)"
                                + (" | %{customdata[4]}" if multi_pitcher_mode else "")
                                + "<br>Angular Velocity: %{y:.1f}°/s"
                                + "<br>Time: %{x:.0f} ms rel BR"
                                + "<extra></extra>"
                            ),
                            name=(
                                f"Control Group | Shoulder – Pitch {take_order[take_id]} ({take_velocity[take_id]:.1f} MPH)"
                            ) if control_group_take else (
                                f"{take_group_label} | Shoulder – {take_date_map[take_id]} | "
                                f"Pitch {take_order[take_id]} ({take_velocity[take_id]:.1f} MPH)"
                            ) if comparison_grouping_enabled else None,
                            showlegend=False,
                            legendgroup=legendgroup,
                            legendgrouptitle_text=None
                        )
                    )
                    # Legend-only trace (once per Shoulder IR + Date)
                    legend_key = ("Control Group", "Shoulder IR") if control_group_take else None
                    if control_group_take and legend_key not in legend_keys_added:
                        fig.add_trace(
                            go.Scatter(
                                x=[None],
                                y=[None],
                                mode="lines",
                                line=dict(
                                    color="red",
                                    dash=date_dash_map[take_date_map[take_id]],
                                    width=4
                                ),
                                name=(
                                    f"Control Group | Shoulder IR AV"
                                    if (comparison_grouping_enabled and control_group_take) else
                                    f"{take_group_label} | Shoulder IR AV | {take_date_map[take_id]} | {pitcher_name}"
                                    if (comparison_grouping_enabled and multi_pitcher_mode) else
                                    f"{take_group_label} | Shoulder IR AV | {take_date_map[take_id]}"
                                    if comparison_grouping_enabled else
                                    f"Shoulder IR AV | {take_date_map[take_id]} | {pitcher_name}"
                                    if multi_pitcher_mode else
                                    f"Shoulder IR AV | {take_date_map[take_id]}"
                                ),
                                showlegend=True,
                                legendgroup=legendgroup,
                                legendgrouptitle_text=None
                            )
                        )
                        legend_keys_added.add(legend_key)
            if not norm_frames:
                continue

            if display_mode == "Individual Throws":
                legendgroup = "Control_Group_Pelvis" if control_group_take else f"Pelvis_{take_date_map[take_id]}"
                pitcher_name = take_pitcher_map.get(take_id, "")
                # Actual data trace (no legend)
                fig.add_trace(
                    go.Scatter(
                        x=norm_frames,
                        y=norm_values,
                        mode="lines",
                        line=dict(
                            color="blue",
                            dash=date_dash_map[take_date_map[take_id]]
                        ),
                        customdata=[[ "Pelvis", take_date_map[take_id], take_order[take_id], take_velocity[take_id], hover_pitcher_name ]] * len(norm_frames),
                        hovertemplate=(
                            "%{customdata[0]} – %{customdata[1]} | "
                            "Pitch %{customdata[2]} (%{customdata[3]:.1f} MPH)"
                            + (" | %{customdata[4]}" if multi_pitcher_mode else "")
                            + "<br>Angular Velocity: %{y:.1f}°/s"
                            + "<br>Time: %{x:.0f} ms rel BR"
                            + "<extra></extra>"
                        ),
                        name=(
                            f"Control Group | Pelvis – Pitch {take_order[take_id]} ({take_velocity[take_id]:.1f} MPH)"
                        ) if control_group_take else (
                            f"{take_group_label} | Pelvis – {take_date_map[take_id]} | "
                            f"Pitch {take_order[take_id]} ({take_velocity[take_id]:.1f} MPH)"
                        ) if comparison_grouping_enabled else None,
                        showlegend=False,
                        legendgroup=legendgroup,
                        legendgrouptitle_text=None
                    )
                )
                # Legend-only trace (once per Pelvis + Date)
                legend_key = ("Control Group", "Pelvis") if control_group_take else None
                if control_group_take and legend_key not in legend_keys_added:
                    fig.add_trace(
                        go.Scatter(
                            x=[None],
                            y=[None],
                            mode="lines",
                            line=dict(
                                color="blue",
                                dash=date_dash_map[take_date_map[take_id]],
                                width=4
                            ),
                        name=(
                            f"Control Group | Pelvis AV"
                            if (comparison_grouping_enabled and control_group_take) else
                            f"{take_group_label} | Pelvis AV | {take_date_map[take_id]} | {pitcher_name}"
                            if (comparison_grouping_enabled and multi_pitcher_mode) else
                            f"{take_group_label} | Pelvis AV | {take_date_map[take_id]}"
                            if comparison_grouping_enabled else
                            f"Pelvis AV | {take_date_map[take_id]} | {pitcher_name}"
                            if multi_pitcher_mode else
                            f"Pelvis AV | {take_date_map[take_id]}"
                        ),
                            showlegend=True,
                            legendgroup=legendgroup,
                            legendgrouptitle_text=None
                        )
                    )
                    legend_keys_added.add(legend_key)

        # --- Store peak summary for table ---
        kinematic_peak_rows = []

        if display_mode == "Grouped":
            color_map = {
                "Pelvis": "blue",
                "Torso": "orange",
                "Elbow": "green",
                "Shoulder": "red"
            }
            grouped_peak_time_reference = {}

            # --- Condensed legend: track (Segment, Date) pairs ---
            legend_keys_added = set()
            peak_marker_traces = []
            peak_marker_annotations = []

            for label, curves in [
                ("Pelvis", grouped_pelvis),
                ("Torso", grouped_torso),
                ("Elbow", grouped_elbow),
                ("Shoulder", grouped_shoulder_ir)
            ]:
                if not curves:
                    continue

                # Group curves by date
                from collections import defaultdict
                curves_by_date = defaultdict(dict)
                for take_id, d in curves.items():
                    date = take_date_map[take_id]
                    pitcher_name = take_pitcher_map.get(take_id, "")
                    group_label = take_group_map.get(take_id, "")
                    if comparison_grouping_enabled and is_control_group_label(group_label):
                        date_key = group_label
                    elif comparison_grouping_enabled:
                        date_key = group_label if group_mode_aggregate_across_pitchers else ((group_label, pitcher_name, date) if multi_pitcher_mode else (group_label, date))
                    else:
                        date_key = (pitcher_name, date) if multi_pitcher_mode else date
                    curves_by_date[date_key][take_id] = d
                for date_key, curves_date in curves_by_date.items():
                    if comparison_grouping_enabled and date_key == "Control Group":
                        group_label = "Control Group"
                        pitcher_name = ""
                        date = "Selected Takes"
                    elif comparison_grouping_enabled and show_group_pitcher_breakout:
                        group_label, pitcher_name, date = date_key
                    elif comparison_grouping_enabled:
                        group_label = date_key
                        date = "Selected Takes"
                        pitcher_name = ""
                    elif multi_pitcher_mode and not comparison_grouping_enabled:
                        pitcher_name, date = date_key
                        group_label = ""
                    else:
                        date = date_key
                        pitcher_name = ""
                        group_label = ""
                    x_date, y_date, q1_date, q3_date = aggregate_curves(curves_date, "Mean")
                    avg_velocity = (
                        float(np.mean([
                            take_velocity[tid]
                            for tid in curves_date.keys()
                            if tid in take_velocity and take_velocity[tid] is not None
                        ]))
                        if any(
                            tid in take_velocity and take_velocity[tid] is not None
                            for tid in curves_date.keys()
                        ) else None
                    )
                    color = color_map[label]
                    # Smoothing
                    if len(y_date) >= 11:
                        y_date = savgol_filter(y_date, window_length=7, polyorder=3)
                    dash = date_dash_map.get(date, "solid")
                    legendgroup = f"{label}_{date}_{pitcher_name}" if show_group_pitcher_breakout else f"{label}_{date}"
                    # --- IQR band (draw first so the line color stays visually true on top) ---
                    if show_ks_signal_iqr_band:
                        fig.add_trace(
                            go.Scatter(
                                x=x_date + x_date[::-1],
                                y=q3_date + q1_date[::-1],
                                fill="toself",
                                fillcolor=to_rgba(color, alpha=0.30),
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo="skip",
                                legendgroup=legendgroup
                            )
                        )
                    # --- Grouped curve (no legend, but legendgroup set) ---
                    fig.add_trace(
                        go.Scatter(
                            x=x_date,
                            y=y_date,
                            mode="lines",
                            line=dict(
                                width=4,
                                color=color,
                                dash=dash,
                            ),
                            customdata=[[label, date, group_label, pitcher_name]] * len(x_date),
                            hovertemplate=(
                                (f"{group_label}<br>" if comparison_grouping_enabled else "")
                                + ("%{customdata[0]}" if comparison_grouping_enabled else "%{customdata[0]} | %{customdata[1]}")
                                + (" | %{customdata[3]}" if show_group_pitcher_breakout else "")
                                + (f"<br>Avg Velocity: {avg_velocity:.1f} mph" if avg_velocity is not None else "")
                                + "<br>Angular Velocity: %{y:.1f}°/s"
                                + "<br>Time: %{x:.0f} ms rel BR"
                                + "<extra></extra>"
                            ),
                            showlegend=False,
                            legendgroup=legendgroup
                        )
                    )
                    # --- Legend-only trace (once per Segment + Date, legendgroup set) ---
                    legend_key = (label, date, pitcher_name) if show_group_pitcher_breakout else (label, date)
                    if legend_key not in legend_keys_added:
                        fig.add_trace(
                            go.Scatter(
                                x=[None],
                                y=[None],
                                mode="lines",
                                line=dict(
                                    color=color,
                                    dash=dash,
                                    width=4
                                ),
                            name=(
                                    f"{group_label} | {label} AV | {date} | {pitcher_name}"
                                    if (comparison_grouping_enabled and show_group_pitcher_breakout) else
                                    f"{group_label} | {label} AV | {date}"
                                    if comparison_grouping_enabled else
                                    f"{label} AV | {date} | {pitcher_name}"
                                    if show_group_pitcher_breakout else
                                    f"{label} AV | {date}"
                                ),
                                showlegend=True,
                                legendgroup=legendgroup
                            )
                        )
                        legend_keys_added.add(legend_key)
                    # --- Peak arrow and marker for this grouped curve ---
                    if len(y_date) > 0:
                        # Restrict pelvis & torso peak search to FP → BR
                        if label in ["Pelvis", "Torso"] and fp_event_frames:
                            fp_rel = rel_frame_to_ms(int(np.median(fp_event_frames)))
                            valid_idxs = [
                                i for i, xf in enumerate(x_date)
                                if fp_rel <= xf <= 0
                            ]
                            if not valid_idxs:
                                continue
                            max_idx = max(valid_idxs, key=lambda i: y_date[i])
                        else:
                            # Elbow / Shoulder IR use full window
                            max_idx = int(np.argmax(y_date))
                        max_x = x_date[max_idx]
                        max_y = y_date[max_idx]
                        reference_time_ms_grouped = None
                        if label == "Pelvis" and fp_event_frames:
                            fp_rel = rel_frame_to_ms(int(np.median(fp_event_frames)))
                            reference_time_ms_grouped = max_x - fp_rel
                        elif label == "Torso":
                            pelvis_peak_time = grouped_peak_time_reference.get((date_key, "Pelvis"))
                            if pelvis_peak_time is not None:
                                reference_time_ms_grouped = max_x - pelvis_peak_time
                        elif label == "Elbow":
                            torso_peak_time = grouped_peak_time_reference.get((date_key, "Torso"))
                            if torso_peak_time is not None:
                                reference_time_ms_grouped = max_x - torso_peak_time
                        elif label == "Shoulder":
                            elbow_peak_time = grouped_peak_time_reference.get((date_key, "Elbow"))
                            if elbow_peak_time is not None:
                                reference_time_ms_grouped = max_x - elbow_peak_time

                        grouped_peak_time_reference[(date_key, label)] = max_x

                        local_y_min = min(y_date) if len(y_date) > 0 else max_y
                        local_y_max = max(y_date) if len(y_date) > 0 else max_y
                        local_y_span = max(local_y_max - local_y_min, 1)
                        peak_marker_y = max_y + max(0.07 * local_y_span, 55)

                        kinematic_peak_rows.append({
                            **({"Group": group_label} if comparison_grouping_enabled else {}),
                            **({"Pitcher": pitcher_name} if show_group_pitcher_breakout else {}),
                            "Session Date": date,
                            "Velocity (mph)": (
                                float(np.mean([
                                    take_velocity[tid]
                                    for tid in curves_date.keys()
                                    if tid in take_velocity and take_velocity[tid] is not None
                                ]))
                                if any(
                                    tid in take_velocity and take_velocity[tid] is not None
                                    for tid in curves_date.keys()
                                ) else None
                            ),
                            "Segment": segment_display_name(label),
                            "Peak Value (°/s)": max_y,
                            "Peak Time from Reference (ms)": reference_time_ms_grouped
                        })
                        peak_marker_traces.append(
                            go.Scatter(
                                x=[max_x],
                                y=[max_y],
                                mode="markers",
                                marker=dict(
                                    symbol="circle",
                                    size=10,
                                    color=color,
                                    opacity=0,
                                ),
                                showlegend=False,
                                legendgroup=legendgroup,
                                customdata=[[
                                    label,
                                    date,
                                    group_label,
                                    pitcher_name,
                                    max_y,
                                    max_x,
                                    peak_marker_y,
                                ]],
                                hovertemplate=(
                                    ("%{customdata[2]} | " if comparison_grouping_enabled else "")
                                    + "%{customdata[0]} | %{customdata[1]}"
                                    + (" | %{customdata[3]}" if show_group_pitcher_breakout else "")
                                    + "<br>Peak Angular Velocity: %{customdata[4]:.1f}°/s"
                                    + "<br>Peak Time: %{customdata[5]:.0f} ms rel BR"
                                    + "<extra></extra>"
                                ),
                            )
                        )
                        peak_marker_annotations.append(
                            dict(
                                x=max_x,
                                y=max_y,
                                xref="x",
                                yref="y",
                                text="",
                                showarrow=True,
                                arrowhead=2,
                                arrowsize=1.2,
                                arrowwidth=2,
                                arrowcolor=color,
                                ax=0,
                                ay=-40,
                            )
                        )
            for peak_marker_trace in peak_marker_traces:
                fig.add_trace(peak_marker_trace)
            for peak_marker_annotation in peak_marker_annotations:
                fig.add_annotation(**peak_marker_annotation)

        # Median Refined Foot Plant (zero-cross) event
        if fp_event_frames:
            add_event_iqr_band(fig, fp_event_frames, "green", show_ks_fp_iqr_band)
            median_fp_frame = rel_frame_to_ms(int(np.median(fp_event_frames)))

            fig.add_vline(
                x=median_fp_frame,
                line_width=3,
                line_dash="dash",
                line_color="green",
                opacity=0.9
            )
            fig.add_annotation(
                x=median_fp_frame,
                y=1.055,
                xref="x",
                yref="paper",
                text="FP",
                showarrow=False,
                font=dict(color="green", size=14),
                align="center"
            )

        # Median Max Shoulder ER event
        if mer_event_frames:
            add_event_iqr_band(fig, mer_event_frames, "red", show_ks_fp_iqr_band)
            median_mer_frame = rel_frame_to_ms(int(np.median(mer_event_frames)))

            fig.add_vline(
                x=median_mer_frame,
                line_width=3,
                line_dash="dash",
                line_color="red",
                opacity=0.9
            )
            fig.add_annotation(
                x=median_mer_frame,
                y=1.055,
                xref="x",
                yref="paper",
                text="MER",
                showarrow=False,
                font=dict(color="red", size=14),
                align="center"
            )

        # Normalized Ball Release reference line
        add_event_iqr_band(fig, [0] * max(len(take_ids), 1), "blue", show_ks_fp_iqr_band)
        fig.add_vline(
            x=0,
            line_width=3,
            line_dash="dash",
            line_color="blue",
            opacity=0.9
        )
        fig.add_annotation(
            x=0,
            y=1.055,
            xref="x",
            yref="paper",
            text="BR",
            showarrow=False,
            font=dict(color="blue", size=14),
            align="center"
        )

        grouped_visible_y_vals = []
        yaxis_range = None
        if display_mode == "Grouped":
            for trace in fig.data:
                if getattr(trace, "type", None) != "scatter":
                    continue
                if getattr(trace, "mode", None) != "lines":
                    continue
                if getattr(trace, "fill", None) == "toself":
                    continue

                trace_y = getattr(trace, "y", None)
                if trace_y is None:
                    continue

                grouped_visible_y_vals.extend(
                    v for v in trace_y
                    if v is not None and np.isfinite(v)
                )

            if grouped_visible_y_vals:
                y_min = min(grouped_visible_y_vals)
                y_max = max(grouped_visible_y_vals)
                y_span = max(y_max - y_min, 1)
                yaxis_range = [y_min - (0.10 * y_span), y_max + (0.22 * y_span)]

        fig.update_layout(
            xaxis_title="Time Relative to Ball Release (ms)",
            yaxis_title="Angular Velocity",
            yaxis=dict(
                ticksuffix="°/s",
                range=yaxis_range,
            ),
            xaxis_range=[window_start_ms, window_end_ms],
            height=600,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.30,
                xanchor="center",
                x=0.5,
                groupclick="togglegroup"
            ),
            hoverlabel=dict(
                namelength=-1,
                font_size=13
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="kinematic_sequence_plot",
            config={
                "toImageButtonOptions": {
                    "filename": "kinematic_sequence"
                }
            },
        )

        # --- Kinematic Sequence Peak Summary Table (Individual Throws) ---
        if display_mode == "Individual Throws":

            individual_rows = []

            for take_id in take_ids:
                if take_id not in br_frames:
                    continue

                br_frame = br_frames[take_id]

                # Helper to compute peak and frame
                def peak_and_frame(curves, invert=False):
                    if take_id not in curves:
                        return None, None

                    vals = curves[take_id]["value"]
                    frames = curves[take_id]["frame"]
                    if not vals:
                        return None, None

                    if invert:
                        idx = int(np.argmax(vals))
                    else:
                        idx = int(np.argmax(vals))

                    return vals[idx], frames[idx]

                pelvis_peak, pelvis_frame = peak_and_frame(grouped_pelvis)
                # Pelvis peak timing from Foot Plant (zero-cross), in ms (250 Hz)
                pelvis_time_ms = None
                fp_abs = foot_plant_zero_cross_frames.get(take_id)  # absolute frame
                br_abs = br_frames.get(take_id)  # absolute frame

                if pelvis_frame is not None and fp_abs is not None and br_abs is not None:
                    fp_rel = fp_abs - br_abs  # FP relative to BR (frames)
                    pelvis_time_ms = pelvis_frame - rel_frame_to_ms(fp_rel)
                torso_peak, torso_frame = peak_and_frame(grouped_torso)
                elbow_peak, elbow_frame = peak_and_frame(grouped_elbow)
                shoulder_peak, shoulder_frame = peak_and_frame(grouped_shoulder_ir)
                torso_time_from_pelvis_ms = (
                    torso_frame - pelvis_frame
                    if torso_frame is not None and pelvis_frame is not None else None
                )
                elbow_time_from_torso_ms = (
                    elbow_frame - torso_frame
                    if elbow_frame is not None and torso_frame is not None else None
                )
                shoulder_time_from_elbow_ms = (
                    shoulder_frame - elbow_frame
                    if shoulder_frame is not None and elbow_frame is not None else None
                )

                individual_rows.append({
                    **({"Group": take_group_map.get(take_id, "")} if comparison_grouping_enabled else {}),
                    **({"Pitcher": take_pitcher_map.get(take_id)} if multi_pitcher_mode else {}),
                    "Session Date": take_date_map[take_id],
                    "Pitch": take_order[take_id],
                    "Velocity (mph)": take_velocity[take_id],
                    "Pelvis Rotation Peak (°/s)": pelvis_peak,
                    "Pelvis Rotation Time from FP (ms)": pelvis_time_ms,
                    "Torso Rotation Peak (°/s)": torso_peak,
                    "Torso Rotation Time from Peak Pelvis (ms)": torso_time_from_pelvis_ms,
                    "Elbow Extension Peak (°/s)": elbow_peak,
                    "Elbow Extension Time from Peak Torso (ms)": elbow_time_from_torso_ms,
                    "Shoulder Internal Rotation Peak (°/s)": shoulder_peak,
                    "Shoulder Internal Rotation Time from Peak Elbow (ms)": shoulder_time_from_elbow_ms
                })

            if individual_rows:
                import pandas as pd

                st.markdown("### Kinematic Sequence - Individual Throws")

                df_individual = pd.DataFrame(individual_rows)

                # Sort logically: date → pitch order
                sort_cols = ["Session Date", "Pitch"]
                if comparison_grouping_enabled and "Group" in df_individual.columns:
                    sort_cols = ["Group"] + sort_cols
                if multi_pitcher_mode and "Pitcher" in df_individual.columns:
                    sort_cols = ["Pitcher"] + sort_cols
                df_individual = df_individual.sort_values(sort_cols)

                index_cols = ["Session Date", "Velocity (mph)"]
                if comparison_grouping_enabled and "Group" in df_individual.columns:
                    index_cols = ["Group"] + index_cols
                if multi_pitcher_mode and "Pitcher" in df_individual.columns:
                    index_cols = (
                        (["Group"] if comparison_grouping_enabled and "Group" in df_individual.columns else [])
                        + ["Pitcher", "Session Date", "Velocity (mph)"]
                    )

                segment_metric_map = {
                    "Pelvis Rotation Peak (°/s)": ("Pelvis Rotation", "Peak (°/s)"),
                    "Pelvis Rotation Time from FP (ms)": ("Pelvis Rotation", "Peak Time from Foot Plant (ms)"),
                    "Torso Rotation Peak (°/s)": ("Torso Rotation", "Peak (°/s)"),
                    "Torso Rotation Time from Peak Pelvis (ms)": ("Torso Rotation", "Peak Time from Peak Pelvis (ms)"),
                    "Elbow Extension Peak (°/s)": ("Elbow Extension", "Peak (°/s)"),
                    "Elbow Extension Time from Peak Torso (ms)": ("Elbow Extension", "Peak Time from Peak Torso (ms)"),
                    "Shoulder Internal Rotation Peak (°/s)": ("Shoulder Internal Rotation", "Peak (°/s)"),
                    "Shoulder Internal Rotation Time from Peak Elbow (ms)": ("Shoulder Internal Rotation", "Peak Time from Peak Elbow (ms)"),
                }
                value_cols = list(segment_metric_map.keys())
                df_individual_display = df_individual[index_cols + value_cols].set_index(index_cols)
                df_individual_display.columns = pd.MultiIndex.from_tuples(
                    [segment_metric_map[col] for col in value_cols]
                )

                segment_order = [
                    "Pelvis Rotation",
                    "Torso Rotation",
                    "Elbow Extension",
                    "Shoulder Internal Rotation",
                ]
                ordered_cols = []
                for seg in segment_order:
                    segment_metrics = [
                        "Peak (°/s)",
                        {
                            "Pelvis Rotation": "Peak Time from Foot Plant (ms)",
                            "Torso Rotation": "Peak Time from Peak Pelvis (ms)",
                            "Elbow Extension": "Peak Time from Peak Torso (ms)",
                            "Shoulder Internal Rotation": "Peak Time from Peak Elbow (ms)",
                        }[seg],
                    ]
                    for metric_name in segment_metrics:
                        if (seg, metric_name) in df_individual_display.columns:
                            ordered_cols.append((seg, metric_name))
                if ordered_cols:
                    df_individual_display = df_individual_display[ordered_cols]

                segment_colors = {
                    "Pelvis Rotation": "#DBEAFE",
                    "Torso Rotation": "#FED7AA",
                    "Elbow Extension": "#DCFCE7",
                    "Shoulder Internal Rotation": "#FEE2E2",
                }

                def style_segment_headers(headers):
                    return [
                        f"background-color: {segment_colors.get(header, '#FFFFFF')}; color: #111827;"
                        if header in segment_colors else ""
                        for header in headers
                    ]

                def fmt(val, decimals=2):
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        return ""
                    return f"{val:.{decimals}f}"

                styled_individual = (
                    df_individual_display
                    .style
                    .format(lambda x: fmt(x, 1) if isinstance(x, (int, float, np.floating)) else x)
                    .apply_index(style_segment_headers, axis="columns", level=0)
                    .set_table_styles([
                        {"selector": "th", "props": [("text-align", "center")]},
                        {"selector": "th.row_heading", "props": [("text-align", "center")]},
                        {"selector": "th.index_name", "props": [("text-align", "center")]},
                        *(
                            [
                                {
                                    "selector": "th.row_heading.level0",
                                    "props": [("min-width", "80px"), ("max-width", "80px")]
                                }
                            ]
                            if df_individual_display.index.names and df_individual_display.index.names[0] == "Group"
                            else []
                        ),
                    ])
                    .set_properties(**{"text-align": "center", "font-weight": "500"})
                )

                try:
                    st.dataframe(styled_individual, use_container_width=True)
                except KeyError:
                    # Streamlit Cloud can error on some Styler/MultiIndex combinations;
                    # fall back to the plain dataframe instead of breaking the page.
                    st.dataframe(df_individual_display, use_container_width=True)

        # --- Kinematic Sequence Peak Summary Table (Segment-Grouped) ---
        if display_mode == "Grouped" and kinematic_peak_rows:
            import pandas as pd

            st.markdown("### Kinematic Sequence - Grouped")

            df = pd.DataFrame(kinematic_peak_rows)
            index_cols = ["Session Date", "Velocity (mph)"]
            if comparison_grouping_enabled and "Group" in df.columns:
                index_cols = ["Group"] + index_cols
            if multi_pitcher_mode and "Pitcher" in df.columns:
                index_cols = (
                    (["Group"] if comparison_grouping_enabled and "Group" in df.columns else [])
                    + ["Pitcher", "Session Date", "Velocity (mph)"]
                )

            df_pivot = df.pivot_table(
                index=index_cols,
                columns="Segment",
                values=["Peak Value (°/s)", "Peak Time from Reference (ms)"],
                aggfunc="first"
            )

            # Reorder to (Segment, Metric) like the original grouped summary layout
            df_pivot = df_pivot.swaplevel(0, 1, axis=1)
            metric_map = {
                "Peak Value (°/s)": "Peak (°/s)",
            }
            segment_reference_metric_map = {
                "Pelvis Rotation": "Peak Time from Foot Plant (ms)",
                "Torso Rotation": "Peak Time from Peak Pelvis (ms)",
                "Elbow Extension": "Peak Time from Peak Torso (ms)",
                "Shoulder Internal Rotation": "Peak Time from Peak Elbow (ms)",
            }
            df_pivot.columns = pd.MultiIndex.from_tuples(
                [
                    (
                        seg,
                        metric_map.get(metric, segment_reference_metric_map.get(seg, metric))
                    )
                    for seg, metric in df_pivot.columns
                ]
            )
            segment_order = [
                "Pelvis Rotation",
                "Torso Rotation",
                "Elbow Extension",
                "Shoulder Internal Rotation",
            ]
            ordered_cols = []
            for seg in segment_order:
                segment_metrics = [
                    "Peak (°/s)",
                    segment_reference_metric_map.get(seg, "Peak Time from Reference (ms)"),
                ]
                for metric_name in segment_metrics:
                    if (seg, metric_name) in df_pivot.columns:
                        ordered_cols.append((seg, metric_name))
            if ordered_cols:
                df_pivot = df_pivot[ordered_cols]

            segment_colors = {
                "Pelvis Rotation": "#DBEAFE",
                "Torso Rotation": "#FED7AA",
                "Elbow Extension": "#DCFCE7",
                "Shoulder Internal Rotation": "#FEE2E2",
            }

            def style_segments(col):
                column_name = col.name
                seg = column_name[0] if isinstance(column_name, tuple) else column_name
                if seg in segment_colors:
                    return [f"background-color: {segment_colors[seg]}"] * len(df_pivot)
                return [""] * len(df_pivot)

            def style_segment_headers(headers):
                return [
                    f"background-color: {segment_colors.get(header, '#FFFFFF')}; color: #111827;"
                    if header in segment_colors else ""
                    for header in headers
                ]

            df_display = df_pivot.copy()
            if "Velocity (mph)" in df_display.index.names:
                velocity_level = df_display.index.names.index("Velocity (mph)")
                formatted_index = []
                for idx in df_display.index:
                    idx_list = list(idx) if isinstance(idx, tuple) else [idx]
                    velocity_value = idx_list[velocity_level]
                    if velocity_value is not None and not pd.isna(velocity_value):
                        idx_list[velocity_level] = f"{velocity_value:.1f}"
                    formatted_index.append(tuple(idx_list) if isinstance(idx, tuple) else idx_list[0])
                df_display.index = pd.MultiIndex.from_tuples(formatted_index, names=df_display.index.names)
            for col in df_display.columns:
                if col[1] == "Peak (°/s)":
                    df_display[col] = df_display[col].map(lambda x: "" if x is None or pd.isna(x) else f"{x:.0f}")
                elif "Time" in col[1]:
                    df_display[col] = df_display[col].map(lambda x: "" if x is None or pd.isna(x) else f"{x:.0f}")

            styled = (
                df_display
                .style
                .apply(style_segments, axis=0)
                .apply_index(style_segment_headers, axis="columns", level=0)
                .set_table_styles([
                    {"selector": "th", "props": [("text-align", "center")]},
                    {"selector": "th.row_heading", "props": [("text-align", "center")]},
                    {"selector": "th.index_name", "props": [("text-align", "center")]},
                    *(
                        [
                            {
                                "selector": "th.row_heading.level0",
                                "props": [("min-width", "80px"), ("max-width", "80px")]
                            }
                        ]
                        if df_display.index.names and df_display.index.names[0] == "Group"
                        else []
                    ),
                ])
                .set_properties(**{"text-align": "center", "font-weight": "500"})
            )
            st.dataframe(styled, use_container_width=True)

        kinematic_sequence_definitions = {
            "Pelvis Angular Velocity": "how fast the hips are rotating.",
            "Torso Angular Velocity": "how fast the shoulders are rotating.",
            "Elbow Angular Velocity": "how fast the elbow is straightening.",
            "Shoulder Angular Velocity": "how fast the shoulder rotates during the throw.",
        }
        st.markdown("### Kinematic Sequence Definitions")
        for segment, definition in kinematic_sequence_definitions.items():
            st.markdown(
                (
                    f"<div style='font-size:1.15rem; line-height:1.6; margin:0.35rem 0 0.9rem 0;'>"
                    f"<strong>{segment}:</strong> {definition}"
                    f"</div>"
                ),
                unsafe_allow_html=True,
            )


with tab_joint:
    st.subheader("Kinematics")
    render_group_selection_summary()
    st.markdown(
        """
        <style>
        .joint-controls-label {
            font-size: 0.8rem;
            font-weight: 700;
            color: #6b7280;
            margin-bottom: 0.1rem;
        }

        div[data-testid="stSegmentedControl"] label,
        div[data-testid="stSegmentedControl"] div[role="radiogroup"] label,
        div[data-testid="stSegmentedControl"] div[role="radiogroup"] p,
        div[data-testid="stToggle"] label,
        div[data-testid="stToggle"] p {
            font-size: 1rem !important;
            font-weight: 400 !important;
        }

        .joint-toggle-label {
            margin-top: -0.1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    joint_view_mode = st.session_state.get("joint_view_mode", "Single")

    kinematic_options = [
        # Arm and hand
        "Elbow Extension Velocity",
        "Elbow Flexion",
        "Forearm Pronation/Supination",
        "Hand Speed",

        # Lower body
        "Hip-Shoulder Separation",
        "Lead Knee Flexion",
        "Lead Knee Flexion/Extension Velocity",
        "Pelvic Lateral Tilt",
        "Pelvis Rotation",
        "Pelvis Rotational Velocity",

        # Shoulder
        "Shoulder Abduction",
        "Shoulder Horizontal Abduction",
        "Shoulder Rotation",
        "Shoulder Rotation Velocity",

        # Trunk and whole-body movement
        "Center of Mass Velocity (Anterior/Posterior)",
        "Torso-Pelvis Rotational Velocity",
        "Trunk Forward Tilt",
        "Trunk Lateral Tilt",
        "Trunk Rotation",
        "Trunk Rotational Velocity",
    ]

    kinematic_definitions = {
        "Elbow Flexion": {
            "definition": "Angle of the elbow (forearm relative to upper arm) showing how bent the arm is during the throw. 0° = fully straight; >90° at foot plant = inside 90.",
        },
        "Hand Speed": {
            "definition": "How fast the throwing hand is moving in space during the throw (total speed in all directions).",
        },
        "Center of Mass Velocity (Anterior/Posterior)": {
            "definition": "Forward/backward speed of the body's center of mass toward or away from home plate. Positive = moving toward home plate; negative = moving back toward the mound.",
        },
        "Shoulder Rotation": {
            "definition": "How much the throwing arm rotates back and forward at the shoulder.",
        },
        "Shoulder Rotation Velocity": {
            "definition": "How fast the shoulder rotates during the throw. Peak internal rotation velocity = how quickly the arm turns forward. ~90° = goalpost position; moving forward toward 0° = arm rotating forward.",
        },
        "Shoulder Abduction": {
            "definition": "How far the arm is lifted away from the body. 0° = arms at your side; 90° = straight out (T-pose).",
        },
        "Shoulder Horizontal Abduction": {
            "definition": "How far the upper arm moves forward or backward relative to the trunk. 0° = T-pose; positive = arm moves behind you.",
        },
        "Lead Knee Flexion": {
            "definition": "Angle of the front knee (lower leg relative to upper leg). 0° = fully straight; higher values = more bend.",
        },
        "Lead Knee Flexion/Extension Velocity": {
            "definition": "How fast the front knee is bending or straightening.",
        },
        "Trunk Forward Tilt": {
            "definition": "Forward/backward lean of the upper body. 0° = upright; positive = leaning forward; negative = leaning back.",
        },
        "Trunk Lateral Tilt": {
            "definition": "Side-to-side lean of the upper body. 0° = upright; positive = leaning toward the lead leg side.",
        },
        "Trunk Rotation": {
            "definition": "How much the shoulders are turned toward home plate. -90° = open/sideways; 0° = square to home plate.",
        },
        "Pelvis Rotation": {
            "definition": "How much the hips are turned toward home plate. -90° = open/sideways; 0° = square to home plate.",
        },
        "Pelvic Lateral Tilt": {
            "definition": "Side-to-side tilt of the hips. 0° = level; positive = lead leg side drops; negative = trail side drops.",
        },
        "Hip-Shoulder Separation": {
            "definition": "Difference between hip and shoulder rotation. Positive = hips are opening ahead of the shoulders.",
        },
        "Pelvis Rotational Velocity": {
            "definition": "How fast the hips are rotating.",
        },
        "Trunk Rotational Velocity": {
            "definition": "How fast the shoulders are rotating.",
        },
        "Torso-Pelvis Rotational Velocity": {
            "definition": "How fast the shoulders are rotating relative to the hips.",
        },
        "Elbow Extension Velocity": {
            "definition": "How fast the elbow is straightening.",
        },
        "Forearm Pronation/Supination": {
            "definition": "Rotation of the forearm (palm turning down vs. up).",
        },
    }

    energy_definitions = {
        "Trunk-Shoulder Energy Flow (RTA_DIST_L | RTA_DIST_R)": {
            "definition": (
                "Measures how the trunk loads and then transfers energy to the throwing arm. "
                "Negative = the trunk is absorbing energy (loading), positive = the trunk is "
                "sending energy to the arm (throwing)."
            ),
        },
        "Arm Energy Flow (LAR_PROX | RAR_PROX)": {
            "definition": (
                "Measures how the throwing arm receives and responds to energy from the trunk "
                "at the shoulder connection. Positive values -> the arm is loading "
                "(receiving energy). Negative values -> the arm is being accelerated by the trunk."
            ),
        },
        "Glove Side Trunk-Shoulder Energy Flow": {
            "definition": (
                "Measures how the trunk loads and then transfers energy to the glove-side arm."
            ),
        },
        "Glove Arm Energy Flow": {
            "definition": (
                "Measures how the glove-side arm receives and responds to energy from the trunk "
                "at the shoulder connection."
            ),
        },
        "Trunk-Shoulder Elevation/Depression Energy Flow": {
            "definition": (
                "Energy flow into (+) or out of (-) the trunk at the shoulder due to shoulder "
                "elevation/depression (vertical abduction/adduction)."
            ),
        },
        "Trunk-Shoulder Horizontal Abd/Add Energy Flow": {
            "definition": (
                "Energy flow into (+) or out of (-) the trunk at the shoulder due to shoulder "
                "horizontal abduction/adduction (scap load)."
            ),
        },
        "Trunk-Shoulder Rotational Energy Flow": {
            "definition": (
                "Energy flow into (+) or out of (-) the trunk at the shoulder due to shoulder "
                "internal/external rotation."
            ),
        },
        "Arm Elevation/Depression Energy Flow": {
            "definition": (
                "Energy flow into (+) or out of (-) the upper arm at the shoulder due to "
                "shoulder elevation/depression (vertical abduction/adduction)."
            ),
        },
        "Arm Horizontal Abd/Add Energy Flow": {
            "definition": (
                "Energy flow into (+) or out of (-) the upper arm at the shoulder due to "
                "shoulder horizontal abduction/adduction (scap load)."
            ),
        },
        "Arm Rotational Energy Flow": {
            "definition": (
                "Energy flow into (+) or out of (-) the upper arm at the shoulder due to "
                "shoulder internal/external rotation."
            ),
        },
        "Throwing Shoulder Rotational Torque (Relative to Trunk)": {
            "definition": (
                "The rotational torque at the throwing shoulder over time, measured relative "
                "to the trunk, representing the rotational load acting at the shoulder joint "
                "throughout the pitching motion."
            ),
        },
    }

    def get_kinematic_unit(kinematic_name):
        if "Velocity" in kinematic_name and "Hand Speed" not in kinematic_name and "Center of Mass Velocity" not in kinematic_name:
            return "°/s"
        if kinematic_name in {"Hand Speed", "Center of Mass Velocity (Anterior/Posterior)"}:
            return "m/s"
        return "°"

    compare_energy_metrics = []
    compare_energy_display_mode = "Grouped"
    compare_energy_window_mode = "Peak Knee Height View"

    if joint_view_mode == "Comparison":
        compare_top_left, compare_top_right = st.columns([1.3, 4.7])
        with compare_top_left:
            st.markdown('<div class="joint-controls-label">View Mode</div>', unsafe_allow_html=True)
            joint_view_mode = st.segmented_control(
                "View Mode",
                ["Single", "Comparison"],
                default="Comparison",
                key="joint_view_mode",
                label_visibility="collapsed",
            )
        with compare_top_right:
            st.markdown("")
        control_left_col, control_right_col = st.columns(2)
        with control_left_col:
            st.markdown('<div class="joint-controls-label">Display Mode</div>', unsafe_allow_html=True)
            display_mode = st.segmented_control(
                "Kinematics Display Mode",
                ["Individual Throws", "Grouped"],
                default="Grouped",
                key="joint_display_mode_compare",
                label_visibility="collapsed",
            )
            joint_window_mode = "Foot Plant to Ball Release View"
            st.markdown('<div class="joint-controls-label joint-toggle-label">Options</div>', unsafe_allow_html=True)
            left_event_col, left_signal_col = st.columns(2)
            with left_event_col:
                show_joint_fp_iqr_band = st.toggle(
                    "Event Bands",
                    value=False,
                    key="joint_show_fp_iqr_band_compare",
                    help="Shows the middle 50% range for event timing across selected throws.",
                )
            with left_signal_col:
                show_joint_signal_iqr_band = st.toggle(
                    "Signal Bands",
                    value=True,
                    key="joint_show_signal_iqr_band_compare",
                    help="Shows the middle 50% range around each grouped mean line.",
                )
            selected_kinematics = st.multiselect(
                "Select Kinematics",
                options=kinematic_options,
                default=[],
                help=(
                    "Select one or more kinematics to plot. "
                    "Hover any line in the chart to see that metric's definition."
                ),
                key="joint_angles_select_compare"
            )
        with control_right_col:
            st.markdown('<div class="joint-controls-label">Display Mode</div>', unsafe_allow_html=True)
            compare_energy_display_mode = st.segmented_control(
                "Energy Flow Display Mode",
                ["Individual Throws", "Grouped"],
                default="Grouped",
                key="joint_energy_display_mode_compare"
                ,
                label_visibility="collapsed",
            )
            st.markdown('<div class="joint-controls-label">View Window</div>', unsafe_allow_html=True)
            compare_energy_window_mode = st.segmented_control(
                "Energy Flow View",
                ["Peak Knee Height View", "Foot Plant to Ball Release View"],
                default="Peak Knee Height View",
                key="joint_energy_window_mode_compare",
                label_visibility="collapsed",
            )
            st.markdown('<div class="joint-controls-label joint-toggle-label">Options</div>', unsafe_allow_html=True)
            right_event_col, right_signal_col = st.columns(2)
            with right_event_col:
                show_compare_energy_fp_iqr_band = st.toggle(
                    "Event Bands",
                    value=False,
                    key="joint_energy_show_fp_iqr_band_compare",
                    help="Shows the middle 50% range for event timing across selected throws.",
                )
            with right_signal_col:
                show_compare_energy_signal_iqr_band = st.toggle(
                    "Signal Bands",
                    value=True,
                    key="joint_energy_show_signal_iqr_band_compare",
                    help="Shows the middle 50% range around each grouped mean line.",
                )
            compare_energy_metrics = st.multiselect(
                "Select Energy Flow Metrics",
                [
                    "Trunk-Shoulder Energy Flow (RTA_DIST_L | RTA_DIST_R)",
                    "Arm Energy Flow (LAR_PROX | RAR_PROX)",
                    "Glove Side Trunk-Shoulder Energy Flow",
                    "Glove Arm Energy Flow",
                    "Trunk-Shoulder Rotational Energy Flow",
                    "Trunk-Shoulder Elevation/Depression Energy Flow",
                    "Trunk-Shoulder Horizontal Abd/Add Energy Flow",
                    "Arm Rotational Energy Flow",
                    "Arm Elevation/Depression Energy Flow",
                    "Arm Horizontal Abd/Add Energy Flow",
                    "Throwing Shoulder Rotational Torque (Relative to Trunk)",
                    *NEW_TRUNK_PELVIS_ENERGY_METRICS,
                ],
                default=[],
                key="joint_energy_metrics_compare"
            )
    else:
        display_col, options_col, spacer_col = st.columns([1.45, 1.75, 2.2])
        with display_col:
            st.markdown('<div class="joint-controls-label">Display Mode</div>', unsafe_allow_html=True)
            display_mode = st.segmented_control(
                "Select Display Mode",
                ["Individual Throws", "Grouped"],
                default="Grouped",
                key="joint_display_mode",
                label_visibility="collapsed",
            )
        with options_col:
            st.markdown('<div class="joint-controls-label joint-toggle-label">Options</div>', unsafe_allow_html=True)
            joint_event_col, joint_signal_col = st.columns(2)
            with joint_event_col:
                show_joint_fp_iqr_band = st.toggle(
                    "Event Bands",
                    value=False,
                    key="joint_show_fp_iqr_band",
                    help="Shows the middle 50% range for event timing across selected throws.",
                )
            with joint_signal_col:
                show_joint_signal_iqr_band = st.toggle(
                    "Signal Bands",
                    value=True,
                    key="joint_show_signal_iqr_band",
                    help="Shows the middle 50% range around each grouped mean line.",
                )
        with spacer_col:
            st.markdown("")
        st.markdown(
            """
            <style>
            div[data-testid="stHorizontalBlock"] + div[data-testid="stVerticalBlock"] {
                margin-top: -0.2rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        window_col, mode_col, second_row_spacer = st.columns([2.35, 1.15, 2.75])
        with window_col:
            st.markdown('<div class="joint-controls-label">View Window</div>', unsafe_allow_html=True)
            joint_window_mode = st.segmented_control(
                "Kinematics View",
                ["Peak Knee Height View", "Foot Plant to Ball Release View"],
                default="Peak Knee Height View",
                key="joint_window_mode",
                label_visibility="collapsed",
            )
        with mode_col:
            st.markdown('<div class="joint-controls-label">View Mode</div>', unsafe_allow_html=True)
            joint_view_mode = st.segmented_control(
                "View Mode",
                ["Single", "Comparison"],
                default="Single",
                key="joint_view_mode",
                label_visibility="collapsed",
            )
        with second_row_spacer:
            st.markdown("")
        kinematics_select_col, kinematics_select_spacer = st.columns([2.35, 3.65])
        with kinematics_select_col:
            selected_kinematics = st.multiselect(
                "Select Kinematics",
                options=kinematic_options,
                default=[],
                help=(
                    "Select one or more kinematics to plot. "
                    "Hover any line in the chart to see that metric's definition."
                ),
                key="joint_angles_select"
            )
        with kinematics_select_spacer:
            st.markdown("")

    has_kinematics_selection = bool(selected_kinematics)
    show_single_kinematics_empty_state = not has_kinematics_selection and joint_view_mode == "Single"

    if show_single_kinematics_empty_state:
        kinematics_empty_col, kinematics_empty_spacer = st.columns([2.35, 3.65])
        with kinematics_empty_col:
            st.info("Select at least one kinematic.")
        with kinematics_empty_spacer:
            st.markdown("")

    # --- Color map for joint types ---
    joint_color_map = {
        "Elbow Flexion": "purple",
        "Hand Speed": "deeppink",
        "Center of Mass Velocity (Anterior/Posterior)": "cyan",
        "Shoulder Rotation": "teal",
        "Shoulder Rotation Velocity": "magenta",
        "Shoulder Abduction": "orange",
        "Shoulder Horizontal Abduction": "brown",
        "Forearm Pronation/Supination": "crimson",
        "Pelvis Rotational Velocity": "navy",
        "Trunk Rotational Velocity": "darkorange",
        "Torso-Pelvis Rotational Velocity": "dodgerblue",
        "Elbow Extension Velocity": "limegreen",
    }
    joint_color_map.update({
        "Trunk Forward Tilt": "blue",
        "Trunk Lateral Tilt": "green",
        "Trunk Rotation": "#E9FF70"
    })
    joint_color_map.update({
        "Pelvis Rotation": "darkblue",
        "Pelvic Lateral Tilt": "#FF4FA3",
        "Hip-Shoulder Separation": "darkred"
    })
    joint_color_map.update({
        "Lead Knee Flexion": "darkgreen"
    })
    joint_color_map.update({
        "Lead Knee Flexion/Extension Velocity": "olive"
    })

    # --- Load joint data conditionally ---
    joint_data = {}

    def load_joint_by_handedness(loader_fn):
        merged = {}
        for hand, ids in take_ids_by_handedness.items():
            if ids:
                merged.update(loader_fn(ids, hand))
        return merged

    # --- Pelvis / Trunk rotational velocity (z_data) ---
    if "Pelvis Rotational Velocity" in selected_kinematics:
        joint_data["Pelvis Rotational Velocity"] = get_pelvis_angular_velocity(take_ids)

    if "Trunk Rotational Velocity" in selected_kinematics:
        joint_data["Trunk Rotational Velocity"] = get_torso_angular_velocity(take_ids)

    if "Torso-Pelvis Rotational Velocity" in selected_kinematics:
        joint_data["Torso-Pelvis Rotational Velocity"] = get_torso_pelvis_angular_velocity(take_ids)

    if "Elbow Flexion" in selected_kinematics:
        joint_data["Elbow Flexion"] = load_joint_by_handedness(get_elbow_flexion_angle)

    if "Hand Speed" in selected_kinematics:
        joint_data["Hand Speed"] = load_joint_by_handedness(get_hand_speed)

    if "Center of Mass Velocity (Anterior/Posterior)" in selected_kinematics:
        joint_data["Center of Mass Velocity (Anterior/Posterior)"] = get_center_of_mass_velocity_x(take_ids)

    if "Shoulder Rotation" in selected_kinematics:
        joint_data["Shoulder Rotation"] = load_joint_by_handedness(get_shoulder_er_angle)

    if "Shoulder Rotation Velocity" in selected_kinematics:
        joint_data["Shoulder Rotation Velocity"] = load_joint_by_handedness(get_shoulder_ir_velocity)

    if "Shoulder Abduction" in selected_kinematics:
        joint_data["Shoulder Abduction"] = load_joint_by_handedness(get_shoulder_abduction_angle)

    if "Shoulder Horizontal Abduction" in selected_kinematics:
        joint_data["Shoulder Horizontal Abduction"] = load_joint_by_handedness(
            get_shoulder_horizontal_abduction_angle
        )

    if "Forearm Pronation/Supination" in selected_kinematics:
        joint_data["Forearm Pronation/Supination"] = load_joint_by_handedness(
            get_forearm_pron_sup_angle
        )

    if "Lead Knee Flexion" in selected_kinematics:
        joint_data["Lead Knee Flexion"] = load_joint_by_handedness(get_front_knee_flexion_angle)

    if "Lead Knee Flexion/Extension Velocity" in selected_kinematics:
        joint_data["Lead Knee Flexion/Extension Velocity"] = load_joint_by_handedness(
            get_front_knee_extension_velocity
        )

    # --- Load Torso Angle components conditionally ---
    needs_torso_angle_data = any(
        metric in selected_kinematics
        for metric in ["Trunk Forward Tilt", "Trunk Lateral Tilt", "Trunk Rotation"]
    )
    torso_angle_data = get_torso_angle_components(take_ids) if needs_torso_angle_data else {}

    if "Trunk Forward Tilt" in selected_kinematics:
        joint_data["Trunk Forward Tilt"] = {
            k: {"frame": v["frame"], "value": v["x"]}
            for k, v in torso_angle_data.items()
        }

    if "Trunk Lateral Tilt" in selected_kinematics:
        joint_data["Trunk Lateral Tilt"] = {
            k: {"frame": v["frame"], "value": v["y"]}
            for k, v in torso_angle_data.items()
        }

    if "Trunk Rotation" in selected_kinematics:
        joint_data["Trunk Rotation"] = {
            k: {"frame": v["frame"], "value": v["z"]}
            for k, v in torso_angle_data.items()
        }

    if "Pelvis Rotation" in selected_kinematics:
        joint_data["Pelvis Rotation"] = get_pelvis_angle(take_ids)

    if "Pelvic Lateral Tilt" in selected_kinematics:
        joint_data["Pelvic Lateral Tilt"] = get_pelvic_lateral_tilt(take_ids)

    if "Hip-Shoulder Separation" in selected_kinematics:
        joint_data["Hip-Shoulder Separation"] = get_hip_shoulder_separation(take_ids)

    if "Elbow Extension Velocity" in selected_kinematics:
        joint_data["Elbow Extension Velocity"] = load_joint_by_handedness(get_elbow_angular_velocity)

    # --- Helper for extracting value at a specific time (ms) ---
    def value_at_time_ms(times_ms, values, target_time_ms):
        if target_time_ms in times_ms:
            return values[times_ms.index(target_time_ms)]
        return None

    import pandas as pd
    summary_rows = []
    compare_energy_summary_rows = []

    fig = go.Figure()

    # --- Date-based colors (Joint Angles ONLY) ---
    unique_dates = sorted(set(take_date_map.values()))
    date_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ]
    date_color_map = {
        d: date_palette[i % len(date_palette)]
        for i, d in enumerate(unique_dates)
    }

    # --- Date-based line dash style map (for visual distinction) ---
    date_dash_map = {}
    dash_styles = ["solid", "dash", "dot", "dashdot"]
    for i, d in enumerate(unique_dates):
        date_dash_map[d] = dash_styles[i % len(dash_styles)]
    use_group_colors_joint = (
        comparison_grouping_enabled
        and len(selected_kinematics) == 1
        and len(group_color_map) >= 2
    )

    # --- Per-take normalization and plotting ---
    grouped = {}
    grouped_by_date = {}
    mound_only_selected = mound_only_sidebar
    median_pkh_frame = None
    if mound_only_selected and knee_event_frames:
        median_pkh_frame = int(np.median(knee_event_frames))

    if joint_window_mode == "Foot Plant to Ball Release View":
        median_fp_frame = int(np.median(fp_event_frames)) if fp_event_frames else None
        joint_window_start = (median_fp_frame - 25) if median_fp_frame is not None else window_start
        joint_window_end = 25
    else:
        joint_window_start = window_start
        joint_window_end = 50
        # For mound throws, ensure the window includes PKH and 20 frames before it.
        if median_pkh_frame is not None:
            joint_window_start = min(window_start, median_pkh_frame - 20)

    joint_window_start_ms = rel_frame_to_ms(joint_window_start)
    joint_window_end_ms = rel_frame_to_ms(joint_window_end)

    # For condensed legend: track which (kinematic, date) pairs have legend entries
    legend_keys_added = set()
    summary_knee_frame = None
    if median_pkh_frame is not None:
        summary_knee_frame = median_pkh_frame
    elif knee_event_frames:
        summary_knee_frame = int(np.median(knee_event_frames))

    # Reuse take_order and take_velocity from Kinematic Sequence section if available
    peak_positive_kinematics = {
        "Shoulder Rotation Velocity",
        "Trunk Rotational Velocity",
        "Torso-Pelvis Rotational Velocity",
        "Pelvis Rotational Velocity",
        "Elbow Extension Velocity",
    }
    collapse_control_group_in_comparison = joint_view_mode == "Comparison" and bool(control_take_ids)
    right_hand_mirror_kinematics = {
        "Shoulder Horizontal Abduction",
        "Shoulder Rotation",
    }
    left_hand_mirror_kinematics = {
        "Trunk Forward Tilt",
        "Trunk Lateral Tilt",
        "Trunk Rotation",
        "Pelvic Lateral Tilt",
        "Pelvis Rotation",
        "Hip-Shoulder Separation",
    }
    for kinematic, data_dict in joint_data.items():
        grouped[kinematic] = {}

        for take_id in take_ids:
            if take_id not in data_dict or take_id not in br_frames:
                continue

            # --- Support both "value" (angles) and "z" (rotational velocities) dicts ---
            if "value" in data_dict[take_id]:
                values = data_dict[take_id]["value"]
                frames = data_dict[take_id]["frame"]
            elif "x" in data_dict[take_id]:
                values = data_dict[take_id]["x"]
                frames = data_dict[take_id]["frame"]
            elif "z" in data_dict[take_id]:
                values = data_dict[take_id]["z"]
                frames = data_dict[take_id]["frame"]
            else:
                continue
            br = br_frames[take_id]
            sign_flip = 1.0
            if kinematic in peak_positive_kinematics:
                valid_vals = [v for v in values if v is not None]
                if valid_vals:
                    dominant_peak = max(valid_vals, key=lambda x: abs(x))
                    if dominant_peak < 0:
                        sign_flip = -1.0

            norm_f, norm_v = [], []
            for f, v in zip(frames, values):
                if v is None:
                    continue

                rel = f - br
                if joint_window_start <= rel <= joint_window_end:
                    norm_f.append(rel_frame_to_ms(rel))

                    # --- Handedness normalization ---
                    take_hand = take_handedness.get(take_id)
                    handedness_factor = 1.0

                    # Keep selected angle directions aligned to a shared orientation.
                    if "Velocity" not in kinematic and take_hand == "R" and kinematic in right_hand_mirror_kinematics:
                        handedness_factor = -1.0

                    # Mirror left-handed trunk tilt curves to right-handed orientation.
                    if take_hand == "L" and kinematic in left_hand_mirror_kinematics:
                        handedness_factor = -1.0

                    norm_v.append(sign_flip * handedness_factor * v)

            grouped[kinematic][take_id] = {"frame": norm_f, "value": norm_v}

            # --- Store by date for grouped plotting ---
            date = take_date_map[take_id]
            group_label = take_group_map.get(take_id, "Ungrouped")
            pitcher_name = take_pitcher_map.get(take_id, "")
            control_group_take = is_control_group_label(group_label)
            hover_pitcher_name = "" if control_group_take else pitcher_name
            if comparison_grouping_enabled and control_group_take:
                date_key = group_label
            elif comparison_grouping_enabled:
                date_key = group_label if group_mode_aggregate_across_pitchers else ((group_label, pitcher_name, date) if multi_pitcher_mode else (group_label, date))
            else:
                date_key = (pitcher_name, date) if multi_pitcher_mode else date
            grouped_by_date.setdefault(date_key, {}).setdefault(kinematic, {})[take_id] = {
                "frame": norm_f,
                "value": norm_v
            }
            trace_color = (
                group_color_map.get(group_label, joint_color_map[kinematic])
                if use_group_colors_joint else
                joint_color_map[kinematic]
            )

            if display_mode == "Individual Throws":
                if collapse_control_group_in_comparison and control_group_take:
                    continue
                # Use kinematic color and date-based dash for individual throws
                fig.add_trace(
                    go.Scatter(
                        x=norm_f,
                        y=norm_v,
                        mode="lines",
                        customdata=[[hover_pitcher_name]] * len(norm_f),
                        hovertemplate=(
                            "<b>%{fullData.name}</b><br>"
                            f"{kinematic}: %{{y:.1f}}{get_kinematic_unit(kinematic)}<br>"
                            "Time: %{x:.1f} ms"
                            + ("<br>Pitcher: %{customdata[0]}" if show_group_pitcher_breakout else "")
                            + "<extra></extra>"
                        ),
                        line=dict(
                            color=trace_color,
                            dash=date_dash_map[take_date_map[take_id]]
                        ),
                        name=(
                            f"Control Group | {kinematic} – Pitch {take_order[take_id]} ({take_velocity[take_id]:.1f} mph)"
                            if (comparison_grouping_enabled and control_group_take) else
                            (
                                f"{group_label} | {kinematic} – {take_date_map[take_id]} | "
                                f"Pitch {take_order[take_id]} ({take_velocity[take_id]:.1f} mph) | {pitcher_name}"
                            ) if (show_group_pitcher_breakout and comparison_grouping_enabled) else
                            (
                                f"{group_label} | {kinematic} – {take_date_map[take_id]} | "
                                f"Pitch {take_order[take_id]} ({take_velocity[take_id]:.1f} mph)"
                            ) if comparison_grouping_enabled else
                            (
                            f"{kinematic} – {take_date_map[take_id]} | Pitch {take_order[take_id]} "
                            f"({take_velocity[take_id]:.1f} mph) | {pitcher_name}"
                            if show_group_pitcher_breakout else
                            f"{kinematic} – {take_date_map[take_id]} | Pitch {take_order[take_id]} ({take_velocity[take_id]:.1f} mph)"
                            )
                        ),
                        showlegend=False
                    )
                )
                # Add one legend-only trace per (kinematic, date) (shows color + dash)
                legend_key = (kinematic, date_key)
                if control_group_take and legend_key not in legend_keys_added:
                    fig.add_trace(
                        go.Scatter(
                            x=[None],
                            y=[None],
                            mode="lines",
                            line=dict(
                                color=trace_color,
                                dash=date_dash_map[date],
                                width=4
                            ),
                            name=(
                                f"Control Group | {kinematic}"
                                if (comparison_grouping_enabled and control_group_take) else
                                f"{group_label} | {kinematic} | {date} | {pitcher_name}"
                                if (show_group_pitcher_breakout and comparison_grouping_enabled) else
                                f"{group_label} | {kinematic} | {date}"
                                if comparison_grouping_enabled else
                                f"{kinematic} | {date} | {pitcher_name}"
                                if show_group_pitcher_breakout else
                                f"{kinematic} | {date}"
                            ),
                            showlegend=True
                        )
                    )
                    legend_keys_added.add(legend_key)

    if display_mode == "Individual Throws" and collapse_control_group_in_comparison:
        control_group_curves = grouped_by_date.get("Control Group", {})
        for kinematic, curves in control_group_curves.items():
            if not curves:
                continue

            x, y, q1, q3 = aggregate_curves(curves, "Mean")
            if len(y) >= 11:
                y = savgol_filter(y, window_length=11, polyorder=3)

            color = (
                group_color_map.get("Control Group", joint_color_map.get(kinematic, "#444"))
                if use_group_colors_joint else
                joint_color_map.get(kinematic, "#444")
            )
            legendgroup = f"Control_Group_{kinematic}"

            if show_joint_signal_iqr_band:
                fig.add_trace(
                    go.Scatter(
                        x=x + x[::-1],
                        y=q3 + q1[::-1],
                        fill="toself",
                        fillcolor=to_rgba(color, 0.35),
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                        legendgroup=legendgroup
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(width=4, color=color),
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        f"{kinematic}: %{{y:.1f}}{get_kinematic_unit(kinematic)}<br>"
                        "Time: %{x:.1f} ms<extra></extra>"
                    ),
                    name=f"Control Group | {kinematic}",
                    showlegend=True,
                    legendgroup=legendgroup
                )
            )

    # --- Summary table: Individual Throws ---
    if display_mode == "Individual Throws":
        for kinematic, curves in grouped.items():
            for take_id, d in curves.items():
                frames = d["frame"]
                values = d["value"]
                if not values:
                    continue

                max_val = np.max(values)
                # sd_val = np.std(values)  # removed as not used below

                br_val = value_at_time_ms(frames, values, 0)

                fp_val = None
                if fp_event_frames:
                    median_fp = rel_frame_to_ms(int(np.median(fp_event_frames)))
                    fp_val = value_at_time_ms(frames, values, median_fp)

                # value at MER (same frame used in plot)
                mer_val = None
                if take_id in shoulder_er_max_frames:
                    mer_frame_rel = shoulder_er_max_frames[take_id] - br_frames[take_id]
                    mer_val = value_at_time_ms(frames, values, rel_frame_to_ms(mer_frame_rel))

                # value at per-take PKH frame (fallback to summary knee frame)
                pkh_val = None
                if take_id in knee_peak_frames:
                    pkh_frame_rel = knee_peak_frames[take_id] - br_frames[take_id]
                    pkh_val = value_at_time_ms(frames, values, rel_frame_to_ms(pkh_frame_rel))
                elif summary_knee_frame is not None:
                    pkh_val = value_at_time_ms(frames, values, rel_frame_to_ms(summary_knee_frame))

                summary_rows.append({
                    **({"Group": take_group_map.get(take_id, "")} if comparison_grouping_enabled else {}),
                    **({"Pitcher": take_pitcher_map.get(take_id)} if show_group_pitcher_breakout else {}),
                    "Kinematic": kinematic + (" (°/s)" if "Velocity" in kinematic else ""),
                    "Session Date": take_date_map[take_id],
                    "Average Velocity": take_velocity[take_id],
                    "Max": max_val,
                    "Peak Knee Height": pkh_val,
                    "Foot Plant": fp_val,
                    "Ball Release": br_val,
                    "Max External Rotation": mer_val
                })

    # --- Grouped plot (mean + IQR per date) ---
    if display_mode == "Grouped":
        for date_key, kin_dict in grouped_by_date.items():
            if comparison_grouping_enabled and date_key == "Control Group":
                group_label = "Control Group"
                pitcher_name = ""
                date = "Selected Takes"
            elif comparison_grouping_enabled and show_group_pitcher_breakout:
                group_label, pitcher_name, date = date_key
            elif comparison_grouping_enabled:
                group_label = date_key
                date = "Selected Takes"
                pitcher_name = ""
            elif multi_pitcher_mode and not comparison_grouping_enabled:
                pitcher_name, date = date_key
                group_label = ""
            else:
                date = date_key
                pitcher_name = ""
                group_label = ""
            for kinematic, curves in kin_dict.items():
                if not curves:
                    continue

                x, y, q1, q3 = aggregate_curves(curves, "Mean")
                avg_velocity = np.mean([take_velocity[tid] for tid in curves.keys()])

                # Smooth grouped curve ONLY
                if len(y) >= 11:
                    y = savgol_filter(y, window_length=11, polyorder=3)

                color = (
                    group_color_map.get(group_label, joint_color_map.get(kinematic, "#444"))
                    if use_group_colors_joint else
                    joint_color_map.get(kinematic, "#444")
                )
                dash = date_dash_map.get(date, "solid")

                # IQR band (draw first so the line color stays visually true on top)
                if show_joint_signal_iqr_band:
                    fig.add_trace(
                        go.Scatter(
                            x=x + x[::-1],
                            y=q3 + q1[::-1],
                            fill="toself",
                            fillcolor=to_rgba(color, 0.35),
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo="skip"
                        )
                    )

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        hovertemplate=(
                            (f"<b>{group_label}</b><br>" if comparison_grouping_enabled else "<b>%{fullData.name}</b><br>")
                            + (f"Avg Velocity: {avg_velocity:.1f} mph<br>" if avg_velocity is not None else "")
                            + 
                            f"{kinematic}: %{{y:.1f}}{get_kinematic_unit(kinematic)}<br>"
                            "Time: %{x:.1f} ms<extra></extra>"
                        ),
                        line=dict(width=4, color=color, dash=dash),
                        name=(
                            f"{group_label} | {kinematic} – {date} | {pitcher_name}"
                            if (show_group_pitcher_breakout and comparison_grouping_enabled) else
                            f"{group_label} | {kinematic} – {date}"
                            if comparison_grouping_enabled else
                            f"{kinematic} – {date} | {pitcher_name}"
                            if show_group_pitcher_breakout else
                            f"{kinematic} – {date}"
                        )
                    )
                )

                max_val = np.max(y)
                br_val = value_at_time_ms(x, y, 0)

                fp_val = None
                if fp_event_frames:
                    median_fp = rel_frame_to_ms(int(np.median(fp_event_frames)))
                    fp_val = value_at_time_ms(x, y, median_fp)

                max_vals = [np.max(d["value"]) for d in curves.values() if d["value"]]
                sd_val = np.std(max_vals)

                # value at MER from grouped mean curve
                mer_val = None
                if mer_event_frames:
                    median_mer = rel_frame_to_ms(int(np.median(mer_event_frames)))
                    mer_val = value_at_time_ms(x, y, median_mer)

                # value at summary PKH frame from grouped mean curve
                pkh_val = None
                if summary_knee_frame is not None:
                    pkh_val = value_at_time_ms(x, y, rel_frame_to_ms(summary_knee_frame))

                summary_rows.append({
                    **({"Group": group_label} if comparison_grouping_enabled else {}),
                    **({"Pitcher": pitcher_name} if show_group_pitcher_breakout else {}),
                    "Kinematic": kinematic + (" (°/s)" if "Velocity" in kinematic else ""),
                    "Session Date": date,
                    "Average Velocity": np.mean([take_velocity[tid] for tid in curves.keys()]),
                    "Max": max_val,
                    "Peak Knee Height": pkh_val,
                    "Foot Plant": fp_val,
                    "Ball Release": br_val,
                    "Max External Rotation": mer_val,
                    "Standard Deviation": sd_val
                })

    # --- Event lines and annotations (match Kinematic Sequence styling) ---
    if median_pkh_frame is not None:
        add_event_iqr_band(fig, knee_event_frames, "gold", show_joint_fp_iqr_band)
        median_pkh_time_ms = rel_frame_to_ms(median_pkh_frame)
        fig.add_vline(
            x=median_pkh_time_ms,
            line_width=3,
            line_dash="dash",
            line_color="gold",
            opacity=0.9
        )
        fig.add_annotation(
            x=median_pkh_time_ms,
            y=1.055,
            xref="x",
            yref="paper",
            text="PKH",
            showarrow=False,
            font=dict(color="gold", size=14),
            align="center"
        )
    elif knee_event_frames:
        # Non-mound fallback: keep a single knee marker when PKH is not enabled.
        add_event_iqr_band(fig, knee_event_frames, "gold", show_joint_fp_iqr_band)
        median_knee_frame = rel_frame_to_ms(int(np.median(knee_event_frames)))
        fig.add_vline(
            x=median_knee_frame,
            line_width=3,
            line_dash="dash",
            line_color="gold",
            opacity=0.9
        )
        fig.add_annotation(
            x=median_knee_frame,
            y=1.055,
            xref="x",
            yref="paper",
            text="Knee",
            showarrow=False,
            font=dict(color="gold", size=14),
            align="center"
        )

    if fp_event_frames:
        add_event_iqr_band(fig, fp_event_frames, "green", show_joint_fp_iqr_band)
        median_fp_frame = rel_frame_to_ms(int(np.median(fp_event_frames)))
        fig.add_vline(
            x=median_fp_frame,
            line_width=3,
            line_dash="dash",
            line_color="green",
            opacity=0.9
        )
        fig.add_annotation(
            x=median_fp_frame,
            y=1.055,
            xref="x",
            yref="paper",
            text="FP",
            showarrow=False,
            font=dict(color="green", size=14),
            align="center"
        )

    if mer_event_frames:
        add_event_iqr_band(fig, mer_event_frames, "red", show_joint_fp_iqr_band)
        median_mer_frame = rel_frame_to_ms(int(np.median(mer_event_frames)))
        fig.add_vline(
            x=median_mer_frame,
            line_width=3,
            line_dash="dash",
            line_color="red",
            opacity=0.9
        )
        fig.add_annotation(
            x=median_mer_frame,
            y=1.055,
            xref="x",
            yref="paper",
            text="MER",
            showarrow=False,
            font=dict(color="red", size=14),
            align="center"
        )

    # Ball Release reference
    add_event_iqr_band(fig, [0] * max(len(take_ids), 1), "blue", show_joint_fp_iqr_band)
    fig.add_vline(
        x=0,
        line_width=3,
        line_dash="dash",
        line_color="blue",
        opacity=0.9
    )
    fig.add_annotation(
        x=0,
        y=1.055,
        xref="x",
        yref="paper",
        text="BR",
        showarrow=False,
        font=dict(color="blue", size=14),
        align="center"
    )

    fig.update_layout(
        xaxis_title="Time Relative to Ball Release (ms)",
        yaxis_title="Kinematics",
        yaxis=dict(),
        xaxis_range=[joint_window_start_ms, joint_window_end_ms],
        height=600,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.30,
            xanchor="center",
            x=0.5
        ),
        hoverlabel=dict(
            namelength=-1,
            font_size=13
        )
    )

    if joint_view_mode == "Comparison":
        plot_left_col, plot_right_col = st.columns(2)
        with plot_left_col:
            if has_kinematics_selection:
                st.markdown("#### Kinematics")
                st.plotly_chart(fig, use_container_width=True, key="joint_plot_compare_left")
            else:
                st.info("Select at least one kinematic to render the left-side plot.")

        with plot_right_col:
            if compare_energy_metrics:
                st.markdown("#### Energy Flow")
            if not compare_energy_metrics:
                st.info("Select at least one energy flow metric to render the right-side plot.")
            elif not take_ids:
                st.info("No takes available for Energy Flow.")
            else:
                energy_color_map = {
                    "Trunk-Shoulder Energy Flow (RTA_DIST_L | RTA_DIST_R)": "#4C1D95",
                    "Arm Energy Flow (LAR_PROX | RAR_PROX)": "#7C2D12",
                    "Glove Side Trunk-Shoulder Energy Flow": "#E11D48",
                    "Glove Arm Energy Flow": "#14B8A6",
                    "Trunk-Shoulder Rotational Energy Flow": "#DC2626",
                    "Trunk-Shoulder Elevation/Depression Energy Flow": "#2563EB",
                    "Trunk-Shoulder Horizontal Abd/Add Energy Flow": "#16A34A",
                    "Arm Rotational Energy Flow": "#F59E0B",
                    "Arm Elevation/Depression Energy Flow": "#06B6D4",
                    "Arm Horizontal Abd/Add Energy Flow": "#9333EA",
                    "Throwing Shoulder Rotational Torque (Relative to Trunk)": "#FB8C00",
                    **NEW_TRUNK_PELVIS_ENERGY_COLOR_MAP,
                }

                compare_energy_data_by_metric = {}

                def load_compare_energy_by_handedness(loader_fn):
                    merged = {}
                    for hand, ids in take_ids_by_handedness.items():
                        if ids:
                            merged.update(loader_fn(ids, hand))
                    return merged

                for metric in compare_energy_metrics:
                    if metric == "Trunk-Shoulder Energy Flow (RTA_DIST_L | RTA_DIST_R)":
                        compare_energy_data_by_metric[metric] = load_compare_energy_by_handedness(get_distal_arm_segment_power)
                    elif metric == "Arm Energy Flow (LAR_PROX | RAR_PROX)":
                        compare_energy_data_by_metric[metric] = load_compare_energy_by_handedness(get_arm_proximal_energy_transfer)
                    elif metric == "Glove Side Trunk-Shoulder Energy Flow":
                        compare_energy_data_by_metric[metric] = load_compare_energy_by_handedness(get_glove_side_trunk_shoulder_energy_flow)
                    elif metric == "Glove Arm Energy Flow":
                        compare_energy_data_by_metric[metric] = load_compare_energy_by_handedness(get_glove_arm_energy_flow)
                    elif metric == "Trunk-Shoulder Rotational Energy Flow":
                        compare_energy_data_by_metric[metric] = load_compare_energy_by_handedness(get_trunk_shoulder_rot_energy_flow)
                    elif metric == "Trunk-Shoulder Elevation/Depression Energy Flow":
                        compare_energy_data_by_metric[metric] = load_compare_energy_by_handedness(get_trunk_shoulder_elev_energy_flow)
                    elif metric == "Trunk-Shoulder Horizontal Abd/Add Energy Flow":
                        compare_energy_data_by_metric[metric] = load_compare_energy_by_handedness(get_trunk_shoulder_horizabd_energy_flow)
                    elif metric == "Arm Rotational Energy Flow":
                        compare_energy_data_by_metric[metric] = load_compare_energy_by_handedness(get_arm_rot_energy_flow)
                    elif metric == "Arm Elevation/Depression Energy Flow":
                        compare_energy_data_by_metric[metric] = load_compare_energy_by_handedness(get_arm_elev_energy_flow)
                    elif metric == "Arm Horizontal Abd/Add Energy Flow":
                        compare_energy_data_by_metric[metric] = load_compare_energy_by_handedness(get_arm_horizabd_energy_flow)
                    elif metric == "Throwing Shoulder Rotational Torque (Relative to Trunk)":
                        mmt_data = {}
                        if take_ids_by_handedness.get("R"):
                            mmt_data.update(
                                get_energy_flow_from_segment(
                                    take_ids_by_handedness["R"],
                                    "RT_SHOULDER_RTA_MMT",
                                    component="z"
                                )
                            )
                        if take_ids_by_handedness.get("L"):
                            mmt_data.update(
                                get_energy_flow_from_segment(
                                    take_ids_by_handedness["L"],
                                    "LT_SHOULDER_RTA_MMT",
                                    component="z"
                                )
                            )
                        compare_energy_data_by_metric[metric] = mmt_data
                    elif metric in NEW_TRUNK_PELVIS_ENERGY_METRICS:
                        segment_name, category_name = NEW_TRUNK_PELVIS_ENERGY_METRIC_MAP[metric]
                        compare_energy_data_by_metric[metric] = get_energy_flow_from_category_segment(
                            take_ids,
                            category_name,
                            segment_name,
                            component="x",
                        )

                compare_energy_data_by_metric = {
                    k: v for k, v in compare_energy_data_by_metric.items() if v
                }

                if not compare_energy_data_by_metric:
                    st.warning("No energy flow data found for the selected metrics.")
                else:
                    energy_fig = go.Figure()
                    collapse_control_group_energy = bool(control_take_ids)
                    unique_dates = sorted(set(take_date_map.values()))
                    dash_styles = ["solid", "dash", "dot", "dashdot"]
                    date_dash_map = {
                        d: dash_styles[i % len(dash_styles)]
                        for i, d in enumerate(unique_dates)
                    }

                    energy_legend_keys = set()
                    compare_energy_median_pkh_frame = None
                    if mound_only_sidebar and knee_event_frames:
                        compare_energy_median_pkh_frame = int(np.median(knee_event_frames))

                    if compare_energy_window_mode == "Foot Plant to Ball Release View":
                        compare_energy_median_fp_frame = int(np.median(fp_event_frames)) if fp_event_frames else None
                        energy_window_start = (
                            compare_energy_median_fp_frame - 25
                            if compare_energy_median_fp_frame is not None else
                            window_start
                        )
                        energy_window_end = 25
                    else:
                        energy_window_start = window_start
                        energy_window_end = 50
                        if compare_energy_median_pkh_frame is not None:
                            energy_window_start = min(window_start, compare_energy_median_pkh_frame - 20)

                    energy_window_start_ms = rel_frame_to_ms(energy_window_start)
                    energy_window_end_ms = rel_frame_to_ms(energy_window_end)

                    for metric, energy_data in compare_energy_data_by_metric.items():
                        metric_color = energy_color_map.get(metric, "#444")
                        grouped_by_date = {}

                        for take_id, d in energy_data.items():
                            if take_id not in br_frames:
                                continue

                            frames = d["frame"]
                            values = d["value"]
                            br = br_frames[take_id]

                            norm_f, norm_v = [], []
                            for f, v in zip(frames, values):
                                rel = f - br
                                if energy_window_start <= rel <= energy_window_end:
                                    norm_f.append(rel_frame_to_ms(rel))
                                    norm_v.append(v)

                            date = take_date_map[take_id]
                            pitcher_name = take_pitcher_map.get(take_id, "")
                            group_label = take_group_map.get(take_id, "")
                            control_group_take = is_control_group_label(group_label)
                            if comparison_grouping_enabled and control_group_take:
                                date_key = "Control Group"
                            else:
                                date_key = (pitcher_name, date) if multi_pitcher_mode else date
                            grouped_by_date.setdefault(date_key, {})[take_id] = {
                                "frame": norm_f,
                                "value": norm_v
                            }

                            peak_val = None
                            if norm_v:
                                peak_idx = int(np.argmax(np.abs(np.array(norm_v, dtype=float))))
                                peak_val = norm_v[peak_idx]

                            br_val = value_at_time_ms(norm_f, norm_v, 0)
                            fp_val = None
                            if fp_event_frames:
                                median_fp = rel_frame_to_ms(int(np.median(fp_event_frames)))
                                fp_val = value_at_time_ms(norm_f, norm_v, median_fp)

                            mer_val = None
                            if mer_event_frames:
                                median_mer = rel_frame_to_ms(int(np.median(mer_event_frames)))
                                mer_val = value_at_time_ms(norm_f, norm_v, median_mer)

                            if compare_energy_display_mode == "Individual Throws":
                                compare_energy_summary_rows.append({
                                    **({"Pitcher": pitcher_name} if multi_pitcher_mode else {}),
                                    "Metric": metric,
                                    "Session Date": date,
                                    "Average Velocity": take_velocity[take_id],
                                    "Peak": peak_val,
                                    "Foot Plant": fp_val,
                                    "Ball Release": br_val,
                                    "Max External Rotation": mer_val,
                                })

                            if compare_energy_display_mode == "Individual Throws":
                                if collapse_control_group_energy and control_group_take:
                                    continue
                                legendgroup = f"{metric}_{pitcher_name}_{date}" if multi_pitcher_mode else f"{metric}_{date}"
                                energy_fig.add_trace(
                                    go.Scatter(
                                        x=norm_f,
                                        y=norm_v,
                                        mode="lines",
                                        line=dict(
                                            color=metric_color,
                                            dash=date_dash_map[date]
                                        ),
                                        customdata=[[metric, date, take_order[take_id], take_velocity[take_id], pitcher_name]] * len(norm_f),
                                        hovertemplate=(
                                            ("%{customdata[4]} | %{customdata[1]}" if multi_pitcher_mode else "%{customdata[1]}")
                                            + "<br>%{customdata[0]}: %{y:.1f}"
                                            + "<br>Pitch %{customdata[2]} (%{customdata[3]:.1f} mph)"
                                            + "<br>Time: %{x:.0f} ms rel BR"
                                            + "<extra></extra>"
                                        ),
                                        showlegend=False,
                                        legendgroup=legendgroup
                                    )
                                )
                                legend_key = (metric, date_key)
                                if legend_key not in energy_legend_keys:
                                    energy_fig.add_trace(
                                        go.Scatter(
                                            x=[None],
                                            y=[None],
                                            mode="lines",
                                            line=dict(
                                                color=metric_color,
                                                dash=date_dash_map[date],
                                                width=4
                                            ),
                                            name=(
                                                f"{metric} | {date} | {pitcher_name}"
                                                if multi_pitcher_mode else
                                                f"{metric} | {date}"
                                            ),
                                            showlegend=True,
                                            legendgroup=legendgroup
                                        )
                                    )
                                    energy_legend_keys.add(legend_key)

                        if compare_energy_display_mode == "Individual Throws" and collapse_control_group_energy:
                            control_curves = grouped_by_date.get("Control Group", {})
                            if control_curves:
                                x, y, q1, q3 = aggregate_curves(control_curves, "Mean")
                                legendgroup = f"{metric}_Control_Group"

                                if show_compare_energy_signal_iqr_band:
                                    energy_fig.add_trace(
                                        go.Scatter(
                                            x=x + x[::-1],
                                            y=q3 + q1[::-1],
                                            fill="toself",
                                            fillcolor=to_rgba(metric_color, alpha=0.35),
                                            line=dict(width=0),
                                            showlegend=False,
                                            hoverinfo="skip",
                                            legendgroup=legendgroup
                                        )
                                    )

                                energy_fig.add_trace(
                                    go.Scatter(
                                        x=x,
                                        y=y,
                                        mode="lines",
                                        line=dict(width=4, color=metric_color),
                                        hovertemplate=(
                                            "Control Group"
                                            + "<br>%{fullData.name}: %{y:.1f}"
                                            + "<br>Time: %{x:.0f} ms rel BR"
                                            + "<extra></extra>"
                                        ),
                                        name=f"Control Group | {metric}",
                                        showlegend=True,
                                        legendgroup=legendgroup
                                    )
                                )

                        if compare_energy_display_mode == "Grouped":
                            for date_key, curves in grouped_by_date.items():
                                control_group_curves = comparison_grouping_enabled and date_key == "Control Group"
                                if control_group_curves:
                                    pitcher_name = ""
                                    date = "Selected Takes"
                                elif multi_pitcher_mode:
                                    pitcher_name, date = date_key
                                else:
                                    date = date_key
                                    pitcher_name = ""
                                x, y, q1, q3 = aggregate_curves(curves, "Mean")
                                dash_style = date_dash_map.get(date, "solid")
                                legendgroup = (
                                    f"{metric}_Control_Group"
                                    if control_group_curves else
                                    f"{metric}_{pitcher_name}_{date}" if multi_pitcher_mode else f"{metric}_{date}"
                                )

                                peak_val = None
                                if y:
                                    peak_idx = int(np.argmax(np.abs(np.array(y, dtype=float))))
                                    peak_val = y[peak_idx]

                                br_val = value_at_time_ms(x, y, 0)
                                fp_val = None
                                if fp_event_frames:
                                    median_fp = rel_frame_to_ms(int(np.median(fp_event_frames)))
                                    fp_val = value_at_time_ms(x, y, median_fp)

                                mer_val = None
                                if mer_event_frames:
                                    median_mer = rel_frame_to_ms(int(np.median(mer_event_frames)))
                                    mer_val = value_at_time_ms(x, y, median_mer)

                                peak_vals = []
                                for curve in curves.values():
                                    if curve["value"]:
                                        curve_arr = np.array(curve["value"], dtype=float)
                                        peak_vals.append(float(curve_arr[np.argmax(np.abs(curve_arr))]))

                                compare_energy_summary_rows.append({
                                    **({"Pitcher": pitcher_name} if multi_pitcher_mode else {}),
                                    "Metric": metric,
                                    "Session Date": date,
                                    "Average Velocity": np.mean([take_velocity[tid] for tid in curves.keys()]),
                                    "Peak": peak_val,
                                    "Foot Plant": fp_val,
                                    "Ball Release": br_val,
                                    "Max External Rotation": mer_val,
                                    "Standard Deviation": (np.std(peak_vals) if peak_vals else None),
                                })
                                avg_velocity = np.mean([take_velocity[tid] for tid in curves.keys()])

                                if show_compare_energy_signal_iqr_band:
                                    energy_fig.add_trace(
                                        go.Scatter(
                                            x=x + x[::-1],
                                            y=q3 + q1[::-1],
                                            fill="toself",
                                            fillcolor=to_rgba(metric_color, alpha=0.35),
                                            line=dict(width=0),
                                            showlegend=False,
                                            hoverinfo="skip",
                                            legendgroup=legendgroup
                                        )
                                    )
                                energy_fig.add_trace(
                                    go.Scatter(
                                        x=x,
                                        y=y,
                                        mode="lines",
                                        line=dict(width=4, color=metric_color, dash=dash_style),
                                        customdata=[[metric, date, pitcher_name]] * len(x),
                                        hovertemplate=(
                                            ("Control Group" if control_group_curves else "%{customdata[2]} | %{customdata[1]}" if multi_pitcher_mode else "%{customdata[1]}")
                                            + (f"<br>Avg Velocity: {avg_velocity:.1f} mph" if avg_velocity is not None else "")
                                            + "<br>%{customdata[0]}: %{y:.1f}"
                                            + "<br>Time: %{x:.0f} ms rel BR"
                                            + "<extra></extra>"
                                        ),
                                        showlegend=False,
                                        legendgroup=legendgroup
                                    )
                                )
                                energy_fig.add_trace(
                                    go.Scatter(
                                        x=[None],
                                        y=[None],
                                        mode="lines",
                                        line=dict(color=metric_color, dash=dash_style, width=4),
                                        name=(
                                            f"Control Group | {metric}"
                                            if control_group_curves else
                                            f"{metric} | {date} | {pitcher_name}"
                                            if multi_pitcher_mode else
                                            f"{metric} | {date}"
                                        ),
                                        showlegend=True,
                                        legendgroup=legendgroup
                                    )
                    )

                    if compare_energy_median_pkh_frame is not None:
                        add_event_iqr_band(energy_fig, knee_event_frames, "gold", show_compare_energy_fp_iqr_band)
                        median_pkh = rel_frame_to_ms(compare_energy_median_pkh_frame)
                        energy_fig.add_vline(x=median_pkh, line_width=3, line_dash="dash", line_color="gold")
                        energy_fig.add_annotation(
                            x=median_pkh,
                            y=1.06,
                            xref="x",
                            yref="paper",
                            text="PKH",
                            showarrow=False,
                            font=dict(color="gold", size=13, family="Arial"),
                            align="center"
                        )
                    elif knee_event_frames:
                        add_event_iqr_band(energy_fig, knee_event_frames, "gold", show_compare_energy_fp_iqr_band)
                        median_knee = rel_frame_to_ms(int(np.median(knee_event_frames)))
                        energy_fig.add_vline(x=median_knee, line_width=3, line_dash="dash", line_color="gold")
                        energy_fig.add_annotation(
                            x=median_knee,
                            y=1.06,
                            xref="x",
                            yref="paper",
                            text="Knee",
                            showarrow=False,
                            font=dict(color="gold", size=13, family="Arial"),
                            align="center"
                        )

                    if fp_event_frames:
                        add_event_iqr_band(energy_fig, fp_event_frames, "green", show_compare_energy_fp_iqr_band)
                        median_fp = rel_frame_to_ms(int(np.median(fp_event_frames)))
                        energy_fig.add_vline(x=median_fp, line_width=3, line_dash="dash", line_color="green")
                        energy_fig.add_annotation(
                            x=median_fp,
                            y=1.06,
                            xref="x",
                            yref="paper",
                            text="FP",
                            showarrow=False,
                            font=dict(color="green", size=13, family="Arial"),
                            align="center"
                        )
                    if mer_event_frames:
                        add_event_iqr_band(energy_fig, mer_event_frames, "red", show_compare_energy_fp_iqr_band)
                        median_mer = rel_frame_to_ms(int(np.median(mer_event_frames)))
                        energy_fig.add_vline(x=median_mer, line_width=3, line_dash="dash", line_color="red")
                        energy_fig.add_annotation(
                            x=median_mer,
                            y=1.06,
                            xref="x",
                            yref="paper",
                            text="MER",
                            showarrow=False,
                            font=dict(color="red", size=13, family="Arial"),
                            align="center"
                        )
                    add_event_iqr_band(energy_fig, [0] * max(len(take_ids), 1), "blue", show_compare_energy_fp_iqr_band)
                    energy_fig.add_vline(x=0, line_width=3, line_dash="dash", line_color="blue")
                    energy_fig.add_annotation(
                        x=0,
                        y=1.06,
                        xref="x",
                        yref="paper",
                        text="BR",
                        showarrow=False,
                        font=dict(color="blue", size=13, family="Arial"),
                        align="center"
                    )

                    energy_fig.update_layout(
                        xaxis_title="Time Relative to Ball Release (ms)",
                        yaxis_title="Energy Flow / Segment Power",
                        xaxis_range=[energy_window_start_ms, energy_window_end_ms],
                        height=600,
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=-0.30,
                            xanchor="center",
                            x=0.5,
                            groupclick="togglegroup"
                        ),
                        hoverlabel=dict(
                            namelength=-1,
                            font_size=13
                        )
                    )
                    st.plotly_chart(energy_fig, use_container_width=True, key="joint_plot_compare_right_energy")
    else:
        if show_single_kinematics_empty_state:
            st.markdown("")
        else:
            st.plotly_chart(fig, use_container_width=True, key="joint_plot_single")

    # --- Kinematics Table ---
    has_compare_energy_summary = (
        joint_view_mode == "Comparison"
        and bool(compare_energy_metrics)
        and bool(compare_energy_summary_rows)
    )
    combined_summary_mode = (
        not show_single_kinematics_empty_state
        and bool(summary_rows)
        and has_compare_energy_summary
    )
    rendered_summary_heading = False
    if not show_single_kinematics_empty_state and summary_rows:
        st.markdown("### Summary" if combined_summary_mode else "### Kinematics Summary")
        rendered_summary_heading = True
        df_summary = pd.DataFrame(summary_rows)
        # Reorder columns explicitly
        base_columns = [
            "Kinematic",
            "Session Date",
            "Average Velocity",
            "Max",
            "Peak Knee Height",
            "Foot Plant",
            "Ball Release",
            "Max External Rotation"
        ]
        if joint_window_mode == "Foot Plant to Ball Release View":
            base_columns.remove("Peak Knee Height")
        if comparison_grouping_enabled:
            base_columns = ["Group"] + base_columns
        if show_group_pitcher_breakout:
            base_columns = ["Pitcher"] + base_columns

        if display_mode == "Grouped":
            column_order = base_columns + ["Standard Deviation"]
        else:
            column_order = base_columns

        df_summary = df_summary[column_order]

        import numpy as np
        def fmt(val, decimals=2):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return ""
            return f"{val:.{decimals}f}"

        def normalize_kinematic_name(display_name):
            return display_name.replace(" (°/s)", "")

        def format_with_unit(val, unit, decimals=2, prefix=""):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return ""
            return f"{prefix}{val:.{decimals}f} {unit}".strip()

        measurement_columns = [
            "Max",
            "Peak Knee Height",
            "Foot Plant",
            "Ball Release",
            "Max External Rotation",
        ]
        if joint_window_mode == "Foot Plant to Ball Release View":
            measurement_columns.remove("Peak Knee Height")

        for idx, row in df_summary.iterrows():
            kinematic_name = normalize_kinematic_name(row["Kinematic"])
            kinematic_unit = get_kinematic_unit(kinematic_name)

            if "Average Velocity" in df_summary.columns:
                df_summary.at[idx, "Average Velocity"] = fmt(row["Average Velocity"], 1)

            for col in measurement_columns:
                if col in df_summary.columns:
                    df_summary.at[idx, col] = format_with_unit(
                        row[col], kinematic_unit, decimals=2
                    )

            if display_mode == "Grouped" and "Standard Deviation" in df_summary.columns:
                df_summary.at[idx, "Standard Deviation"] = format_with_unit(
                    row["Standard Deviation"], kinematic_unit, decimals=2, prefix="±"
                )

        styled_summary = (
            df_summary
            .style
            .hide(axis="index")
            # Center headers
            .set_table_styles([
                {"selector": "th", "props": [("text-align", "center")]}
            ])
            # Center label columns
            .set_properties(
                subset=["Kinematic", "Session Date"],
                **{"text-align": "center"}
            )
            # Center numeric columns
            .set_properties(
                subset=[c for c in df_summary.columns if c not in ["Kinematic", "Session Date"]],
                **{"text-align": "center"}
            )
        )
        summary_column_config = {}
        if "Group" in df_summary.columns:
            summary_column_config["Group"] = st.column_config.TextColumn(
                "Group",
                width="small",
            )

        st.dataframe(
            styled_summary,
            use_container_width=True,
            column_config=summary_column_config or None,
            hide_index=True,
        )

    if has_compare_energy_summary:
        if not rendered_summary_heading:
            st.markdown("### Energy Flow Summary")
            rendered_summary_heading = True
        df_energy_summary = pd.DataFrame(compare_energy_summary_rows)

        def fmt_energy(val, decimals=2):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return ""
            return f"{val:.{decimals}f}"

        energy_base_columns = [
            "Metric",
            "Session Date",
            "Average Velocity",
            "Peak",
            "Foot Plant",
            "Ball Release",
            "Max External Rotation",
        ]
        if multi_pitcher_mode and "Pitcher" in df_energy_summary.columns:
            energy_base_columns = ["Pitcher"] + energy_base_columns
        if compare_energy_display_mode == "Grouped" and "Standard Deviation" in df_energy_summary.columns:
            energy_column_order = energy_base_columns + ["Standard Deviation"]
        else:
            energy_column_order = energy_base_columns

        df_energy_summary = df_energy_summary[energy_column_order]

        for idx, row in df_energy_summary.iterrows():
            if "Average Velocity" in df_energy_summary.columns:
                df_energy_summary.at[idx, "Average Velocity"] = fmt_energy(row["Average Velocity"], 1)
            for col in ["Peak", "Foot Plant", "Ball Release", "Max External Rotation"]:
                if col in df_energy_summary.columns:
                    df_energy_summary.at[idx, col] = fmt_energy(row[col], 1)
            if compare_energy_display_mode == "Grouped" and "Standard Deviation" in df_energy_summary.columns:
                df_energy_summary.at[idx, "Standard Deviation"] = (
                    f"±{row['Standard Deviation']:.1f}" if row["Standard Deviation"] is not None and not (isinstance(row["Standard Deviation"], float) and np.isnan(row["Standard Deviation"])) else ""
                )

        styled_energy_summary = (
            df_energy_summary
            .style
            .hide(axis="index")
            .set_table_styles([
                {"selector": "th", "props": [("text-align", "center")]}
            ])
            .set_properties(
                subset=[c for c in df_energy_summary.columns],
                **{"text-align": "center"}
            )
        )

        st.dataframe(
            styled_energy_summary,
            use_container_width=True,
            hide_index=True,
        )

    has_compare_energy_definitions = (
        joint_view_mode == "Comparison"
        and bool(compare_energy_metrics)
        and any(metric in energy_definitions for metric in compare_energy_metrics)
    )
    combined_definitions_mode = (
        not show_single_kinematics_empty_state
        and bool(selected_kinematics)
        and has_compare_energy_definitions
    )
    rendered_definitions_heading = False
    if not show_single_kinematics_empty_state:
        st.markdown("### Definitions" if combined_definitions_mode else "### Kinematic Definitions")
        rendered_definitions_heading = True
        for metric in selected_kinematics:
            metric_info = kinematic_definitions.get(metric, {})
            if not metric_info:
                continue
            st.markdown(
                (
                    f"<div style='font-size:1.15rem; line-height:1.6; margin:0.35rem 0 0.9rem 0;'>"
                    f"<strong>{metric}:</strong> {metric_info.get('definition', '')}"
                    f"</div>"
                ),
                unsafe_allow_html=True,
            )

    if joint_view_mode == "Comparison" and compare_energy_metrics:
        defined_compare_energy_metrics = [
            metric for metric in compare_energy_metrics if metric in energy_definitions
        ]
        if defined_compare_energy_metrics:
            if not rendered_definitions_heading:
                st.markdown("### Energy Flow Definitions")
                rendered_definitions_heading = True
            for metric in defined_compare_energy_metrics:
                metric_info = energy_definitions.get(metric, {})
                st.markdown(
                    (
                        f"<div style='font-size:1.15rem; line-height:1.6; margin:0.35rem 0 0.9rem 0;'>"
                        f"<strong>{metric}:</strong> {metric_info.get('definition', '')}"
                        f"</div>"
                    ),
                    unsafe_allow_html=True,
                )

# --------------------------------------------------
# Energy Flow Tab
# --------------------------------------------------


# --------------------------------------------------
# Helper: Compute peak distal arm segment power (W) per take
# --------------------------------------------------
def compute_peak_segment_power(energy_data, br_frames, fp_event_frames):
    """
    Compute peak distal arm segment power (W) per take
    restricted to Foot Plant → Ball Release.
    Uses the most negative (minimum) value in the window.
    """
    peak_map = {}

    if not fp_event_frames:
        return peak_map

    median_fp_rel = int(np.median(fp_event_frames))

    for take_id, d in energy_data.items():
        if take_id not in br_frames:
            continue

        br = br_frames[take_id]
        frames = np.array(d["frame"], dtype=int)
        values = np.array(d["value"], dtype=float)

        rel = frames - br

        # STRICT biomechanical window: Foot Plant → Ball Release
        mask = (rel >= median_fp_rel) & (rel <= 0)

        if not np.any(mask):
            continue

        peak_map[take_id] = float(np.nanmin(values[mask]))

    return peak_map


with tab_energy:
    st.subheader("Energy Flow")
    render_group_selection_summary()

    st.markdown(
        """
        <style>
        .energy-controls-label {
            font-size: 0.8rem;
            font-weight: 700;
            color: #6b7280;
            margin-bottom: 0.1rem;
        }

        div[data-testid="stSegmentedControl"] label,
        div[data-testid="stSegmentedControl"] div[role="radiogroup"] label,
        div[data-testid="stSegmentedControl"] div[role="radiogroup"] p,
        div[data-testid="stToggle"] label,
        div[data-testid="stToggle"] p {
            font-size: 1rem !important;
            font-weight: 400 !important;
        }

        .energy-toggle-label {
            margin-top: -0.1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    energy_display_col, energy_options_col, energy_spacer_col = st.columns([1.45, 1.75, 2.2])
    with energy_display_col:
        st.markdown('<div class="energy-controls-label">Display Mode</div>', unsafe_allow_html=True)
        display_mode = st.segmented_control(
            "Select Display Mode",
            ["Individual Throws", "Grouped"],
            default="Grouped",
            key="energy_display_mode",
            label_visibility="collapsed",
        )
    with energy_options_col:
        st.markdown('<div class="energy-controls-label energy-toggle-label">Options</div>', unsafe_allow_html=True)
        energy_event_col, energy_signal_col = st.columns(2)
        with energy_event_col:
            show_energy_fp_iqr_band = st.toggle(
                "Event Bands",
                value=False,
                key="energy_show_fp_iqr_band",
                help="Shows the middle 50% range for event timing across selected throws.",
            )
        with energy_signal_col:
            show_energy_signal_iqr_band = st.toggle(
                "Signal Bands",
                value=True,
                key="energy_show_signal_iqr_band",
                help="Shows the middle 50% range around each grouped mean line.",
            )
    with energy_spacer_col:
        st.markdown("")

    energy_window_col, energy_window_spacer = st.columns([2.35, 3.65])
    with energy_window_col:
        st.markdown('<div class="energy-controls-label">View Window</div>', unsafe_allow_html=True)
        energy_window_mode = st.segmented_control(
            "Energy Flow View",
            ["Peak Knee Height View", "Foot Plant to Ball Release View"],
            default="Peak Knee Height View",
            key="energy_window_mode",
            label_visibility="collapsed",
        )
    with energy_window_spacer:
        st.markdown("")

    energy_select_col, energy_select_spacer = st.columns([3, 3])
    with energy_select_col:
        energy_metrics = st.multiselect(
            "Select Energy Flow Metrics",
            [
                "Trunk-Shoulder Energy Flow (RTA_DIST_L | RTA_DIST_R)",
                "Arm Energy Flow (LAR_PROX | RAR_PROX)",
                "Glove Side Trunk-Shoulder Energy Flow",
                "Glove Arm Energy Flow",
                "Trunk-Shoulder Rotational Energy Flow",
                "Trunk-Shoulder Elevation/Depression Energy Flow",
                "Trunk-Shoulder Horizontal Abd/Add Energy Flow",
                "Arm Rotational Energy Flow",
                "Arm Elevation/Depression Energy Flow",
                "Arm Horizontal Abd/Add Energy Flow",
                "Throwing Shoulder Rotational Torque (Relative to Trunk)",
                *NEW_TRUNK_PELVIS_ENERGY_METRICS,
            ],
            default=[]
        )
    with energy_select_spacer:
        st.markdown("")

    if not energy_metrics:
        energy_empty_col, energy_empty_spacer = st.columns([3, 3])
        with energy_empty_col:
            st.info("Select at least one energy flow metric.")
        with energy_empty_spacer:
            st.markdown("")

    if not take_ids:
        st.info("No takes available for Energy Flow.")

    # --- Fixed color map for Energy Flow metrics (high-contrast palette) ---
    energy_color_map = {
        "Trunk-Shoulder Energy Flow (RTA_DIST_L | RTA_DIST_R)": "#4C1D95",  # deep indigo / purple
        "Arm Energy Flow (LAR_PROX | RAR_PROX)": "#7C2D12",  # dark brown
        "Glove Side Trunk-Shoulder Energy Flow": "#E11D48",
        "Glove Arm Energy Flow": "#14B8A6",
        "Trunk-Shoulder Rotational Energy Flow": "#DC2626",  # strong red
        "Trunk-Shoulder Elevation/Depression Energy Flow": "#2563EB",  # vivid blue
        "Trunk-Shoulder Horizontal Abd/Add Energy Flow": "#16A34A",     # strong green
        "Arm Rotational Energy Flow": "#F59E0B",        # amber
        "Arm Elevation/Depression Energy Flow": "#06B6D4",  # cyan
        "Arm Horizontal Abd/Add Energy Flow": "#9333EA",     # violet
        "Throwing Shoulder Rotational Torque (Relative to Trunk)": "#FB8C00",
        **NEW_TRUNK_PELVIS_ENERGY_COLOR_MAP,
    }

    # --- Load all selected metrics ---
    energy_data_by_metric = {}

    def load_energy_by_handedness(loader_fn):
        merged = {}
        for hand, ids in take_ids_by_handedness.items():
            if ids:
                merged.update(loader_fn(ids, hand))
        return merged

    for metric in energy_metrics:
        if metric == "Trunk-Shoulder Energy Flow (RTA_DIST_L | RTA_DIST_R)":
            energy_data_by_metric[metric] = load_energy_by_handedness(get_distal_arm_segment_power)
        elif metric == "Arm Energy Flow (LAR_PROX | RAR_PROX)":
            energy_data_by_metric[metric] = load_energy_by_handedness(get_arm_proximal_energy_transfer)
        elif metric == "Glove Side Trunk-Shoulder Energy Flow":
            energy_data_by_metric[metric] = load_energy_by_handedness(get_glove_side_trunk_shoulder_energy_flow)
        elif metric == "Glove Arm Energy Flow":
            energy_data_by_metric[metric] = load_energy_by_handedness(get_glove_arm_energy_flow)
        elif metric == "Trunk-Shoulder Rotational Energy Flow":
            energy_data_by_metric[metric] = load_energy_by_handedness(get_trunk_shoulder_rot_energy_flow)
        elif metric == "Trunk-Shoulder Elevation/Depression Energy Flow":
            energy_data_by_metric[metric] = load_energy_by_handedness(get_trunk_shoulder_elev_energy_flow)
        elif metric == "Trunk-Shoulder Horizontal Abd/Add Energy Flow":
            energy_data_by_metric[metric] = load_energy_by_handedness(get_trunk_shoulder_horizabd_energy_flow)
        elif metric == "Arm Rotational Energy Flow":
            energy_data_by_metric[metric] = load_energy_by_handedness(get_arm_rot_energy_flow)
        elif metric == "Arm Elevation/Depression Energy Flow":
            energy_data_by_metric[metric] = load_energy_by_handedness(get_arm_elev_energy_flow)
        elif metric == "Arm Horizontal Abd/Add Energy Flow":
            energy_data_by_metric[metric] = load_energy_by_handedness(get_arm_horizabd_energy_flow)
        elif metric == "Throwing Shoulder Rotational Torque (Relative to Trunk)":
            mmt_data = {}
            if take_ids_by_handedness.get("R"):
                mmt_data.update(
                    get_energy_flow_from_segment(
                        take_ids_by_handedness["R"],
                        "RT_SHOULDER_RTA_MMT",
                        component="z"
                    )
                )
            if take_ids_by_handedness.get("L"):
                mmt_data.update(
                    get_energy_flow_from_segment(
                        take_ids_by_handedness["L"],
                        "LT_SHOULDER_RTA_MMT",
                        component="z"
                    )
                )
            energy_data_by_metric[metric] = mmt_data
        elif metric in NEW_TRUNK_PELVIS_ENERGY_METRICS:
            segment_name, category_name = NEW_TRUNK_PELVIS_ENERGY_METRIC_MAP[metric]
            energy_data_by_metric[metric] = get_energy_flow_from_category_segment(
                take_ids,
                category_name,
                segment_name,
                component="x",
            )

    energy_data_by_metric = {
        k: v for k, v in energy_data_by_metric.items() if v
    }

    if not energy_data_by_metric:
        st.warning("No energy flow data found for the selected metrics.")

    fig = go.Figure()

    # --- Date dash styles (same as KS) ---
    unique_dates = sorted(set(take_date_map.values()))
    dash_styles = ["solid", "dash", "dot", "dashdot"]
    date_dash_map = {
        d: dash_styles[i % len(dash_styles)]
        for i, d in enumerate(unique_dates)
    }

    legend_keys_added = set()
    energy_median_pkh_frame = None
    if mound_only_sidebar and knee_event_frames:
        energy_median_pkh_frame = int(np.median(knee_event_frames))

    if energy_window_mode == "Foot Plant to Ball Release View":
        energy_median_fp_frame = int(np.median(fp_event_frames)) if fp_event_frames else None
        energy_window_start = (
            energy_median_fp_frame - 25
            if energy_median_fp_frame is not None else
            window_start
        )
        energy_window_end = 25
    else:
        energy_window_start = window_start
        energy_window_end = 50
        if energy_median_pkh_frame is not None:
            energy_window_start = min(window_start, energy_median_pkh_frame - 20)

    energy_window_start_ms = rel_frame_to_ms(energy_window_start)
    energy_window_end_ms = rel_frame_to_ms(energy_window_end)
    use_group_colors_energy = (
        comparison_grouping_enabled
        and len(energy_metrics) == 1
        and len(group_color_map) >= 2
    )

    # -------------------------------
    # Normalize to Ball Release and Plot
    # -------------------------------
    for metric, energy_data in energy_data_by_metric.items():
        metric_color = energy_color_map.get(metric, "#444")

        grouped_power = {}
        grouped_by_date = {}

        for take_id, d in energy_data.items():
            if take_id not in br_frames:
                continue

            frames = d["frame"]
            values = d["value"]
            br = br_frames[take_id]

            norm_f, norm_v = [], []
            for f, v in zip(frames, values):
                rel = f - br
                if energy_window_start <= rel <= energy_window_end:
                    norm_f.append(rel_frame_to_ms(rel))
                    norm_v.append(v)

            grouped_power[take_id] = {"frame": norm_f, "value": norm_v}

            date = take_date_map[take_id]
            group_label = take_group_map.get(take_id, "Ungrouped")
            pitcher_name = take_pitcher_map.get(take_id, "")
            control_group_take = is_control_group_label(group_label)
            hover_pitcher_name = "" if control_group_take else pitcher_name
            if comparison_grouping_enabled and control_group_take:
                date_key = group_label
            elif comparison_grouping_enabled:
                date_key = group_label if group_mode_aggregate_across_pitchers else ((group_label, pitcher_name, date) if multi_pitcher_mode else (group_label, date))
            else:
                date_key = (pitcher_name, date) if multi_pitcher_mode else date
            grouped_by_date.setdefault(date_key, {})[take_id] = {
                "frame": norm_f,
                "value": norm_v
            }
            trace_color = (
                group_color_map.get(group_label, metric_color)
                if use_group_colors_energy else
                metric_color
            )

            if display_mode == "Individual Throws":
                legendgroup = (
                    f"{group_label}_{metric}_{pitcher_name}_{date}"
                    if (comparison_grouping_enabled and show_group_pitcher_breakout) else
                    f"{group_label}_{metric}_{date}"
                    if comparison_grouping_enabled else
                    f"{metric}_{pitcher_name}_{date}"
                    if show_group_pitcher_breakout else
                    f"{metric}_{date}"
                )
                fig.add_trace(
                    go.Scatter(
                        x=norm_f,
                        y=norm_v,
                        mode="lines",
                        line=dict(
                            color=trace_color,
                            dash=date_dash_map[date]
                        ),
                        customdata=[[metric, date, take_order[take_id], take_velocity[take_id], hover_pitcher_name]] * len(norm_f),
                        hovertemplate=(
                            ("%{customdata[4]} | %{customdata[1]}" if show_group_pitcher_breakout else "%{customdata[1]}")
                            + "<br>%{customdata[0]}: %{y:.1f}"
                            + "<br>Pitch %{customdata[2]} (%{customdata[3]:.1f} mph)"
                            + "<br>Time: %{x:.0f} ms rel BR"
                            + "<extra></extra>"
                        ),
                        name=(
                            f"Control Group | {metric} – Pitch {take_order[take_id]} ({take_velocity[take_id]:.1f} mph)"
                            if (comparison_grouping_enabled and control_group_take) else
                            f"{group_label} | {metric} – {date} | Pitch {take_order[take_id]} ({take_velocity[take_id]:.1f} mph) | {pitcher_name}"
                            if (comparison_grouping_enabled and show_group_pitcher_breakout) else
                            f"{group_label} | {metric} – {date} | Pitch {take_order[take_id]} ({take_velocity[take_id]:.1f} mph)"
                            if comparison_grouping_enabled else None
                        ),
                        showlegend=False,
                        legendgroup=legendgroup
                    )
                )
                legend_key = (metric, date_key)
                if control_group_take and legend_key not in legend_keys_added:
                    fig.add_trace(
                        go.Scatter(
                            x=[None],
                            y=[None],
                            mode="lines",
                            line=dict(
                                color=trace_color,
                                dash=date_dash_map[date],
                                width=4
                            ),
                            name=(
                                f"Control Group | {metric}"
                                if (comparison_grouping_enabled and control_group_take) else
                                f"{group_label} | {metric} | {date} | {pitcher_name}"
                                if (show_group_pitcher_breakout and comparison_grouping_enabled) else
                                f"{group_label} | {metric} | {date}"
                                if comparison_grouping_enabled else
                            f"{metric} | {date} | {pitcher_name}"
                            if show_group_pitcher_breakout else
                            f"{metric} | {date}"
                        ),
                            showlegend=True,
                            legendgroup=legendgroup
                        )
                    )
                    legend_keys_added.add(legend_key)

        # -------------------------------
        # Grouped (Mean + IQR per date)
        # -------------------------------
        if display_mode == "Grouped":
            for date_key, curves in grouped_by_date.items():
                if comparison_grouping_enabled and date_key == "Control Group":
                    group_label = "Control Group"
                    pitcher_name = ""
                    date = "Selected Takes"
                elif comparison_grouping_enabled and show_group_pitcher_breakout:
                    group_label, pitcher_name, date = date_key
                elif comparison_grouping_enabled:
                    group_label = date_key
                    date = "Selected Takes"
                    pitcher_name = ""
                elif multi_pitcher_mode and not comparison_grouping_enabled:
                    pitcher_name, date = date_key
                    group_label = ""
                else:
                    date = date_key
                    pitcher_name = ""
                    group_label = ""
                x, y, q1, q3 = aggregate_curves(curves, "Mean")
                avg_velocity = np.mean([take_velocity[tid] for tid in curves.keys()])
                legendgroup = (
                    f"{group_label}_{metric}_{pitcher_name}_{date}"
                    if (comparison_grouping_enabled and show_group_pitcher_breakout) else
                    f"{group_label}_{metric}_{date}"
                    if comparison_grouping_enabled else
                    f"{metric}_{pitcher_name}_{date}"
                    if show_group_pitcher_breakout else
                    f"{metric}_{date}"
                )

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        line=dict(
                            width=4,
                            color=(
                                group_color_map.get(group_label, metric_color)
                                if use_group_colors_energy else
                                metric_color
                            ),
                            dash=date_dash_map.get(date, "solid")
                        ),
                        customdata=[[metric, date, group_label, pitcher_name]] * len(x),
                        hovertemplate=(
                            (f"{group_label}<br>" if comparison_grouping_enabled else "")
                            + ("%{customdata[0]}" if comparison_grouping_enabled else "%{customdata[3]} | %{customdata[1]}" if show_group_pitcher_breakout else "%{customdata[1]}")
                            + (" | %{customdata[3]}" if show_group_pitcher_breakout and comparison_grouping_enabled else "")
                            + (f"<br>Avg Velocity: {avg_velocity:.1f} mph" if avg_velocity is not None else "")
                            + "<br>%{customdata[0]}: %{y:.1f}"
                            + "<br>Time: %{x:.0f} ms rel BR"
                            + "<extra></extra>"
                        ),
                        showlegend=False,
                        legendgroup=legendgroup
                    )
                )

                if show_energy_signal_iqr_band:
                    fig.add_trace(
                        go.Scatter(
                            x=x + x[::-1],
                            y=q3 + q1[::-1],
                            fill="toself",
                            fillcolor=to_rgba(
                                group_color_map.get(group_label, metric_color)
                                if use_group_colors_energy else
                                metric_color,
                                alpha=0.35
                            ),
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo="skip",
                            legendgroup=legendgroup
                        )
                    )

                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="lines",
                        line=dict(
                            color=(
                                group_color_map.get(group_label, metric_color)
                                if use_group_colors_energy else
                                metric_color
                            ),
                            dash=date_dash_map.get(date, "solid"),
                            width=4
                        ),
                        name=(
                            f"{group_label} | {metric} | {date} | {pitcher_name}"
                            if (show_group_pitcher_breakout and comparison_grouping_enabled) else
                            f"{group_label} | {metric} | {date}"
                            if comparison_grouping_enabled else
                            f"{metric} | {date} | {pitcher_name}"
                            if show_group_pitcher_breakout else
                            f"{metric} | {date}"
                        ),
                        showlegend=True,
                        legendgroup=legendgroup
                    )
                )

    # -------------------------------
    # Event Lines (with text labels above)
    # -------------------------------
    if energy_median_pkh_frame is not None:
        add_event_iqr_band(fig, knee_event_frames, "gold", show_energy_fp_iqr_band)
        median_pkh = rel_frame_to_ms(energy_median_pkh_frame)
        fig.add_vline(x=median_pkh, line_width=3, line_dash="dash", line_color="gold")
        fig.add_annotation(
            x=median_pkh,
            y=1.06,
            xref="x",
            yref="paper",
            text="PKH",
            showarrow=False,
            font=dict(color="gold", size=13, family="Arial"),
            align="center"
        )
    elif knee_event_frames:
        add_event_iqr_band(fig, knee_event_frames, "gold", show_energy_fp_iqr_band)
        median_knee = rel_frame_to_ms(int(np.median(knee_event_frames)))
        fig.add_vline(x=median_knee, line_width=3, line_dash="dash", line_color="gold")
        fig.add_annotation(
            x=median_knee,
            y=1.06,
            xref="x",
            yref="paper",
            text="Knee",
            showarrow=False,
            font=dict(color="gold", size=13, family="Arial"),
            align="center"
        )

    if fp_event_frames:
        add_event_iqr_band(fig, fp_event_frames, "green", show_energy_fp_iqr_band)
        median_fp = rel_frame_to_ms(int(np.median(fp_event_frames)))
        fig.add_vline(x=median_fp, line_width=3, line_dash="dash", line_color="green")
        fig.add_annotation(
            x=median_fp,
            y=1.06,
            xref="x",
            yref="paper",
            text="FP",
            showarrow=False,
            font=dict(color="green", size=13, family="Arial"),
            align="center"
        )

    if mer_event_frames:
        add_event_iqr_band(fig, mer_event_frames, "red", show_energy_fp_iqr_band)
        median_mer = rel_frame_to_ms(int(np.median(mer_event_frames)))
        fig.add_vline(x=median_mer, line_width=3, line_dash="dash", line_color="red")
        fig.add_annotation(
            x=median_mer,
            y=1.06,
            xref="x",
            yref="paper",
            text="MER",
            showarrow=False,
            font=dict(color="red", size=13, family="Arial"),
            align="center"
        )

    add_event_iqr_band(fig, [0] * max(len(take_ids), 1), "blue", show_energy_fp_iqr_band)
    fig.add_vline(x=0, line_width=3, line_dash="dash", line_color="blue")
    fig.add_annotation(
        x=0,
        y=1.06,
        xref="x",
        yref="paper",
        text="BR",
        showarrow=False,
        font=dict(color="blue", size=13, family="Arial"),
        align="center"
    )

    fig.update_layout(
        xaxis_title="Time Relative to Ball Release (ms)",
        yaxis_title="Energy Flow / Segment Power",
        xaxis_range=[energy_window_start_ms, energy_window_end_ms],
        height=600,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.30,
            xanchor="center",
            x=0.5,
            groupclick="togglegroup"
        ),
        hoverlabel=dict(
            namelength=-1,
            font_size=13
        )
    )

    st.plotly_chart(fig, use_container_width=True, key="energy_plot_main_tab")

    defined_energy_metrics = [
        metric for metric in energy_metrics if metric in energy_definitions
    ]
    if defined_energy_metrics:
        st.markdown("### Energy Flow Definitions")
        for metric in defined_energy_metrics:
            metric_info = energy_definitions.get(metric, {})
            st.markdown(
                (
                    f"<div style='font-size:1.15rem; line-height:1.6; margin:0.35rem 0 0.9rem 0;'>"
                    f"<strong>{metric}:</strong> {metric_info.get('definition', '')}"
                    f"</div>"
                ),
                unsafe_allow_html=True,
            )






# Connect to DB for the existing biomechanics viewer tabs
conn = get_connection()
cur = conn.cursor()

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
    st.subheader("Compensation Analysis")
    render_group_selection_summary()

    energy_plot_options = st.session_state.get("tab1_energy_plot_options", ["Torso Power"])
    if not energy_plot_options:
        energy_plot_options = ["Torso Power"]

    take_rows = get_compensation_take_rows_from_sidebar()

    if not take_rows:
        st.warning("No takes found for the current sidebar selection.")

    # ---- Build session-scoped pitch numbers (reset per pitcher/session date) ----
    from collections import defaultdict

    pitch_number_map = {}
    takes_by_date = defaultdict(list)

    for tid, file_name, velo, take_date, throw_type, pitcher_name, handedness in take_rows:
        takes_by_date[(pitcher_name, take_date)].append((int(tid), file_name))

    for (_pitcher_name, take_date), ordered_rows in sorted(takes_by_date.items(), key=lambda item: (item[0][0], item[0][1])):
        ordered = sorted(ordered_rows, key=lambda x: x[1])
        for idx, (tid, _fname) in enumerate(ordered, start=1):
            pitch_number_map[int(tid)] = idx

    rows = []
    for tid, file_name, velo, take_date, throw_type, pitcher_name, handedness in take_rows:
        tid = int(tid)
        pitch_number = pitch_number_map.get(tid, 1)
        label = (
            f"{pitcher_name} | {take_date.strftime('%Y-%m-%d')} | {throw_type} | "
            f"Pitch {pitch_number} ({velo:.1f} mph)"
        )

        handedness = handedness if handedness in ("R", "L") else "R"
        rear_knee = "RT_KNEE" if handedness == "R" else "LT_KNEE"
        torso_segment = "RTA_DIST_R" if handedness == "R" else "RTA_DIST_L"
        arm_segment = "RAR" if handedness == "R" else "LAR"
        shoulder_stp_segment = "RTA_RAR" if handedness == "R" else "RTA_LAR"

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
            """, (tid, shoulder_segment, int(mer_frame)))
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

        # Peak Arm Energy (MER-windowed max: +/-30 frames around MER)
        cur.execute("""
            SELECT frame, x_data FROM time_series_data ts
            JOIN segments s ON ts.segment_id = s.segment_id
            JOIN categories c ON ts.category_id = c.category_id
            WHERE ts.take_id = %s AND c.category_name = 'SEGMENT_ENERGIES' AND s.segment_name = %s
            ORDER BY frame
        """, (tid, arm_segment))
        df_arm = pd.DataFrame(cur.fetchall(), columns=["frame", "x_data"])

        if not df_arm.empty and not np.isnan(max_knee_frame):
            mer_frame_local = int(peak_shoulder_frame) if peak_shoulder_frame is not None and not (isinstance(peak_shoulder_frame, float) and np.isnan(peak_shoulder_frame)) else None
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
            "Pitcher": pitcher_name,
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
    with tab1:
        st.warning("No valid data found for the current sidebar selection.")

if rows:
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
    pitchers = pitcher_names
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
        "posterior_cuff",
    ]
    selected_movement = st.selectbox(
        "Movement",
        options=movement_options,
        format_func=format_biodex_movement_label,
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
        disabled=selected_athlete is None or selected_protocol_type == "reactive_eccentric",
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
        biodex_plot_label_parts.append(format_biodex_movement_label(selected_movement))
    if selected_protocol_type != "reactive_eccentric" and selected_speed_deg_per_sec:
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
        biodex_upload_files_state = st.session_state.get("biodex_test_upload_files") or []
        biodex_upload_signature = tuple(
            (getattr(item, "name", ""), getattr(item, "size", 0))
            for item in biodex_upload_files_state
        )
        detected_biodex_start_datetime = None
        if biodex_upload_signature and st.session_state.get("biodex_test_upload_signature") != biodex_upload_signature:
            try:
                detected_df, _detected_numeric_columns = prepare_biodex_dataframe(biodex_upload_files_state[0])
                if not detected_df.empty:
                    detected_biodex_start_datetime = pd.to_datetime(detected_df["Time"].iloc[0], errors="coerce")
                    if pd.notna(detected_biodex_start_datetime):
                        detected_biodex_start_datetime = detected_biodex_start_datetime.to_pydatetime()
                        st.session_state["biodex_test_upload_date"] = detected_biodex_start_datetime.date()
                        st.session_state["biodex_test_upload_time"] = detected_biodex_start_datetime.time().replace(microsecond=0)
                        st.session_state["biodex_test_detected_datetime"] = detected_biodex_start_datetime.isoformat()
                        st.session_state["biodex_test_use_file_datetime"] = True
            except Exception:
                st.session_state.pop("biodex_test_detected_datetime", None)
            finally:
                st.session_state["biodex_test_upload_signature"] = biodex_upload_signature
        elif st.session_state.get("biodex_test_detected_datetime"):
            detected_biodex_start_datetime = pd.to_datetime(
                st.session_state.get("biodex_test_detected_datetime"),
                errors="coerce",
            )
            if pd.notna(detected_biodex_start_datetime):
                detected_biodex_start_datetime = detected_biodex_start_datetime.to_pydatetime()

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
                options=["d2_shoulder_pattern", "shoulder_er_ir", "posterior_cuff"],
                format_func=format_biodex_movement_label,
                key="biodex_test_upload_movement",
                disabled=selected_biodex_test_athlete_id is None,
            )
            selected_biodex_test_throwing_context = st.selectbox(
                "Throwing Context",
                options=get_biodex_throwing_context_options(),
                format_func=format_biodex_throwing_context_label,
                key="biodex_test_upload_throwing_context",
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
                disabled=(
                    selected_biodex_test_athlete_id is None
                    or selected_biodex_test_protocol == "reactive_eccentric"
                ),
            )
            selected_biodex_test_date = st.date_input(
                "Test Date",
                key="biodex_test_upload_date",
                disabled=selected_biodex_test_athlete_id is None,
            )
            selected_biodex_test_time = st.time_input(
                "Test Time",
                key="biodex_test_upload_time",
                disabled=selected_biodex_test_athlete_id is None,
            )
            use_biodex_file_datetime = st.checkbox(
                "Use file start timestamp when available",
                key="biodex_test_use_file_datetime",
                disabled=selected_biodex_test_athlete_id is None,
            )

        if detected_biodex_start_datetime is not None:
            st.caption(
                "Detected Biodex file start timestamp: "
                f"`{detected_biodex_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}`"
            )
        if not table_has_column(cur, "biodex_tests", "throwing_context"):
            st.info(
                "The `Throwing Context` dropdown is live in the UI. "
                "To persist it in `biodex_tests`, add a `throwing_context` column in the database."
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
                    file_start_datetime = pd.to_datetime(biodex_df["Time"].iloc[0], errors="coerce")
                    if pd.notna(file_start_datetime):
                        file_start_datetime = file_start_datetime.to_pydatetime()
                    else:
                        file_start_datetime = None

                    manual_test_datetime = datetime.combine(
                        selected_biodex_test_date,
                        selected_biodex_test_time,
                    )
                    stored_test_datetime = (
                        file_start_datetime
                        if use_biodex_file_datetime and file_start_datetime is not None
                        else manual_test_datetime
                    )
                    effective_biodex_test_speed = get_biodex_effective_speed(
                        selected_biodex_test_protocol,
                        selected_biodex_test_speed,
                    )
                    selected_athlete_name = athlete_options_test[int(selected_biodex_test_athlete_id)]["athlete_name"]
                    test_name_parts = [
                        selected_athlete_name,
                        selected_biodex_test_protocol.replace("_", " ").title(),
                        format_biodex_movement_label(selected_biodex_test_movement),
                        format_biodex_throwing_context_label(selected_biodex_test_throwing_context),
                    ]
                    if effective_biodex_test_speed is not None:
                        test_name_parts.append(f"{int(effective_biodex_test_speed)} deg/s")
                    test_name = " | ".join(test_name_parts)
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
                        speed_deg_per_sec=effective_biodex_test_speed,
                        test_date=stored_test_datetime,
                        source_file_name=uploaded_file.name,
                        notes=entered_biodex_test_notes,
                        throwing_context=selected_biodex_test_throwing_context,
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
                    "movement": selected_biodex_test_movement,
                    "protocol_type": selected_biodex_test_protocol,
                    "test_date": stored_test_datetime,
                    "throwing_context": selected_biodex_test_throwing_context,
                    "manual_rom_end": None,
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

        with st.expander("Restore Stored Test"):
            st.caption("Load a previously saved raw Biodex test back into this preview workspace without re-uploading the file.")
            restorable_tests_df = fetch_biodex_tests_for_restore(cur)
            if restorable_tests_df.empty:
                st.info("No stored Biodex tests are available to restore yet.")
            else:
                restore_options = restorable_tests_df["biodex_test_id"].tolist()
                restore_labels = {}
                for _, row in restorable_tests_df.iterrows():
                    test_date_label = (
                        pd.to_datetime(row["test_date"]).strftime("%Y-%m-%d %H:%M")
                        if pd.notna(row["test_date"])
                        else "No datetime"
                    )
                    processing_label = (
                        "Unprocessed"
                        if int(row["processing_run_count"]) == 0
                        else f"{int(row['processing_run_count'])} saved run(s)"
                    )
                    restore_labels[int(row["biodex_test_id"])] = (
                        f"{row['athlete_name']} | {test_date_label} | "
                        f"{format_biodex_movement_label(row['movement'])} | "
                        f"{row['source_file_name']} | {processing_label}"
                    )

                selected_restore_biodex_test_ids = st.multiselect(
                    "Stored Biodex test(s)",
                    options=restore_options,
                    format_func=lambda test_id: restore_labels.get(test_id, f"Test {test_id}"),
                    default=restore_options,
                    key="biodex_test_restore_selection",
                )
                restore_action_col1, restore_action_col2 = st.columns(2)
                with restore_action_col1:
                    load_selected_restore_tests = st.button(
                        "Load Selected Tests into Preview",
                        key="biodex_test_restore_button",
                        use_container_width=True,
                        disabled=not selected_restore_biodex_test_ids,
                    )
                with restore_action_col2:
                    load_all_restore_tests = st.button(
                        "Load All Stored Tests",
                        key="biodex_test_restore_all_button",
                        use_container_width=True,
                        disabled=not restore_options,
                    )

                if load_selected_restore_tests or load_all_restore_tests:
                    target_restore_ids = (
                        [int(test_id) for test_id in restore_options]
                        if load_all_restore_tests
                        else [int(test_id) for test_id in selected_restore_biodex_test_ids]
                    )
                    restored_preview_items = []
                    restore_errors = []
                    for biodex_test_id in target_restore_ids:
                        try:
                            restored_preview_items.append(
                                build_biodex_preview_item_from_db(
                                    cur,
                                    biodex_test_id=int(biodex_test_id),
                                )
                            )
                        except Exception as exc:
                            restore_errors.append(f"Test {int(biodex_test_id)}: {exc}")

                    current_previews = st.session_state.get("biodex_test_uploaded_previews", [])
                    existing_by_id = {
                        int(item["biodex_test_id"]): item
                        for item in current_previews
                    }
                    for restored_item in restored_preview_items:
                        existing_by_id[int(restored_item["biodex_test_id"])] = restored_item

                    merged_previews = list(existing_by_id.values())
                    st.session_state["biodex_test_uploaded_previews"] = merged_previews
                    if restored_preview_items:
                        st.session_state["biodex_test_preview_file"] = restored_preview_items[0]["name"]
                        st.success(
                            f"Loaded {len(restored_preview_items)} stored Biodex test(s) back into Upload & Process."
                        )
                    if restore_errors:
                        st.error("Some tests could not be restored:\n\n" + "\n".join(restore_errors))

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
            preview_movement = preview_item.get("movement", selected_biodex_test_movement)
            preview_protocol_type = preview_item.get("protocol_type", selected_biodex_test_protocol)

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

                if "Position_Deg" in preview_item["numeric_columns"]:
                    position_preview_fig = go.Figure()
                    position_preview_fig.add_trace(go.Scatter(
                        x=preview_df["Elapsed Seconds"],
                        y=preview_df["Position_Deg"],
                        mode="lines",
                        name="Position_Deg",
                    ))
                    position_preview_title = "Stored Raw Position Preview"
                    if preview_item.get("movement", selected_biodex_test_movement) == "d2_shoulder_pattern":
                        position_preview_title = "D2 Shoulder Pattern Position"
                    position_preview_fig.update_layout(
                        title=position_preview_title,
                        xaxis_title="Elapsed Time (s)",
                        yaxis_title="Position_Deg",
                        height=500,
                    )
                    st.plotly_chart(position_preview_fig, use_container_width=True)

                if (
                    preview_movement == "posterior_cuff"
                    and preview_protocol_type == "reactive_eccentric"
                ):
                    st.markdown("### Single-Rep File Alignment Preview")
                    st.caption(
                        "For Posterior Cuff reactive eccentric work, each uploaded file is treated as one rep. "
                        "Instead of detecting reps within a file, this view aligns whole files to one another."
                    )
                    posterior_rep_candidates = [
                        item for item in uploaded_previews
                        if item.get("movement") == "posterior_cuff"
                        and item.get("protocol_type") == "reactive_eccentric"
                    ]
                    posterior_rep_candidate_by_id = {
                        int(item["biodex_test_id"]): item
                        for item in posterior_rep_candidates
                    }
                    restorable_posterior_tests_df = fetch_biodex_tests_for_restore(cur)
                    restorable_posterior_tests_df = restorable_posterior_tests_df.loc[
                        (restorable_posterior_tests_df["movement"] == "posterior_cuff")
                        & (restorable_posterior_tests_df["protocol_type"] == "reactive_eccentric")
                    ].copy()

                    posterior_rep_options = restorable_posterior_tests_df["biodex_test_id"].astype(int).tolist()
                    default_posterior_rep_selection = list(posterior_rep_candidate_by_id.keys())
                    posterior_rep_labels = {}
                    for _, row in restorable_posterior_tests_df.iterrows():
                        test_date_label = (
                            pd.to_datetime(row["test_date"]).strftime("%Y-%m-%d %H:%M")
                            if pd.notna(row["test_date"])
                            else "No datetime"
                        )
                        posterior_rep_labels[int(row["biodex_test_id"])] = (
                            f"{row['athlete_name']} | {test_date_label} | "
                            f"{row['source_file_name']}"
                        )

                    single_rep_control_col, single_rep_plot_col = st.columns([0.35, 1.0], vertical_alignment="top")
                    with single_rep_control_col:
                        selected_posterior_rep_files = st.multiselect(
                            "Files to align",
                            options=posterior_rep_options,
                            format_func=lambda test_id: posterior_rep_labels.get(int(test_id), f"Test {int(test_id)}"),
                            default=default_posterior_rep_selection,
                            key="posterior_cuff_single_rep_files",
                        )
                        posterior_x_axis_mode = st.selectbox(
                            "Alignment view",
                            options=["filtered_position_window_normalized", "position_window_normalized", "zero_to_position_136_ascent_normalized", "zero_to_common_smoothed_rom_end_normalized", "zero_to_peak_normalized", "normalized_duration", "raw_time"],
                            format_func=lambda value: {
                                "filtered_position_window_normalized": "Filtered ROM Start -> End Normalized",
                                "position_window_normalized": "ROM Start -> End Normalized",
                                "zero_to_position_136_ascent_normalized": "0 Torque Rise -> 136° On Ascent",
                                "zero_to_common_smoothed_rom_end_normalized": "0 Torque Rise -> Stabilized Peak ROM End",
                                "zero_to_peak_normalized": "0 Torque -> Peak Positive Normalized",
                                "normalized_duration": "Normalized Rep Duration",
                                "raw_time": "Raw Time Around Anchor",
                            }.get(value, value.replace("_", " ").title()),
                            key="posterior_cuff_single_rep_x_axis_mode",
                        )
                        if posterior_x_axis_mode in {"position_window_normalized", "filtered_position_window_normalized"}:
                            posterior_anchor_mode = "zero_torque_rise"
                            st.caption(
                                "This posterior cuff workflow defines the torque window from ROM start to ROM end "
                                "using the position signal rather than a torque-event endpoint."
                            )
                        else:
                            posterior_anchor_mode = st.selectbox(
                                "Alignment anchor",
                                options=[
                                    "zero_torque_rise",
                                    "positive_rise_onset",
                                    "negative_torque_onset",
                                    "peak_negative_torque",
                                    "peak_positive_torque",
                                ],
                                format_func=lambda value: {
                                    "zero_torque_rise": "0 Torque Rise",
                                    "positive_rise_onset": "Positive Rise Onset",
                                    "negative_torque_onset": "Negative Torque Onset",
                                    "peak_negative_torque": "Peak Negative Torque",
                                    "peak_positive_torque": "Peak Positive Torque",
                                }.get(value, value.replace("_", " ").title()),
                                key="posterior_cuff_single_rep_anchor",
                            )
                        posterior_n_points = st.number_input(
                            "Aligned points per file",
                            min_value=51,
                            max_value=500,
                            value=201,
                            step=10,
                            key="posterior_cuff_single_rep_points",
                        )
                        posterior_common_rom_end_tolerance_deg = 2.5
                        posterior_common_rom_end_hold_time_seconds = 0.08
                        if posterior_x_axis_mode == "zero_to_common_smoothed_rom_end_normalized":
                            posterior_common_rom_end_tolerance_deg = st.slider(
                                "Shared ROM band tolerance (deg)",
                                min_value=1.0,
                                max_value=6.0,
                                value=2.5,
                                step=0.5,
                                key="posterior_cuff_common_rom_end_tolerance_deg",
                            )
                            posterior_common_rom_end_hold_time_seconds = st.slider(
                                "Shared ROM end hold (s)",
                                min_value=0.00,
                                max_value=0.30,
                                value=0.08,
                                step=0.01,
                                key="posterior_cuff_common_rom_end_hold_time_seconds",
                            )
                        posterior_filtered_cutoff_hz = st.slider(
                            "Filtered position smoothing cutoff (Hz)",
                            min_value=1.0,
                            max_value=8.0,
                            value=4.0,
                            step=0.5,
                            key="posterior_cuff_filtered_position_cutoff_hz",
                        )
                        st.caption(
                            "Lower cutoff = smoother ROM line in the filtered position plot only."
                        )

                    selected_posterior_rep_items = []
                    for biodex_test_id in selected_posterior_rep_files:
                        biodex_test_id = int(biodex_test_id)
                        if biodex_test_id in posterior_rep_candidate_by_id:
                            selected_posterior_rep_items.append(posterior_rep_candidate_by_id[biodex_test_id])
                            continue
                        try:
                            restored_item = build_biodex_preview_item_from_db(cur, biodex_test_id=biodex_test_id)
                        except Exception:
                            continue
                        posterior_rep_candidate_by_id[biodex_test_id] = restored_item
                        selected_posterior_rep_items.append(restored_item)

                    posterior_reps_long_df, posterior_mean_df, posterior_alignment_metadata, posterior_rom_reps_long_df, posterior_rom_mean_df = extract_single_rep_file_aligned_curves(
                        selected_posterior_rep_items,
                        anchor_mode=posterior_anchor_mode,
                        value_col="Torque_Nm",
                        time_col="Elapsed Seconds",
                        n_points=int(posterior_n_points),
                        x_axis_mode=posterior_x_axis_mode,
                        common_rom_end_tolerance_deg=float(posterior_common_rom_end_tolerance_deg),
                        common_rom_end_hold_time_seconds=float(posterior_common_rom_end_hold_time_seconds),
                        rom_display_lowpass_cutoff_hz=(
                            1.0
                            if posterior_x_axis_mode == "raw_time" and posterior_anchor_mode == "zero_torque_rise"
                            else None
                        ),
                    )
                    zero_rise_rom_reps_long_df = pd.DataFrame()
                    zero_rise_rom_mean_df = pd.DataFrame()
                    fifty_rom_reps_long_df = pd.DataFrame()
                    fifty_rom_mean_df = pd.DataFrame()
                    ninety_rom_reps_long_df = pd.DataFrame()
                    ninety_rom_mean_df = pd.DataFrame()
                    ninety_five_rom_reps_long_df = pd.DataFrame()
                    ninety_five_rom_mean_df = pd.DataFrame()
                    five_to_ninety_eight_torque_reps_long_df = pd.DataFrame()
                    five_to_ninety_eight_torque_mean_df = pd.DataFrame()
                    five_to_ninety_eight_position_reps_long_df = pd.DataFrame()
                    five_to_ninety_eight_position_mean_df = pd.DataFrame()
                    zero_to_ninety_eight_torque_reps_long_df = pd.DataFrame()
                    zero_to_ninety_eight_torque_mean_df = pd.DataFrame()
                    zero_to_ninety_eight_position_reps_long_df = pd.DataFrame()
                    zero_to_ninety_eight_position_mean_df = pd.DataFrame()
                    if selected_posterior_rep_items:
                        (
                            _zero_rise_torque_reps_long_df,
                            _zero_rise_torque_mean_df,
                            _zero_rise_alignment_metadata,
                            zero_rise_rom_reps_long_df,
                            zero_rise_rom_mean_df,
                        ) = extract_single_rep_file_aligned_curves(
                            selected_posterior_rep_items,
                            anchor_mode="zero_torque_rise",
                            value_col="Torque_Nm",
                            time_col="Elapsed Seconds",
                            n_points=int(posterior_n_points),
                            x_axis_mode="raw_time",
                        )
                        (
                            fifty_rom_reps_long_df,
                            fifty_rom_mean_df,
                            _fifty_rom_alignment_metadata,
                        ) = extract_position_fraction_aligned_curves(
                            selected_posterior_rep_items,
                            rom_fraction=0.50,
                            time_col="Elapsed Seconds",
                            position_col="Position_Deg",
                            n_points=int(posterior_n_points),
                            lowpass_cutoff_hz=1.0,
                        )
                        (
                            ninety_rom_reps_long_df,
                            ninety_rom_mean_df,
                            _ninety_rom_alignment_metadata,
                        ) = extract_position_fraction_aligned_curves(
                            selected_posterior_rep_items,
                            rom_fraction=0.90,
                            time_col="Elapsed Seconds",
                            position_col="Position_Deg",
                            n_points=int(posterior_n_points),
                            lowpass_cutoff_hz=1.0,
                        )
                        (
                            ninety_five_rom_reps_long_df,
                            ninety_five_rom_mean_df,
                            _ninety_five_rom_alignment_metadata,
                        ) = extract_position_fraction_aligned_curves(
                            selected_posterior_rep_items,
                            rom_fraction=0.95,
                            time_col="Elapsed Seconds",
                            position_col="Position_Deg",
                            n_points=int(posterior_n_points),
                            lowpass_cutoff_hz=1.0,
                        )
                        (
                            five_to_ninety_eight_torque_reps_long_df,
                            five_to_ninety_eight_torque_mean_df,
                            five_to_ninety_eight_position_reps_long_df,
                            five_to_ninety_eight_position_mean_df,
                            _five_to_ninety_eight_alignment_metadata,
                        ) = extract_torque_fraction_window_curves(
                            selected_posterior_rep_items,
                            start_fraction=0.05,
                            end_fraction=0.98,
                            value_col="Torque_Nm",
                            time_col="Elapsed Seconds",
                            position_col="Position_Deg",
                            n_points=int(posterior_n_points),
                        )
                        (
                            zero_to_ninety_eight_torque_reps_long_df,
                            zero_to_ninety_eight_torque_mean_df,
                            zero_to_ninety_eight_position_reps_long_df,
                            zero_to_ninety_eight_position_mean_df,
                            _zero_to_ninety_eight_alignment_metadata,
                        ) = extract_torque_fraction_window_curves(
                            selected_posterior_rep_items,
                            start_fraction=0.05,
                            end_fraction=0.98,
                            start_mode="zero_torque_rise",
                            value_col="Torque_Nm",
                            time_col="Elapsed Seconds",
                            position_col="Position_Deg",
                            n_points=int(posterior_n_points),
                        )

                    with single_rep_plot_col:
                        if posterior_reps_long_df.empty or posterior_mean_df.empty:
                            st.warning("Select at least one valid single-rep file to build the aligned overlay.")
                        else:
                            x_axis_title = str(posterior_mean_df.attrs.get("x_axis_title", "Aligned Time (s)"))
                            anchor_x = float(posterior_mean_df.attrs.get("anchor_x", 0.0))
                            posterior_align_fig = go.Figure()
                            for rep_number, rep_df in posterior_reps_long_df.groupby("rep_number"):
                                file_name = rep_df["file_name"].iloc[0]
                                posterior_align_fig.add_trace(go.Scatter(
                                    x=rep_df["alignment_x"],
                                    y=rep_df["torque_nm"],
                                    mode="lines",
                                    line=dict(width=1.5),
                                    opacity=0.45,
                                    name=file_name,
                                ))
                            posterior_align_fig.add_trace(go.Scatter(
                                x=posterior_mean_df["alignment_x"],
                                y=posterior_mean_df["upper_band"],
                                mode="lines",
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo="skip",
                            ))
                            posterior_align_fig.add_trace(go.Scatter(
                                x=posterior_mean_df["alignment_x"],
                                y=posterior_mean_df["lower_band"],
                                mode="lines",
                                line=dict(width=0),
                                fill="tonexty",
                                name="±1 SD",
                            ))
                            posterior_align_fig.add_trace(go.Scatter(
                                x=posterior_mean_df["alignment_x"],
                                y=posterior_mean_df["mean_torque_nm"],
                                mode="lines",
                                line=dict(width=4),
                                name="Mean Torque",
                            ))
                            posterior_align_fig.add_vline(
                                x=anchor_x,
                                line_width=2,
                                line_dash="dot",
                                line_color="rgba(255,255,255,0.45)",
                            )
                            posterior_align_fig.add_annotation(
                                x=anchor_x,
                                y=1.03,
                                xref="x",
                                yref="paper",
                                text=str(posterior_mean_df.attrs.get("anchor_label", "Alignment Anchor")),
                                showarrow=False,
                                font=dict(size=11),
                            )
                            secondary_anchor_x = posterior_mean_df.attrs.get("secondary_anchor_x")
                            secondary_anchor_label = posterior_mean_df.attrs.get("secondary_anchor_label")
                            if secondary_anchor_x is not None and secondary_anchor_label:
                                posterior_align_fig.add_vline(
                                    x=float(secondary_anchor_x),
                                    line_width=2,
                                    line_dash="dot",
                                    line_color="rgba(255,255,255,0.30)",
                                )
                                posterior_align_fig.add_annotation(
                                    x=float(secondary_anchor_x),
                                    y=1.03,
                                    xref="x",
                                    yref="paper",
                                    text=str(secondary_anchor_label),
                                    showarrow=False,
                                    font=dict(size=11),
                                )
                            tertiary_anchor_x = posterior_mean_df.attrs.get("tertiary_anchor_x")
                            tertiary_anchor_label = posterior_mean_df.attrs.get("tertiary_anchor_label")
                            if tertiary_anchor_x is not None and tertiary_anchor_label:
                                posterior_align_fig.add_vline(
                                    x=float(tertiary_anchor_x),
                                    line_width=2,
                                    line_dash="dot",
                                    line_color="rgba(255,255,255,0.22)",
                                )
                                posterior_align_fig.add_annotation(
                                    x=float(tertiary_anchor_x),
                                    y=1.03,
                                    xref="x",
                                    yref="paper",
                                    text=str(tertiary_anchor_label),
                                    showarrow=False,
                                    font=dict(size=11),
                                )
                            posterior_align_fig.update_layout(
                                title="Posterior Cuff Reactive Eccentric: Across-File Alignment",
                                xaxis_title=x_axis_title,
                                yaxis_title="Torque_Nm",
                                height=500,
                            )
                            st.plotly_chart(
                                posterior_align_fig,
                                use_container_width=True,
                                key=f"posterior_cuff_single_rep_plot_{posterior_anchor_mode}_{posterior_x_axis_mode}_{posterior_n_points}_{len(selected_posterior_rep_items)}",
                            )
                            if not posterior_rom_reps_long_df.empty and not posterior_rom_mean_df.empty:
                                posterior_rom_fig = go.Figure()
                                for rep_number, rep_df in posterior_rom_reps_long_df.groupby("rep_number"):
                                    file_name = rep_df["file_name"].iloc[0]
                                    posterior_rom_fig.add_trace(go.Scatter(
                                        x=rep_df["alignment_x"],
                                        y=rep_df["position_deg"],
                                        mode="lines",
                                        line=dict(width=1.5),
                                        opacity=0.45,
                                        name=file_name,
                                    ))
                                posterior_rom_fig.add_trace(go.Scatter(
                                    x=posterior_rom_mean_df["alignment_x"],
                                    y=posterior_rom_mean_df["upper_band"],
                                    mode="lines",
                                    line=dict(width=0),
                                    showlegend=False,
                                    hoverinfo="skip",
                                ))
                                posterior_rom_fig.add_trace(go.Scatter(
                                    x=posterior_rom_mean_df["alignment_x"],
                                    y=posterior_rom_mean_df["lower_band"],
                                    mode="lines",
                                    line=dict(width=0),
                                    fill="tonexty",
                                    name="±1 SD",
                                ))
                                posterior_rom_fig.add_trace(go.Scatter(
                                    x=posterior_rom_mean_df["alignment_x"],
                                    y=posterior_rom_mean_df["mean_position_deg"],
                                    mode="lines",
                                    line=dict(width=4),
                                    name="Mean Position",
                                ))
                                posterior_rom_fig.add_vline(
                                    x=anchor_x,
                                    line_width=2,
                                    line_dash="dot",
                                    line_color="rgba(255,255,255,0.45)",
                                )
                                posterior_rom_fig.add_annotation(
                                    x=anchor_x,
                                    y=1.03,
                                    xref="x",
                                    yref="paper",
                                    text=str(posterior_rom_mean_df.attrs.get("anchor_label", "Alignment Anchor")),
                                    showarrow=False,
                                    font=dict(size=11),
                                )
                                if secondary_anchor_x is not None and secondary_anchor_label:
                                    posterior_rom_fig.add_vline(
                                        x=float(secondary_anchor_x),
                                        line_width=2,
                                        line_dash="dot",
                                        line_color="rgba(255,255,255,0.30)",
                                    )
                                    posterior_rom_fig.add_annotation(
                                        x=float(secondary_anchor_x),
                                        y=1.03,
                                        xref="x",
                                        yref="paper",
                                        text=str(secondary_anchor_label),
                                        showarrow=False,
                                        font=dict(size=11),
                                    )
                                if tertiary_anchor_x is not None and tertiary_anchor_label:
                                    posterior_rom_fig.add_vline(
                                        x=float(tertiary_anchor_x),
                                        line_width=2,
                                        line_dash="dot",
                                        line_color="rgba(255,255,255,0.22)",
                                    )
                                    posterior_rom_fig.add_annotation(
                                        x=float(tertiary_anchor_x),
                                        y=1.03,
                                        xref="x",
                                        yref="paper",
                                        text=str(tertiary_anchor_label),
                                        showarrow=False,
                                        font=dict(size=11),
                                    )
                                quaternary_anchor_x = posterior_rom_mean_df.attrs.get("quaternary_anchor_x")
                                quaternary_anchor_label = posterior_rom_mean_df.attrs.get("quaternary_anchor_label")
                                if quaternary_anchor_x is not None and quaternary_anchor_label:
                                    posterior_rom_fig.add_vline(
                                        x=float(quaternary_anchor_x),
                                        line_width=2,
                                        line_dash="dot",
                                        line_color="rgba(144,238,144,0.55)",
                                    )
                                    posterior_rom_fig.add_annotation(
                                        x=float(quaternary_anchor_x),
                                        y=1.03,
                                        xref="x",
                                        yref="paper",
                                        text=str(quaternary_anchor_label),
                                        showarrow=False,
                                        font=dict(size=11, color="rgba(144,238,144,0.95)"),
                                    )
                                posterior_rom_fig.update_layout(
                                    title=(
                                        "Posterior Cuff Reactive Eccentric: Across-File Range of Motion (1 Hz Filtered)"
                                        if posterior_x_axis_mode == "raw_time" and posterior_anchor_mode == "zero_torque_rise"
                                        else "Posterior Cuff Reactive Eccentric: Across-File Range of Motion"
                                    ),
                                    xaxis_title=x_axis_title,
                                    yaxis_title="Position_Deg",
                                    height=500,
                                )
                                st.plotly_chart(
                                    posterior_rom_fig,
                                    use_container_width=True,
                                    key=f"posterior_cuff_single_rep_rom_plot_{posterior_anchor_mode}_{posterior_x_axis_mode}_{posterior_n_points}_{len(selected_posterior_rep_items)}",
                                )
                                if not zero_rise_rom_reps_long_df.empty and not zero_rise_rom_mean_df.empty:
                                    zero_rise_rom_fig = go.Figure()
                                    for rep_number, rep_df in zero_rise_rom_reps_long_df.groupby("rep_number"):
                                        file_name = rep_df["file_name"].iloc[0]
                                        zero_rise_rom_fig.add_trace(go.Scatter(
                                            x=rep_df["alignment_x"],
                                            y=rep_df["position_deg"],
                                            mode="lines",
                                            line=dict(width=1.5),
                                            opacity=0.45,
                                            name=file_name,
                                        ))
                                    zero_rise_rom_fig.add_trace(go.Scatter(
                                        x=zero_rise_rom_mean_df["alignment_x"],
                                        y=zero_rise_rom_mean_df["upper_band"],
                                        mode="lines",
                                        line=dict(width=0),
                                        showlegend=False,
                                        hoverinfo="skip",
                                    ))
                                    zero_rise_rom_fig.add_trace(go.Scatter(
                                        x=zero_rise_rom_mean_df["alignment_x"],
                                        y=zero_rise_rom_mean_df["lower_band"],
                                        mode="lines",
                                        line=dict(width=0),
                                        fill="tonexty",
                                        name="±1 SD",
                                    ))
                                    zero_rise_rom_fig.add_trace(go.Scatter(
                                        x=zero_rise_rom_mean_df["alignment_x"],
                                        y=zero_rise_rom_mean_df["mean_position_deg"],
                                        mode="lines",
                                        line=dict(width=4),
                                        name="Mean Position",
                                    ))
                                    zero_rise_anchor_x = float(zero_rise_rom_mean_df.attrs.get("anchor_x", 0.0))
                                    zero_rise_rom_fig.add_vline(
                                        x=zero_rise_anchor_x,
                                        line_width=2,
                                        line_dash="dot",
                                        line_color="rgba(255,255,255,0.45)",
                                    )
                                    zero_rise_rom_fig.add_annotation(
                                        x=zero_rise_anchor_x,
                                        y=1.03,
                                        xref="x",
                                        yref="paper",
                                        text="0 Torque Rise",
                                        showarrow=False,
                                        font=dict(size=11),
                                    )
                                    zero_rise_rom_fig.update_layout(
                                        title="Posterior Cuff ROM When Torque Is Aligned at 0 Torque Rise",
                                        xaxis_title=str(zero_rise_rom_mean_df.attrs.get("x_axis_title", "Aligned Time (s)")),
                                        yaxis_title="Position_Deg",
                                        height=500,
                                    )
                                    st.plotly_chart(
                                        zero_rise_rom_fig,
                                        use_container_width=True,
                                        key=f"posterior_cuff_zero_rise_rom_plot_{posterior_n_points}_{len(selected_posterior_rep_items)}",
                                    )
                                if not fifty_rom_reps_long_df.empty and not fifty_rom_mean_df.empty:
                                    fifty_rom_fig = go.Figure()
                                    for rep_number, rep_df in fifty_rom_reps_long_df.groupby("rep_number"):
                                        file_name = rep_df["file_name"].iloc[0]
                                        fifty_rom_fig.add_trace(go.Scatter(
                                            x=rep_df["alignment_x"],
                                            y=rep_df["position_deg"],
                                            mode="lines",
                                            line=dict(width=1.5),
                                            opacity=0.45,
                                            name=file_name,
                                        ))
                                    fifty_rom_fig.add_trace(go.Scatter(
                                        x=fifty_rom_mean_df["alignment_x"],
                                        y=fifty_rom_mean_df["upper_band"],
                                        mode="lines",
                                        line=dict(width=0),
                                        showlegend=False,
                                        hoverinfo="skip",
                                    ))
                                    fifty_rom_fig.add_trace(go.Scatter(
                                        x=fifty_rom_mean_df["alignment_x"],
                                        y=fifty_rom_mean_df["lower_band"],
                                        mode="lines",
                                        line=dict(width=0),
                                        fill="tonexty",
                                        name="±1 SD",
                                    ))
                                    fifty_rom_fig.add_trace(go.Scatter(
                                        x=fifty_rom_mean_df["alignment_x"],
                                        y=fifty_rom_mean_df["mean_position_deg"],
                                        mode="lines",
                                        line=dict(width=4),
                                        name="Mean Position",
                                    ))
                                    fifty_rom_anchor_x = float(fifty_rom_mean_df.attrs.get("anchor_x", 0.0))
                                    fifty_rom_fig.add_vline(
                                        x=fifty_rom_anchor_x,
                                        line_width=2,
                                        line_dash="dot",
                                        line_color="rgba(255,255,255,0.45)",
                                    )
                                    fifty_rom_fig.add_annotation(
                                        x=fifty_rom_anchor_x,
                                        y=1.03,
                                        xref="x",
                                        yref="paper",
                                        text=str(fifty_rom_mean_df.attrs.get("anchor_label", "50% ROM")),
                                        showarrow=False,
                                        font=dict(size=11),
                                    )
                                    fifty_rom_fig.update_layout(
                                        title="Posterior Cuff ROM When Aligned at 50% ROM (1 Hz Filtered)",
                                        xaxis_title=str(fifty_rom_mean_df.attrs.get("x_axis_title", "Aligned Time (s)")),
                                        yaxis_title="Position_Deg",
                                        height=500,
                                    )
                                    st.plotly_chart(
                                        fifty_rom_fig,
                                        use_container_width=True,
                                        key=f"posterior_cuff_fifty_rom_plot_{posterior_n_points}_{len(selected_posterior_rep_items)}",
                                    )
                                if not ninety_rom_reps_long_df.empty and not ninety_rom_mean_df.empty:
                                    ninety_rom_fig = go.Figure()
                                    for rep_number, rep_df in ninety_rom_reps_long_df.groupby("rep_number"):
                                        file_name = rep_df["file_name"].iloc[0]
                                        ninety_rom_fig.add_trace(go.Scatter(
                                            x=rep_df["alignment_x"],
                                            y=rep_df["position_deg"],
                                            mode="lines",
                                            line=dict(width=1.5),
                                            opacity=0.45,
                                            name=file_name,
                                        ))
                                    ninety_rom_fig.add_trace(go.Scatter(
                                        x=ninety_rom_mean_df["alignment_x"],
                                        y=ninety_rom_mean_df["upper_band"],
                                        mode="lines",
                                        line=dict(width=0),
                                        showlegend=False,
                                        hoverinfo="skip",
                                    ))
                                    ninety_rom_fig.add_trace(go.Scatter(
                                        x=ninety_rom_mean_df["alignment_x"],
                                        y=ninety_rom_mean_df["lower_band"],
                                        mode="lines",
                                        line=dict(width=0),
                                        fill="tonexty",
                                        name="±1 SD",
                                    ))
                                    ninety_rom_fig.add_trace(go.Scatter(
                                        x=ninety_rom_mean_df["alignment_x"],
                                        y=ninety_rom_mean_df["mean_position_deg"],
                                        mode="lines",
                                        line=dict(width=4),
                                        name="Mean Position",
                                    ))
                                    ninety_rom_anchor_x = float(ninety_rom_mean_df.attrs.get("anchor_x", 0.0))
                                    ninety_rom_fig.add_vline(
                                        x=ninety_rom_anchor_x,
                                        line_width=2,
                                        line_dash="dot",
                                        line_color="rgba(255,255,255,0.45)",
                                    )
                                    ninety_rom_fig.add_annotation(
                                        x=ninety_rom_anchor_x,
                                        y=1.03,
                                        xref="x",
                                        yref="paper",
                                        text=str(ninety_rom_mean_df.attrs.get("anchor_label", "90% ROM")),
                                        showarrow=False,
                                        font=dict(size=11),
                                    )
                                    ninety_rom_fig.update_layout(
                                        title="Posterior Cuff ROM When Aligned at 90% ROM (1 Hz Filtered)",
                                        xaxis_title=str(ninety_rom_mean_df.attrs.get("x_axis_title", "Aligned Time (s)")),
                                        yaxis_title="Position_Deg",
                                        height=500,
                                    )
                                    st.plotly_chart(
                                        ninety_rom_fig,
                                        use_container_width=True,
                                        key=f"posterior_cuff_ninety_rom_plot_{posterior_n_points}_{len(selected_posterior_rep_items)}",
                                    )
                                if not ninety_five_rom_reps_long_df.empty and not ninety_five_rom_mean_df.empty:
                                    ninety_five_rom_fig = go.Figure()
                                    for rep_number, rep_df in ninety_five_rom_reps_long_df.groupby("rep_number"):
                                        file_name = rep_df["file_name"].iloc[0]
                                        ninety_five_rom_fig.add_trace(go.Scatter(
                                            x=rep_df["alignment_x"],
                                            y=rep_df["position_deg"],
                                            mode="lines",
                                            line=dict(width=1.5),
                                            opacity=0.45,
                                            name=file_name,
                                        ))
                                    ninety_five_rom_fig.add_trace(go.Scatter(
                                        x=ninety_five_rom_mean_df["alignment_x"],
                                        y=ninety_five_rom_mean_df["upper_band"],
                                        mode="lines",
                                        line=dict(width=0),
                                        showlegend=False,
                                        hoverinfo="skip",
                                    ))
                                    ninety_five_rom_fig.add_trace(go.Scatter(
                                        x=ninety_five_rom_mean_df["alignment_x"],
                                        y=ninety_five_rom_mean_df["lower_band"],
                                        mode="lines",
                                        line=dict(width=0),
                                        fill="tonexty",
                                        name="±1 SD",
                                    ))
                                    ninety_five_rom_fig.add_trace(go.Scatter(
                                        x=ninety_five_rom_mean_df["alignment_x"],
                                        y=ninety_five_rom_mean_df["mean_position_deg"],
                                        mode="lines",
                                        line=dict(width=4),
                                        name="Mean Position",
                                    ))
                                    ninety_five_rom_anchor_x = float(ninety_five_rom_mean_df.attrs.get("anchor_x", 0.0))
                                    ninety_five_rom_fig.add_vline(
                                        x=ninety_five_rom_anchor_x,
                                        line_width=2,
                                        line_dash="dot",
                                        line_color="rgba(255,255,255,0.45)",
                                    )
                                    ninety_five_rom_fig.add_annotation(
                                        x=ninety_five_rom_anchor_x,
                                        y=1.03,
                                        xref="x",
                                        yref="paper",
                                        text=str(ninety_five_rom_mean_df.attrs.get("anchor_label", "95% ROM")),
                                        showarrow=False,
                                        font=dict(size=11),
                                    )
                                    ninety_five_rom_fig.update_layout(
                                        title="Posterior Cuff ROM When Aligned at 95% ROM (1 Hz Filtered)",
                                        xaxis_title=str(ninety_five_rom_mean_df.attrs.get("x_axis_title", "Aligned Time (s)")),
                                        yaxis_title="Position_Deg",
                                        height=500,
                                    )
                                    st.plotly_chart(
                                        ninety_five_rom_fig,
                                        use_container_width=True,
                                        key=f"posterior_cuff_ninety_five_rom_plot_{posterior_n_points}_{len(selected_posterior_rep_items)}",
                                    )
                                if not five_to_ninety_eight_torque_reps_long_df.empty and not five_to_ninety_eight_torque_mean_df.empty:
                                    five_to_ninety_eight_torque_fig = go.Figure()
                                    for rep_number, rep_df in five_to_ninety_eight_torque_reps_long_df.groupby("rep_number"):
                                        file_name = rep_df["file_name"].iloc[0]
                                        five_to_ninety_eight_torque_fig.add_trace(go.Scatter(
                                            x=rep_df["alignment_x"],
                                            y=rep_df["torque_nm"],
                                            mode="lines",
                                            line=dict(width=1.5),
                                            opacity=0.45,
                                            name=file_name,
                                        ))
                                    five_to_ninety_eight_torque_fig.add_trace(go.Scatter(
                                        x=five_to_ninety_eight_torque_mean_df["alignment_x"],
                                        y=five_to_ninety_eight_torque_mean_df["upper_band"],
                                        mode="lines",
                                        line=dict(width=0),
                                        showlegend=False,
                                        hoverinfo="skip",
                                    ))
                                    five_to_ninety_eight_torque_fig.add_trace(go.Scatter(
                                        x=five_to_ninety_eight_torque_mean_df["alignment_x"],
                                        y=five_to_ninety_eight_torque_mean_df["lower_band"],
                                        mode="lines",
                                        line=dict(width=0),
                                        fill="tonexty",
                                        name="±1 SD",
                                    ))
                                    five_to_ninety_eight_torque_fig.add_trace(go.Scatter(
                                        x=five_to_ninety_eight_torque_mean_df["alignment_x"],
                                        y=five_to_ninety_eight_torque_mean_df["mean_torque_nm"],
                                        mode="lines",
                                        line=dict(width=4),
                                        name="Mean Torque",
                                    ))
                                    five_to_ninety_eight_torque_fig.add_vline(
                                        x=0.0,
                                        line_width=2,
                                        line_dash="dot",
                                        line_color="rgba(255,255,255,0.45)",
                                    )
                                    five_to_ninety_eight_torque_fig.add_annotation(
                                        x=0.0,
                                        y=1.03,
                                        xref="x",
                                        yref="paper",
                                        text="5% Torque",
                                        showarrow=False,
                                        font=dict(size=11),
                                    )
                                    five_to_ninety_eight_torque_fig.add_vline(
                                        x=100.0,
                                        line_width=2,
                                        line_dash="dot",
                                        line_color="rgba(255,255,255,0.30)",
                                    )
                                    five_to_ninety_eight_torque_fig.add_annotation(
                                        x=100.0,
                                        y=1.03,
                                        xref="x",
                                        yref="paper",
                                        text="98% Torque",
                                        showarrow=False,
                                        font=dict(size=11),
                                    )
                                    five_to_ninety_eight_torque_fig.update_layout(
                                        title="Posterior Cuff Torque from 5% to 98% Peak Positive Torque",
                                        xaxis_title=str(five_to_ninety_eight_torque_mean_df.attrs.get("x_axis_title", "5% to 98% Peak Positive Torque (%)")),
                                        yaxis_title="Torque_Nm",
                                        height=500,
                                    )
                                    st.plotly_chart(
                                        five_to_ninety_eight_torque_fig,
                                        use_container_width=True,
                                        key=f"posterior_cuff_five_to_ninety_eight_torque_plot_{posterior_n_points}_{len(selected_posterior_rep_items)}",
                                    )
                                if not five_to_ninety_eight_position_reps_long_df.empty and not five_to_ninety_eight_position_mean_df.empty:
                                    five_to_ninety_eight_position_fig = go.Figure()
                                    for rep_number, rep_df in five_to_ninety_eight_position_reps_long_df.groupby("rep_number"):
                                        file_name = rep_df["file_name"].iloc[0]
                                        five_to_ninety_eight_position_fig.add_trace(go.Scatter(
                                            x=rep_df["alignment_x"],
                                            y=rep_df["position_deg"],
                                            mode="lines",
                                            line=dict(width=1.5),
                                            opacity=0.45,
                                            name=file_name,
                                        ))
                                    five_to_ninety_eight_position_fig.add_trace(go.Scatter(
                                        x=five_to_ninety_eight_position_mean_df["alignment_x"],
                                        y=five_to_ninety_eight_position_mean_df["upper_band"],
                                        mode="lines",
                                        line=dict(width=0),
                                        showlegend=False,
                                        hoverinfo="skip",
                                    ))
                                    five_to_ninety_eight_position_fig.add_trace(go.Scatter(
                                        x=five_to_ninety_eight_position_mean_df["alignment_x"],
                                        y=five_to_ninety_eight_position_mean_df["lower_band"],
                                        mode="lines",
                                        line=dict(width=0),
                                        fill="tonexty",
                                        name="±1 SD",
                                    ))
                                    five_to_ninety_eight_position_fig.add_trace(go.Scatter(
                                        x=five_to_ninety_eight_position_mean_df["alignment_x"],
                                        y=five_to_ninety_eight_position_mean_df["mean_position_deg"],
                                        mode="lines",
                                        line=dict(width=4),
                                        name="Mean Position",
                                    ))
                                    five_to_ninety_eight_position_fig.add_vline(
                                        x=0.0,
                                        line_width=2,
                                        line_dash="dot",
                                        line_color="rgba(255,255,255,0.45)",
                                    )
                                    five_to_ninety_eight_position_fig.add_annotation(
                                        x=0.0,
                                        y=1.03,
                                        xref="x",
                                        yref="paper",
                                        text="5% Torque",
                                        showarrow=False,
                                        font=dict(size=11),
                                    )
                                    five_to_ninety_eight_position_fig.add_vline(
                                        x=100.0,
                                        line_width=2,
                                        line_dash="dot",
                                        line_color="rgba(255,255,255,0.30)",
                                    )
                                    five_to_ninety_eight_position_fig.add_annotation(
                                        x=100.0,
                                        y=1.03,
                                        xref="x",
                                        yref="paper",
                                        text="98% Torque",
                                        showarrow=False,
                                        font=dict(size=11),
                                    )
                                    five_to_ninety_eight_position_fig.update_layout(
                                        title="Posterior Cuff Position Degrees over 5% to 98% Peak Positive Torque",
                                        xaxis_title=str(five_to_ninety_eight_position_mean_df.attrs.get("x_axis_title", "5% to 98% Peak Positive Torque (%)")),
                                        yaxis_title="Position_Deg",
                                        height=500,
                                    )
                                    st.plotly_chart(
                                        five_to_ninety_eight_position_fig,
                                        use_container_width=True,
                                        key=f"posterior_cuff_five_to_ninety_eight_position_plot_{posterior_n_points}_{len(selected_posterior_rep_items)}",
                                    )
                                if not zero_to_ninety_eight_torque_reps_long_df.empty and not zero_to_ninety_eight_torque_mean_df.empty:
                                    zero_to_ninety_eight_torque_fig = go.Figure()
                                    for rep_number, rep_df in zero_to_ninety_eight_torque_reps_long_df.groupby("rep_number"):
                                        file_name = rep_df["file_name"].iloc[0]
                                        zero_to_ninety_eight_torque_fig.add_trace(go.Scatter(
                                            x=rep_df["alignment_x"],
                                            y=rep_df["torque_nm"],
                                            mode="lines",
                                            line=dict(width=1.5),
                                            opacity=0.45,
                                            name=file_name,
                                        ))
                                    zero_to_ninety_eight_torque_fig.add_trace(go.Scatter(
                                        x=zero_to_ninety_eight_torque_mean_df["alignment_x"],
                                        y=zero_to_ninety_eight_torque_mean_df["upper_band"],
                                        mode="lines",
                                        line=dict(width=0),
                                        showlegend=False,
                                        hoverinfo="skip",
                                    ))
                                    zero_to_ninety_eight_torque_fig.add_trace(go.Scatter(
                                        x=zero_to_ninety_eight_torque_mean_df["alignment_x"],
                                        y=zero_to_ninety_eight_torque_mean_df["lower_band"],
                                        mode="lines",
                                        line=dict(width=0),
                                        fill="tonexty",
                                        name="±1 SD",
                                    ))
                                    zero_to_ninety_eight_torque_fig.add_trace(go.Scatter(
                                        x=zero_to_ninety_eight_torque_mean_df["alignment_x"],
                                        y=zero_to_ninety_eight_torque_mean_df["mean_torque_nm"],
                                        mode="lines",
                                        line=dict(width=4),
                                        name="Mean Torque",
                                    ))
                                    zero_to_ninety_eight_torque_fig.add_vline(
                                        x=0.0,
                                        line_width=2,
                                        line_dash="dot",
                                        line_color="rgba(255,255,255,0.45)",
                                    )
                                    zero_to_ninety_eight_torque_fig.add_annotation(
                                        x=0.0,
                                        y=1.03,
                                        xref="x",
                                        yref="paper",
                                        text="0 Torque Rise",
                                        showarrow=False,
                                        font=dict(size=11),
                                    )
                                    zero_to_ninety_eight_torque_fig.add_vline(
                                        x=100.0,
                                        line_width=2,
                                        line_dash="dot",
                                        line_color="rgba(255,255,255,0.30)",
                                    )
                                    zero_to_ninety_eight_torque_fig.add_annotation(
                                        x=100.0,
                                        y=1.03,
                                        xref="x",
                                        yref="paper",
                                        text="98% Torque",
                                        showarrow=False,
                                        font=dict(size=11),
                                    )
                                    zero_to_ninety_eight_torque_fig.update_layout(
                                        title="Posterior Cuff Torque from 0 Torque Rise to 98% Peak Positive Torque",
                                        xaxis_title=str(zero_to_ninety_eight_torque_mean_df.attrs.get("x_axis_title", "0 Torque Rise to 98% Peak Positive Torque (%)")),
                                        yaxis_title="Torque_Nm",
                                        height=500,
                                    )
                                    st.plotly_chart(
                                        zero_to_ninety_eight_torque_fig,
                                        use_container_width=True,
                                        key=f"posterior_cuff_zero_to_ninety_eight_torque_plot_{posterior_n_points}_{len(selected_posterior_rep_items)}",
                                    )
                                if not zero_to_ninety_eight_position_reps_long_df.empty and not zero_to_ninety_eight_position_mean_df.empty:
                                    zero_to_ninety_eight_position_fig = go.Figure()
                                    for rep_number, rep_df in zero_to_ninety_eight_position_reps_long_df.groupby("rep_number"):
                                        file_name = rep_df["file_name"].iloc[0]
                                        zero_to_ninety_eight_position_fig.add_trace(go.Scatter(
                                            x=rep_df["alignment_x"],
                                            y=rep_df["position_deg"],
                                            mode="lines",
                                            line=dict(width=1.5),
                                            opacity=0.45,
                                            name=file_name,
                                        ))
                                    zero_to_ninety_eight_position_fig.add_trace(go.Scatter(
                                        x=zero_to_ninety_eight_position_mean_df["alignment_x"],
                                        y=zero_to_ninety_eight_position_mean_df["upper_band"],
                                        mode="lines",
                                        line=dict(width=0),
                                        showlegend=False,
                                        hoverinfo="skip",
                                    ))
                                    zero_to_ninety_eight_position_fig.add_trace(go.Scatter(
                                        x=zero_to_ninety_eight_position_mean_df["alignment_x"],
                                        y=zero_to_ninety_eight_position_mean_df["lower_band"],
                                        mode="lines",
                                        line=dict(width=0),
                                        fill="tonexty",
                                        name="±1 SD",
                                    ))
                                    zero_to_ninety_eight_position_fig.add_trace(go.Scatter(
                                        x=zero_to_ninety_eight_position_mean_df["alignment_x"],
                                        y=zero_to_ninety_eight_position_mean_df["mean_position_deg"],
                                        mode="lines",
                                        line=dict(width=4),
                                        name="Mean Position",
                                    ))
                                    zero_to_ninety_eight_position_fig.add_vline(
                                        x=0.0,
                                        line_width=2,
                                        line_dash="dot",
                                        line_color="rgba(255,255,255,0.45)",
                                    )
                                    zero_to_ninety_eight_position_fig.add_annotation(
                                        x=0.0,
                                        y=1.03,
                                        xref="x",
                                        yref="paper",
                                        text="0 Torque Rise",
                                        showarrow=False,
                                        font=dict(size=11),
                                    )
                                    zero_to_ninety_eight_position_fig.add_vline(
                                        x=100.0,
                                        line_width=2,
                                        line_dash="dot",
                                        line_color="rgba(255,255,255,0.30)",
                                    )
                                    zero_to_ninety_eight_position_fig.add_annotation(
                                        x=100.0,
                                        y=1.03,
                                        xref="x",
                                        yref="paper",
                                        text="98% Torque",
                                        showarrow=False,
                                        font=dict(size=11),
                                    )
                                    zero_to_ninety_eight_position_fig.update_layout(
                                        title="Posterior Cuff Position Degrees from 0 Torque Rise to 98% Peak Positive Torque",
                                        xaxis_title=str(zero_to_ninety_eight_position_mean_df.attrs.get("x_axis_title", "0 Torque Rise to 98% Peak Positive Torque (%)")),
                                        yaxis_title="Position_Deg",
                                        height=500,
                                    )
                                    st.plotly_chart(
                                        zero_to_ninety_eight_position_fig,
                                        use_container_width=True,
                                        key=f"posterior_cuff_zero_to_ninety_eight_position_plot_{posterior_n_points}_{len(selected_posterior_rep_items)}",
                                    )
                                raw_position_items = []
                                for rep_item in selected_posterior_rep_items:
                                    rep_df = rep_item["df"].copy()
                                    if "Elapsed Seconds" not in rep_df.columns or "Position_Deg" not in rep_df.columns:
                                        continue
                                    rep_df["Elapsed Seconds"] = pd.to_numeric(rep_df["Elapsed Seconds"], errors="coerce")
                                    rep_df["Position_Deg"] = pd.to_numeric(rep_df["Position_Deg"], errors="coerce")
                                    rep_df = rep_df.dropna(subset=["Elapsed Seconds", "Position_Deg"]).reset_index(drop=True)
                                    if rep_df.empty:
                                        continue
                                    position_bounds = detect_position_deg_rep_bounds(
                                        rep_df["Elapsed Seconds"].to_numpy(dtype=float),
                                        rep_df["Position_Deg"].to_numpy(dtype=float)
                                    )
                                    filtered_position, _filtered_fs, _clean_position = smooth_position_deg_signal(
                                        rep_df["Elapsed Seconds"].to_numpy(dtype=float),
                                        rep_df["Position_Deg"].to_numpy(dtype=float),
                                        lowpass_cutoff_hz=float(posterior_filtered_cutoff_hz),
                                    )
                                    raw_position_items.append({
                                        "rep_item": rep_item,
                                        "file_name": rep_item["name"],
                                        "rep_df": rep_df,
                                        "position_bounds": position_bounds,
                                        "filtered_position": filtered_position,
                                    })

                                if raw_position_items:
                                    posterior_raw_only_position_fig = go.Figure()
                                    for position_item in raw_position_items:
                                        file_name = position_item["file_name"]
                                        rep_df = position_item["rep_df"]
                                        posterior_raw_only_position_fig.add_trace(go.Scatter(
                                            x=rep_df["Elapsed Seconds"],
                                            y=rep_df["Position_Deg"],
                                            mode="lines",
                                            line=dict(width=1.75),
                                            opacity=0.8,
                                            name=file_name,
                                        ))
                                    posterior_raw_only_position_fig.update_layout(
                                        title="Posterior Cuff Reactive Eccentric: Raw Position Signals",
                                        xaxis_title="Elapsed Time (s)",
                                        yaxis_title="Position_Deg",
                                        height=450,
                                    )
                                    st.plotly_chart(
                                        posterior_raw_only_position_fig,
                                        use_container_width=True,
                                        key=f"posterior_cuff_single_rep_raw_only_position_plot_{len(raw_position_items)}",
                                    )

                                    posterior_filtered_position_fig = go.Figure()
                                    for position_item in raw_position_items:
                                        file_name = position_item["file_name"]
                                        rep_df = position_item["rep_df"]
                                        filtered_position = np.asarray(position_item["filtered_position"], dtype=float)
                                        posterior_filtered_position_fig.add_trace(go.Scatter(
                                            x=rep_df["Elapsed Seconds"],
                                            y=filtered_position,
                                            mode="lines",
                                            line=dict(width=2.5),
                                            opacity=0.9,
                                            name=file_name,
                                        ))
                                    posterior_filtered_position_fig.update_layout(
                                        title="Posterior Cuff Reactive Eccentric: Low-Pass Filtered Position Signals",
                                        xaxis_title="Elapsed Time (s)",
                                        yaxis_title="Position_Deg",
                                        height=450,
                                    )
                                    st.plotly_chart(
                                        posterior_filtered_position_fig,
                                        use_container_width=True,
                                        key=f"posterior_cuff_single_rep_filtered_position_plot_{len(raw_position_items)}",
                                    )

                                    manual_editor_options = [
                                        int(position_item["rep_item"]["biodex_test_id"])
                                        for position_item in raw_position_items
                                        if position_item["rep_item"].get("biodex_test_id") is not None
                                    ]
                                    if manual_editor_options:
                                        manual_editor_labels = {
                                            int(position_item["rep_item"]["biodex_test_id"]): position_item["file_name"]
                                            for position_item in raw_position_items
                                            if position_item["rep_item"].get("biodex_test_id") is not None
                                        }
                                        st.markdown("#### Manual ROM End Override")
                                        st.caption(
                                            "Pick a stored rep, set the ROM end time from the filtered position view, and save it. "
                                            "That saved endpoint will override the automatic ROM end in the posterior cuff torque normalization modes that use ROM end."
                                        )
                                        manual_editor_test_id = st.selectbox(
                                            "Rep to edit",
                                            options=manual_editor_options,
                                            format_func=lambda test_id: manual_editor_labels.get(int(test_id), f"Test {int(test_id)}"),
                                            key="posterior_cuff_manual_rom_end_test_id",
                                        )
                                        manual_editor_item = next(
                                            position_item
                                            for position_item in raw_position_items
                                            if int(position_item["rep_item"]["biodex_test_id"]) == int(manual_editor_test_id)
                                        )
                                        manual_editor_rep_df = manual_editor_item["rep_df"]
                                        manual_editor_filtered_position = np.asarray(manual_editor_item["filtered_position"], dtype=float)
                                        manual_editor_bounds = manual_editor_item["position_bounds"]
                                        manual_editor_existing = manual_editor_item["rep_item"].get("manual_rom_end") or {}
                                        if (
                                            manual_editor_existing.get("time_seconds") is not None
                                            and np.isfinite(float(manual_editor_existing["time_seconds"]))
                                        ):
                                            default_manual_end_time = float(manual_editor_existing["time_seconds"])
                                        else:
                                            default_manual_end_time = float(
                                                manual_editor_rep_df["Elapsed Seconds"].iloc[int(manual_editor_bounds["end_idx"])]
                                            )
                                        manual_editor_step = 0.01
                                        if len(manual_editor_rep_df) > 1:
                                            time_diffs = np.diff(manual_editor_rep_df["Elapsed Seconds"].to_numpy(dtype=float))
                                            positive_diffs = time_diffs[np.isfinite(time_diffs) & (time_diffs > 0)]
                                            if positive_diffs.size > 0:
                                                manual_editor_step = max(0.001, float(np.nanmedian(positive_diffs)))
                                        manual_end_time_seconds = st.slider(
                                            "Manual ROM end time (s)",
                                            min_value=float(manual_editor_rep_df["Elapsed Seconds"].min()),
                                            max_value=float(manual_editor_rep_df["Elapsed Seconds"].max()),
                                            value=float(default_manual_end_time),
                                            step=float(manual_editor_step),
                                            key="posterior_cuff_manual_rom_end_time_seconds",
                                        )
                                        manual_end_idx = int(np.argmin(np.abs(
                                            manual_editor_rep_df["Elapsed Seconds"].to_numpy(dtype=float) - float(manual_end_time_seconds)
                                        )))
                                        manual_end_position = float(manual_editor_filtered_position[manual_end_idx])
                                        st.caption(
                                            "Drag the slider to scrub the filtered ROM line. "
                                            f"Selected endpoint: `{float(manual_editor_rep_df['Elapsed Seconds'].iloc[manual_end_idx]):.3f} s`, "
                                            f"`{manual_end_position:.1f} deg`."
                                        )
                                        manual_editor_fig = go.Figure()
                                        manual_editor_fig.add_trace(go.Scatter(
                                            x=manual_editor_rep_df["Elapsed Seconds"],
                                            y=manual_editor_filtered_position,
                                            mode="lines",
                                            line=dict(width=3),
                                            name=f"{manual_editor_item['file_name']} (Filtered)",
                                        ))
                                        manual_editor_fig.add_vline(
                                            x=float(manual_editor_rep_df["Elapsed Seconds"].iloc[manual_end_idx]),
                                            line_width=2,
                                            line_dash="dot",
                                            line_color="rgba(255,184,108,0.75)",
                                        )
                                        manual_editor_fig.add_trace(go.Scatter(
                                            x=[float(manual_editor_rep_df["Elapsed Seconds"].iloc[manual_end_idx])],
                                            y=[manual_end_position],
                                            mode="markers",
                                            marker=dict(size=11, symbol="diamond", color="#ffb86c"),
                                            name="Manual ROM End",
                                        ))
                                        manual_editor_fig.update_layout(
                                            title="Posterior Cuff Reactive Eccentric: Manual ROM End Editor",
                                            xaxis_title="Elapsed Time (s)",
                                            yaxis_title="Position_Deg",
                                            height=420,
                                        )
                                        st.plotly_chart(
                                            manual_editor_fig,
                                            use_container_width=True,
                                            key=f"posterior_cuff_manual_rom_end_editor_{int(manual_editor_test_id)}",
                                        )
                                        manual_action_col1, manual_action_col2 = st.columns(2)
                                        with manual_action_col1:
                                            save_manual_rom_end = st.button(
                                                "Save Manual ROM End",
                                                key="posterior_cuff_save_manual_rom_end",
                                                use_container_width=True,
                                            )
                                        with manual_action_col2:
                                            clear_manual_rom_end = st.button(
                                                "Clear Manual ROM End Override",
                                                key="posterior_cuff_clear_manual_rom_end",
                                                use_container_width=True,
                                            )

                                        if save_manual_rom_end:
                                            try:
                                                upsert_biodex_manual_landmark(
                                                    cur,
                                                    conn,
                                                    biodex_test_id=int(manual_editor_test_id),
                                                    landmark_type="posterior_cuff_rom_end",
                                                    sample_index=int(manual_end_idx + 1),
                                                    time_seconds=float(manual_editor_rep_df["Elapsed Seconds"].iloc[manual_end_idx]),
                                                    position_deg=float(manual_end_position),
                                                )
                                            except Exception as exc:
                                                conn.rollback()
                                                st.error(f"Could not save the manual ROM end: {exc}")
                                            else:
                                                updated_manual_landmark = fetch_biodex_manual_landmark(
                                                    cur,
                                                    biodex_test_id=int(manual_editor_test_id),
                                                    landmark_type="posterior_cuff_rom_end",
                                                )
                                                updated_previews = []
                                                for preview_item_entry in st.session_state.get("biodex_test_uploaded_previews", []):
                                                    if int(preview_item_entry.get("biodex_test_id", -1)) == int(manual_editor_test_id):
                                                        preview_item_entry = dict(preview_item_entry)
                                                        preview_item_entry["manual_rom_end"] = updated_manual_landmark
                                                    updated_previews.append(preview_item_entry)
                                                st.session_state["biodex_test_uploaded_previews"] = updated_previews
                                                st.success("Saved the manual ROM end override for this rep.")
                                                st.rerun()

                                        if clear_manual_rom_end:
                                            try:
                                                delete_biodex_manual_landmark(
                                                    cur,
                                                    conn,
                                                    biodex_test_id=int(manual_editor_test_id),
                                                    landmark_type="posterior_cuff_rom_end",
                                                )
                                            except Exception as exc:
                                                conn.rollback()
                                                st.error(f"Could not clear the manual ROM end override: {exc}")
                                            else:
                                                updated_previews = []
                                                for preview_item_entry in st.session_state.get("biodex_test_uploaded_previews", []):
                                                    if int(preview_item_entry.get("biodex_test_id", -1)) == int(manual_editor_test_id):
                                                        preview_item_entry = dict(preview_item_entry)
                                                        preview_item_entry["manual_rom_end"] = None
                                                    updated_previews.append(preview_item_entry)
                                                st.session_state["biodex_test_uploaded_previews"] = updated_previews
                                                st.success("Cleared the manual ROM end override for this rep.")
                                                st.rerun()

                                    common_smoothed_rom_end_values = []
                                    for position_item in raw_position_items:
                                        position_bounds = position_item["position_bounds"]
                                        smooth_position = np.asarray(position_bounds["smooth_position"], dtype=float)
                                        final_plateau_value = position_bounds.get("final_plateau_value")
                                        if final_plateau_value is None or not np.isfinite(final_plateau_value):
                                            continue
                                        common_smoothed_rom_end_values.append(float(final_plateau_value))

                                    common_smoothed_rom_end = (
                                        float(np.nanmedian(common_smoothed_rom_end_values))
                                        if common_smoothed_rom_end_values
                                        else None
                                    )
                                    posterior_raw_position_fig = go.Figure()
                                    target_angle_136 = 136.0
                                    any_manual_rom_override = False
                                    for position_item in raw_position_items:
                                        file_name = position_item["file_name"]
                                        rep_df = position_item["rep_df"]
                                        position_bounds = position_item["position_bounds"]
                                        rep_item = position_item["rep_item"]
                                        smooth_position = np.asarray(position_bounds["smooth_position"], dtype=float)
                                        start_idx = int(position_bounds["start_idx"])
                                        end_idx = int(position_bounds["end_idx"])
                                        manual_rom_end = rep_item.get("manual_rom_end") or {}
                                        if (
                                            manual_rom_end.get("time_seconds") is not None
                                            and np.isfinite(float(manual_rom_end["time_seconds"]))
                                        ):
                                            manual_end_time = float(manual_rom_end["time_seconds"])
                                            end_idx = int(np.argmin(np.abs(
                                                rep_df["Elapsed Seconds"].to_numpy(dtype=float) - manual_end_time
                                            )))
                                            any_manual_rom_override = True
                                        elif posterior_x_axis_mode == "zero_to_position_136_ascent_normalized":
                                            threshold_end_idx = find_first_position_ascent_threshold(
                                                smooth_position,
                                                start_idx,
                                                target_angle_136,
                                            )
                                            if threshold_end_idx is not None:
                                                end_idx = int(threshold_end_idx)
                                        elif common_smoothed_rom_end is not None:
                                            common_end_idx = find_first_common_rom_band_entry(
                                                smooth_position,
                                                start_idx,
                                                position_bounds.get("fs"),
                                                common_smoothed_rom_end,
                                                angle_tolerance_deg=float(posterior_common_rom_end_tolerance_deg),
                                                velocity_tolerance_deg_per_second=15.0,
                                                hold_time_seconds=float(posterior_common_rom_end_hold_time_seconds),
                                            )
                                            if common_end_idx is not None:
                                                end_idx = int(common_end_idx)
                                        posterior_raw_position_fig.add_trace(go.Scatter(
                                            x=rep_df["Elapsed Seconds"],
                                            y=rep_df["Position_Deg"],
                                            mode="lines",
                                            line=dict(width=1.5),
                                            opacity=0.45,
                                            name=file_name,
                                        ))
                                        posterior_raw_position_fig.add_trace(go.Scatter(
                                            x=rep_df["Elapsed Seconds"],
                                            y=smooth_position,
                                            mode="lines",
                                            line=dict(width=2.5, dash="dash"),
                                            opacity=0.9,
                                            name=f"{file_name} (Smoothed)",
                                        ))
                                        posterior_raw_position_fig.add_trace(go.Scatter(
                                            x=[float(rep_df["Elapsed Seconds"].iloc[start_idx])],
                                            y=[float(smooth_position[start_idx])],
                                            mode="markers",
                                            marker=dict(size=10, symbol="circle", color="#7bd389"),
                                            name=f"{file_name} Start",
                                            showlegend=False,
                                        ))
                                        posterior_raw_position_fig.add_trace(go.Scatter(
                                            x=[float(rep_df["Elapsed Seconds"].iloc[end_idx])],
                                            y=[float(smooth_position[end_idx])],
                                            mode="markers",
                                            marker=dict(size=10, symbol="diamond", color="#ffb86c"),
                                            name=f"{file_name} End",
                                            showlegend=False,
                                        ))
                                    if posterior_x_axis_mode == "zero_to_position_136_ascent_normalized":
                                        posterior_raw_position_fig.add_hline(
                                            y=target_angle_136,
                                            line_width=1.5,
                                            line_dash="dot",
                                            line_color="rgba(255,184,108,0.55)",
                                        )
                                        posterior_raw_position_fig.add_annotation(
                                            x=1.0,
                                            y=target_angle_136,
                                            xref="paper",
                                            yref="y",
                                            xanchor="right",
                                            yanchor="bottom",
                                            text="136° On Ascent",
                                            showarrow=False,
                                            font=dict(size=11, color="rgba(255,184,108,0.95)"),
                                        )
                                    elif common_smoothed_rom_end is not None:
                                        posterior_raw_position_fig.add_hline(
                                            y=common_smoothed_rom_end,
                                            line_width=1.5,
                                            line_dash="dot",
                                            line_color="rgba(255,184,108,0.55)",
                                        )
                                        posterior_raw_position_fig.add_annotation(
                                            x=1.0,
                                            y=common_smoothed_rom_end,
                                            xref="paper",
                                            yref="y",
                                            xanchor="right",
                                            yanchor="bottom",
                                            text="Common Smoothed ROM End",
                                            showarrow=False,
                                            font=dict(size=11, color="rgba(255,184,108,0.95)"),
                                        )
                                    if len(raw_position_items) == 1:
                                        _file_name, _rep_df, position_bounds = raw_position_items[0]
                                        posterior_raw_position_fig.add_hrect(
                                            y0=float(position_bounds.get("plateau_band_low", np.nan)),
                                            y1=float(position_bounds.get("plateau_band_high", np.nan)),
                                            fillcolor="rgba(255,184,108,0.10)",
                                            line_width=0,
                                            layer="below",
                                        )
                                    posterior_raw_position_fig.update_layout(
                                        title="Posterior Cuff Reactive Eccentric: Raw Position Signals with Smoothed Start/End",
                                        xaxis_title="Elapsed Time (s)",
                                        yaxis_title="Position_Deg",
                                        height=450,
                                    )
                                    if posterior_x_axis_mode == "zero_to_position_136_ascent_normalized":
                                        raw_position_note = (
                                            "Dashed line = smoothed Position_Deg, green marker = detected start, "
                                            "orange marker = manual ROM end override when saved, otherwise 136° ascent end"
                                        )
                                    elif posterior_x_axis_mode == "zero_to_common_smoothed_rom_end_normalized":
                                        raw_position_note = (
                                            "Dashed line = smoothed Position_Deg, green marker = detected start, "
                                            "orange marker = manual ROM end override when saved, otherwise first entry into shared stabilized ROM band"
                                        )
                                    else:
                                        raw_position_note = (
                                            "Dashed line = smoothed Position_Deg, green marker = detected start, "
                                            "orange marker = manual ROM end override when saved, otherwise detected end"
                                        )
                                    posterior_raw_position_fig.add_annotation(
                                        x=1.0,
                                        y=1.10,
                                            xref="paper",
                                            yref="paper",
                                        xanchor="right",
                                        yanchor="bottom",
                                        text=raw_position_note,
                                        showarrow=False,
                                        font=dict(size=11),
                                    )
                                    st.plotly_chart(
                                        posterior_raw_position_fig,
                                        use_container_width=True,
                                        key=f"posterior_cuff_single_rep_raw_position_plot_{len(raw_position_items)}",
                                    )

                    if posterior_alignment_metadata:
                        posterior_summary_rows = []
                        for rep_meta in posterior_alignment_metadata:
                            posterior_summary_rows.append({
                                "Rep File": rep_meta["file_name"],
                                "Anchor": rep_meta["anchor_label"],
                                "Anchor Time (s)": rep_meta["anchor_time"],
                                "Anchor Torque (Nm)": rep_meta["anchor_torque"],
                                "0 Torque Rise Time (s)": rep_meta.get("zero_torque_rise_time"),
                                "Peak Positive Time (s)": rep_meta.get("peak_positive_time"),
                                "0 Torque -> Peak Positive (s)": rep_meta.get("zero_to_peak_duration_s"),
                            })
                        st.markdown("### Single-Rep Alignment Summary")
                        st.dataframe(pd.DataFrame(posterior_summary_rows), use_container_width=True)
                else:
                    st.markdown("### Rep Detection Preview")
                    preview_controls_col, preview_plot_col = st.columns([0.35, 1.0], vertical_alignment="top")

                    with preview_controls_col:
                        is_shoulder_er_ir_speed_preview = (
                            preview_movement == "shoulder_er_ir"
                            and preview_protocol_type == "speed"
                        )
                        preview_threshold = 20.0
                        preview_min_samples = 15
                        preview_buffer_samples = 20
                        preview_n_points = 101
                        preview_landmark_prominence = 0.12
                        preview_position_cutoff_hz = 0.5
                        preview_position_drop_fraction = 0.60
                        if is_shoulder_er_ir_speed_preview:
                            st.info("Auto-detecting Shoulder ER/IR speed reps from position.")
                            st.caption("Recommended defaults: 0.5 Hz smoothing and 0.60 dip-depth threshold.")
                            with st.expander("Adjust detection if needed", expanded=False):
                                if st.button(
                                    "Use recommended defaults",
                                    key="biodex_test_preview_erir_detection_defaults",
                                    use_container_width=True,
                                ):
                                    st.session_state["biodex_test_preview_position_cutoff_hz"] = 0.5
                                    st.session_state["biodex_test_preview_position_drop_fraction"] = 0.60
                                    st.rerun()
                                preview_position_cutoff_hz = st.slider(
                                    "Position smoothing cutoff (Hz)",
                                    min_value=0.5,
                                    max_value=4.0,
                                    value=0.5,
                                    step=0.1,
                                    key="biodex_test_preview_position_cutoff_hz",
                                )
                                preview_position_drop_fraction = st.slider(
                                    "Rep threshold depth into dip",
                                    min_value=0.05,
                                    max_value=0.60,
                                    value=0.60,
                                    step=0.01,
                                    key="biodex_test_preview_position_drop_fraction",
                                )
                                st.caption(
                                    "Lower smoothing follows more noise. Higher dip depth requires a deeper position excursion before counting a rep."
                                )
                        else:
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
                        if (
                            preview_movement == "d2_shoulder_pattern"
                            and preview_protocol_type != "speed"
                        ):
                            st.caption(
                                "D2 preview uses `Position_Deg` cycles for rep windows. "
                                "Buffer and normalized points apply before D2 landmark alignment."
                            )
                        elif (
                            preview_movement == "d2_shoulder_pattern"
                            and preview_protocol_type == "speed"
                        ):
                            st.caption(
                                "D2 Speed preview uses torque spike bursts as reps, similar to Shoulder ER/IR. "
                                "Threshold, minimum active samples, buffer, and landmark prominence all apply here."
                            )

                    preview_rep_windows = []
                    preview_position_detection_metadata = None
                    preview_processing_version = "shoulder_er_ir_landmark_v1"
                    preview_landmark_reps_long_df = pd.DataFrame()
                    preview_landmark_mean_df = pd.DataFrame()
                    preview_landmark_aligned_rep_metadata = []
                    if is_shoulder_er_ir_speed_preview:
                        preview_rep_windows, preview_position_detection_metadata = detect_shoulder_er_ir_speed_reps(
                            preview_df,
                            position_col="Position_Deg",
                            time_col="Elapsed Seconds",
                            lowpass_cutoff_hz=float(preview_position_cutoff_hz),
                            min_samples=int(preview_min_samples),
                            buffer_samples=int(preview_buffer_samples),
                            drop_fraction=float(preview_position_drop_fraction),
                        )
                        preview_reps_long_df, preview_mean_df, preview_aligned_rep_metadata = extract_position_window_normalized_biodex_reps(
                            preview_df,
                            preview_rep_windows,
                            time_col="Elapsed Seconds",
                            value_col="Torque_Nm",
                            n_points=int(preview_n_points),
                        )
                        preview_processing_version = "shoulder_er_ir_speed_position_window_normalized_v1"
                    elif (
                        preview_movement == "d2_shoulder_pattern"
                        and preview_protocol_type != "speed"
                    ):
                        preview_rep_windows = detect_d2_biodex_reps(
                            preview_df,
                            position_col="Position_Deg",
                            buffer_samples=int(preview_buffer_samples),
                        )
                        preview_reps_long_df, preview_mean_df, preview_aligned_rep_metadata = extract_d2_landmark_aligned_biodex_reps(
                            preview_df,
                            preview_rep_windows,
                            time_col="Elapsed Seconds",
                            position_col="Position_Deg",
                            value_col="Torque_Nm",
                            n_points=int(preview_n_points),
                        )
                        preview_processing_version = "d2_shoulder_pattern_landmark_v1"
                    elif (
                        preview_movement == "d2_shoulder_pattern"
                        and preview_protocol_type == "speed"
                    ):
                        preview_rep_windows = detect_d2_speed_biodex_reps(
                            preview_df,
                            value_col="Torque_Nm",
                            threshold=float(preview_threshold),
                            min_samples=int(preview_min_samples),
                            tail_buffer_samples=int(preview_buffer_samples),
                        )
                        preview_reps_long_df, preview_mean_df, preview_aligned_rep_metadata = extract_d2_speed_landmark_aligned_biodex_reps(
                            preview_df,
                            preview_rep_windows,
                            time_col="Elapsed Seconds",
                            value_col="Torque_Nm",
                            n_points=int(preview_n_points),
                            prominence_ratio=float(preview_landmark_prominence),
                        )
                        preview_processing_version = "d2_shoulder_pattern_speed_landmark_v1"
                    else:
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

                    if is_shoulder_er_ir_speed_preview:
                        with preview_controls_col:
                            if preview_rep_windows:
                                st.success(f"{len(preview_rep_windows)} position reps detected.")
                            else:
                                st.warning("No position reps detected.")
                            st.caption(
                                f"Using {float(preview_position_cutoff_hz):.1f} Hz smoothing "
                                f"and {float(preview_position_drop_fraction):.2f} dip depth."
                            )

                    with preview_plot_col:
                        preview_plot_suffix = (
                            f"{preview_item['biodex_test_id']}_"
                            f"{preview_movement}_"
                            f"{preview_protocol_type}_"
                            f"{preview_threshold}_"
                            f"{preview_min_samples}_"
                            f"{preview_buffer_samples}_"
                            f"{preview_n_points}_"
                            f"{preview_landmark_prominence}_"
                            f"{preview_position_cutoff_hz}_"
                            f"{preview_position_drop_fraction}"
                        )
                        preview_raw_fig = go.Figure()
                        if (
                            is_shoulder_er_ir_speed_preview
                            and preview_position_detection_metadata is not None
                        ):
                            preview_smooth_position = np.asarray(
                                preview_position_detection_metadata["smooth_position"],
                                dtype=float,
                            )
                            preview_raw_fig.add_trace(go.Scatter(
                                x=preview_df["Elapsed Seconds"],
                                y=preview_smooth_position,
                                mode="lines",
                                line=dict(width=2.75),
                                name=f"{preview_item['name']} (Filtered Position)",
                            ))
                        else:
                            preview_raw_fig.add_trace(go.Scatter(
                                x=preview_df["Elapsed Seconds"],
                                y=preview_df["Torque_Nm"],
                                mode="lines",
                                name=f"{preview_item['name']} (Raw)",
                            ))

                        preview_shapes = []
                        preview_elapsed_seconds = pd.to_numeric(
                            preview_df["Elapsed Seconds"],
                            errors="coerce",
                        ).to_numpy(dtype=float)
                        preview_valid_elapsed = preview_elapsed_seconds[np.isfinite(preview_elapsed_seconds)]
                        preview_dt = 0.0
                        if preview_valid_elapsed.size > 1:
                            preview_diff = np.diff(preview_valid_elapsed)
                            preview_diff = preview_diff[np.isfinite(preview_diff) & (preview_diff > 0)]
                            if preview_diff.size:
                                preview_dt = float(np.median(preview_diff))
                        preview_min_visible_width = max(preview_dt * 2.0, 0.25)
                        for rep_number, (start_idx, end_idx) in enumerate(preview_rep_windows, start=1):
                            x0 = float(preview_df.iloc[start_idx]["Elapsed Seconds"])
                            x1 = float(preview_df.iloc[end_idx]["Elapsed Seconds"])
                            if not np.isfinite(x0) or not np.isfinite(x1):
                                continue
                            if x1 < x0:
                                x0, x1 = x1, x0
                            if (x1 - x0) < preview_min_visible_width:
                                x_center = (x0 + x1) / 2.0
                                half_width = preview_min_visible_width / 2.0
                                x0 = x_center - half_width
                                x1 = x_center + half_width
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
                            preview_raw_fig.add_vline(
                                x=x0,
                                line_width=1,
                                line_dash="dot",
                                line_color="rgba(0, 123, 255, 0.30)",
                            )
                            preview_raw_fig.add_vline(
                                x=x1,
                                line_width=1,
                                line_dash="dot",
                                line_color="rgba(0, 123, 255, 0.30)",
                            )
                            preview_raw_fig.add_annotation(
                                x=(x0 + x1) / 2.0,
                                y=1.02,
                                xref="x",
                                yref="paper",
                                text=f"Rep {rep_number}",
                                showarrow=False,
                            )

                        if (
                            is_shoulder_er_ir_speed_preview
                            and preview_position_detection_metadata is not None
                        ):
                            preview_raw_fig.add_hline(
                                y=float(preview_position_detection_metadata["active_threshold"]),
                                line_width=1.5,
                                line_dash="dot",
                                line_color="rgba(255,184,108,0.55)",
                            )
                            preview_raw_fig.add_annotation(
                                x=1.0,
                                y=float(preview_position_detection_metadata["active_threshold"]),
                                xref="paper",
                                yref="y",
                                xanchor="right",
                                yanchor="bottom",
                                text="Rep Threshold",
                                showarrow=False,
                                font=dict(size=11, color="rgba(255,184,108,0.95)"),
                            )

                        preview_raw_fig.update_layout(
                            title=(
                                "Detected Position Reps"
                                if is_shoulder_er_ir_speed_preview
                                else "Detected Torque Reps"
                            ),
                            xaxis_title="Elapsed Time (s)",
                            yaxis_title=(
                                "Position_Deg"
                                if is_shoulder_er_ir_speed_preview
                                else "Torque_Nm"
                            ),
                            shapes=preview_shapes,
                            height=500,
                        )
                        st.plotly_chart(
                            preview_raw_fig,
                            use_container_width=True,
                            key=f"biodex_test_preview_raw_plot_{preview_plot_suffix}",
                        )
                        if len(preview_rep_windows) > 1:
                            st.caption(
                                f"{len(preview_rep_windows)} reps were detected. "
                                "If you only see the first one, the plot is zoomed in; adjusting any control or using Plotly's autoscale/reset axes will bring the full trace back."
                            )

                        if preview_reps_long_df.empty or preview_mean_df.empty:
                            if preview_rep_windows:
                                if is_shoulder_er_ir_speed_preview:
                                    st.warning(
                                        "Rep windows were detected, but torque normalization could not be completed with the current settings."
                                    )
                                else:
                                    st.warning(
                                        "Rep windows were detected, but landmark alignment could not be completed with the current settings."
                                    )
                            else:
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
                            for boundary_pct, label in zip(
                                preview_mean_df.attrs.get("landmark_boundary_pct", []),
                                preview_mean_df.attrs.get("landmark_labels", []),
                            ):
                                preview_avg_fig.add_vline(
                                    x=float(boundary_pct),
                                    line_width=2,
                                    line_dash="dot",
                                    line_color="rgba(255,255,255,0.45)",
                                )
                                preview_avg_fig.add_annotation(
                                    x=float(boundary_pct),
                                    y=1.03,
                                    xref="x",
                                    yref="paper",
                                    text=label,
                                    showarrow=False,
                                    font=dict(size=11),
                                )
                            preview_avg_fig.update_layout(
                                title=preview_mean_df.attrs.get("title", "Landmark-Aligned Average Torque Curve Across Detected Reps"),
                                xaxis_title="Movement Cycle (%)",
                                yaxis_title="Torque_Nm",
                                height=500,
                            )
                            st.plotly_chart(
                                preview_avg_fig,
                                use_container_width=True,
                                key=f"biodex_test_preview_avg_plot_{preview_plot_suffix}",
                            )
                            if (
                                is_shoulder_er_ir_speed_preview
                                and "Position_Deg" in preview_df.columns
                            ):
                                preview_position_rows = []
                                preview_position_curves = []
                                preview_percent_axis = np.linspace(0.0, 100.0, int(preview_n_points))
                                for rep_number, (start_idx, end_idx) in enumerate(preview_rep_windows, start=1):
                                    rep_position_df = preview_df.iloc[int(start_idx):int(end_idx) + 1].copy()
                                    rep_position_df["Position_Deg"] = pd.to_numeric(
                                        rep_position_df["Position_Deg"],
                                        errors="coerce",
                                    )
                                    rep_position_df = rep_position_df.dropna(subset=["Position_Deg"]).reset_index(drop=True)
                                    if len(rep_position_df) < 3:
                                        continue
                                    rep_position_values = rep_position_df["Position_Deg"].to_numpy(dtype=float)
                                    rep_pct = np.linspace(0.0, 100.0, len(rep_position_values))
                                    interp_position = np.interp(
                                        preview_percent_axis,
                                        rep_pct,
                                        rep_position_values,
                                    )
                                    preview_position_curves.append(interp_position)
                                    preview_position_rows.append(pd.DataFrame({
                                        "rep_number": rep_number,
                                        "movement_pct": preview_percent_axis,
                                        "position_deg": interp_position,
                                    }))

                                if preview_position_curves:
                                    preview_position_long_df = pd.concat(preview_position_rows, ignore_index=True)
                                    preview_position_mean_arr = np.vstack(preview_position_curves)
                                    preview_position_mean_df = pd.DataFrame({
                                        "movement_pct": preview_percent_axis,
                                        "mean_position_deg": np.nanmean(preview_position_mean_arr, axis=0),
                                        "std_position_deg": np.nanstd(preview_position_mean_arr, axis=0),
                                    })
                                    preview_position_mean_df["upper_band"] = (
                                        preview_position_mean_df["mean_position_deg"]
                                        + preview_position_mean_df["std_position_deg"]
                                    )
                                    preview_position_mean_df["lower_band"] = (
                                        preview_position_mean_df["mean_position_deg"]
                                        - preview_position_mean_df["std_position_deg"]
                                    )

                                    preview_position_fig = go.Figure()
                                    for rep_number, rep_position_plot_df in preview_position_long_df.groupby("rep_number"):
                                        preview_position_fig.add_trace(go.Scatter(
                                            x=rep_position_plot_df["movement_pct"],
                                            y=rep_position_plot_df["position_deg"],
                                            mode="lines",
                                            line=dict(width=1),
                                            opacity=0.35,
                                            name=f"Rep {rep_number}",
                                        ))
                                    preview_position_fig.add_trace(go.Scatter(
                                        x=preview_position_mean_df["movement_pct"],
                                        y=preview_position_mean_df["upper_band"],
                                        mode="lines",
                                        line=dict(width=0),
                                        showlegend=False,
                                        hoverinfo="skip",
                                    ))
                                    preview_position_fig.add_trace(go.Scatter(
                                        x=preview_position_mean_df["movement_pct"],
                                        y=preview_position_mean_df["lower_band"],
                                        mode="lines",
                                        line=dict(width=0),
                                        fill="tonexty",
                                        name="±1 SD",
                                    ))
                                    preview_position_fig.add_trace(go.Scatter(
                                        x=preview_position_mean_df["movement_pct"],
                                        y=preview_position_mean_df["mean_position_deg"],
                                        mode="lines",
                                        line=dict(width=4),
                                        name="Mean Position",
                                    ))
                                    preview_position_fig.update_layout(
                                        title="Position Start -> End Normalized Position Comparison",
                                        xaxis_title="Movement Cycle (%)",
                                        yaxis_title="Position_Deg",
                                        height=500,
                                    )
                                    st.plotly_chart(
                                        preview_position_fig,
                                        use_container_width=True,
                                        key=f"biodex_test_preview_position_normalized_plot_{preview_plot_suffix}",
                                    )
                            if (
                                not is_shoulder_er_ir_speed_preview
                                and preview_movement == "shoulder_er_ir"
                                and preview_protocol_type == "speed"
                                and not preview_landmark_reps_long_df.empty
                                and not preview_landmark_mean_df.empty
                            ):
                                preview_landmark_fig = go.Figure()
                                for rep_number, rep_df in preview_landmark_reps_long_df.groupby("rep_number"):
                                    preview_landmark_fig.add_trace(go.Scatter(
                                        x=rep_df["movement_pct"],
                                        y=rep_df["torque_nm"],
                                        mode="lines",
                                        line=dict(width=1),
                                        opacity=0.35,
                                        name=f"Rep {rep_number}",
                                    ))
                                preview_landmark_fig.add_trace(go.Scatter(
                                    x=preview_landmark_mean_df["movement_pct"],
                                    y=preview_landmark_mean_df["upper_band"],
                                    mode="lines",
                                    line=dict(width=0),
                                    showlegend=False,
                                    hoverinfo="skip",
                                ))
                                preview_landmark_fig.add_trace(go.Scatter(
                                    x=preview_landmark_mean_df["movement_pct"],
                                    y=preview_landmark_mean_df["lower_band"],
                                    mode="lines",
                                    line=dict(width=0),
                                    fill="tonexty",
                                    name="±1 SD",
                                ))
                                preview_landmark_fig.add_trace(go.Scatter(
                                    x=preview_landmark_mean_df["movement_pct"],
                                    y=preview_landmark_mean_df["mean_torque_nm"],
                                    mode="lines",
                                    line=dict(width=4),
                                    name="Mean Torque",
                                ))
                                for boundary_pct, label in zip(
                                    preview_landmark_mean_df.attrs.get("landmark_boundary_pct", []),
                                    preview_landmark_mean_df.attrs.get("landmark_labels", []),
                                ):
                                    preview_landmark_fig.add_vline(
                                        x=float(boundary_pct),
                                        line_width=2,
                                        line_dash="dot",
                                        line_color="rgba(255,255,255,0.45)",
                                    )
                                    preview_landmark_fig.add_annotation(
                                        x=float(boundary_pct),
                                        y=1.03,
                                        xref="x",
                                        yref="paper",
                                        text=label,
                                        showarrow=False,
                                        font=dict(size=11),
                                    )
                                preview_landmark_fig.update_layout(
                                    title="Landmark-Aligned Torque Comparison Across Detected Reps",
                                    xaxis_title="Movement Cycle (%)",
                                    yaxis_title="Torque_Nm",
                                    height=500,
                                )
                                st.plotly_chart(
                                    preview_landmark_fig,
                                    use_container_width=True,
                                    key=f"biodex_test_preview_landmark_plot_{preview_plot_suffix}",
                                )

                            if st.button(
                                "Save Detection Results",
                                key=f"biodex_test_save_detection_{preview_item['biodex_test_id']}",
                                use_container_width=True,
                            ):
                                try:
                                    biodex_processing_run_id = insert_biodex_processing_run(
                                        cur,
                                        biodex_test_id=int(preview_item["biodex_test_id"]),
                                        processing_version=preview_processing_version,
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
                options=["d2_shoulder_pattern", "shoulder_er_ir", "posterior_cuff"],
                format_func=format_biodex_movement_label,
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
                disabled=(
                    selected_compare_athlete_id is None
                    or selected_compare_protocol == "reactive_eccentric"
                ),
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
                speed_deg_per_sec=get_biodex_effective_speed(
                    selected_compare_protocol,
                    selected_compare_speed,
                ),
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
                review_speed = selected_review_row["speed_deg_per_sec"]
                st.metric(
                    "Speed",
                    f"{int(review_speed)} deg/s" if pd.notna(review_speed) else "N/A",
                )
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
