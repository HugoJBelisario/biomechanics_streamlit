import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import psycopg2
import numpy as np
from scipy.stats import linregress
import plotly.graph_objects as go

interp_points = np.linspace(0, 100, 100)

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

def get_shoulder_er_max_frame(take_id, handedness, cur):
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
    end_frame   = peak_arm_energy_frame + 15

    # Constrain MER search so it cannot occur after Ball Release
    br_frame = get_ball_release_frame(take_id, handedness, cur)
    if br_frame is not None:
        end_frame = min(int(end_frame), int(br_frame))

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

# Title
st.title("Biomechanics Viewer")

# Connect to DB
conn = get_connection()
cur = conn.cursor()

tab1, tab2, tab3, tab4 = st.tabs(["Compensation Analysis", "Session Comparison", "0-10 Report", "Time-Series"])
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
            knee_window = df_knee[
                (df_knee["frame"] >= drive_start_frame - 100) &
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

    energy_plot_options = st.multiselect(
        "Energy Flow Type",
        [
            "Torso Power",
            "STP Elevation",
            "STP Horizontal Abduction",
            "STP Rotational"
        ],
        default=["Torso Power"],
        key="tab1_energy_plot_options"
    )

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
        peak_shoulder_frame = get_shoulder_er_max_frame(tid, handedness, cur)
        if peak_shoulder_frame is None:
            continue

        df_power["x_data"] = pd.to_numeric(df_power["x_data"], errors="coerce").fillna(0)

        df_power_window = df_power[
            (df_power["frame"] >= peak_shoulder_frame - 50) &
            (df_power["frame"] < peak_shoulder_frame)
            ]

        drive_start_frame = df_power_window[df_power_window["x_data"] < -3000]["frame"].min()
        if pd.isna(drive_start_frame):
            continue

        # Query rear knee
        cur.execute("""
            SELECT frame, x_data FROM time_series_data ts
            JOIN segments s ON ts.segment_id = s.segment_id
            JOIN categories c ON ts.category_id = c.category_id
            WHERE ts.take_id = %s AND c.category_name = 'JOINT_ANGLES' AND s.segment_name = %s
            ORDER BY frame
        """, (tid, rear_knee))
        df_knee = pd.DataFrame(cur.fetchall(), columns=["frame", "x_data"])
        if df_knee.empty:
            continue

        # --- Rear knee anchor (Pulldown-safe) ---
        df_knee["x_data"] = pd.to_numeric(df_knee["x_data"], errors="coerce")
        df_knee = df_knee.dropna(subset=["x_data"])

        if throw_type == "Pulldown":
            knee_window = df_knee[
                (df_knee["frame"] >= drive_start_frame - 100) &
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

        df_after = df_power[df_power["frame"] > max_knee_frame].copy()
        if df_after.empty:
            continue

        neg_peak_idx = df_after["x_data"].idxmin()
        neg_peak_frame = df_after.loc[neg_peak_idx, "frame"]
        df_after_peak = df_after[df_after["frame"] > neg_peak_frame]
        zero_cross = df_after_peak[df_after_peak["x_data"] >= 0]

        torso_end_frame = (int(zero_cross.iloc[0]["frame"]) - 1) if not zero_cross.empty else int(df_after["frame"].iloc[-1])
        df_segment = df_power[(df_power["frame"] >= max_knee_frame) & (df_power["frame"] <= torso_end_frame)]
        auc_total = np.trapezoid(df_segment["x_data"], df_segment["frame"])

        # Peak Arm Energy (MER-windowed max: ±20 frames around MER)
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

        # Find MER (shoulder layback) frame already computed as peak_shoulder_frame
        mer_frame = int(peak_shoulder_frame)
        MER_WINDOW = 20

        # Ball Release frame (hard right-bound)
        br_frame = get_ball_release_frame(tid, handedness, cur)

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

        # Peak within constrained window
        if windowed_energy.isna().all():
            peak_idx = int(df_arm_window["frame"].sub(mer_frame).abs().idxmin())
        else:
            peak_idx = int(windowed_energy.idxmax())

        arm_peak_frame = int(df_arm.loc[peak_idx, "frame"])
        arm_peak_value = float(df_arm.loc[peak_idx, "x_data"])

        df_to_peak = df_power[(df_power["frame"] >= max_knee_frame) & (df_power["frame"] <= arm_peak_frame)]
        auc_to_peak = np.trapezoid(df_to_peak["x_data"], df_to_peak["frame"])
        auc_pct = (auc_to_peak / auc_total * 100) if auc_total else 0

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

        if not df_stp.empty:
            df_stp["x_data"] = pd.to_numeric(df_stp["x_data"], errors="coerce").fillna(0)

            df_stp_seg = df_stp[
                (df_stp["frame"] >= max_knee_frame) &
                (df_stp["frame"] <= torso_end_frame)
            ]

            auc_stp_total = (
                np.trapezoid(df_stp_seg["x_data"], df_stp_seg["frame"])
                if not df_stp_seg.empty else np.nan
            )

            df_stp_to_peak = df_stp[
                (df_stp["frame"] >= max_knee_frame) &
                (df_stp["frame"] <= arm_peak_frame)
            ]

            auc_stp_to_peak = (
                np.trapezoid(df_stp_to_peak["x_data"], df_stp_to_peak["frame"])
                if not df_stp_to_peak.empty else np.nan
            )
        else:
            auc_stp_total = np.nan
            auc_stp_to_peak = np.nan

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

        if not df_stp_habd.empty:
            df_stp_habd["x_data"] = pd.to_numeric(df_stp_habd["x_data"], errors="coerce").fillna(0)

            df_habd_seg = df_stp_habd[
                (df_stp_habd["frame"] >= max_knee_frame) &
                (df_stp_habd["frame"] <= torso_end_frame)
            ]

            auc_stp_habd_total = (
                np.trapezoid(df_habd_seg["x_data"], df_habd_seg["frame"])
                if not df_habd_seg.empty else np.nan
            )

            df_habd_to_peak = df_stp_habd[
                (df_stp_habd["frame"] >= max_knee_frame) &
                (df_stp_habd["frame"] <= arm_peak_frame)
            ]

            auc_stp_habd_to_peak = (
                np.trapezoid(df_habd_to_peak["x_data"], df_habd_to_peak["frame"])
                if not df_habd_to_peak.empty else np.nan
            )
        else:
            auc_stp_habd_total = np.nan
            auc_stp_habd_to_peak = np.nan

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
        """, (tid, shoulder_stp_segment))

        df_stp_rot = pd.DataFrame(cur.fetchall(), columns=["frame", "x_data"])

        if not df_stp_rot.empty:
            df_stp_rot["x_data"] = pd.to_numeric(df_stp_rot["x_data"], errors="coerce").fillna(0)

            df_rot_seg = df_stp_rot[
                (df_stp_rot["frame"] >= max_knee_frame) &
                (df_stp_rot["frame"] <= torso_end_frame)
            ]

            auc_stp_rot_total = (
                np.trapezoid(df_rot_seg["x_data"], df_rot_seg["frame"])
                if not df_rot_seg.empty else np.nan
            )

            df_rot_to_peak = df_stp_rot[
                (df_stp_rot["frame"] >= max_knee_frame) &
                (df_stp_rot["frame"] <= arm_peak_frame)
            ]

            auc_stp_rot_to_peak = (
                np.trapezoid(df_rot_to_peak["x_data"], df_rot_to_peak["frame"])
                if not df_rot_to_peak.empty else np.nan
            )
        else:
            auc_stp_rot_total = np.nan
            auc_stp_rot_to_peak = np.nan

        rows.append({
            "take_id": tid,
            "Session Date": take_date.strftime("%Y-%m-%d"),
            "Throw Type": throw_type,
            "Max Knee Flexion Frame": int(max_knee_frame),
            "Velocity": velo,
            "AUC (Drive → 0)": round(float(auc_total), 2),
            "AUC (Drive → Peak Arm Energy)": round(float(auc_to_peak), 2),
            "Peak Arm Energy": round(float(arm_peak_value), 2),
            "% Total Energy Into Layback": round(float(auc_pct), 1),
            "STP Elevation AUC (Drive → 0)": round(float(auc_stp_total), 2) if pd.notna(auc_stp_total) else np.nan,
            "STP Elevation AUC (Drive → Peak Arm Energy)": round(float(auc_stp_to_peak), 2) if pd.notna(auc_stp_to_peak) else np.nan,
            "STP HorizAbd AUC (Drive → 0)": round(float(auc_stp_habd_total), 2) if pd.notna(auc_stp_habd_total) else np.nan,
            "STP HorizAbd AUC (Drive → Peak Arm Energy)": round(float(auc_stp_habd_to_peak), 2) if pd.notna(auc_stp_habd_to_peak) else np.nan,
            "STP Rotational AUC (Drive → 0)": round(float(auc_stp_rot_total), 2) if pd.notna(auc_stp_rot_total) else np.nan,
            "STP Rotational AUC (Drive → Peak Arm Energy)": round(float(auc_stp_rot_to_peak), 2) if pd.notna(auc_stp_rot_to_peak) else np.nan,
        })

# Guard for empty rows
if rows:
    df_tab1 = pd.DataFrame(rows)

    # ---- Exclude Takes (Tab 1) ----
    df_tab1 = df_tab1.copy()

    # Normalize Velocity column name if pitch_velo is used
    if "pitch_velo" in df_tab1.columns and "Velocity" not in df_tab1.columns:
        df_tab1 = df_tab1.rename(columns={"pitch_velo": "Velocity"})

    if not df_tab1.empty:
        # Build readable labels identical in style to Tab 3
        def make_label_tab1(row):
            try:
                auc0 = float(row["AUC (Drive → 0)"])
            except Exception:
                auc0 = float("nan")
            try:
                auc_peak = float(row["AUC (Drive → Peak Arm Energy)"])
            except Exception:
                auc_peak = float("nan")
            return f"{row['Session Date']} | {row['Velocity']} mph | {auc0:.2f} → {auc_peak:.2f}"

        df_tab1["label"] = df_tab1.apply(make_label_tab1, axis=1)

        exclude_labels_tab1 = st.multiselect(
            "Exclude Takes",
            options=df_tab1["label"].tolist(),
            key="exclude_takes_tab1"
        )

        # Map labels back to take_ids
        exclude_take_ids_tab1 = df_tab1[
            df_tab1["label"].isin(exclude_labels_tab1)
        ]["take_id"].tolist()

        # Filter out excluded takes
        df_tab1 = df_tab1[~df_tab1["take_id"].isin(exclude_take_ids_tab1)]

    # Prepare regressions
    df_tab1["Velocity"] = pd.to_numeric(df_tab1["Velocity"], errors="coerce")
    df_tab1["AUC (Drive → 0)"] = pd.to_numeric(df_tab1["AUC (Drive → 0)"], errors="coerce")
    df_tab1["AUC (Drive → Peak Arm Energy)"] = pd.to_numeric(df_tab1["AUC (Drive → Peak Arm Energy)"], errors="coerce")
    df_tab1 = df_tab1.dropna(subset=["Velocity", "AUC (Drive → 0)", "AUC (Drive → Peak Arm Energy)"])
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
    # --- Normalize Throw Type: ensure always present and filled ---
    df_tab1["Throw Type"] = df_tab1["Throw Type"].fillna("Mound")
else:
    st.warning("No valid data found for this pitcher/date.")
    st.stop()
# --- Date-based color map for Tab 1 ---
date_color_cycle = px.colors.qualitative.Bold

fig = go.Figure()
# For marker symbols per metric
metric_symbol_map = {
    "Torso Power": ["circle", "triangle-up"],
    "STP Elevation": ["diamond"],
    "STP Horizontal Abduction": ["square"],
    "STP Rotational": ["pentagon"],
}
# For regression line dashes per metric
metric_dash_map = {
    "Torso Power": ["dash", "dot"],
    "STP Elevation": ["longdash"],
    "STP Horizontal Abduction": [None],
    "STP Rotational": ["dot"],
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
    if len(sub) < 2:
        continue
    color = date_color_cycle[i % len(date_color_cycle)]
    x = sub["Velocity"]
    for energy_plot_option in energy_plot_options:
        # Torso Power: plot both AUC → 0 and AUC → Peak
        if energy_plot_option == "Torso Power":
            # --- AUC Drive → 0 ---
            y0 = sub["AUC (Drive → 0)"]
            if y0.size >= 2:
                slope0, intercept0, r0, _, _ = linregress(x, y0)
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit0 = slope0 * x_fit + intercept0
                fig.add_trace(go.Scatter(
                    x=x, y=y0, mode="markers",
                    marker=dict(color=color, symbol=metric_symbol_map[energy_plot_option][0]),
                    name=f"{date} | {throw_type} | {metric_trace_names[energy_plot_option][0]}",
                    hovertext=[f"{date} | {throw_type} | {metric_trace_names[energy_plot_option][0]}"] * len(x),
                    hovertemplate="%{hovertext}<br>Velocity: %{x:.1f} mph<br>Value: %{y:.2f}<extra></extra>",
                ))
                fig.add_trace(go.Scatter(
                    x=x_fit, y=y_fit0, mode="lines",
                    line=dict(color=color, dash=metric_dash_map[energy_plot_option][0]),
                    name=f"R²={r0**2:.2f}",
                    hovertext=[f"{date} | {throw_type} | R²={r0**2:.2f}"] * len(x_fit),
                    hovertemplate="%{hovertext}<br>Velocity: %{x:.1f} mph<br>Predicted: %{y:.2f}<extra></extra>",
                ))
            # --- AUC Drive → Peak Arm Energy ---
            y1 = sub["AUC (Drive → Peak Arm Energy)"]
            if y1.size >= 2:
                slope1, intercept1, r1, _, _ = linregress(x, y1)
                y_fit1 = slope1 * x_fit + intercept1
                fig.add_trace(go.Scatter(
                    x=x, y=y1, mode="markers",
                    marker=dict(color=color, symbol=metric_symbol_map[energy_plot_option][1]),
                    name=f"{date} | {throw_type} | {metric_trace_names[energy_plot_option][1]}",
                    hovertext=[f"{date} | {throw_type} | {metric_trace_names[energy_plot_option][1]}"] * len(x),
                    hovertemplate="%{hovertext}<br>Velocity: %{x:.1f} mph<br>Value: %{y:.2f}<extra></extra>",
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
priority_cols = ["Session Date", "Throw Type", "Max Knee Flexion Frame"]
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
        df_knee_pre = df_knee[df_knee["frame"] < drive_start_frame]
        if df_knee_pre.empty:
            continue
        max_knee_frame = df_knee_pre.loc[df_knee_pre["x_data"].idxmin(), "frame"]
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
    # Two columns: controls (left), charts (right)
    left, right = st.columns([0.4, 1.4], vertical_alignment="top")
    with left:
        # --- Select pitcher ---
        selected_pitcher_comp = st.selectbox("Select Pitcher", pitchers, key="comp_pitcher")
        # --- Throw type selection (default = Mound) ---
        cur.execute("""
            SELECT DISTINCT COALESCE(t.throw_type, 'Mound') AS throw_type
            FROM takes t
            JOIN athletes a ON t.athlete_id = a.athlete_id
            WHERE a.athlete_name = %s
            ORDER BY throw_type
        """, (selected_pitcher_comp,))
        throw_type_options_comp = [r[0] for r in cur.fetchall()] or ["Mound", "Pulldown"]
        default_throw_types_comp = ["Mound"] if "Mound" in throw_type_options_comp else [throw_type_options_comp[0]]

        selected_throw_types_comp = st.multiselect(
            "Throw Type(s)",
            options=throw_type_options_comp,
            default=default_throw_types_comp,
            key="throw_types_comp"
        )

        if not selected_throw_types_comp:
            selected_throw_types_comp = default_throw_types_comp

        display_mode_tab2 = st.radio(
            "Display Mode",
            ["Grouped Average", "Individual Throws"],
            index=0,
            horizontal=True,
            key="tab2_display_mode"
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

        # --- Get min and max velocity for session1 (filtered by throw type) ---
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
            st.warning(f"No {', '.join(selected_throw_types_comp)} throws found for {session1_date}")
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

        # --- Get min and max velocity for session2 (filtered by throw type) ---
        placeholders_tt = ",".join(["%s"] * len(selected_throw_types_comp))
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
            st.warning(f"No {', '.join(selected_throw_types_comp)} throws found for {session2_date}")
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

        # Shoulder component selector
        component = st.selectbox("Shoulder Axis", ["Horizontal Abduction/Adduction (X)", "Abduction/Adduction (Y)", "Internal/External Rotation (Z)"], index=2, key="shoulder_component")
        comp_col = {"Horizontal Abduction/Adduction (X)": "x_data", "Abduction/Adduction (Y)": "y_data", "Internal/External Rotation (Z)": "z_data"}[component]
        use_abs = st.checkbox("Absolute Value", value=True, key="shoulder_abs")

        # --- Reference group controls ---
        include_ref = st.checkbox("Reference Group", value=True, key="ref_include")

        # Set default reference group to 95+ and All Pitchers
        default_ref_mode = "All Pitchers"
        default_ref_velo_min = 95.0
        # Use default only on first load
        ref_mode = st.radio(
            "Reference Population",
            ["Selected Pitcher", "All Pitchers"],
            index=0 if st.session_state.get("ref_mode") == "Selected Pitcher" else 1,
            horizontal=True, key="ref_mode"
        )

        # --- Compute default bounds for the velocity slider (for reference group) ---
        if ref_mode == "Selected Pitcher":
            cur.execute("""
                SELECT MIN(t.pitch_velo), MAX(t.pitch_velo)
                FROM takes t
                JOIN athletes a ON t.athlete_id = a.athlete_id
                WHERE a.athlete_name = %s
            """, (selected_pitcher_comp,))
        else:
            cur.execute("SELECT MIN(pitch_velo), MAX(pitch_velo) FROM takes;")
        rvmin, rvmax = cur.fetchone() or (None, None)
        if rvmin is None or rvmax is None:
            rvmin, rvmax = 70.0, 100.0
        rvmin = float(f"{rvmin:.1f}")
        rvmax = float(f"{rvmax:.1f}")
        if rvmin >= rvmax:
            rvmax = rvmin + 0.1

        # Set default for ref_velo if not present or incomplete

        velo_range_ref = st.slider(
            "Velocity Range for Reference Group",
            min_value=int(rvmin), max_value=int(rvmax),
            value=(int(st.session_state.get("ref_velo", (default_ref_velo_min, rvmax))[0]), int(st.session_state.get("ref_velo", (default_ref_velo_min, rvmax))[1])),
            step=1,
            key="ref_velo"
        )

        # --- Show pitchers included in reference group ---
        if include_ref:
            _, _, ref_pitchers = load_reference_curves_player_mean(
                ref_mode, selected_pitcher_comp,
                velo_range_ref[0], velo_range_ref[1],
                comp_col, use_abs,
                throw_types=["Mound"]
            )
            st.markdown("**Pitchers in Reference Group:**")
            if ref_pitchers:
                if len(ref_pitchers) == 1:
                    st.info("Only one pitcher found in reference group for the selected settings.")
                else:
                    st.code("\n".join(ref_pitchers), language="text")  # Show pitchers in a code box
            else:
                st.info("No pitchers found in reference group for the selected settings.")




    def load_and_interpolate_curves(date, velo_min, velo_max, comp_col, use_abs, throw_types=None):
        """
        Returns:
            shoulder_curves (list[np.ndarray])  # one per take, length 100
            torso_curves    (list[np.ndarray])  # one per take, length 100
            peak_arm_time_pcts (list[float])    # one per take
        """
        if date is not None:
            if throw_types:
                placeholders_tt = ",".join(["%s"] * len(throw_types))
                cur.execute(f"""
                    SELECT t.take_id, t.pitch_velo
                    FROM takes t
                    JOIN athletes a ON t.athlete_id = a.athlete_id
                    WHERE a.athlete_name = %s AND t.take_date = %s
                      AND COALESCE(t.throw_type, 'Mound') IN ({placeholders_tt})
                      AND t.pitch_velo BETWEEN %s AND %s
                    ORDER BY t.file_name
                """, (selected_pitcher_comp, date, *throw_types, velo_min, velo_max))
            else:
                cur.execute("""
                    SELECT t.take_id, t.pitch_velo
                    FROM takes t
                    JOIN athletes a ON t.athlete_id = a.athlete_id
                    WHERE a.athlete_name = %s AND t.take_date = %s
                    AND t.pitch_velo BETWEEN %s AND %s
                    ORDER BY t.file_name
                """, (selected_pitcher_comp, date, velo_min, velo_max))
        else:
            if throw_types:
                placeholders_tt = ",".join(["%s"] * len(throw_types))
                cur.execute(f"""
                    SELECT t.take_id, t.pitch_velo
                    FROM takes t
                    JOIN athletes a ON t.athlete_id = a.athlete_id
                    WHERE COALESCE(t.throw_type, 'Mound') IN ({placeholders_tt})
                      AND t.pitch_velo BETWEEN %s AND %s
                    ORDER BY t.file_name
                """, (*throw_types, velo_min, velo_max))
            else:
                cur.execute("""
                    SELECT t.take_id, t.pitch_velo
                    FROM takes t
                    JOIN athletes a ON t.athlete_id = a.athlete_id
                    WHERE t.pitch_velo BETWEEN %s AND %s
                    ORDER BY t.file_name
                """, (velo_min, velo_max))

        takes = cur.fetchall()
        if not takes:
            return None, None, None

        shoulder_curves, torso_curves = [], []
        peak_arm_time_pcts = []

        for tid, _velo in takes:
            # Torso power (to find drive start and for torso curve)
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
            drive_start_frame = df_power[df_power["x_data"] < -3000]["frame"].min()
            if pd.isna(drive_start_frame):
                continue

            # Rear knee (to find max pre-drive knee flexion frame)
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

        if not shoulder_curves or not torso_curves:
            return None, None, None

        return shoulder_curves, torso_curves, peak_arm_time_pcts

    # ---- Compute curves and render charts ----
    with right:
        # --- 1) Loop per throw type for session 1 ---
        curves_s1_by_type = {}
        peak_arm_time_pct_by_type = {}
        for tt in selected_throw_types_comp:
            s1_sh_curves, s1_to_curves, s1_peak_arm_times = load_and_interpolate_curves(
                session1_date, velo_range1[0], velo_range1[1], comp_col, use_abs,
                throw_types=[tt]
            )
            curves_s1_by_type[tt] = dict(shoulder=s1_sh_curves, torso=s1_to_curves)
            peak_arm_time_pct_by_type[tt] = s1_peak_arm_times

        # --- 2) Loop per throw type for session 2 ---
        curves_s2_by_type = {}
        peak_arm_time_pct_by_type_2 = {}
        for tt in selected_throw_types_comp:
            s2_sh_curves, s2_to_curves, s2_peak_arm_times = load_and_interpolate_curves(
                session2_date, velo_range2[0], velo_range2[1], comp_col, use_abs,
                throw_types=[tt]
            )
            curves_s2_by_type[tt] = dict(shoulder=s2_sh_curves, torso=s2_to_curves)
            peak_arm_time_pct_by_type_2[tt] = s2_peak_arm_times

        mean_ref_shoulder, mean_ref_torso = None, None
        if include_ref and display_mode_tab2 == "Grouped Average":
            mean_ref_shoulder, mean_ref_torso, _ = load_reference_curves_player_mean(
                ref_mode, selected_pitcher_comp,
                velo_range_ref[0], velo_range_ref[1],
                comp_col, use_abs,
                throw_types=["Mound"]
            )

        # Color encodes throw type (solid lines)
        throw_type_color = {
            "Mound": "#1f77b4",     # blue
            "Pulldown": "#d62728"   # red
        }

        # Only plot if at least one session has at least one valid curve
        has_any_curve = (
            any(v["shoulder"] not in (None, []) for v in curves_s1_by_type.values())
            and any(v["shoulder"] not in (None, []) for v in curves_s2_by_type.values())
        )
        if has_any_curve:
            # ===================== SHOULDER =====================
            fig_shoulder = go.Figure()
            # Plot session 1
            for tt, v in curves_s1_by_type.items():
                curves = v["shoulder"]
                if not curves:
                    continue

                color_tt = throw_type_color.get(tt, "#444")

                if display_mode_tab2 == "Grouped Average":
                    mean_curve = np.nanmean(np.vstack(curves), axis=0)
                    fig_shoulder.add_trace(go.Scatter(
                        x=interp_points, y=mean_curve,
                        mode="lines",
                        line=dict(color=color_tt, width=3),
                        name=f"{session1_date} | {tt}",
                        legendgroup=f"s1_{tt}"
                    ))
                else:
                    for i, c in enumerate(curves):
                        fig_shoulder.add_trace(go.Scatter(
                            x=interp_points, y=c,
                            mode="lines",
                            line=dict(color=color_tt, width=2),
                            opacity=0.35,
                            name=f"{session1_date} | {tt} | Pitch {i+1}",
                            legendgroup=f"s1_{tt}"
                        ))

            # Plot session 2
            for tt, v in curves_s2_by_type.items():
                curves = v["shoulder"]
                if not curves:
                    continue

                color_tt = throw_type_color.get(tt, "#444")

                if display_mode_tab2 == "Grouped Average":
                    mean_curve = np.nanmean(np.vstack(curves), axis=0)
                    fig_shoulder.add_trace(go.Scatter(
                        x=interp_points, y=mean_curve,
                        mode="lines",
                        line=dict(color=color_tt, width=3),
                        name=f"{session2_date} | {tt}",
                        legendgroup=f"s2_{tt}"
                    ))
                else:
                    for i, c in enumerate(curves):
                        fig_shoulder.add_trace(go.Scatter(
                            x=interp_points, y=c,
                            mode="lines",
                            line=dict(color=color_tt, width=2),
                            opacity=0.35,
                            name=f"{session2_date} | {tt} | Pitch {i+1}",
                            legendgroup=f"s2_{tt}"
                        ))
            # Reference group
            if mean_ref_shoulder is not None:
                fig_shoulder.add_trace(go.Scatter(
                    x=interp_points, y=mean_ref_shoulder,
                    mode='lines', name="Reference (mean)",
                    line=dict(color='red')
                ))

            # y-bounds for vertical markers
            yvals_sh = []
            for v in curves_s1_by_type.values():
                curves = v["shoulder"]
                if curves:
                    for c in curves:
                        yvals_sh.extend(list(c))
            for v in curves_s2_by_type.values():
                curves = v["shoulder"]
                if curves:
                    for c in curves:
                        yvals_sh.extend(list(c))
            if mean_ref_shoulder is not None:
                yvals_sh += list(mean_ref_shoulder)
            y0_sh, y1_sh = float(np.nanmin(yvals_sh)), float(np.nanmax(yvals_sh))

            # Legend entries for vertical lines
            fig_shoulder.add_trace(go.Scatter(
                x=[None], y=[None], mode='lines',
                line=dict(dash='dot', color='gray'), name="Max Layback"
            ))
            fig_shoulder.add_trace(go.Scatter(
                x=[None], y=[None], mode='lines',
                line=dict(dash='dot', color='green'), name="Peak Arm Energy"
            ))

            # Draw Max Layback (x=50) and Peak Arm Energy for session 1 (use mean across all session1 curves)
            shapes_list = [dict(type="line", x0=50, x1=50, y0=y0_sh, y1=y1_sh, line=dict(dash="dot", color="gray"))]
            # For peak arm energy, use mean across all session1 curves
            peak_arm_time_pct = None
            all_peak = []
            for tt in selected_throw_types_comp:
                vals = peak_arm_time_pct_by_type.get(tt)
                if vals:
                    all_peak.extend([v for v in vals if v is not None])
            if all_peak:
                peak_arm_time_pct = float(np.nanmean(all_peak))
            if peak_arm_time_pct is not None:
                shapes_list.append(dict(type="line", x0=peak_arm_time_pct, x1=peak_arm_time_pct,
                                        y0=y0_sh, y1=y1_sh, line=dict(dash="dot", color="green")))
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
            # Plot session 1
            for tt, v in curves_s1_by_type.items():
                curves = v["torso"]
                if not curves:
                    continue

                color_tt = throw_type_color.get(tt, "#444")

                if display_mode_tab2 == "Grouped Average":
                    mean_curve = np.nanmean(np.vstack(curves), axis=0)
                    fig_torso.add_trace(go.Scatter(
                        x=interp_points, y=mean_curve,
                        mode="lines",
                        line=dict(color=color_tt, width=3),
                        name=f"{session1_date} | {tt}",
                        legendgroup=f"s1_{tt}"
                    ))
                else:
                    for i, c in enumerate(curves):
                        fig_torso.add_trace(go.Scatter(
                            x=interp_points, y=c,
                            mode="lines",
                            line=dict(color=color_tt, width=2),
                            opacity=0.35,
                            name=f"{session1_date} | {tt} | Pitch {i+1}",
                            legendgroup=f"s1_{tt}"
                        ))

            # Plot session 2
            for tt, v in curves_s2_by_type.items():
                curves = v["torso"]
                if not curves:
                    continue

                color_tt = throw_type_color.get(tt, "#444")

                if display_mode_tab2 == "Grouped Average":
                    mean_curve = np.nanmean(np.vstack(curves), axis=0)
                    fig_torso.add_trace(go.Scatter(
                        x=interp_points, y=mean_curve,
                        mode="lines",
                        line=dict(color=color_tt, width=3),
                        name=f"{session2_date} | {tt}",
                        legendgroup=f"s2_{tt}"
                    ))
                else:
                    for i, c in enumerate(curves):
                        fig_torso.add_trace(go.Scatter(
                            x=interp_points, y=c,
                            mode="lines",
                            line=dict(color=color_tt, width=2),
                            opacity=0.35,
                            name=f"{session2_date} | {tt} | Pitch {i+1}",
                            legendgroup=f"s2_{tt}"
                        ))
            # Reference group
            if mean_ref_torso is not None:
                fig_torso.add_trace(go.Scatter(
                    x=interp_points, y=mean_ref_torso,
                    mode='lines', name="Reference (mean)",
                    line=dict(color='red')
                ))

            yvals_to = []
            for v in curves_s1_by_type.values():
                curves = v["torso"]
                if curves:
                    for c in curves:
                        yvals_to.extend(list(c))
            for v in curves_s2_by_type.values():
                curves = v["torso"]
                if curves:
                    for c in curves:
                        yvals_to.extend(list(c))
            if mean_ref_torso is not None:
                yvals_to += list(mean_ref_torso)
            y0_to, y1_to = float(np.nanmin(yvals_to)), float(np.nanmax(yvals_to))

            fig_torso.add_trace(go.Scatter(
                x=[None], y=[None], mode='lines',
                line=dict(dash='dot', color='gray'), name="Max Layback"
            ))
            fig_torso.add_trace(go.Scatter(
                x=[None], y=[None], mode='lines',
                line=dict(dash='dot', color='green'), name="Peak Arm Energy"
            ))

            shapes_list_torso = [dict(type="line", x0=50, x1=50, y0=y0_to, y1=y1_to, line=dict(dash="dot", color="gray"))]
            # For peak arm energy, use mean across all session1 curves
            peak_arm_time_pct_to = None
            all_peak_to = []
            for tt in selected_throw_types_comp:
                vals = peak_arm_time_pct_by_type.get(tt)
                if vals:
                    all_peak_to.extend([v for v in vals if v is not None])
            if all_peak_to:
                peak_arm_time_pct_to = float(np.nanmean(all_peak_to))
            if peak_arm_time_pct_to is not None:
                shapes_list_torso.append(dict(type="line", x0=peak_arm_time_pct_to, x1=peak_arm_time_pct_to,
                                              y0=y0_to, y1=y1_to, line=dict(dash="dot", color="green")))
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
                SELECT t.take_id, t.pitch_velo, a.handedness
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
                SELECT t.take_id, t.pitch_velo, a.handedness
                FROM takes t
                JOIN athletes a ON t.athlete_id = a.athlete_id
                WHERE a.athlete_name = %s
                  AND t.take_date IN ({placeholders_dates})
                  AND COALESCE(t.throw_type, 'Mound') IN ({placeholders_tt})
                ORDER BY t.file_name
            """, (pitcher, *selected_dates, *selected_throw_types_010))
            take_rows_010.extend(cur.fetchall())


    rows_010 = []
    for take_id_010, pitch_velo_010, handedness_local in take_rows_010:
        # Determine handedness-specific segment names for this take
        if handedness_local == "R":
            shoulder_velo_segment = "RT_SHOULDER_ANGULAR_VELOCITY"
            hip_velo_segment   = "RT_HIP_ANGULAR_VELOCITY"
            knee_velo_segment  = "RT_KNEE_ANGULAR_VELOCITY"
            ankle_velo_segment = "RT_ANKLE_ANGULAR_VELOCITY"
            lead_knee_velo_segment = "LT_KNEE_ANGULAR_VELOCITY"
            elbow_velo_segment = "RT_ELBOW_ANGULAR_VELOCITY"
            shank_seg_name = "LSK"
            hand_segment = "RHA"
        else:
            shoulder_velo_segment = "LT_SHOULDER_ANGULAR_VELOCITY"
            hip_velo_segment   = "LT_HIP_ANGULAR_VELOCITY"
            knee_velo_segment  = "LT_KNEE_ANGULAR_VELOCITY"
            ankle_velo_segment = "LT_ANKLE_ANGULAR_VELOCITY"
            lead_knee_velo_segment = "RT_KNEE_ANGULAR_VELOCITY"
            elbow_velo_segment = "LT_ELBOW_ANGULAR_VELOCITY"
            shank_seg_name = "RSK"
            hand_segment = "LHA"

        # Get ball release frame for this take
        br_frame_010 = get_ball_release_frame(take_id_010, handedness_local, cur)

        # --- Compute peak glove-side knee height frame before BR ---
        knee_peak_frame_pre_br_010 = None
        if br_frame_010 is not None:
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
            velo_segment = shoulder_velo_segment
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
        """, (take_id_010, velo_segment, velo_segment, velo_segment, velo_segment))
        data = cur.fetchall()
        if not data:
            continue
        # Convert to array
        arr = np.array(data, dtype=float)
        frames = arr[:, 0].astype(int)
        arr = arr[:, 1:]   # now x = arr[:,0], y = arr[:,1], z = arr[:,2]

        # Determine the shoulder IR velocity peak frame (for time restriction)
        z_vals = arr[:, 2]
        sh_ir_peak_idx = np.nanargmax(np.abs(z_vals))
        sh_ir_peak_frame = frames[sh_ir_peak_idx]
        if selected_metric_010 == "Max Shoulder Internal Rotation Velocity":
            z_vals = arr[:, 2]
            # ---------------------------------------------
            # ER-centered windowing for throwing arm metrics
            # ---------------------------------------------
            if sh_er_max_frame_010 is not None:
                er_frame = int(sh_er_max_frame_010)
                win_mask = (
                    (frames >= er_frame - 50) &
                    (frames <= er_frame + 50)
                )
                # Only apply window if it yields data
                if np.any(win_mask):
                    frames = frames[win_mask]
                    z_vals = z_vals[win_mask]
            if handedness_local == "R":
                # RHP IR = most positive
                vals = np.array([np.nanmax(z_vals)])
            else:
                # LHP IR = most negative
                vals = np.array([np.nanmin(z_vals)])
        elif selected_metric_010 == "Max Shoulder External Rotation Velocity":
            z_vals = arr[:, 2]
            # ---------------------------------------------
            # ER-centered windowing for throwing arm metrics
            # ---------------------------------------------
            if sh_er_max_frame_010 is not None:
                er_frame = int(sh_er_max_frame_010)
                win_mask = (
                    (frames >= er_frame - 50) &
                    (frames <= er_frame + 30)
                )
                # Only apply window if it yields data
                if np.any(win_mask):
                    frames = frames[win_mask]
                    z_vals = z_vals[win_mask]
            if handedness_local == "R":
                # RHP ER = most negative
                vals = np.array([np.nanmin(z_vals)])
            else:
                # LHP ER = most positive
                vals = np.array([np.nanmax(z_vals)])
        elif selected_metric_010 == "Max Elbow Extension Velocity":
            x_vals = arr[:, 0]
            # ---------------------------------------------
            # ER-centered windowing for throwing arm metrics
            # ---------------------------------------------
            if sh_er_max_frame_010 is not None:
                er_frame = int(sh_er_max_frame_010)
                win_mask = (
                    (frames >= er_frame - 50) &
                    (frames <= er_frame + 30)
                )
                # Only apply window if it yields data
                if np.any(win_mask):
                    frames = frames[win_mask]
                    x_vals = x_vals[win_mask]
            # Elbow extension velocity: always the most negative value (both handedness)
            vals = np.array([np.nanmin(x_vals)])
        elif selected_metric_010 == "Max Torso Angular Velocity":
            if torso_axis == "X (Extension)":
                vals = np.array([np.nanmax(arr[:, 0])])

            elif torso_axis == "X (Flexion)":
                vals = np.array([np.nanmin(arr[:, 0])])

            elif torso_axis == "Y":
                y_vals = arr[:, 1]

                # Right-handed: most negative Y
                if handedness_local == "R":
                    vals = np.array([np.nanmin(y_vals)])

                # Left-handed: most positive Y
                else:
                    vals = np.array([np.nanmax(y_vals)])

            elif torso_axis == "Z":
                z_vals = arr[:, 2]
                if handedness_local == "R":
                    # RHP: most positive Z
                    vals = np.array([np.nanmax(z_vals)])
                else:
                    # LHP: most negative Z
                    vals = np.array([np.nanmin(z_vals)])

            else:
                vals = np.array([])
        elif selected_metric_010 == "Max Torso–Pelvis Angular Velocity":

            # X-axis split into Extension (positive) and Flexion (negative)
            if torso_pelvis_axis == "X (Extension)":
                x_vals = arr[:, 0]
                raw_val = np.nanmax(x_vals)
                vals = np.array([raw_val])


            elif torso_pelvis_axis == "X (Flexion)":
                x_vals = arr[:, 0]
                raw_val = np.nanmin(x_vals)
                vals = np.array([abs(raw_val)])  # normalize flexion to positive

            # Y-axis: handedness-aware, pre-BR windowed logic
            elif torso_pelvis_axis == "Y":
                # ---------------------------------------------
                # Pre-BR windowed torso–pelvis Y angular velocity
                # Glove-side peak, normalized to POSITIVE
                # Window: 50 frames before Ball Release
                # ---------------------------------------------
                y_vals = arr[:, 1]
                frame_vals = frames

                if br_frame_010 is not None:
                    start = br_frame_010 - 50
                    end = br_frame_010
                    mask = (frame_vals >= start) & (frame_vals <= end)
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

            # Z-axis: always return the positive maxima for the Z component, but restrict to values before Shoulder ER Max frame
            elif torso_pelvis_axis == "Z":
                z_vals = arr[:, 2]

                # Reapply window: ONLY use values before Shoulder ER Max frame
                if sh_er_max_frame_010 is not None:
                    mask = frames < sh_er_max_frame_010
                    z_window = z_vals[mask]
                else:
                    z_window = z_vals

                # Fallback if window empty
                if z_window.size == 0:
                    z_window = z_vals

                # Return the positive maxima within this window
                vals = np.array([np.nanmax(z_window)])

            else:
                vals = np.array([])
        elif selected_metric_010 == "Max Pelvis Angular Velocity":
            # X-axis: always most negative for both handedness
            if pelvis_axis == "X":
                vals = np.array([np.nanmin(arr[:, 0])])
            # Z-axis: use absolute maxima regardless of handedness
            elif pelvis_axis == "Z":
                z_vals = arr[:, 2]
                idx = np.nanargmax(np.abs(z_vals))
                vals = np.array([z_vals[idx]])
            else:
                vals = np.array([])
        elif selected_metric_010 == "Max Pelvis Angle (Z)":
            z_vals = arr[:, 2]

            if handedness_local == "R":
                # RHP → most negative
                raw_val = np.nanmin(z_vals)
            else:
                # LHP → (peak value) - 90
                raw_val = np.nanmax(z_vals) - 90

            # Normalize for UI
            vals = np.array([abs(raw_val)])
        elif selected_metric_010 == "Max COM Velocity":
            if com_axis == "X":
                vals = arr[:, 0]
            elif com_axis == "Y":
                y_vals = arr[:, 1]

                # Restrict to frames before max shoulder ER (layback), if available
                if sh_er_max_frame_010 is not None:
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
                vals = np.array([np.nanmin(z_vals)])  # COM Z = most negative max for both handedness
            else:
                vals = np.array([])
        elif selected_metric_010 == "Max Hip Extension Velocity":
            x_vals = arr[:, 0]

            # Window: 100 frames before Shoulder ER Max → Shoulder ER Max
            if sh_er_max_frame_010 is not None:
                start = sh_er_max_frame_010 - 100
                end = sh_er_max_frame_010
                mask = (frames >= start) & (frames <= end)
                window_vals = x_vals[mask]
            else:
                window_vals = x_vals

            # Fallback if window empty
            if window_vals.size == 0:
                window_vals = x_vals

            # Most negative maxima (true peak extension)
            vals = np.array([np.nanmin(window_vals)])
        elif selected_metric_010 == "Max Knee Extension Velocity":
            x_vals = arr[:, 0]  # knee extension velocity typically in X

            # Restrict window to frames BEFORE Shoulder ER Max
            if sh_er_max_frame_010 is not None:
                mask = frames < sh_er_max_frame_010
                window_vals = x_vals[mask]
            else:
                window_vals = x_vals

            # Fallback if window empty
            if window_vals.size == 0:
                window_vals = x_vals

            # Most positive maxima before ER max
            vals = np.array([np.nanmax(window_vals)])
        elif selected_metric_010 == "Max Lead Knee Extension Velocity":
            # Simplified window: BR ± 25 frames
            if br_frame_010 is not None:
                start = max(0, br_frame_010 - 25)
                end = br_frame_010 + 25
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

            # Restrict window to frames BEFORE Shoulder ER Max
            if sh_er_max_frame_010 is not None:
                mask = frames < sh_er_max_frame_010
                window_vals = x_vals[mask]
            else:
                window_vals = x_vals

            # Fallback if window empty
            if window_vals.size == 0:
                window_vals = x_vals

            # Most negative maxima before ER max (true ankle extension peak)
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
            # Pelvis anterior tilt = pelvis X-angle (positive tilt) at BR frame
            x_vals = arr[:, 0]
            frame_vals = frames

            if br_frame_010 is not None and frame_vals.size > 0:
                # nearest-frame match
                idx = np.argmin(np.abs(frame_vals - br_frame_010))
                pel_x_at_br = x_vals[idx]
                vals = np.array([pel_x_at_br])
            else:
                vals = np.array([np.nan])
        elif selected_metric_010 == "Max Torso–Pelvis Angle (Z)":
            z_vals = arr[:, 2]
            if handedness_local == "R":
                # Right-handed: most negative Z
                vals = np.array([np.nanmin(z_vals)])
            else:
                # Left-handed: most positive Z
                vals = np.array([np.nanmax(z_vals)])
        elif selected_metric_010 == "Max Torso–Pelvis Angle (X-Extended)":
            x_vals = arr[:, 0]

            # Case 1 — if any positive values exist → choose the most positive
            pos_vals = x_vals[x_vals > 0]
            if pos_vals.size > 0:
                raw_val = np.nanmax(pos_vals)
            else:
                # Case 2 — all values are negative → choose the value closest to 0 (largest)
                raw_val = np.nanmax(x_vals)

            # Normalize for UI
            vals = np.array([abs(raw_val)])

        elif selected_metric_010 == "Max Torso–Pelvis Angle (X-Flexed)":
            # Always take the most negative value (both handedness)
            x_vals = arr[:, 0]
            raw_val = np.nanmin(x_vals)

            # Normalize for UI (absolute)
            vals = np.array([abs(raw_val)])
        elif selected_metric_010 == "Max Torso–Pelvis Angle (Y-Glove Side)":
            y_vals = arr[:, 1]
            if handedness_local == "R":
                # Right-handed: Glove Side = most negative Y
                vals = np.array([np.nanmin(y_vals)])
            else:
                # Left-handed: Glove Side = most positive Y
                vals = np.array([np.nanmax(y_vals)])

        elif selected_metric_010 == "Max Torso–Pelvis Angle (Y-Arm Side)":
            y_vals = arr[:, 1]
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
            if sh_er_max_frame_010 is not None:
                er_frame = int(sh_er_max_frame_010)
                win_mask = (
                    (frames >= er_frame - 50) &
                    (frames <= er_frame + 30)
                )
                # Only apply window if it yields data
                if np.any(win_mask):
                    frames = frames[win_mask]
                    x_vals = x_vals[win_mask]
            # Always take the most negative value (both handedness)
            vals = np.array([np.nanmin(x_vals)])
        elif selected_metric_010 == "Max Shoulder Horizontal Abduction":
            x_vals = arr[:, 0]

            if handedness_local == "R":
                # Right-handed: use most negative
                raw_val = np.nanmin(x_vals)
            else:
                # Left-handed: use most positive
                raw_val = np.nanmax(x_vals)

            # Normalize for UI
            vals = np.array([abs(raw_val)])

        elif selected_metric_010 == "Max Shoulder External Rotation":
            z_vals = arr[:, 2]

            # Use ER frame selected from arm-energy anchor
            if sh_er_max_frame_010 is not None and frames.size > 0:
                nearest_idx = int(np.argmin(np.abs(frames - int(sh_er_max_frame_010))))
                raw_val = float(z_vals[nearest_idx])
            else:
                # Fallback only if ER frame unavailable
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

            # Determine HA peak frame based on handedness
            if handedness_local == "R":
                # Right-handed: HA = most negative X
                ha_idx = np.nanargmin(x_vals)
            else:
                # Left-handed: HA = most positive X
                ha_idx = np.nanargmax(x_vals)

            # Extract ER at that frame: z_data
            raw_er = z_vals[ha_idx]

            # Normalize for UI (absolute)
            vals = np.array([abs(raw_er)])

        elif selected_metric_010 == "Max Hand Speed":
            # Hand CG speed magnitude (no sign, no windowing)
            # arr columns: x, y, z CG velocity components
            x_vals = arr[:, 0]
            y_vals = arr[:, 1]
            z_vals = arr[:, 2]
            speed = np.sqrt(x_vals**2 + y_vals**2 + z_vals**2)
            vals = np.array([np.nanmax(speed)])
        # Convert vals to a scalar for storage
        raw_value = float(np.nanmax(vals)) if vals.size > 0 else np.nan

        # For most metrics we report absolute magnitude, but for
        # Max Torso–Pelvis Angular Velocity we want the signed value
        if selected_metric_010 == "Max Torso–Pelvis Angular Velocity":
            metric_value = raw_value
        else:
            metric_value = abs(raw_value)

        rows_010.append({
            "take_id": take_id_010,
            "Velocity": pitch_velo_010,
            selected_metric_010: metric_value
        })

    if rows_010:
        # Rebuild dataframe including take_id so we can filter correctly
        df_010 = pd.DataFrame(rows_010)

        # Attach pitcher names and dates to df_010 (for labels & plotting)
        cur.execute(f"""
            SELECT t.take_id, a.athlete_name, t.take_date
            FROM takes t
            JOIN athletes a ON t.athlete_id = a.athlete_id
            WHERE t.take_id IN ({",".join(["%s"] * len(df_010))})
        """, tuple(df_010["take_id"].tolist()))

        take_lookup = {
            row[0]: (row[1], row[2].strftime("%Y-%m-%d"))
            for row in cur.fetchall()
        }

        df_010["Pitcher"] = df_010["take_id"].map(lambda x: take_lookup[x][0])
        df_010["Date"]    = df_010["take_id"].map(lambda x: take_lookup[x][1])

        # Build rich, human-readable exclude labels
        def make_exclude_label(row):
            return (
                f"{row['Pitcher']} | "
                f"{row['Date']} | "
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
            for (pitcher_name, date_str), sub in df_010.groupby(["Pitcher", "Date"]):
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

                # Scatter points
                fig_010.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(color=color),
                    name=f"{pitcher_name} | {date_str}"
                ))

                # Regression line
                fig_010.add_trace(go.Scatter(
                    x=x_fit,
                    y=y_fit,
                    mode="lines",
                    line=dict(color=color, dash="dash"),
                    name=f"R² = {r_value**2:.2f}"
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
                height=500
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
        SELECT t.take_id, t.take_date
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
        fp_aligned_frames = []
        er_aligned_frames = []
        max_x_aligned = None

        for take_id, take_date in takes_ts:
            # --- Ball Release frame ---
            br_frame = get_ball_release_frame(take_id, handedness_ts, cur)
            if br_frame is None:
                continue

            # --- Foot Plant frame (same logic as Tab 3) ---
            sh_er_max_frame = get_shoulder_er_max_frame(take_id, handedness_ts, cur)
            # Store Shoulder ER aligned to Ball Release
            if sh_er_max_frame is not None:
                er_aligned_frames.append(sh_er_max_frame - br_frame)
            fp_start_candidate = get_lead_ankle_prox_x_peak_frame(take_id, handedness_ts, cur)

            ankle_min_frame = None
            if fp_start_candidate is not None and sh_er_max_frame is not None:
                ankle_min_frame = get_ankle_min_frame(
                    take_id, handedness_ts,
                    fp_start_candidate,
                    sh_er_max_frame,
                    cur
                )

            zero_cross_frame = None
            if ankle_min_frame is not None:
                zero_cross_frame = get_zero_cross_frame(
                    take_id, handedness_ts,
                    ankle_min_frame,
                    sh_er_max_frame,
                    cur
                )

            if zero_cross_frame is not None:
                fp_frame = zero_cross_frame
            elif ankle_min_frame is not None:
                fp_frame = ankle_min_frame
            else:
                fp_frame = fp_start_candidate

            # Store FP aligned to Ball Release
            if fp_frame is not None:
                fp_aligned_frames.append(fp_frame - br_frame)

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

            # Align time so that Ball Release maps to 0
            t_aligned = frames - br_frame
            # Track the max aligned x so we can set a valid upper bound for x-axis range
            try:
                cur_max = float(np.nanmax(t_aligned))
                if max_x_aligned is None or cur_max > max_x_aligned:
                    max_x_aligned = cur_max
            except Exception:
                pass

            date_str = take_date.strftime("%Y-%m-%d")

            # Assign consistent color per date
            if date_str not in date_colors:
                date_colors[date_str] = color_cycle[len(date_colors) % len(color_cycle)]

            fig_ts.add_trace(
                go.Scatter(
                    x=t_aligned,
                    y=vals,
                    mode="lines",
                    line=dict(color=date_colors[date_str]),
                    name=date_str,
                    legendgroup=date_str,
                    opacity=0.4,
                    showlegend=(date_str not in [t.name for t in fig_ts.data])
                )
            )

        fig_ts.add_vline(
            x=0,
            line_dash="dot",
            line_color="green",
            annotation_text="BR",
            annotation_position="top",
            annotation_font=dict(
                size=13,
                color="green",
                family="Arial"
            )
        )

        # --- Median Foot Plant ---
        if fp_aligned_frames:
            median_fp = float(np.median(fp_aligned_frames))
            fig_ts.add_vline(
                x=median_fp,
                line_dash="dash",
                line_color="orange",
                annotation_text="FP",
                annotation_position="top",
                annotation_font=dict(
                    size=13,
                    color="orange",
                    family="Arial"
                )
            )
            x_start = median_fp - 50
        else:
            x_start = None

        # --- Median Shoulder ER ---
        if er_aligned_frames:
            median_er = float(np.median(er_aligned_frames))
            fig_ts.add_vline(
                x=median_er,
                line_dash="dot",
                line_color="purple",
                annotation_text="MER",
                annotation_position="top",
                annotation_font=dict(
                    size=13,
                    color="purple",
                    family="Arial"
                )
            )

        # End plot 50 frames after Ball Release
        x_end = 50

        fig_ts.update_layout(
            title=f"Arm Proximal Energy Transfer — {arm_prox_segment} (Aligned to Ball Release)",
            xaxis_title="Aligned Time (Ball Release = 0)",
            yaxis_title="Power",
            xaxis=dict(range=[x_start, x_end]) if x_start is not None else {},
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