import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import psycopg2
import numpy as np
from scipy.stats import linregress
import plotly.graph_objects as go
import plotly.express as px

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
            "STP Rotational"
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
        """, (tid, shoulder_stp_segment))
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
        # Only use valid rows for regression/plotting
        sub_valid = sub.dropna(subset=["Velocity", "AUC (Drive → 0)", "AUC (Drive → Peak Arm Energy)"])
        if len(sub_valid) < 2:
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
        "Max Scap Retraction",
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
            "Max Scap Retraction",
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
        elif selected_metric_010 == "Max Scap Retraction":
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
        elif selected_metric_010 == "Max Scap Retraction":
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
            else:
                max_scap_frame = int(mer_anchor) if mer_anchor is not None else None

            if max_scap_frame is not None:
                pre_mask = frames <= max_scap_frame
                pre_frames = frames[pre_mask]
                pre_vel = x_vel[pre_mask]
                if pre_vel.size > 0:
                    # Last positive->negative zero-cross before max scap retraction.
                    zc_candidates = np.where((pre_vel[:-1] >= 0) & (pre_vel[1:] < 0))[0]
                    if zc_candidates.size > 0:
                        start_frame = int(pre_frames[zc_candidates[-1] + 1])
                    else:
                        # Fallback: closest-to-zero frame before the max.
                        start_frame = int(pre_frames[int(np.argmin(np.abs(pre_vel)))])

                    leadup_mask = (frames >= start_frame) & (frames <= max_scap_frame)
                    if np.any(leadup_mask):
                        x_w = x_vel[leadup_mask]
                    else:
                        x_w = pre_vel
                else:
                    x_w = x_vel
            else:
                x_w = x_vel

            vals = np.array([np.nanmin(x_w)])
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

        rows_010.append({
            "take_id": take_id_010,
            "Throw Type": (throw_type_local if throw_type_local is not None else "Mound"),
            "Velocity": pitch_velo_010,
            selected_metric_010: metric_value
        })

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
