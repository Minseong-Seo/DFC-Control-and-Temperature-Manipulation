import pandas as pd
import numpy as np
from pathlib import Path

# =====================================================
# User Settings
# =====================================================

ORIGINAL = Path(
    "/Volumes/T7/DFC/R_parsing_origin_Ioniq5/"
    "bms_01241248820_2023-09_r.csv"
)

DFC60 = Path(
    "/Volumes/T7/DFC/DFC_origin_Ioniq5/DFC80_t60/"
    "bms_01241248820_2023-09_DFC80_t60.csv"
)

DFC30 = Path(
    "/Volumes/T7/DFC/DFC_origin_Ioniq5/DFC80_t30/"
    "bms_01241248820_2023-09_DFC80_t30.csv"
)

OUTPUT = Path(
    "/Volumes/T7/DFC/New_DFC_Sessions_t30_only.csv"
)

# =====================================================
# Read CSV
# =====================================================

df_org = pd.read_csv(ORIGINAL)
df_60 = pd.read_csv(DFC60)
df_30 = pd.read_csv(DFC30)

for df in (df_org, df_60, df_30):
    df["time"] = pd.to_datetime(df["time"])


# =====================================================
# Charging Session 추출
# =====================================================

def get_charge_sessions(df):

    charging = df["charging"].fillna(0).astype(int)

    sessions = []

    in_session = False
    start = None

    for i, value in enumerate(charging):

        if value == 1 and not in_session:
            start = i
            in_session = True

        elif value == 0 and in_session:

            sessions.append((start, i - 1))

            in_session = False

    if in_session:
        sessions.append((start, len(df) - 1))

    return sessions

sessions = get_charge_sessions(df_org)

print(f"Charging Sessions : {len(sessions)}")

# =====================================================
# 각 충전 세션의 시간 이동 여부 확인
# =====================================================

results = []

for session_id, (start_idx, end_idx) in enumerate(sessions):

    # 세션 구간의 time
    org_time = df_org.loc[start_idx:end_idx, "time"].reset_index(drop=True)
    t60_time = df_60.loc[start_idx:end_idx, "time"].reset_index(drop=True)
    t30_time = df_30.loc[start_idx:end_idx, "time"].reset_index(drop=True)

    # 행별 시간차(초)
    diff60 = (t60_time - org_time).dt.total_seconds()
    diff30 = (t30_time - org_time).dt.total_seconds()

    # 시간이 실제로 변경된 행
    changed60 = diff60 != 0
    changed30 = diff30 != 0

    results.append({

        "session_id": session_id,

        "start_idx": start_idx,
        "end_idx": end_idx,

        "start_time": df_org.loc[start_idx, "time"],
        "end_time": df_org.loc[end_idx, "time"],

        "start_soc": df_org.loc[start_idx, "soc"],
        "end_soc": df_org.loc[end_idx, "soc"],

        "applied_t60": changed60.any(),
        "applied_t30": changed30.any(),

        "changed_rows_t60": int(changed60.sum()),
        "changed_rows_t30": int(changed30.sum()),

        "max_delay_min_t60": diff60.max() / 60,
        "max_delay_min_t30": diff30.max() / 60,
    })

result_df = pd.DataFrame(results)

print(result_df)

# =====================================================
# t60에서는 적용 안되고
# t30에서는 적용된 세션만 추출
# =====================================================

new_sessions = result_df[
    (~result_df["applied_t60"]) &
    (result_df["applied_t30"])
].copy()

print("\n===================================")
print("New DFC Sessions (t30 only)")
print("===================================")

print(new_sessions)

new_sessions.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nSaved : {OUTPUT}")


# =====================================================
# 시간이 변경된 행만 출력
# =====================================================

session = 0

start_idx = sessions[session][0]
end_idx = sessions[session][1]

compare = pd.DataFrame({

    "index": range(start_idx, end_idx+1),

    "original": df_org.loc[start_idx:end_idx,"time"].values,

    "t60": df_60.loc[start_idx:end_idx,"time"].values,

    "t30": df_30.loc[start_idx:end_idx,"time"].values,

    "soc": df_org.loc[start_idx:end_idx,"soc"].values

})

compare["t60_changed"] = compare["original"] != compare["t60"]
compare["t30_changed"] = compare["original"] != compare["t30"]

changed = compare[
    compare["t60_changed"] | compare["t30_changed"]
]

print(changed)

changed.to_csv(
    "/Volumes/T7/DFC/session0_changed_rows.csv",
    index=False,
    encoding="utf-8-sig"
)