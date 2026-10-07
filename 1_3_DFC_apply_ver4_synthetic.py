import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path, PureWindowsPath
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==========================================================
# User settings
# ==========================================================

# 합성차량은 CSV 전체 ID로 선택하고, 저장할 때만 짧은 이름을 사용한다.
SYNTHETIC_VEHICLES = {
    "01241228082_S02": "8082_S02",
    "01241228094_S10": "8094_S10",
}
COMPOSITION_CSV = Path(r"C:\Users\BSG 202608\Downloads\synthetic_vehicle_composition.csv")
VEHICLE_MODELS = ["EV6"]  # 기존 함수 초기화용. 합성 모드는 구성표에서 차종을 결정한다.

# SOC별로 허용된 t_buffer(분)만 실행한다. 차량 모델당 총 52개 조합.
BUFFER_BY_SOC = {
    50: [0, 5, 10],
    60: [0, 5, 10, 15, 20],
    70: [0, 5, 10, 15, 20],
    80: [0, 5, 10, 15, 20, 30],
    83: [0, 5, 10, 15, 20, 30],
    85: [0, 5, 10, 15, 20, 30],
    88: [0, 5, 10, 15, 20, 30, 40],
    90: [0, 5, 10, 15, 20, 30, 40],
    92: [0, 5, 10, 15, 20, 30, 40],
}

DFC_STAY_VALUES = list(BUFFER_BY_SOC)
TIME_BUFFER_MIN_VALUES = [0]  # 기존 초기화·검증용. 실제 반복은 BUFFER_BY_SOC 사용.

# 아래 DEBUG_DEVICES는 기존 폴더 처리 함수 전용이다.
# 합성 실행 대상은 SYNTHETIC_VEHICLES로 선택한다.
# 특정 차량만 처리하려면 ID 리스트를 입력한다.
# 전체 차량은 DEBUG_DEVICES = None
# DEBUG_DEVICES = [
#     "01241228082",
#     "01241228094"
# ]
DEBUG_DEVICES = [
    "01241228082",
    "01241228094"
]
# 특정 월만 처리하려면 "YYYY-MM" 리스트를 입력한다.
# 전체 월은 TARGET_YMS = None
# 예: TARGET_YMS = ["2023-08", "2023-09"]
TARGET_YMS = None

# 기존 폴더 처리 함수의 건너뛰기 설정.
# 합성 실행은 출처 검증의 일관성을 위해 기존 결과도 다시 계산한다.
SKIP_EXISTING = True

# 개별 DFC CSV 저장 여부
# True: DFC 결과 CSV와 summary 모두 저장
# False: 개별 CSV는 저장하지 않고 summary만 저장
WRITE_OUTPUTS = True

# 멀티프로세스 작업자 수
WORKERS = 8
INPUT_ROOT_DIR = Path(r"Z:\Processed_Data\DFC")
OUTPUT_ROOT_DIR = Path(r"D:\DFC_synthetic_base_condition")

if not VEHICLE_MODELS or any(
    vehicle_model not in {"Ioniq5", "EV6"}
    for vehicle_model in VEHICLE_MODELS
):
    raise ValueError(
        "VEHICLE_MODELS에는 'Ioniq5' 또는 'EV6'만 입력할 수 있습니다."
    )


def normalize_filter_values(values):
    if values is None:
        return None
    if isinstance(values, str):
        return [values]
    return [str(value) for value in values]


DEBUG_DEVICES = normalize_filter_values(DEBUG_DEVICES)
TARGET_YMS = normalize_filter_values(TARGET_YMS)

# 현재 실행 조건. 실행 직전에 configure_runtime()이 갱신한다.
VEHICLE_MODEL = VEHICLE_MODELS[0]
DFC_STAY = DFC_STAY_VALUES[0]
TIME_BUFFER_MIN = TIME_BUFFER_MIN_VALUES[0]
TIME_BUFFER = pd.Timedelta(minutes=TIME_BUFFER_MIN)
DFC_SUFFIX = f"_DFC{DFC_STAY}_t{TIME_BUFFER_MIN}"


def configure_runtime(
    vehicle_model: str,
    dfc_stay: int,
    time_buffer_min: int,
) -> None:
    """차량과 DFC 조건에 맞춰 현재 실행 설정을 갱신한다."""
    global VEHICLE_MODEL
    global DFC_STAY
    global TIME_BUFFER_MIN
    global TIME_BUFFER
    global DFC_SUFFIX

    VEHICLE_MODEL = vehicle_model
    DFC_STAY = int(dfc_stay)
    TIME_BUFFER_MIN = int(time_buffer_min)
    TIME_BUFFER = pd.Timedelta(minutes=TIME_BUFFER_MIN)
    DFC_SUFFIX = f"_DFC{DFC_STAY}_t{TIME_BUFFER_MIN}"

# ─────────────────────────────────────────────────────────────
# 충전후 구간 불필요한 데이터 삭제 (벡터화, 인덱스 안전)
# ─────────────────────────────────────────────────────────────
def remove_consecutive_ones(data):
    if 'R_aftercharg' not in data.columns:
        return data

    s = data['R_aftercharg'].fillna(0).astype(int)
    grp = (s != s.shift(fill_value=s.iloc[0])).cumsum()          # 연속 구간 라벨
    group_sizes = grp.map(grp.value_counts())                    # 각 행이 속한 구간 길이
    pos_from_start = data.groupby(grp).cumcount()
    pos_from_end = data.iloc[::-1].groupby(grp.iloc[::-1]).cumcount()[::-1]

    # ─────────────────────────────────────────────
    # ① 보호 대상 R_aftercharg 그룹 찾기
    #    조건: 해당 after 구간 바로 직전 R_charg 구간의 "시작 SOC ≥ 95"
    # ─────────────────────────────────────────────
    protect_groups = set()

    if ('R_charg' in data.columns) and ('soc' in data.columns):
        # R_aftercharg == 1 인 그룹들만 대상
        after_groups = grp[s == 1].unique()

        for g in after_groups:
            # 이 그룹에 속한 인덱스들
            idxs = np.flatnonzero(grp.values == g)
            if len(idxs) == 0:
                continue

            start_idx = int(idxs[0])  # after 구간 시작 행 인덱스

            # 맨 앞이면 바로 직전 구간이 없으므로 패스
            if start_idx == 0:
                continue

            prev_idx = start_idx - 1

            # 직전 행이 R_charg==1 이 아니면 패스
            rc_prev = int(data.loc[prev_idx, 'R_charg']) if not pd.isna(data.loc[prev_idx, 'R_charg']) else 0
            if rc_prev != 1:
                continue

            # 직전 R_charg 구간의 "시작 인덱스" 찾기 (연속 1 구간의 맨 앞)
            k = prev_idx
            while k > 0:
                val = data.loc[k-1, 'R_charg']
                rc = int(val) if not pd.isna(val) else 0
                if rc != 1:
                    break
                k -= 1
            start_charg_idx = k

            # 그 구간 시작 시점 SOC ≥ 95 인지 확인
            soc_start = pd.to_numeric(data.loc[start_charg_idx, 'soc'], errors='coerce')
            if pd.notna(soc_start) and soc_start >= 95:
                protect_groups.add(g)

    # 각 행이 보호 대상 그룹인지 여부 시리즈
    protect_flag = grp.isin(protect_groups)

    # ─────────────────────────────────────────────
    # ② keep 마스크 구성
    #    - s==0: 항상 유지
    #    - s==1 & 보호 그룹: 전부 유지 (중간행 삭제 금지)
    #    - s==1 & 비보호 그룹:
    #        · 길이<3 → 전부 유지
    #        · 길이≥3 → 처음/마지막만 유지
    # ─────────────────────────────────────────────
    keep = (
        (s == 0) |
        ((s == 1) & protect_flag) |
        ((s == 1) & ~protect_flag & (group_sizes < 3)) |
        ((s == 1) & ~protect_flag & (group_sizes >= 3) & ((pos_from_start == 0) | (pos_from_end == 0)))
    )

    return data.loc[keep].reset_index(drop=True)


def DFC(data, collect_stats=False):
    data = remove_consecutive_ones(data)

    # time 파싱 안전화
    if 'time' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['time']):
        data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # 실제로 시간축이 이동된 행을 표시하는 칼럼
    # DFC가 적용되지 않은 행은 빈칸으로 유지한다.
    data['DFC_applied'] = ''
    # ─ 충전 구간 경계 수집
    charg = []
    if data.loc[0, 'R_charg'] == 1:
        charg.append(0)
    for i in range(len(data) - 1):
        if data.loc[i, 'R_charg'] != data.loc[i + 1, 'R_charg']:
            charg.append(i + 1)
    if data.loc[len(data) - 1, 'R_charg'] == 1:
        charg.append(len(data) - 1)

    # cend±1 근접 체크
    def any_after_near(idx):
        lo = max(0, idx - 1)
        hi = min(len(data) - 1, idx + 1)
        return (data.loc[lo:hi, 'R_aftercharg'] == 1).any()

    # DFC 적용 충전구간
    dfc_charg = []
    for i in range(len(charg) - 1):
        if data.loc[charg[i], 'R_charg'] == 1 and any_after_near(charg[i + 1] - 1):
            dfc_charg.append(charg[i])
            dfc_charg.append(charg[i + 1] - 1)

    # ─ 충전단계2: 정확히 DFC_STAY 기준 + 쌍 유지
    charg_2_pairs = []   # (delay_start, charge_end)
    delay_start = []
    for j in range(0, len(dfc_charg) - 1, 2):
        start_j, end_j = dfc_charg[j], dfc_charg[j + 1]
        found = False
        # (1) 구간 내부에서 {DFC_STAY-1}→DFC_STAY가 되는 순간을 찾으면 그 다음 인덱스를 지연 시작으로 사용
        for i in range(start_j, max(start_j, end_j)):  # [start_j, end_j-1]
            if (data.loc[i, 'soc'] < DFC_STAY) and (data.loc[i + 1, 'soc'] == DFC_STAY):
                charg_2_pairs.append((i + 1, end_j))
                delay_start.append(i + 1)
                found = True
                break
        # (2) 위가 없고, **충전 시작 SOC가 DFC_STAY 이상이면 충전 시작 시점부터 지연 시작**
        if (not found) and (data.loc[start_j, 'soc'] >= DFC_STAY):
            charg_2_pairs.append((start_j, end_j))
            delay_start.append(start_j)

    # # ─ aftercharge 종료점(원형)
    # after = []
    # if data.loc[0, 'R_aftercharg'] == 1:
    #     after.append(0)
    # for i in range(len(data) - 1):
    #     if data.loc[i, 'R_aftercharg'] != data.loc[i + 1, 'R_aftercharg']:
    #         after.append(i + 1)
    # if data.loc[len(data) - 1, 'R_aftercharg'] == 1:
    #     after.append(len(data) - 1)
    #
    # end_aftercharg = []
    # for i in range(len(after) - 1):
    #     if data.loc[after[i], 'R_aftercharg'] == 1:
    #         end_aftercharg.append(after[i + 1] - 1)
    #
    # # 종료점 품질 필터
    # remove_end = []
    # for i in range(len(end_aftercharg)):
    #     if data.loc[end_aftercharg[i] - 1, 'soc'] < DFC_STAY:
    #         remove_end.append(end_aftercharg[i])
    # end_aftercharg = [idx for idx in end_aftercharg if idx not in remove_end]

    # ===== DEBUG =====
    print(f"\n[DEBUG] DFC_STAY={DFC_STAY}")
    print(f"[DEBUG] charg_2_pairs={charg_2_pairs}")
    # =================
    # ─ 매칭(넘파이)
    dfc_events = []
    if len(charg_2_pairs) and ('R_charg' in data.columns) and ('R_aftercharg' in data.columns):
        ch = data['R_charg'].fillna(0).astype(int).to_numpy()
        ac = data['R_aftercharg'].fillna(0).astype(int).to_numpy()

        # aftercharge 세그먼트 시작/끝
        transitions_ac = np.diff(np.r_[0, ac, 0])        # +1: start, -1: end+1
        astarts = np.where(transitions_ac == +1)[0]
        aends   = np.where(transitions_ac == -1)[0] - 1

        # 충전 시작 인덱스들
        transitions_ch = np.diff(np.r_[0, ch])           # +1: start
        cstarts = np.where(transitions_ch == +1)[0]

        astarts.sort(); aends.sort(); cstarts.sort()


        for dstart, cend in charg_2_pairs:
            # 🔹 지연 시작점 SOC가 95 이상이면 이 이벤트는 DFC 적용하지 않음
            if 'soc' in data.columns:
                soc_d = pd.to_numeric(data.loc[dstart, 'soc'], errors='coerce')
                if pd.notna(soc_d) and soc_d >= 95:
                    continue

            # 다음 충전 시작
            pos_c = np.searchsorted(cstarts, cend + 1, side='left')
            next_charge_start = cstarts[pos_c] if pos_c < len(cstarts) else None

            # cend 이상에서 시작하는 첫 aftercharge
            pos_a = np.searchsorted(astarts, cend, side='left')
            if pos_a >= len(astarts):
                continue
            astart = astarts[pos_a]

            # 다음 충전 시작 전이어야 함
            if (next_charge_start is not None) and (astart >= next_charge_start):
                continue

            aend = aends[pos_a]  # 동일 세그먼트 끝

            # 시간 계산/적용
            t0 = data.loc[cend, 'time']
            t1 = data.loc[aend, 'time']
            if pd.isna(t0) or pd.isna(t1):
                continue

            delayed_time = (t1 - t0 - TIME_BUFFER)
            if (delayed_time > pd.Timedelta(0)) and (dstart + 1 <= cend):
                # 이벤트 수집
                dfc_events.append({
                    'charge_end_idx': int(cend),
                    'after_end_idx': int(aend),
                    'charge_end_time': t0,
                    'after_end_time': t1,
                    'delay_hours': delayed_time.total_seconds() / 3600.0
                })
                # 실제 보정 적용
                # 현재 DFC 로직상 dstart 행은 유지하고,
                # dstart+1부터 cend까지의 행만 시간축을 이동한다.
                shifted_indices = data.loc[dstart + 1:cend].index

                data.loc[shifted_indices, 'time'] = (
                    data.loc[shifted_indices, 'time']
                    + delayed_time
                )

                # 실제로 시간축이 이동된 행에만 상태를 기록한다.
                data.loc[
                    shifted_indices,
                    'DFC_applied'
                ] = 'DFC_applied'


    # 세분화 컬럼 삭제(존재할 때만)
    columns_to_delete = ['R_charg', 'R_partial_charg', 'R_aftercharg', 'R_uncharg']
    data = data.drop(columns=[c for c in columns_to_delete if c in data.columns], errors='ignore')

    if not collect_stats:
        return data

    # ─ 파일 단위 요약 통계 (delta_t95_event 네이밍 고정)
    delays = pd.to_numeric(pd.Series([e['delay_hours'] for e in dfc_events], dtype='float64'), errors='coerce').dropna()
    N = int(len(delays))
    mean = float(delays.mean()) if N > 0 else 0.0
    std  = float(delays.std(ddof=1)) if N > 1 else 0.0
    summ = float(delays.sum()) if N > 0 else 0.0

    stats = {
        'delta_t95_event_N': N,
        'delta_t95_event_mean_h': mean,
        'delta_t95_event_std_h': std,
        'delta_t95_event_sum_h': summ
    }
    return data, dfc_events, stats

# ─────────────────────────────────────────────────────────────
# 파일 하나 돌리기 (저장 on/off + 요약 리턴)
# ─────────────────────────────────────────────────────────────
def process_DFC_file(file_path, save_path=None, collect_stats=True, write_output=True):
    """
    write_output=False 이면 변환된 DFC CSV를 저장하지 않고 통계만 반환.
    """
    data = pd.read_csv(file_path)
    if collect_stats:
        result = DFC(data, collect_stats=True)
        data = result[0]
        _events = result[1]
        stats = result[2]
    else:
        data = DFC(data, collect_stats=False)
        _events, stats = [], None

    if write_output:
        if save_path is None:
            base, ext = os.path.splitext(file_path)
            save_path = f"{base.rstrip('_r')}{DFC_SUFFIX}{ext}"
        data.to_csv(save_path, index=False)

    return data, stats

# ─────────────────────────────────────────────────────────────
# 요약 CSV (delta_t95_event 네이밍 고정)
# ─────────────────────────────────────────────────────────────
SUMMARY_COLUMNS = [
    'file_stem', 'id_token', 'ym',
    'delta_t95_event_N',
    'delta_t95_event_mean_h',
    'delta_t95_event_std_h',
    'delta_t95_event_sum_h',
]

def parse_id_token_and_ym(p: Path):
    """
    일반 파일과 altitude 파일에서
    차량 ID와 연월을 추출한다.

    예시:
    bms_01241228021_2023-02_r.csv
    bms_altitude_01241597802_2023-12.csv
    """
    parts = p.stem.split("_")

    ym_idx = next(
        (
            i
            for i, part in enumerate(parts)
            if len(part) == 7
            and part[4] == "-"
            and part[:4].isdigit()
            and part[5:].isdigit()
        ),
        None,
    )

    if ym_idx is None or ym_idx == 0:
        return "unknown", "0000-00"

    id_token = parts[ym_idx - 1]
    if re.fullmatch(r"S\d+", id_token) and ym_idx >= 2:
        id_token = "_".join(parts[ym_idx - 2:ym_idx])
    ym = parts[ym_idx]

    return id_token, ym

def _collect_input_files(input_folder, pattern="*.csv"):
    input_dir = Path(input_folder)

    files = [
        p for p in sorted(input_dir.glob(pattern))
        if not p.name.startswith(".")
    ]

    if DEBUG_DEVICES is not None:
        files = [
            p
            for p in files
            if any(
                device_id in p.name
                for device_id in DEBUG_DEVICES
            )
        ]

    if TARGET_YMS is not None:
        files = [
            p
            for p in files
            if any(
                target_ym in p.name
                for target_ym in TARGET_YMS
            )
        ]

    return files

def process_DFC_folder(input_folder, output_folder, summary_csv_path=None,
                       pattern="*.csv", write_outputs=True, skip_existing=True):
    files = _collect_input_files(input_folder, pattern=pattern)
    return _process_files_and_summary(files, output_folder, summary_csv_path,
                                      write_outputs=write_outputs, skip_existing=skip_existing)


def process_DFC_folder_slice(input_folder, output_folder, start_idx=0, end_idx=None,
                             summary_csv_path=None, pattern="*.csv", write_outputs=True,
                             skip_existing=True):
    files = _collect_input_files(input_folder, pattern=pattern)
    if end_idx is None:
        sel = files[start_idx:]
    else:
        sel = files[start_idx:end_idx+1]  # inclusive
    return _process_files_and_summary(sel, output_folder, summary_csv_path,
                                      write_outputs=write_outputs, skip_existing=skip_existing)


def _process_files_and_summary(files, output_folder, summary_csv_path=None,
                               write_outputs=True, skip_existing=True):
    output_dir = Path(output_folder)
    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for p in tqdm(files, desc='Processing Files'):
        try:
            save_path = None
            if write_outputs:
                out_name = p.name.replace('_r.csv', f"{DFC_SUFFIX}.csv")
                save_path = output_dir / out_name

                # ── 여기! 결과가 이미 있으면 스킵 ─────────────────────────
                if skip_existing and save_path.exists():
                    # 필요하면 로그만 남기고 요약에서는 제외(빠르게 돌리기 목적)
                    # tqdm.write(f"[skip] {out_name}")
                    continue
                # ─────────────────────────────────────────────────────────

            _, stats = process_DFC_file(
                str(p),
                save_path=str(save_path) if save_path else None,
                collect_stats=True,
                write_output=write_outputs
            )

            # 요약 수집
            id_token, ym = parse_id_token_and_ym(p)
            if stats is None:
                stats = {
                    'delta_t95_event_N': 0,
                    'delta_t95_event_mean_h': 0.0,
                    'delta_t95_event_std_h': 0.0,
                    'delta_t95_event_sum_h': 0.0
                }

            summary_rows.append({
                'file_stem': p.stem,
                'id_token': id_token,
                'ym': ym,
                'delta_t95_event_N': stats['delta_t95_event_N'],
                'delta_t95_event_mean_h': stats['delta_t95_event_mean_h'],
                'delta_t95_event_std_h': stats['delta_t95_event_std_h'],
                'delta_t95_event_sum_h': stats['delta_t95_event_sum_h'],
            })

        except Exception as e:
            print(f"Error processing {p.name}: {str(e)}")
            id_token, ym = parse_id_token_and_ym(p)
            summary_rows.append({
                'file_stem': p.stem,
                'id_token': id_token,
                'ym': ym,
                'delta_t95_event_N': 0,
                'delta_t95_event_mean_h': 0.0,
                'delta_t95_event_std_h': 0.0,
                'delta_t95_event_sum_h': 0.0,
            })
            continue

    # 요약 CSV 저장
    if summary_csv_path is None:
        summary_csv_path = Path(output_folder) / f"dfc_summary{DFC_SUFFIX}.csv"
    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 요약 저장: {summary_csv_path} (rows={len(summary_df)})")

    return summary_df


def _dfc_one_file_job(args):
    """
    (in_path, out_path_or_None, write_outputs, skip_existing, 실행조건) 받아서
    - DFC 처리(옵션으로 파일 저장)
    - stats dict 반환 (요약용)
    """
    (
        in_path,
        out_path,
        write_outputs,
        skip_existing,
        vehicle_model,
        dfc_stay,
        time_buffer_min,
    ) = args
    configure_runtime(
        vehicle_model,
        dfc_stay,
        time_buffer_min,
    )
    p = Path(in_path)

    # 출력 파일이 있고 스킵이면: 통계도 스킵할지 정책 선택 필요
    # 지금은 "스킵되면 요약에서도 제외" = 기존 단일프로세스와 동일 동작
    if write_outputs and out_path and skip_existing and Path(out_path).exists():
        return ("skip", p.stem, None)

    try:
        _, stats = process_DFC_file(
            str(p),
            save_path=str(out_path) if (write_outputs and out_path) else None,
            collect_stats=True,
            write_output=write_outputs
        )

        if stats is None:
            stats = {
                'delta_t95_event_N': 0,
                'delta_t95_event_mean_h': 0.0,
                'delta_t95_event_std_h': 0.0,
                'delta_t95_event_sum_h': 0.0
            }

        # 요약 row 생성
        id_token, ym = parse_id_token_and_ym(p)
        row = {
            'file_stem': p.stem,
            'id_token': id_token,
            'ym': ym,
            'delta_t95_event_N': int(stats['delta_t95_event_N']),
            'delta_t95_event_mean_h': float(stats['delta_t95_event_mean_h']),
            'delta_t95_event_std_h': float(stats['delta_t95_event_std_h']),
            'delta_t95_event_sum_h': float(stats['delta_t95_event_sum_h']),
        }
        return ("ok", p.stem, row)

    except Exception as e:
        # 에러도 요약에 0으로 남김(기존 로직 유지)
        id_token, ym = parse_id_token_and_ym(p)
        row = {
            'file_stem': p.stem,
            'id_token': id_token,
            'ym': ym,
            'delta_t95_event_N': 0,
            'delta_t95_event_mean_h': 0.0,
            'delta_t95_event_std_h': 0.0,
            'delta_t95_event_sum_h': 0.0,
        }
        return ("error", f"{p.name}: {e}", row)


def process_DFC_folder_mp(
    input_folder,
    output_folder,
    summary_csv_path=None,
    pattern="*.csv",
    write_outputs=True,
    skip_existing=True,
    workers=None,
):
    """
    DFC 멀티프로세스 폴더 처리 + summary 생성
    - write_outputs=True : *_DFCxx_Tyy.csv 저장
    - skip_existing=True : *_DFC.csv 있으면 스킵(요약에서도 제외: 기존과 동일)
    """
    files = _collect_input_files(input_folder, pattern=pattern)
    if not files:
        print("[info] 처리할 CSV가 없습니다.")
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    output_dir = Path(output_folder)
    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)

    # 작업 리스트
    jobs = []
    for p in files:
        out_path = None
        if write_outputs:
            out_name = p.name.replace("_r.csv", f"{DFC_SUFFIX}.csv")
            out_path = str(output_dir / out_name)
        jobs.append(
            (
                str(p),
                out_path,
                write_outputs,
                skip_existing,
                VEHICLE_MODEL,
                DFC_STAY,
                TIME_BUFFER_MIN,
            )
        )

    summary_rows = []
    ok = skip = err = 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_dfc_one_file_job, job) for job in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="DFC Processing (MP)"):
            status, msg, row = fut.result()
            if status == "ok":
                ok += 1
                summary_rows.append(row)
            elif status == "skip":
                skip += 1
                # 스킵은 요약 제외(기존과 동일)
            else:
                err += 1
                print(f"[error] {msg}")
                summary_rows.append(row)

    # summary 저장
    if summary_csv_path is None:
        summary_csv_path = str(output_dir / f"dfc_summary{DFC_SUFFIX}.csv") if write_outputs else str(Path(input_folder) / f"dfc_summary{DFC_SUFFIX}.csv")

    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

    print(f"[done] ok={ok}, skip={skip}, error={err}")
    print(f"✅ 요약 저장: {summary_csv_path} (rows={len(summary_df)})")
    return summary_df

# ==========================================================
# 합성차량 구성표 / 출처 검증 / 실행
# ==========================================================
def load_synthetic_composition():
    """월별 원본 지정만 사용한다. 원본 연도와 데이터 내부 값은 바꾸지 않는다."""
    table = pd.read_csv(COMPOSITION_CSV, dtype=str, keep_default_na=False)
    required = {
        "synthetic_vid", "base_vid", "month_index", "month_code",
        "cluster_bucket", "src_vid", "src_file_stem", "src_non_path", "src_dfc_path",
    }
    if required - set(table.columns):
        raise ValueError(f"구성표 필수 칼럼 누락: {sorted(required - set(table.columns))}")
    selected = table[table["synthetic_vid"].isin(SYNTHETIC_VEHICLES)].copy()
    for full_id in SYNTHETIC_VEHICLES:
        group = selected[selected["synthetic_vid"] == full_id]
        indices = pd.to_numeric(group["month_index"], errors="raise")
        if len(group) != 12 or sorted(indices.tolist()) != list(range(1, 13)):
            raise ValueError(f"{full_id}: month_index 1~12가 각각 한 번 필요합니다.")
        if set(group["month_code"]) != {f"{m:02d}" for m in range(1, 13)}:
            raise ValueError(f"{full_id}: month_code 01~12가 각각 한 번 필요합니다.")
    selected["month_index"] = pd.to_numeric(selected["month_index"])
    return selected.sort_values(["synthetic_vid", "month_index"])


def resolve_composition_row(row):
    """구성표 원본과 동일한 차종/차량/연월의 R 파싱 입력 경로를 결정한다.

    src_non_path에는 R_charg 등이 없으므로 직접 DFC에 넣지 않는다.
    src_dfc_path는 기존 DFC90_t5 결과의 출처 정보일 뿐 재입력하지 않는다.
    """
    source = PureWindowsPath(row["src_non_path"])
    model = source.parent.name
    stem = row["src_file_stem"]
    match = re.fullmatch(r"bms(?:_altitude)?_(\d+)_(\d{4}-\d{2})", stem)
    if model not in {"EV6", "Ioniq5"} or source.stem != stem or not match:
        raise ValueError("구성표 원본 경로/파일명/차종 불일치")
    source_id, ym = match.groups()
    if source_id != row["src_vid"] or ym[-2:] != row["month_code"]:
        raise ValueError("구성표 원본 차량 ID 또는 월 불일치")
    reference = PureWindowsPath(row["src_dfc_path"])
    if (reference.name != f"{stem}_DFC90_t5_Temp_manipulation.csv"
            or reference.parent.name != "DFC90_t5"
            or reference.parent.parent.name != f"Temp_manipulation_DFC_{model}_base"):
        raise ValueError("구성표 일반 원본과 DFC 참조 경로 불일치")
    actual = INPUT_ROOT_DIR / model / "R_parsing_완충후이동주차" / f"{stem}_r.csv"
    actual_id, actual_ym = parse_id_token_and_ym(actual)
    if (actual_id, actual_ym) != (source_id, ym):
        raise ValueError("R 파싱 입력 파일의 차량 ID/연월 불일치")
    return model, ym, actual


def run_synthetic():
    composition = load_synthetic_composition()
    # 모든 대상 원본을 먼저 검증하고, 실패 내용도 디스크에 남긴다.
    records = []
    for row in composition.to_dict("records"):
        record = dict(row)
        record.update(synthetic_alias=SYNTHETIC_VEHICLES[row["synthetic_vid"]],
                      actual_input_path="", source_model="", ym="",
                      mapping_match=False, input_exists=False, error="")
        try:
            model, ym, actual = resolve_composition_row(row)
            record.update(source_model=model, ym=ym, actual_input_path=str(actual),
                          mapping_match=True, input_exists=actual.is_file())
            if not record["input_exists"]:
                raise FileNotFoundError(f"R 파싱 원본이 없습니다: {actual}")
            columns = set(pd.read_csv(actual, nrows=0).columns)
            required = {"time", "soc", "R_charg", "R_aftercharg"}
            if required - columns:
                raise ValueError(f"R 파싱 필수 칼럼 누락: {sorted(required - columns)}")
        except Exception as error:
            record["error"] = str(error)
        records.append(record)
    log_root = OUTPUT_ROOT_DIR / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(log_root / "composition_preflight.csv",
                                 index=False, encoding="utf-8-sig")
    if any(record["error"] for record in records):
        raise ValueError(f"원본 검증 실패. 확인: {log_root / 'composition_preflight.csv'}")

    for alias in SYNTHETIC_VEHICLES.values():
        selected = [r for r in records if r["synthetic_alias"] == alias
                    and (TARGET_YMS is None or r["ym"] in TARGET_YMS)]
        for soc in DFC_STAY_VALUES:
            for buffer in BUFFER_BY_SOC[soc]:
                configure_runtime(alias, soc, buffer)
                condition = DFC_SUFFIX.lstrip("_")
                output_dir = OUTPUT_ROOT_DIR / f"DFC_origin_{alias}" / condition
                summary_dir = OUTPUT_ROOT_DIR / f"DFC_summary_{alias}" / condition
                summary_dir.mkdir(parents=True, exist_ok=True)
                if WRITE_OUTPUTS:
                    output_dir.mkdir(parents=True, exist_ok=True)
                audit_path = summary_dir / "composition_audit.csv"
                audit = []
                for source in selected:
                    entry = dict(source)
                    prefix = "bms_altitude" if source["src_file_stem"].startswith("bms_altitude_") else "bms"
                    output = output_dir / f"{prefix}_{alias}_{source['ym']}{DFC_SUFFIX}.csv"
                    entry.update(output_path=str(output), dfc_condition=condition,
                                 status="pending", error="")
                    audit.append(entry)
                # 기존 파일이 있어도 합성 모드에서는 재계산한다.
                # 과거 파일을 새 구성표에서 생성한 결과로 잘못 인증하지 않기 위함.
                pd.DataFrame(audit).to_csv(audit_path, index=False, encoding="utf-8-sig")
                summaries = []
                with ProcessPoolExecutor(max_workers=WORKERS) as executor:
                    futures = {}
                    for entry in audit:
                        job = (entry["actual_input_path"], entry["output_path"],
                               WRITE_OUTPUTS, False, entry["source_model"], soc, buffer)
                        futures[executor.submit(_dfc_one_file_job, job)] = entry
                    for future in tqdm(as_completed(futures), total=len(futures),
                                       desc=f"{alias} {condition}"):
                        entry = futures[future]
                        try:
                            status, message, stats = future.result()
                            entry["status"] = status if WRITE_OUTPUTS else (
                                "summary_only" if status == "ok" else status)
                            if status == "error":
                                entry["error"] = message
                            elif status == "ok":
                                stats.update(file_stem=Path(entry["output_path"]).stem,
                                             id_token=alias, ym=entry["ym"])
                                summaries.append((entry["month_index"], stats))
                        except Exception as error:
                            entry.update(status="error", error=str(error))
                        pd.DataFrame(audit).to_csv(audit_path, index=False, encoding="utf-8-sig")
                pd.DataFrame([row for _, row in sorted(summaries)], columns=SUMMARY_COLUMNS).to_csv(
                    summary_dir / f"dfc_features_summary{DFC_SUFFIX}_{alias}.csv",
                    index=False, encoding="utf-8-sig")
                if any(entry["status"] == "error" for entry in audit):
                    raise RuntimeError(f"DFC 처리 실패. 확인: {audit_path}")
                print(f"[done] {alias} {condition}: {len(audit)} files; audit={audit_path}")


if __name__ == "__main__":
    run_synthetic()
