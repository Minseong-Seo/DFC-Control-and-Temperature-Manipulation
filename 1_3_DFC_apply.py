import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

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


# ─────────────────────────────────────────────────────────────
# DFC 알고리즘 적용 + 이벤트 통계(선택)
#   - 실제 time shift가 발생한 이벤트들의 delayed_time을 수집
#   - collect_stats=True일 때 (events 리스트, 요약 dict)도 함께 반환
#   - collect_stats=False일 때는 단순히 DFC 적용된 DataFrame만 반환
#   - DFC_START_SOC, TIME_MARGIN은 전역 상수로 사용 및 변경 가능
# ─────────────────────────────────────────────────────────────

# ===== User Settings =====
DFC_START_SOC = int(os.getenv("DFC_START_SOC", "80"))          # %
TIME_MARGIN_MIN = int(os.getenv("TIME_MARGIN_MIN", "60"))      # min

TIME_MARGIN = pd.Timedelta(minutes=TIME_MARGIN_MIN)

DFC_SUFFIX = f"_DFC{DFC_START_SOC}_t{TIME_MARGIN_MIN}"
# ==========================

def DFC(data, collect_stats=False):
    data = remove_consecutive_ones(data)

    # time 파싱 안전화
    if 'time' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['time']):
        data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

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

    # ─ 충전단계2: 정확히 DFC_START_SOC 기준 + 쌍 유지
    charg_2_pairs = []   # (delay_start, charge_end)
    delay_start = []
    for j in range(0, len(dfc_charg) - 1, 2):
        start_j, end_j = dfc_charg[j], dfc_charg[j + 1]
        found = False
        # (1) 구간 내부에서 {DFC_START_SOC-1}→DFC_START_SOC가 되는 순간을 찾으면 그 다음 인덱스를 지연 시작으로 사용
        for i in range(start_j, max(start_j, end_j)):  # [start_j, end_j-1]
            if (data.loc[i, 'soc'] < DFC_START_SOC) and (data.loc[i + 1, 'soc'] == DFC_START_SOC):
                charg_2_pairs.append((i + 1, end_j))
                delay_start.append(i + 1)
                found = True
                break
        # (2) 위가 없고, **충전 시작 SOC가 DFC_START_SOC 이상이면 충전 시작 시점부터 지연 시작**
        if (not found) and (data.loc[start_j, 'soc'] >= DFC_START_SOC):
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
    #     if data.loc[end_aftercharg[i] - 1, 'soc'] < DFC_START_SOC:
    #         remove_end.append(end_aftercharg[i])
    # end_aftercharg = [idx for idx in end_aftercharg if idx not in remove_end]

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

            delayed_time = (t1 - t0 - TIME_MARGIN)
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
                data.loc[dstart + 1: cend, 'time'] = data.loc[dstart + 1: cend, 'time'] + delayed_time


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
    # 파일명 예: bms_01241228021_2023-02_r.csv → id_token=01241228021, ym=2023-02
    id_token, ym = "unknown", "0000-00"
    parts = p.stem.split("_")
    if len(parts) >= 3:
        id_token = parts[1]
        ym = parts[2]
    return id_token, ym

def _collect_input_files(input_folder, pattern="*.csv"):
    input_dir = Path(input_folder)
    files = [p for p in sorted(input_dir.glob(pattern)) if not p.name.startswith(".")]
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
    (in_path, out_path_or_None, write_outputs, skip_existing) 받아서
    - DFC 처리(옵션으로 파일 저장)
    - stats dict 반환 (요약용)
    """
    in_path, out_path, write_outputs, skip_existing = args
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
        jobs.append((str(p), out_path, write_outputs, skip_existing))

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

# ─────────────────────────────────────────────────────────────
# 실행 예시(필요한 부분만 주석 해제해서 사용)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 경로 설정
    input_folder_path = r'/Volumes/T7/DFC/R_parsing_origin_Ioniq5'

    output_root = Path('/Volumes/T7/DFC/DFC_origin_Ioniq5')
    output_folder_path = str(output_root / DFC_SUFFIX.lstrip("_"))

    summary_root = Path('/Volumes/T7/DFC/DFC_summary_Ioniq5')
    summary_folder_path = str(summary_root / DFC_SUFFIX.lstrip("_"))

    os.makedirs(output_folder_path, exist_ok=True)
    os.makedirs(summary_folder_path, exist_ok=True)

    summary_csv_path = os.path.join(
        summary_folder_path,
        f"dfc_features_summary{DFC_SUFFIX}.csv"
    )

    #input_folder_path = r'C:\Users\junny\SynologyDrive\SamsungSTF\Processed_Data\DFC\EV6\R_parsing_완충후이동주차'
    #output_folder_path = r'C:\Users\junny\SynologyDrive\SamsungSTF\Processed_Data\DFC\EV6\DFC_완충후이동주차'
    #summary_folder_path = r'C:\Users\junny\SynologyDrive\SamsungSTF\Processed_Data\DFC\EV6\DFC_수정용_251202'
    #summary_csv_path = os.path.join(summary_folder_path, f'dfc_features_summary{DFC_SUFFIX}.csv')
    #os.makedirs(summary_folder_path, exist_ok=True)

    # #④ 파일 하나만 돌리기 (저장 O, 통계 확인)
    # file_name = "bms_01241228082_2023-10_r.csv"  # ← 여기만 바꿔주면 됨
    # file_path = os.path.join(input_folder_path, file_name)
    # save_path = os.path.join(output_folder_path, file_name.replace('_r.csv', f"{DFC_SUFFIX}.csv"))

    # processed_df, stats = process_DFC_file(
    #     file_path,
    #     save_path=save_path,
    #     collect_stats=True,  # 통계도 보고 싶으면 True
    #     write_output=True  # DFC CSV 저장하고 싶으면 True
    # )

    # print(stats)

    # # ① 폴더 전체 돌리기 (파일이름 오름차순) + 이미 있으면 스킵
    # process_DFC_folder(
    #     input_folder_path,
    #     output_folder_path,
    #     summary_csv_path=summary_csv_path,
    #     pattern="*.csv",
    #     write_outputs=True,
    #     skip_existing=True   # ← 결과 있으면 건너뜀
    # )

    # # ② 폴더 전체 "요약만" 생성 (개별 파일 저장 X) — 스킵 옵션은 영향 적음
    # process_DFC_folder(
    #     input_folder_path,
    #     output_folder_path,
    #     summary_csv_path=summary_csv_path,
    #     pattern="*.csv",
    #     write_outputs=False,
    #     skip_existing=False
    # )

    # ③ 시작~끝 인덱스로 나눠서 돌리기 (이름순, end_idx 포함) + 스킵
    # process_DFC_folder_slice(
    #     input_folder_path,
    #     output_folder_path,
    #     start_idx=644,
    #     end_idx=766,  # ← 0~9 총 10개
    #     summary_csv_path=summary_csv_path,
    #     pattern="*.csv",
    #     write_outputs=True,
    #     skip_existing=True
    # )


    # 멀티프로세스
    process_DFC_folder_mp(
        input_folder_path,
        output_folder_path,
        summary_csv_path=summary_csv_path,
        pattern="*.csv",
        write_outputs=True,
        skip_existing=True,
        workers=8,   # 네트워크 드라이브면 4~8 추천
    )

