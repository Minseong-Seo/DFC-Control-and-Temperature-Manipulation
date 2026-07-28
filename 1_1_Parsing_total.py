import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

TIME_FMT = "%Y-%m-%d %H:%M:%S"

# 차량 선택
VEHICLE_MODEL = "EV6"     # "Ioniq5" 또는 "EV6"

if VEHICLE_MODEL not in {"Ioniq5", "EV6"}:
    raise ValueError("VEHICLE_MODEL은 'Ioniq5' 또는 'EV6'만 가능합니다.")


# time 제외 숫자 컬럼은 coerce 허용
NUM_COLS = ["soc", "pack_current", "pack_volt", "speed"]


# ─────────────────────────────────────────────────────────────
# (공통) 파일당 1회: time 변환 + 숫자 coerce
# ─────────────────────────────────────────────────────────────
def _prep_types_once(data: pd.DataFrame) -> pd.DataFrame:
    # time: 파일당 1회만 변환(강제 포맷)
    data["time"] = pd.to_datetime(data["time"].astype(str).str.strip(), format=TIME_FMT)

    # time 제외 숫자 컬럼은 coerce 허용
    for c in NUM_COLS:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors="coerce")
    return data


# ─────────────────────────────────────────────────────────────
# SOC=0 구간 보정 (원본 유지)
# ─────────────────────────────────────────────────────────────
def fix_soc_zero(data):
    i = 0
    n = len(data)

    while i < n - 1:
        if data.loc[i, 'soc'] > 0.5 and data.loc[i+1, 'soc'] == 0:
            start = i + 1
            end = start
            while end < n and data.loc[end, 'soc'] == 0:
                end += 1

            # current = 0
            if 'pack_current' in data.columns:
                data.loc[start:end-1, 'pack_current'] = 0  # 여기만 0구간까지만

            # 뒤 앵커가 있을 때만 보간
            if end < n:
                data.loc[start:end, 'soc'] = np.linspace(
                    data.loc[i, 'soc'], data.loc[end, 'soc'], end - i + 1
                )[1:]
                if 'pack_volt' in data.columns:
                    data.loc[start:end, 'pack_volt'] = np.linspace(
                        data.loc[i, 'pack_volt'], data.loc[end, 'pack_volt'], end - i + 1
                    )[1:]
        i += 1
    return data


# ─────────────────────────────────────────────────────────────
# 이상 파일 제외 (time은 이미 datetime64로 들어온다고 가정)
# ─────────────────────────────────────────────────────────────
def invalid_data(data):
    # ΔSOC, Δtime 벡터 계산
    dsoc = data['soc'].diff().abs()
    dtime = data['time'].diff()

    # 조건: SOC 변화 ≥ 10 % & 시간 간격 ≥ 12 h
    if ((dsoc >= 10) & (dtime >= pd.Timedelta(hours=12))).any():
        return True
    return False


# ─────────────────────────────────────────────────────────────
# charging 구간 구하기 (벡터화)
# ─────────────────────────────────────────────────────────────
def parse_charging(data):
    n = len(data)
    if n == 0:
        data['charging'] = 0
        return data

    # base charging (원본 의도: pack_current<0 & speed==0)
    charging = ((data['pack_current'] < 0) & (data['speed'] == 0)).astype(np.int8).to_numpy()
    data['charging'] = charging

    # ── outlier 제거 1: 0구간이 1분 이내면 charging=1로 메움 ──
    grp = np.cumsum(np.r_[True, charging[1:] != charging[:-1]])  # 1..k
    data['_chg_grp'] = grp
    g = data.groupby('_chg_grp', sort=False)

    g_val = g['charging'].first()
    g_first = g.apply(lambda x: x.index[0])
    g_last  = g.apply(lambda x: x.index[-1])

    t = data['time']
    g_t_start = t.loc[g_first.values].to_numpy()
    g_t_end   = t.loc[g_last.values].to_numpy()

    zero_groups = g_val[g_val == 0].index.to_numpy()
    if len(zero_groups) > 0:
        z_dur = (g_t_end[zero_groups - 1] - g_t_start[zero_groups - 1])
        short_zero = zero_groups[z_dur <= np.timedelta64(1, 'm')]
        for gid in short_zero:
            s = int(g_first.loc[gid])
            e = int(g_last.loc[gid])
            data.loc[s:e, 'charging'] = 1

    # ── outlier 제거 2: charging=1 구간이 5분 이하면 charging=0 ──
    charging2 = data['charging'].to_numpy().astype(np.int8)
    grp2 = np.cumsum(np.r_[True, charging2[1:] != charging2[:-1]])
    data['_chg_grp2'] = grp2
    g2 = data.groupby('_chg_grp2', sort=False)

    g2_val = g2['charging'].first()
    g2_first = g2.apply(lambda x: x.index[0])
    g2_last  = g2.apply(lambda x: x.index[-1])

    g2_t_start = t.loc[g2_first.values].to_numpy()
    g2_t_end   = t.loc[g2_last.values].to_numpy()

    one_groups = g2_val[g2_val == 1].index.to_numpy()
    if len(one_groups) > 0:
        o_dur = (g2_t_end[one_groups - 1] - g2_t_start[one_groups - 1])
        short_one = one_groups[o_dur <= np.timedelta64(5, 'm')]
        for gid in short_one:
            s = int(g2_first.loc[gid])
            e = int(g2_last.loc[gid])
            data.loc[s:e, 'charging'] = 0

    data.drop(columns=['_chg_grp', '_chg_grp2'], inplace=True, errors='ignore')
    data['charging'] = data['charging'].astype(np.int8)
    return data


# ─────────────────────────────────────────────────────────────
# rest 구간 구하기 (벡터화)
# ─────────────────────────────────────────────────────────────
def parse_rest(data):
    n = len(data)
    if n == 0:
        data['rest'] = 0
        return data

    # 초기 rest: 원본 루프의 실질 결과는 charging==1 구간을 rest=1로 만드는 것과 동일
    data['rest'] = (data['charging'] == 1).astype(np.int8)
    t = data['time']

    # 1) time gap > 5분이면 주변을 rest=1
    dt = t.diff()  # dt[i] = time[i]-time[i-1]
    gap = dt > pd.Timedelta(minutes=5)
    idx = np.flatnonzero(gap.to_numpy())
    if idx.size > 0:
        mark = np.unique(np.r_[idx - 1, idx])  # (i-1, i)
        mark = mark[(mark >= 0) & (mark < n)]
        data.loc[mark, 'rest'] = 1

    # 2) abnormal: 0<=pack_current<=1 구간이 10분 이상이면 rest=1
    ab = (data['pack_current'] >= 0) & (data['pack_current'] <= 1)
    ab_arr = ab.to_numpy()
    if ab_arr.any():
        ab_grp = np.cumsum(np.r_[True, ab_arr[1:] != ab_arr[:-1]])
        data['_ab_grp'] = ab_grp
        g = data.groupby('_ab_grp', sort=False)

        # 그룹이 abnormal인지(첫 값 기준)
        g_val = g.apply(lambda x: bool((0 <= x['pack_current'].iloc[0] <= 1)))
        g_first = g.apply(lambda x: x.index[0])
        g_last  = g.apply(lambda x: x.index[-1])

        for gid in g_val[g_val].index:
            s = int(g_first.loc[gid])
            e = int(g_last.loc[gid])
            if (t.iloc[e] - t.iloc[s]) >= pd.Timedelta(minutes=10):
                data.loc[s:e, 'rest'] = 1

        data.drop(columns=['_ab_grp'], inplace=True, errors='ignore')

    # 3) rest 떨어져있는 문제 수정: 1-0-...-1을 1분 이내면 메움
    r = data['rest'].to_numpy().astype(np.int8)
    r_grp = np.cumsum(np.r_[True, r[1:] != r[:-1]])
    data['_r_grp'] = r_grp
    g = data.groupby('_r_grp', sort=False)

    g_val = g['rest'].first()
    g_first = g.apply(lambda x: x.index[0])
    g_last  = g.apply(lambda x: x.index[-1])

    zero_groups = g_val[g_val == 0].index.to_numpy()
    for gid in zero_groups:
        prev_gid = gid - 1
        next_gid = gid + 1
        if (prev_gid not in g_val.index) or (next_gid not in g_val.index):
            continue
        if (g_val.loc[prev_gid] != 1) or (g_val.loc[next_gid] != 1):
            continue

        left_idx = int(g_last.loc[prev_gid])
        right_idx = int(g_first.loc[next_gid])

        if (t.iloc[right_idx] - t.iloc[left_idx]) <= pd.Timedelta(minutes=1):
            s = int(g_first.loc[gid])
            e = int(g_last.loc[gid])
            data.loc[s:e, 'rest'] = 1

    data.drop(columns=['_r_grp'], inplace=True, errors='ignore')
    data['rest'] = data['rest'].astype(np.int8)
    return data


# ─────────────────────────────────────────────────────────────
# (멀티프로세싱 워커) 파일 1개 처리
# ─────────────────────────────────────────────────────────────
def _one_file_job(args):
    in_path, out_path, overwrite = args

    if (not overwrite) and os.path.exists(out_path):
        return ("skip", os.path.basename(in_path), None)

    try:
        data = pd.read_csv(in_path)

        # time 1회 + 숫자 coerce 1회
        data = _prep_types_once(data)

        # SOC=0 보정
        data = fix_soc_zero(data)

        # invalid 체크
        if invalid_data(data):
            return ("skip_invalid", os.path.basename(in_path), None)

        # 파싱
        data = parse_charging(data)
        data = parse_rest(data)

        data.to_csv(out_path, index=False)
        return ("ok", os.path.basename(in_path), None)

    except Exception as e:
        return ("error", os.path.basename(in_path), str(e))


# ─────────────────────────────────────────────────────────────
# 폴더 전체 처리 (멀티프로세싱)
# ─────────────────────────────────────────────────────────────
def process_folder(merge_folder, parsing_folder, skip_existing=True, workers=8, chunksize=1):
    os.makedirs(parsing_folder, exist_ok=True)

    files = [fn for fn in os.listdir(merge_folder) if fn.lower().endswith('.csv') and not fn.startswith(".")]
    files.sort()

    jobs = []
    overwrite = not skip_existing
    for filename in files:
        in_path = os.path.join(merge_folder, filename)
        out_path = os.path.join(parsing_folder, filename.replace('.csv', '_CR.csv'))
        jobs.append((in_path, out_path, overwrite))

    if not jobs:
        print("[info] 처리할 CSV가 없습니다.")
        return

    ok = skip = skip_inv = err = 0
    errors = []

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_one_file_job, job) for job in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing Files (MP)", unit="file"):
            status, fname, msg = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            elif status == "skip_invalid":
                skip_inv += 1
            else:
                err += 1
                errors.append((fname, msg))

    print(f"[done] ok={ok}, skip_exist={skip}, skip_invalid={skip_inv}, error={err}")
    if errors:
        print("[errors] 아래 파일들 실패:")
        for f, m in errors[:30]:
            print(f"  - {f}: {m}")
        if len(errors) > 30:
            print(f"  ... ({len(errors)-30}개 더 있음)")


# ─────────────────────────────────────────────────────────────
# 선택 파일만 처리 (단일 프로세스: 요청대로 MP 아님)
# ─────────────────────────────────────────────────────────────
def process_selected(merge_folder, parsing_folder, selected_filenames):
    os.makedirs(parsing_folder, exist_ok=True)

    for filename in tqdm(selected_filenames, desc="Processing Selected", unit="file"):
        if not filename.lower().endswith('.csv'):
            print(f"[MISS] {filename} - not a csv")
            continue

        file_path = os.path.join(merge_folder, filename)
        if not os.path.isfile(file_path):
            print(f"[MISS] {filename} - not found")
            continue

        save_path = os.path.join(parsing_folder, filename.replace('.csv', '_CR.csv'))

        try:
            data = pd.read_csv(file_path)

            # time 1회 + 숫자 coerce 1회
            data = _prep_types_once(data)

            # (1) SOC=0 보정
            data = fix_soc_zero(data)

            # (2) invalid → 저장 안 함 (+ 기존 산출물 삭제는 원본 유지)
            if invalid_data(data):
                if os.path.exists(save_path):
                    os.remove(save_path)
                print(f"[SKIP] {filename} - invalid (removed old output if existed)")
                continue

            # (3) 파싱 & 저장
            data = parse_charging(data)
            data = parse_rest(data)
            data.to_csv(save_path, index=False)

        except Exception as e:
            print(f"[SKIP] {filename} - error: {e}")


# ─────────────────────────────────────────────────────────────
# 선택 파일 이후 파일들 처리 (멀티프로세싱)
# ─────────────────────────────────────────────────────────────
def process_folder_resume(merge_folder, parsing_folder, resume_after=None, start_idx=None,
                          skip_existing=True, workers=8, chunksize=1):
    """
    resume_after: 마지막으로 처리한 '파일명'을 넘기면 그 다음 파일부터 시작
                  (CR 출력 파일명을 넘겨도 자동으로 원본명으로 매핑)
    start_idx   : 정렬된 목록에서 시작할 인덱스(0-based). resume_after보다 우선순위 낮음.
    skip_existing: 이미 결과가 있으면 건너뛸지 여부
    """
    os.makedirs(parsing_folder, exist_ok=True)

    files = [f for f in os.listdir(merge_folder) if f.lower().endswith('.csv') and not f.startswith(".")]
    files.sort()

    start = 0
    if resume_after:
        raw_name = (resume_after.replace('_CR.csv', '.csv').replace('_cr.csv', '.csv'))
        if raw_name in files:
            start = files.index(raw_name) + 1
        else:
            print(f"[WARN] '{resume_after}'(원본 '{raw_name}')을 목록에서 찾지 못했어요. 처음부터 시작합니다.")
    elif start_idx is not None:
        if 0 <= start_idx < len(files):
            start = start_idx
        else:
            print(f"[WARN] start_idx {start_idx}가 범위를 벗어났어요. 0부터 시작합니다.")
            start = 0

    subset = files[start:]
    if not subset:
        print("[info] 처리할 파일이 없습니다.")
        return

    jobs = []
    overwrite = not skip_existing
    for filename in subset:
        in_path = os.path.join(merge_folder, filename)
        out_path = os.path.join(parsing_folder, filename.replace('.csv', '_CR.csv'))
        jobs.append((in_path, out_path, overwrite))

    ok = skip = skip_inv = err = 0
    errors = []

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_one_file_job, job) for job in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=f"Processing Files (MP) (from {start}/{len(files)})", unit="file"):
            status, fname, msg = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            elif status == "skip_invalid":
                skip_inv += 1
            else:
                err += 1
                errors.append((fname, msg))

    print(f"[done] ok={ok}, skip_exist={skip}, skip_invalid={skip_inv}, error={err}")
    if errors:
        print("[errors] 아래 파일들 실패:")
        for f, m in errors[:30]:
            print(f"  - {f}: {m}")
        if len(errors) > 30:
            print(f"  ... ({len(errors)-30}개 더 있음)")


# ─────────────────────────────────────────────────────────────
# 파일 경로 (원본 예시 보존)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    # merge_folder_path = '/Volumes/T7/DFC/Ioniq5'
    # parsing_folder_path = '/Volumes/T7/DFC/CR_parsing_Ioniq5'

    ROOT_DIR = "/Volumes/T7/DFC"

    merge_folder_path = os.path.join(
        ROOT_DIR,
        VEHICLE_MODEL
    )

    parsing_folder_path = os.path.join(
        ROOT_DIR,
        f"CR_parsing_{VEHICLE_MODEL}"
    )

    # # 선택 파일 이후 파일들 돌리기
    # process_folder_resume(
    #     merge_folder_path,
    #     parsing_folder_path,
    #     resume_after='bms_altitude_01241364627_2024-01.csv',  # 마지막 완료한 파일명(출력명/원본명 둘 다 OK)
    #     skip_existing=True,
    #     workers=8
    # )

    # 전체 폴더
    process_folder(
        merge_folder_path,
        parsing_folder_path,
        skip_existing=True,
        workers=8
    )

    # # 선택 파일
    # some_files = ['bms_01241228082_2023-10.csv']
    # process_selected(merge_folder_path, parsing_folder_path, some_files)
