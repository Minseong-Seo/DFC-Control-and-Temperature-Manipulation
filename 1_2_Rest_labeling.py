import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# 전역 설정(토글)
# ─────────────────────────────────────────────────────────────
FULLCHARGE_PARKING_MODE = True
MERGE_GAP_MINUTES = 30
MIN_UNCHARG_HOURS = 7.0

TIME_FMT = "%Y-%m-%d %H:%M:%S"


def _ensure_time_once(data: pd.DataFrame) -> pd.DataFrame:
    """
    time 컬럼을 파일당 1회만 datetime으로 변환.
    (이미 datetime64면 아무 것도 하지 않음)
    """
    if "time" in data.columns and (not pd.api.types.is_datetime64_any_dtype(data["time"])):
        data["time"] = pd.to_datetime(data["time"], format=TIME_FMT)
    return data


#미사용-완전충전중 및 미사용-부분충전중
def rest_charging(data):

    #충전구간
    charg = []
    if data.loc[0, 'charging'] == 1:
        charg.append(0)
    for i in range(len(data) - 1):
        if data.loc[i, 'charging'] != data.loc[i + 1, 'charging']:
            charg.append(i + 1)
    if data.loc[len(data) - 1, 'charging'] == 1:
        charg.append(len(data) - 1)

    #완전충전구간
    full = []
    for i in range(len(charg) - 1):
        if data.loc[charg[i], 'charging'] == 1 and any(data.loc[charg[i + 1]-3:charg[i + 1]+6, 'soc'] >= 95):
            full.append(charg[i])
            full.append(charg[i + 1])

    full = list(set(full))
    full.sort()

    #완전충전 및 부분충전 분류
    data['R_charg'] = 0
    data['R_partial_charg'] = 0
    for i in range(0, len(charg) - 1, 2):
        if data.loc[charg[i], 'charging'] == 1:
            data.loc[charg[i]:charg[i + 1]-1, 'R_partial_charg'] = 1

    for i in range(0, len(full) - 1, 2):
        if data.loc[full[i], 'charging'] == 1:
            data.loc[full[i]:full[i + 1]-1, 'R_partial_charg'] = 0
            data.loc[full[i]:full[i + 1]-1, 'R_charg'] = 1

    return data, full


#미사용-충전후
def rest_aftercharg(data, full):

    data['R_aftercharg'] = 0
    # ❗ time 변환 제거 (process_data에서 1회만 수행)

    # ───────────────── 1단계: 완충 후 rest → R_aftercharg ─────────────────
    for i in range(len(data) - 1):
        if (
            data.loc[i, 'R_charg'] == 1 and
            data.loc[i + 1, 'R_charg'] == 0 and
            data.loc[i + 1, 'rest'] == 1
        ):
            j = i  # rest 구간 시작 인덱스
            while j < len(data) and data.loc[j, 'rest'] == 1:
                data.at[j, 'R_aftercharg'] = 1
                j += 1

    # ───────────────── 2단계: 출발 직전 히터 구간 세분화 ─────────────────
    limit_time = pd.Timedelta(minutes=5)
    limit_rest_gap = pd.Timedelta(minutes=30)

    has_curr = 'pack_current' in data.columns

    for i in range(len(data) - 1):
        if (
            data.loc[i, 'R_charg'] == 1 and
            data.loc[i + 1, 'R_charg'] == 1 and
            data.loc[i + 1, 'time'] - data.loc[i, 'time'] > limit_time
        ):
            j = i
            while j < len(data) and data.loc[j, 'R_charg'] == 1:
                j += 1

            if j <= i + 1:
                continue

            rest_gap = data.loc[j - 1, 'time'] - data.loc[i + 1, 'time']

            heater_start_idx = i + 1
            soc_start = data.loc[heater_start_idx, 'soc']
            if soc_start < 95:
                continue

            if has_curr:
                heater_slice = data.loc[i + 1:j - 1]
                pc = heater_slice['pack_current']
                if (pc <= 0).all():
                    continue

            if rest_gap <= limit_rest_gap:
                data.loc[i:j - 1, 'R_aftercharg'] = 1

                k = i + 1
                while k < len(data) and data.loc[k, 'R_charg'] == 1:
                    data.at[k, 'R_charg'] = 0
                    data.at[k, 'R_aftercharg'] = 1
                    k += 1

    return data


def rest_uncharg(
    data,
    enable_merge: bool | None = None,
    min_uncharg_hours: float | None = None,
):
    """
    (원본 주석 유지)
    """

    # 1) 기본 라벨링 ----------------------------------------------------------
    data['R_uncharg'] = 0
    for i in range(len(data)-1):
        if (
            data.loc[i, 'rest'] == 1
            and data.loc[i, 'R_charg'] == 0
            and data.loc[i, 'R_partial_charg'] == 0
            and data.loc[i, 'R_aftercharg'] == 0
        ):
            data.loc[i, 'R_uncharg'] = 1

    # 2) 완충 후 이동주차 모드 on/off ----------------------------------------
    if enable_merge is None:
        enable = FULLCHARGE_PARKING_MODE
    else:
        enable = enable_merge

    if not enable:
        return data

    # 3) time 컬럼 datetime 보장 --------------------------------------------
    # ❗ time 변환 제거: process_data에서 1회 변환이므로 여기서는 타입만 확인
    if not pd.api.types.is_datetime64_any_dtype(data['time']):
        # 예외적으로 혹시라도 남아있으면 안전변환(하지만 정상흐름에선 안 탐)
        data['time'] = pd.to_datetime(data['time'], format=TIME_FMT)

    time_limit = pd.Timedelta(minutes=MERGE_GAP_MINUTES)

    if min_uncharg_hours is None:
        min_uncharg_hours = MIN_UNCHARG_HOURS

    if (min_uncharg_hours is not None) and (min_uncharg_hours > 0):
        min_uncharg_td = pd.Timedelta(hours=min_uncharg_hours)
    else:
        min_uncharg_td = None

    n = len(data)

    def get_blocks(series):
        blks = []
        in_block = False
        start = None
        for i, v in enumerate(series):
            if v == 1 and not in_block:
                start, in_block = i, True
            elif v == 0 and in_block:
                blks.append((start, i - 1))
                in_block = False
        if in_block:
            blks.append((start, n - 1))
        return blks

    after_blocks   = get_blocks(data['R_aftercharg'].fillna(0).astype(int))
    uncharg_blocks = get_blocks(data['R_uncharg'].fillna(0).astype(int))
    charg_blocks   = get_blocks(data['R_charg'].fillna(0).astype(int))

    def passes_uncharg_duration(unc_start: int, unc_end: int) -> bool:
        if min_uncharg_td is None:
            return True
        duration = data.loc[unc_end, 'time'] - data.loc[unc_start, 'time']
        return duration >= min_uncharg_td

    for (aft_start, aft_end) in after_blocks:
        next_unc = next(
            ((u_start, u_end) for (u_start, u_end) in uncharg_blocks
             if u_start > aft_end),
            None
        )
        if not next_unc:
            continue

        unc_start, unc_end = next_unc
        gap = data.loc[unc_start, 'time'] - data.loc[aft_end, 'time']

        if (gap <= time_limit) and passes_uncharg_duration(unc_start, unc_end):
            data.loc[aft_end + 1:unc_end, 'R_aftercharg'] = 1
            data.loc[aft_end + 1:unc_end, 'R_uncharg']    = 0

    for (ch_start, ch_end) in charg_blocks:
        next_unc = next(
            ((u_start, u_end) for (u_start, u_end) in uncharg_blocks
             if u_start > ch_end),
            None
        )
        if not next_unc:
            continue

        unc_start, unc_end = next_unc
        gap = data.loc[unc_start, 'time'] - data.loc[ch_end, 'time']

        if (gap <= time_limit) and passes_uncharg_duration(unc_start, unc_end):
            data.loc[ch_end:unc_end, 'R_aftercharg'] = 1
            data.loc[ch_end:unc_end, 'R_uncharg']    = 0

    return data


#세분화 포함된 엑셀 생성
def process_data(file_path, save_path, enable_merge: bool | None = None):
    data = pd.read_csv(file_path)

    # ✅ time 변환은 여기서만 1회 수행
    data = _ensure_time_once(data)

    data, full = rest_charging(data)
    data = rest_aftercharg(data, full)
    data = rest_uncharg(data, enable_merge=enable_merge)

    data.to_csv(save_path, index=False)


# 파일 돌리기
def process_folder(input_folder, output_folder, skip_existing=True, enable_merge: bool | None = None):
    os.makedirs(output_folder, exist_ok=True)

    files = [fn for fn in os.listdir(input_folder) if fn.lower().endswith('.csv') and not fn.startswith(".")]
    files.sort()

    for filename in tqdm(files, desc='Processing Files'):
        in_path  = os.path.join(input_folder, filename)
        out_name = filename.replace('_CR.csv', '_r.csv')
        out_path = os.path.join(output_folder, out_name)

        if skip_existing and os.path.exists(out_path):
            continue

        try:
            process_data(in_path, out_path, enable_merge=enable_merge)
        except Exception as e:
            tqdm.write(f"[error] {filename}: {e}")
            continue


def resume_after_file(input_folder: str, output_folder: str, name: str | None,
                      skip_existing: bool = True, enable_merge: bool | None = None):
    files = [fn for fn in os.listdir(input_folder) if fn.lower().endswith('.csv') and not fn.startswith(".")]
    files.sort()

    start = 0
    if name:
        target = name.strip().lower().replace('_cr.csv', '.csv')

        lowers = [f.lower() for f in files]
        if target in lowers:
            start = lowers.index(target) + 1
        else:
            cands = [i for i, f in enumerate(lowers) if (target in f) or (name.lower() in f)]
            if len(cands) == 1:
                start = cands[0] + 1
                print(f"[info] 부분 문자열로 유일 매칭: {files[cands[0]]} → 그 다음부터 시작")
            elif len(cands) > 1:
                print(f"[warn] '{name}' 로 {len(cands)}개 매칭. 더 구체적으로 입력하세요:")
                for i in cands[:20]:
                    print("  -", files[i])
                return
            else:
                print(f"[warn] '{name}' 를 찾지 못했습니다. 처음부터 시작합니다.")

    for filename in tqdm(files[start:], desc=f"Processing Files (from {start}/{len(files)})"):
        in_path  = os.path.join(input_folder, filename)
        out_name = filename.replace('_CR.csv', '_r.csv')
        out_path = os.path.join(output_folder, out_name)

        if skip_existing and os.path.exists(out_path):
            continue

        try:
            process_data(in_path, out_path, enable_merge=enable_merge)
        except Exception as e:
            tqdm.write(f"[error] {filename}: {e}")
            continue


def process_until_file(input_folder: str,
                       output_folder: str,
                       name: str,
                       include_name: bool = True,
                       skip_existing: bool = True,
                       enable_merge: bool | None = None):
    files = [fn for fn in os.listdir(input_folder) if fn.lower().endswith('.csv') and not fn.startswith(".")]
    files.sort()
    if not files:
        print("[warn] 입력 폴더에 csv 파일이 없습니다.")
        return

    target = name.strip().lower().replace('_cr.csv', '.csv')
    lowers = [f.lower() for f in files]

    if target in lowers:
        end_idx = lowers.index(target)
    else:
        cands = [i for i, f in enumerate(lowers) if (target in f) or (name.lower() in f)]
        if len(cands) == 1:
            end_idx = cands[0]
            print(f"[info] 부분 문자열로 유일 매칭: {files[end_idx]} → 그 파일까지 처리")
        elif len(cands) > 1:
            print(f"[warn] '{name}' 로 {len(cands)}개 매칭. 더 구체적으로 입력하세요:")
            for i in cands[:20]:
                print("  -", files[i])
            return
        else:
            print(f"[warn] '{name}' 를 찾지 못했습니다. 전체 처리로 대체합니다.")
            end_idx = len(files) - 1

    if not include_name:
        end_idx -= 1
    if end_idx < 0:
        print("[info] include_name=False 이고 기준 파일이 첫 번째라서 처리할 파일이 없습니다.")
        return

    for filename in tqdm(files[:end_idx + 1], desc=f"Processing Files (0..{end_idx})"):
        in_path  = os.path.join(input_folder, filename)
        out_name = filename.replace('_CR.csv', '_r.csv')
        out_path = os.path.join(output_folder, out_name)

        if skip_existing and os.path.exists(out_path):
            continue

        try:
            process_data(in_path, out_path, enable_merge=enable_merge)
        except Exception as e:
            tqdm.write(f"[error] {filename}: {e}")
            continue


def _one_file_job(args):
    in_path, out_path, enable_merge, overwrite = args

    if (not overwrite) and os.path.exists(out_path):
        return ("skip", os.path.basename(in_path), None)

    try:
        process_data(in_path, out_path, enable_merge=enable_merge)
        return ("ok", os.path.basename(in_path), None)
    except Exception as e:
        return ("error", os.path.basename(in_path), str(e))


def process_folder_mp(
    input_folder: str,
    output_folder: str,
    skip_existing: bool = True,
    enable_merge: bool | None = None,
    workers: int | None = None,
    chunksize: int = 1,
):
    os.makedirs(output_folder, exist_ok=True)

    files = [fn for fn in os.listdir(input_folder) if fn.lower().endswith(".csv") and not fn.startswith(".")]
    files.sort()

    jobs = []
    for filename in files:
        in_path = os.path.join(input_folder, filename)
        out_name = filename.replace("_CR.csv", "_r.csv")
        out_path = os.path.join(output_folder, out_name)

        overwrite = not skip_existing
        jobs.append((in_path, out_path, enable_merge, overwrite))

    if not jobs:
        print("[info] 처리할 CSV가 없습니다.")
        return

    ok = skip = err = 0
    errors = []

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_one_file_job, job) for job in jobs]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing Files (MP)"):
            status, fname, msg = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                err += 1
                errors.append((fname, msg))

    print(f"[done] ok={ok}, skip={skip}, error={err}")
    if errors:
        print("[errors] 아래 파일들 실패:")
        for f, m in errors[:30]:
            print(f"  - {f}: {m}")
        if len(errors) > 30:
            print(f"  ... ({len(errors)-30}개 더 있음)")


# ─────────────────────────────────────────────────────────────
# 폴더 전체 코드실행 (예시)
# ─────────────────────────────────────────────────────────────
parsing_folder_path = r'/Volumes/T7/DFC/CR_parsing_Ioniq5'
save_folder_path = r'/Volumes/T7/DFC/R_parsing_origin_Ioniq5'

# # 처음부터 선택파일 까지 '포함'하여 처리
# process_until_file(
#     parsing_folder_path,
#     save_folder_path,
#     'bms_01241228014_2023-09_CR.csv',
#     include_name=True,
#     skip_existing=False,
#     enable_merge=None
# )

# # 전체 파일 돌리기 (전역 토글값 사용)
# process_folder(parsing_folder_path, save_folder_path, skip_existing=True)

# # 선택 파일 이후 파일 돌리기 (전역 토글값 사용)
# resume_after_file(parsing_folder_path, save_folder_path, 'bms_altitude_01241592878_2024-02_CR.csv', skip_existing=True)

# # 데이터 하나만 코드실행
# in_file  = r'/Volumes/T7/DFC/CR_parsing_Ioniq5/bms_01241228055_2023-03_CR.csv'
# out_file = r'/Volumes/T7/DFC/R_parsing_origin Ioniq5/bms_01241228055_2023-03_r.csv'
# process_data(in_file, out_file, enable_merge=None)

if __name__ == "__main__":
    process_folder_mp(
        parsing_folder_path,
        save_folder_path,
        skip_existing=True,
        enable_merge=None,
        workers=8,
        chunksize=1
    )