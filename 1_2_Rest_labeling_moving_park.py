import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 차량 선택
VEHICLE_MODEL = "EV6"     # "Ioniq5" 또는 "EV6"

if VEHICLE_MODEL not in {"Ioniq5", "EV6"}:
    raise ValueError("VEHICLE_MODEL은 'Ioniq5' 또는 'EV6'만 가능합니다.")

# ─────────────────────────────────────────────────────────────
# 전역 설정(토글)
# - 완충 후 이동주차 모드: R_aftercharg 끝난 뒤 ≤ MERGE_GAP_MINUTES 내 시작하는
#   R_uncharg 블록(및 사이 구간)을 R_aftercharg로 흡수
# ─────────────────────────────────────────────────────────────


FULLCHARGE_PARKING_MODE = True     # 완충 후 이동주차 모드 on/off
MERGE_GAP_MINUTES = 30              # 흡수 기준 간격(분)

# R_uncharg 블록 길이(시간) 조건
# - 예: 10.0 → R_uncharg 블록 길이가 10시간 이상일 때만 "완충후 이동주차"로 흡수
# - 0 또는 None 으로 두면 길이 조건을 사용하지 않음(= 기존 로직과 동일)
MIN_UNCHARG_HOURS       = 7.0

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
        if data.loc[charg[i], 'charging'] == 1 and any(data.loc[charg[i + 1]-3:charg[i + 1]+6, 'soc'] >= 95):  # 배터리 설계상 완충기준을 SOC=96.5~97%로 잡음. 넉넉잡아 95를 완충이라고 정의. soc 측정오류 고려하여 범위잡아서 95 이상인 셀 있으면 완충이라고 생각.
            full.append(charg[i])
            full.append(charg[i + 1])

    full = list(set(full))
    full.sort()

    #완전충전 및 부분충전 분류
    data['R_charg'] = 0
    data['R_partial_charg'] = 0
    for i in range(0, len(charg) - 1, 2):
        if data.loc[charg[i], 'charging'] == 1 :
            data.loc[charg[i]:charg[i + 1]-1, 'R_partial_charg'] = 1

    for i in range(0, len(full) - 1, 2):
        if data.loc[full[i], 'charging'] == 1:
            data.loc[full[i]:full[i + 1]-1, 'R_partial_charg'] = 0
            data.loc[full[i]:full[i + 1]-1, 'R_charg'] = 1

    return data, full


#미사용-충전후
def rest_aftercharg(data, full):

    data['R_aftercharg'] = 0
    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

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
    # (charging으로 파싱되지만 R_aftercharg로 세분화)
    limit_time = pd.Timedelta(minutes=5)
    limit_rest_gap = pd.Timedelta(minutes=30)  # 히터 구간 길이 한도

    has_curr = 'pack_current' in data.columns  # 전류 컬럼 존재 여부

    for i in range(len(data) - 1):
        if (
            data.loc[i, 'R_charg'] == 1 and
            data.loc[i + 1, 'R_charg'] == 1 and
            data.loc[i + 1, 'time'] - data.loc[i, 'time'] > limit_time
        ):
            # ── 히터 구간 끝(index) 찾기: R_charg == 1 이 계속되는 구간 ──
            j = i
            while j < len(data) and data.loc[j, 'R_charg'] == 1:
                j += 1  # j는 R_charg == 0이 되는 첫 행

            # rest 구간이 실제로 존재하는지 체크 (i+1 ~ j-1)
            if j <= i + 1:
                continue

            # 히터 구간 길이( i+1 시점과 종료 시점 ) 계산
            rest_gap = data.loc[j - 1, 'time'] - data.loc[i + 1, 'time']

            # ── [추가 조건 1] SOC 조건: 히터 구간 시작 SOC가 95 이상이면 제외 ──
            heater_start_idx = i + 1
            soc_start = data.loc[heater_start_idx, 'soc']  # <-- 숫자 그대로 사용
            if soc_start < 95:
                continue

            # ── [추가 조건 2] current 패턴 검사 ───────────────────────
            # 원격제어 히터 구간이라면 pack_current가 충전/방전이 섞여서 나타날 가능성이 큼.
            # 만약 이 구간에서 current 가 항상 0 이하(충전 방향만)라면
            # "히터 구간이 아니라 그냥 충전"으로 보고 스킵한다.
            if has_curr:
                heater_slice = data.loc[i + 1:j - 1]
                pc = heater_slice['pack_current']  # <-- 숫자 그대로 사용

                # current가 모두 0 이하이면 히터로 보지 않음
                if (pc <= 0).all():
                    continue

            # ── 최종: 30분 이내인 경우에만 aftercharg 라벨 적용 ────────────
            if rest_gap <= limit_rest_gap:
                # 1) i ~ j-1 까지 aftercharg
                data.loc[i:j - 1, 'R_aftercharg'] = 1

                # 2) i+1 부터 R_charg==1 이 끊길 때까지 라벨 변경
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
    1) 기본 라벨링:
       rest == 1 & (R_charg==0, R_partial_charg==0, R_aftercharg==0) → R_uncharg = 1

    2) (옵션) 완충 후 이동주차 모드(FULLCHARGE_PARKING_MODE):
       (a) R_aftercharg 블록 끝난 뒤 ≤ MERGE_GAP_MINUTES 내 시작하는
           R_uncharg 블록이 있으면,
           - (선택) 해당 R_uncharg 블록 길이 ≥ min_uncharg_hours 조건을 만족할 때
             사이 구간+해당 R_uncharg 블록 전체를 R_aftercharg로 흡수
       (b) R_charg 블록 끝난 뒤도 동일 규칙 적용

    enable_merge:
       - None : 전역 FULLCHARGE_PARKING_MODE 값을 따름
       - True : 이 호출에 한해 강제 on
       - False: 이 호출에 한해 강제 off

    min_uncharg_hours:
       - None : 전역 MIN_UNCHARG_HOURS 사용
       - > 0  : 이 호출에 한해 R_uncharg 블록 최소 길이(시간)를 덮어씀
       - <=0  : 길이 조건 사용하지 않음 (gap 조건만 사용)
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
        # 모드 꺼져 있으면 여기서 바로 반환
        return data

    # 3) time 컬럼 datetime 보장 --------------------------------------------
    if not pd.api.types.is_datetime64_any_dtype(data['time']):
        data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

    time_limit = pd.Timedelta(minutes=MERGE_GAP_MINUTES)

    # R_uncharg 최소 길이 기준 설정 -----------------------------------------
    if min_uncharg_hours is None:
        min_uncharg_hours = MIN_UNCHARG_HOURS

    if (min_uncharg_hours is not None) and (min_uncharg_hours > 0):
        min_uncharg_td = pd.Timedelta(hours=min_uncharg_hours)
    else:
        # 0 또는 None 이면 길이 조건 사용 안 함
        min_uncharg_td = None

    n = len(data)

    def get_blocks(series):
        """연속된 1 구간 (start, end) 리스트 반환 (end 포함)"""
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

    # 보조: R_uncharg 블록이 최소 길이 조건을 만족하는지 --------------------
    def passes_uncharg_duration(unc_start: int, unc_end: int) -> bool:
        """
        - min_uncharg_td is None → 길이 조건 사용 안 함 → 항상 True
        - 그 외: 블록 duration >= min_uncharg_td 인지 검사
        """
        if min_uncharg_td is None:
            return True
        duration = data.loc[unc_end, 'time'] - data.loc[unc_start, 'time']
        return duration >= min_uncharg_td

    # (a) 기존 규칙: R_aftercharg 끝 기준 -------------------------------
    for (aft_start, aft_end) in after_blocks:
        # aft_end 이후 첫 번째 R_uncharg 블록
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
            # aft_end 이후부터 R_uncharg 블록 끝까지 흡수 (사이 구간 포함)
            data.loc[aft_end + 1:unc_end, 'R_aftercharg'] = 1
            data.loc[aft_end + 1:unc_end, 'R_uncharg']    = 0

    # (b) 새 규칙: R_charg 끝 기준 ---------------------------------------
    for (ch_start, ch_end) in charg_blocks:
        # ch_end 이후 첫 번째 R_uncharg 블록
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
            # ch_end 이후부터 R_uncharg 블록 끝까지 흡수 (사이 구간 포함)
            data.loc[ch_end:unc_end, 'R_aftercharg'] = 1
            data.loc[ch_end:unc_end, 'R_uncharg']    = 0

    return data



#세분화 포함된 엑셀 생성
def process_data(file_path, save_path, enable_merge: bool | None = None):
    data = pd.read_csv(file_path)

    data, full = rest_charging(data)
    data = rest_aftercharg(data, full)
    data = rest_uncharg(data, enable_merge=enable_merge)  # ← 호출 단위 on/off 가능

    # Save the modified DataFrame to a new CSV file
    data.to_csv(save_path, index=False)

# 파일 돌리기
def process_folder(input_folder, output_folder, skip_existing=True, enable_merge: bool | None = None):
    """
    폴더 내 모든 CSV 처리.
    - skip_existing=True: 이미 *_r.csv 결과가 있으면 건너뜀
    - enable_merge: 완충 후 이동주차 모드 on/off (None이면 전역값 따름)
    """
    os.makedirs(output_folder, exist_ok=True)

    # CSV만 정렬하여 처리
    files = [fn for fn in os.listdir(input_folder) if fn.lower().endswith('.csv') and not fn.startswith('._')]
    files.sort()

    for filename in tqdm(files, desc='Processing Files'):
        in_path  = os.path.join(input_folder, filename)
        out_name = filename.replace('_CR.csv', '_r.csv')
        out_path = os.path.join(output_folder, out_name)

        if skip_existing and os.path.exists(out_path):
            # 있으면 건너뜀 (필요하면 로그 켜기)
            # tqdm.write(f"[skip] {out_name}")
            continue

        try:
            process_data(in_path, out_path, enable_merge=enable_merge)
        except Exception as e:
            tqdm.write(f"[error] {filename}: {e}")
            continue



def resume_after_file(input_folder: str, output_folder: str, name: str | None, skip_existing: bool = True, enable_merge: bool | None = None):
    """
    이름(원본 .csv / 출력 _CR.csv / 일부 문자열)만 주면
    그 '다음 파일'부터 처리 시작.
    - input_folder: CR 파싱된 CSV들이 있는 폴더 (예: .../CR_parsing/)
    - output_folder: 결과 저장 폴더
    - name: 기준 파일명 (예: 'bms_..._2023-12_CR.csv' 또는 일부 문자열). None이면 처음부터.
    - skip_existing: 이미 *_r.csv 있으면 건너뛸지
    - enable_merge: 완충 후 이동주차 모드 on/off (None이면 전역값 따름)
    """

    # 목록 정렬
    files = [fn for fn in os.listdir(input_folder) if fn.lower().endswith('.csv') and not fn.startswith('._')]
    files.sort()

    start = 0
    if name:
        # 기준 문자열 전처리: _CR.csv를 원본 .csv로 맵핑해 검색 유연성 ↑
        target = name.strip().lower().replace('_cr.csv', '.csv')

        # 1) 정확 일치(원본명 기준)
        lowers = [f.lower() for f in files]
        if target in lowers:
            start = lowers.index(target) + 1
        else:
            # 2) 부분 문자열(유일 매칭) 허용
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

    # 재개 루프
    for filename in tqdm(files[start:], desc=f"Processing Files (from {start}/{len(files)})"):
        in_path  = os.path.join(input_folder, filename)
        out_name = filename.replace('_CR.csv', '_r.csv')  # 기존 규칙 유지
        out_path = os.path.join(output_folder, out_name)

        if skip_existing and os.path.exists(out_path):
            # 이미 처리된 산출물 있으면 건너뜀
            # tqdm.write(f"[skip] {out_name}")
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
    """
    처음부터 'name'으로 지정한 파일까지 처리.
    - input_folder: CR 파싱된 CSV들이 있는 폴더 (예: .../CR_parsing/)
    - output_folder: 결과 저장 폴더
    - name        : 종료 기준 파일명 (원본 .csv / 출력 _CR.csv / 일부 문자열 모두 허용)
    - include_name: True면 기준 파일 '포함'하여 처리, False면 '직전'까지 처리
    - skip_existing: 이미 *_r.csv 있으면 건너뛸지
    - enable_merge: 완충 후 이동주차 모드 on/off (None이면 전역값 따름)
    """

    # 목록 정렬
    files = [fn for fn in os.listdir(input_folder) if fn.lower().endswith('.csv') and not fn.startswith('._')]
    files.sort()
    if not files:
        print("[warn] 입력 폴더에 csv 파일이 없습니다.")
        return

    # 기준 문자열 전처리: _CR.csv를 원본 .csv로 맵핑해 검색 유연성 ↑
    target = name.strip().lower().replace('_cr.csv', '.csv')
    lowers = [f.lower() for f in files]

    # 1) 정확 일치(원본명 기준)
    if target in lowers:
        end_idx = lowers.index(target)
    else:
        # 2) 부분 문자열(유일 매칭) 허용
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

    # 포함 여부
    if not include_name:
        end_idx -= 1
    if end_idx < 0:
        print("[info] include_name=False 이고 기준 파일이 첫 번째라서 처리할 파일이 없습니다.")
        return

    # 처리 루프
    for filename in tqdm(files[:end_idx + 1], desc=f"Processing Files (0..{end_idx})"):
        in_path  = os.path.join(input_folder, filename)
        out_name = filename.replace('_CR.csv', '_r.csv')  # 기존 규칙 유지
        out_path = os.path.join(output_folder, out_name)

        if skip_existing and os.path.exists(out_path):
            # 이미 결과가 있으면 건너뜀
            continue

        try:
            process_data(in_path, out_path, enable_merge=enable_merge)
        except Exception as e:
            tqdm.write(f"[error] {filename}: {e}")
            continue


def _one_file_job(args):
    """
    멀티프로세스 워커에서 실행될 1파일 처리.
    args로만 값을 넘겨야 Windows spawn에서 안전합니다.
    """
    in_path, out_path, enable_merge, overwrite = args

    # 이미 결과가 있으면 스킵
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
    """
    폴더 내 모든 CSV를 멀티프로세스로 처리.
    - skip_existing=True면 기존 *_r.csv 있으면 스킵
    - enable_merge: 완충후이동주차 모드 (None이면 전역 FULLCHARGE_PARKING_MODE 따름)
    - workers: 프로세스 개수 (None이면 os.cpu_count())
    - chunksize: 작업 묶음 크기 (파일이 많으면 4~16 정도가 더 빠를 때도 있음)
    """
    os.makedirs(output_folder, exist_ok=True)

    # CSV만 정렬
    files = [fn for fn in os.listdir(input_folder) if fn.lower().endswith(".csv") and not fn.startswith('._')]
    files.sort()

    jobs = []
    for filename in files:
        in_path = os.path.join(input_folder, filename)
        out_name = filename.replace("_CR.csv", "_r.csv")
        out_path = os.path.join(output_folder, out_name)

        # skip_existing=True면 overwrite=False로 전달
        overwrite = not skip_existing
        jobs.append((in_path, out_path, enable_merge, overwrite))

    if not jobs:
        print("[info] 처리할 CSV가 없습니다.")
        return

    ok = skip = err = 0
    errors = []

    # ProcessPoolExecutor: CPU 코어 사용
    with ProcessPoolExecutor(max_workers=workers) as ex:
        # tqdm 진행률: as_completed로 완료될 때마다 1씩 증가
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
ROOT_DIR = "/Volumes/T7/DFC"

parsing_folder_path = os.path.join(
    ROOT_DIR,
    f"CR_parsing_{VEHICLE_MODEL}"
)

save_folder_path = os.path.join(
    ROOT_DIR,
    f"R_parsing_origin_{VEHICLE_MODEL}"
)

# # 처음부터 선택파일 까지 '포함'하여 처리
# process_until_file(
#     parsing_folder_path,
#     save_folder_path,
#     'bms_01241228014_2023-09_CR.csv',
#     include_name=True,
#     skip_existing=False, # False 면 결과파일 있어도 코드 돌림
#     enable_merge=None  # None이면 FULLCHARGE_PARKING_MODE 토글 값 따라감
# )

# # 전체 파일 돌리기 (전역 토글값 사용)
# process_folder(parsing_folder_path, save_folder_path, skip_existing=True) # False 면 결과파일 있어도 코드 돌림


# # 선택 파일 이후 파일 돌리기 (전역 토글값 사용)
# resume_after_file(parsing_folder_path, save_folder_path, 'bms_altitude_01241592878_2024-02_CR.csv', skip_existing=True)
#

# # 데이터 하나만 코드실행
# in_file  = r'C:\Users\junny\SynologyDrive\SamsungSTF\Processed_Data\DFC\EV6\CR_parsing\bms_01241228082_2023-10_CR.csv'
# # out_file = r'Z:\SamsungSTF\Processed_Data\DFC\EV6\R_parsing_완충후이동주차\bms_01241228086_2023-03_r.csv'
# out_file = r'C:\Users\junny\SynologyDrive\SamsungSTF\Processed_Data\DFC\EV6\R_parsing_완충후이동주차\bms_01241228082_2023-10_r.csv'
# # enable_merge:
# #   - None  → 위에 FULLCHARGE_PARKING_MODE 값 따름
# #   - True  → 이 파일만 강제로 완충후 이동주차 모드 ON
# #   - False → 이 파일만 강제로 OFF
# process_data(in_file, out_file, enable_merge=None)


if __name__ == "__main__":
    # CPU 코어 다 쓰면 디스크 I/O 때문에 오히려 느려질 수도 있어요.
    # 처음엔 workers=4~8 정도로 테스트 추천.
    process_folder_mp(
        parsing_folder_path,
        save_folder_path,
        skip_existing=True,
        enable_merge=None,
        workers=8,       # ← 여기 조절!
        chunksize=1
    )