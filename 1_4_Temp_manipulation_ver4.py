import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

import sys

# ==========================================================
# 1. User settings
# ==========================================================

VEHICLE_MODEL = "EV6"  # "Ioniq5" 또는 "EV6"

if VEHICLE_MODEL not in {"Ioniq5", "EV6"}:
    raise ValueError(
        "VEHICLE_MODEL은 'Ioniq5' 또는 'EV6'만 가능합니다."
    )

# DFC 조건
DFC_START_SOC = int(os.getenv("DFC_START_SOC", "80"))
TIME_MARGIN_MIN = int(os.getenv("TIME_MARGIN_MIN", "60"))

DFC_SUFFIX = f"_DFC{DFC_START_SOC}_t{TIME_MARGIN_MIN}"

# Debug: 특정 차량만 처리 (None이면 전체 처리)
DEBUG_DEVICE = "01241228082" #"01241248827"

# 특정 차량에서 특정 월만 처리하려면 "YYYY-MM" 입력
# 차량의 전체 월을 처리하려면 None
TARGET_YM = "2023-08" #"2023-09"
# 결과가 존재하면 덮어쓰기
OVERWRITE = True


# ==========================================================
# 2. Path
# ==========================================================

ROOT_DIR = Path("/Volumes/T7/DFC")

DFC_FOLDER = DFC_SUFFIX.lstrip("_")

INPUT_DIR = (
    ROOT_DIR
    / f"DFC_origin_{VEHICLE_MODEL}"
    / DFC_FOLDER
)

OUTPUT_DIR = (
    ROOT_DIR
    / f"Temp_manipulation_DFC_{VEHICLE_MODEL}"
    / DFC_FOLDER
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_DEVICE_NAME = (
    DEBUG_DEVICE
    if DEBUG_DEVICE is not None
    else "all_devices"
)

LOG_PATH = (
    OUTPUT_DIR
    / (
        f"temperature_manipulation_log_"
        f"{VEHICLE_MODEL}_{LOG_DEVICE_NAME}"
        f"{DFC_SUFFIX}.txt"
    )
)

OUTPUT_SUFFIX = "_Temp_manipulation.csv"


# ==========================================================
# 3. Status
# ==========================================================

DFC_APPLIED = "DFC_applied"

STATUS_MANIPULATED_INCREASED = (
    "Manipulated_temperature_increased"
)

STATUS_MANIPULATED_DECREASED = (
    "Manipulated_temperature_decreased"
)

STATUS_NOT_MANIPULATED = (
    "Not_manipulated_delta_temp_zero"
)

# DFC 구간 각 행의 실제 flat 보정량 평균을 기준으로 한 상태
STATUS_MANIPULATED_INCREASED_BY_MEAN = (
    "Mean_manipulated_temperature_increased"
)

STATUS_MANIPULATED_DECREASED_BY_MEAN = (
    "Mean_manipulated_temperature_decreased"
)

STATUS_NOT_MANIPULATED_BY_MEAN = (
    "Mean_not_manipulated_delta_temp_zero"
)

# ==========================================================
# 4. CSV loading and result-column initialization
# ==========================================================
def read_csv(file_path: Path) -> pd.DataFrame:
    """CSV를 읽고 time 칼럼을 datetime으로 변환한다."""
    try:
        df = pd.read_csv(
            file_path,
            encoding="utf-8",
            low_memory=False,
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            file_path,
            encoding="cp949",
            low_memory=False,
        )

    if "time" not in df.columns:
        raise ValueError(
            f"time 칼럼이 없습니다: {file_path.name}"
        )

    df["time"] = pd.to_datetime(
        df["time"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )

    if df["time"].notna().sum() == 0:
        raise ValueError(
            f"time 칼럼을 datetime으로 변환할 수 없습니다: {file_path.name}"
        )

    # 동일한 time 값을 가진 행의 원래 순서를 보존하기 위해
    # 입력 파일의 행 순서를 보조 정렬 기준으로 사용한다.
    df["_original_row_order"] = np.arange(
        len(df),
        dtype=int,
    )

    df = (
        df.dropna(subset=["time"])
        .sort_values(
            ["time", "_original_row_order"],
            kind="stable",
        )
        .drop(columns=["_original_row_order"])
        .reset_index(drop=True)
    )

    return df


def calculate_module_median(temp_value) -> float:
    """mod_temp_list 한 행에서 모듈 온도의 중앙값을 계산한다."""
    if pd.isna(temp_value):
        return np.nan

    # 음수와 소수점을 포함한 숫자를 모두 추출한다.
    values = re.findall(
        r"-?\d+(?:\.\d+)?",
        str(temp_value),
    )

    if not values:
        return np.nan

    numeric_values = [
        float(value)
        for value in values
    ]

    return float(
        np.median(numeric_values)
    )


def create_temperature_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    mod_temp_manipulated_median을 생성하고,
    온도 조작 결과를 기록할 칼럼을 초기화한다.
    """
    if "mod_temp_list" not in df.columns:
        raise ValueError(
            "mod_temp_list 칼럼이 없습니다."
        )

    # 각 시간 행의 모듈 온도 중앙값을 새로 계산한다.
    df["mod_temp_manipulated_median"] = (
        df["mod_temp_list"]
        .apply(calculate_module_median)
    )

    # DFC 구간 이외의 행은 빈칸으로 유지한다.
    df["temp_manipulation_status"] = ""

    # 실제로 감소시킨 온도 차이를 기록한다.
    # DFC 구간이 아닌 행은 빈칸(NaN)으로 유지한다.
    df["temp_delta"] = np.nan

    return df

# ==========================================================
# 5. DFC region detection
# ==========================================================
def find_dfc_regions(
    df: pd.DataFrame,
) -> list[dict[str, object]]:
    """
    DFC_applied가 기록된 연속 행을 각각 하나의 DFC 구간으로 반환한다.

    각 구간의 첫 행은 DFC 충전 재개 시점,
    마지막 행은 DFC 충전 중단 시점으로 사용한다.
    """
    if "DFC_applied" not in df.columns:
        raise ValueError(
            "DFC_applied 칼럼이 없습니다. "
            "수정된 DFC 생성 코드를 먼저 실행해야 합니다."
        )

    # CSV를 다시 읽으면 빈칸은 NaN이 될 수 있으므로
    # 빈 문자열로 바꾼 뒤 앞뒤 공백을 제거한다.
    applied_status = (
        df["DFC_applied"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # 실제로 시간축이 이동된 행만 True로 표시한다.
    applied_mask = applied_status.eq(
        DFC_APPLIED
    )

    if not applied_mask.any():
        return []

    # False → True 또는 True → False로 바뀌는 지점마다
    # 새로운 연속 구간 번호를 부여한다.
    group_id = applied_mask.ne(
        applied_mask.shift(
            fill_value=False
        )
    ).cumsum()

    regions: list[dict[str, object]] = []

    # DFC_applied가 True인 행만 연속 구간별로 묶는다.
    for _, group in df[
        applied_mask
    ].groupby(
        group_id[applied_mask]
    ):
        start_idx = int(
            group.index[0]
        )
        end_idx = int(
            group.index[-1]
        )

        regions.append(
            {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_time": df.loc[
                    start_idx,
                    "time",
                ],
                "end_time": df.loc[
                    end_idx,
                    "time",
                ],
            }
        )

    return regions

# ==========================================================
# 6. Drive start detection
# ==========================================================
def build_drive_start_indices(
    df: pd.DataFrame,
) -> np.ndarray:
    """
    정지 상태에서 주행 상태로 바뀌는 행의
    인덱스를 미리 계산한다.
    """
    if "speed" not in df.columns:
        raise ValueError(
            "speed 칼럼이 없습니다."
        )

    speed = pd.to_numeric(
        df["speed"],
        errors="coerce",
    ).fillna(0.0)

    # speed > 0이면 주행으로 판단한다.
    is_driving = speed > 0

    # False → True가 되는 첫 행이 주행 시작점이다.
    drive_start_mask = (
        is_driving
        & ~is_driving.shift(
            fill_value=False
        )
    )

    return df.index[
        drive_start_mask
    ].to_numpy(dtype=int)


def find_next_drive_start(
    df: pd.DataFrame,
    drive_start_indices: np.ndarray,
    dfc_start_time: pd.Timestamp,
) -> int | None:
    """
    DFC 충전 재개 이후 가장 가까운
    주행 시작 행의 인덱스를 반환한다.
    """
    if drive_start_indices.size == 0:
        return None

    drive_times = df.loc[
        drive_start_indices,
        "time",
    ].to_numpy(dtype="datetime64[ns]")

    target_time = np.datetime64(
        dfc_start_time,
        "ns",
    )

    # 동일 시각은 제외하고
    # 반드시 이후 주행만 선택한다.
    position = np.searchsorted(
        drive_times,
        target_time,
        side="right",
    )

    if position >= len(drive_start_indices):
        return None

    return int(
        drive_start_indices[position]
    )


# ==========================================================
# 7. Temperature manipulation
# ==========================================================
def manipulate_temperature(
    df: pd.DataFrame,
    region: dict[str, object],
    drive_start_indices: np.ndarray,
    previous_reference_drive_idx: int | None = None,
) -> dict[str, object]:
    """
    이후 주행 시작 온도를 기준으로 DFC 구간의 온도를 flat하게 만든다.

    DFC 구간 내 모든 행의 온도를 주행 시작 시점의 모듈 중앙값으로
    설정한다. 기존 DFC 온도와 주행 시작 온도의 차이는 temp_delta에
    기록한다.
    """
    start_idx = int(region["start_idx"])
    end_idx = int(region["end_idx"])

    dfc_start_time = pd.Timestamp(
        region["start_time"]
    )
    dfc_end_time = pd.Timestamp(
        region["end_time"]
    )

    # DFC 충전 재개 시점의 모듈 중앙값 온도
    dfc_start_temp = pd.to_numeric(
        pd.Series([
            df.loc[start_idx, "mod_temp_manipulated_median"]
        ]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(dfc_start_temp):
        return {
            "status": "Skipped",
            "reason": "invalid_dfc_start_temp",
            "start_time": dfc_start_time,
            "end_time": dfc_end_time,
        }

    # DFC 충전 재개 이후 가장 가까운 주행 시작점 찾기
    drive_start_idx = find_next_drive_start(
        df=df,
        drive_start_indices=drive_start_indices,
        dfc_start_time=dfc_start_time,
    )

    reference_drive_type = "next"

    # 다음 주행 기록이 없으면 이전 DFC 구간에서 실제로 사용했던
    # 주행 시작점을 그대로 재사용한다.
    if drive_start_idx is None:
        drive_start_idx = previous_reference_drive_idx
        reference_drive_type = "previous_dfc"

    if drive_start_idx is None:
        return {
            "status": "Skipped",
            "reason": "no_next_drive_or_previous_dfc_reference",
            "start_time": dfc_start_time,
            "end_time": dfc_end_time,
        }

    drive_start_time = df.loc[
        drive_start_idx,
        "time",
    ]

    # 주행 시작 시점의 모듈 중앙값 온도
    drive_start_temp = pd.to_numeric(
        pd.Series([
            df.loc[drive_start_idx, "mod_temp_manipulated_median"]
        ]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(drive_start_temp):
        return {
            "status": "Skipped",
            "reason": "invalid_drive_start_temp",
            "start_time": dfc_start_time,
            "end_time": dfc_end_time,
            "drive_start_time": drive_start_time,
        }

    # 실제로 시간축이 이동된 DFC 구간의 행
    region_indices = df.loc[
        start_idx:end_idx
    ].index

    # 기존 status 판단에 사용하는 DFC 시작 온도와 주행 시작 온도의 차이.
    # 양수이면 DFC 구간 온도를 낮추고,
    # 음수이면 DFC 구간 온도를 높인다.
    start_delta_temp = float(
        dfc_start_temp - drive_start_temp
    )

    # flat 처리 전 각 행의 온도를 보관한다.
    original_region_temps = pd.to_numeric(
        df.loc[
            region_indices,
            "mod_temp_manipulated_median",
        ],
        errors="coerce",
    )

    # 각 행에서 실제로 적용되는 flat 보정량.
    # 양수: 온도 상승, 음수: 온도 하강
    row_delta_temp = (
        drive_start_temp - original_region_temps
    )

    mean_delta_temp = float(
        row_delta_temp.mean()
    )

    # DFC 시작 온도와 차이가 0이어도 DFC 구간 내부의 온도 변화가
    # 있을 수 있으므로, 항상 주행 시작 온도 하나로 평탄화한다.
    df.loc[
        region_indices,
        "mod_temp_manipulated_median",
    ] = drive_start_temp

    # 실제 행별 flat 보정량을 기록한다.
    df.loc[
        region_indices,
        "temp_delta",
    ] = row_delta_temp

    if start_delta_temp != 0.0:

        # 실제 온도 조정 방향을 상태로 기록한다.
        if -start_delta_temp > 0:
            status = STATUS_MANIPULATED_INCREASED
        else:
            status = STATUS_MANIPULATED_DECREASED

        df.loc[
            region_indices,
            "temp_manipulation_status",
        ] = status

    else:
        # 두 기준 온도가 같으므로 온도 차이 기준의 증감은 없지만,
        # 위에서 DFC 구간 전체를 주행 시작 온도로 평탄화했다.
        df.loc[
            region_indices,
            "temp_manipulation_status",
        ] = STATUS_NOT_MANIPULATED

        status = STATUS_NOT_MANIPULATED

    # DFC 구간 전체의 실제 행별 보정량 평균을 기준으로 한 추가 status
    if mean_delta_temp > 0.0:
        mean_status = STATUS_MANIPULATED_INCREASED_BY_MEAN
    elif mean_delta_temp < 0.0:
        mean_status = STATUS_MANIPULATED_DECREASED_BY_MEAN
    else:
        mean_status = STATUS_NOT_MANIPULATED_BY_MEAN

    return {
        "status": status,
        "reason": "",
        "start_time": dfc_start_time,
        "end_time": dfc_end_time,
        "drive_start_idx": int(drive_start_idx),
        "drive_start_time": drive_start_time,
        "reference_drive_type": reference_drive_type,
        "dfc_start_temp": float(dfc_start_temp),
        "drive_start_temp": float(drive_start_temp),
        # 시작점 기준 실제 온도 조정량.
        # 양수: 온도 상승, 음수: 온도 하강
        "delta_temp": -start_delta_temp,
        # DFC 구간 각 행의 실제 flat 보정량 평균.
        "mean_delta_temp": mean_delta_temp,
        "mean_status": mean_status,
    }

# ==========================================================
# 8. File collection and output path
# ==========================================================
def _collect_input_files(
    input_folder,
    pattern="*.csv",
) -> list[Path]:
    """설정에 따라 처리할 DFC CSV 목록을 반환한다."""
    input_dir = Path(input_folder)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"입력 폴더가 없습니다: {input_dir}"
        )

    target_files = [
        file_path
        for file_path in sorted(
            input_dir.glob(pattern)
        )
        if not file_path.name.startswith(".")
        and not file_path.name.endswith(
            OUTPUT_SUFFIX
        )
    ]
    if DEBUG_DEVICE is not None:
        target_files = [
            file_path
            for file_path in target_files
            if DEBUG_DEVICE in file_path.name
        ]

    if TARGET_YM is not None:
        target_files = [
            file_path
            for file_path in target_files
            if TARGET_YM in file_path.name
        ]

    return target_files

def make_output_path(
    input_path: Path,
) -> Path:
    """출력 파일 경로를 생성한다."""
    output_name = (
        f"{input_path.stem}"
        f"{OUTPUT_SUFFIX}"
    )

    return (
        OUTPUT_DIR
        / output_name
    )


# ==========================================================
# 9. Console output
# ==========================================================
def print_region_result(
    region_number: int,
    result: dict[str, object],
) -> None:
    """각 DFC 구간의 처리 결과를 출력한다."""
    print(
        f"\n[Region {region_number}]"
    )
    print(
        f"DFC start      : "
        f"{result.get('start_time')}"
    )
    print(
        f"DFC end        : "
        f"{result.get('end_time')}"
    )

    if result["status"] == "Skipped":
        print(
            "Status         : Skipped"
        )
        print(
            f"Reason         : "
            f"{result.get('reason', '')}"
        )
        return

    print(
        f"Reference drive: "
        f"{result.get('drive_start_time')}"
    )
    print(
        f"Reference type : "
        f"{result.get('reference_drive_type')}"
    )
    print(
        f"DFC start temp : "
        f"{result.get('dfc_start_temp'):.3f}"
    )
    print(
        f"Drive temp     : "
        f"{result.get('drive_start_temp'):.3f}"
    )
    print(
        f"Delta temp     : "
        f"{result.get('delta_temp'):.3f}"
    )
    print(
        f"Mean delta temp: "
        f"{result.get('mean_delta_temp'):.3f}"
    )
    print(
        f"Status         : "
        f"{result.get('status')}"
    )
    print(
        f"Mean status    : "
        f"{result.get('mean_status')}"
    )


# ==========================================================
# 10. Single-file processing
# ==========================================================
def process_single_file(
    input_path: Path,
) -> dict[str, int]:
    """
    DFC 파일 하나를 처리하고
    새로운 CSV로 저장한다.
    """
    output_path = make_output_path(
        input_path
    )

    if (
        output_path.exists()
        and not OVERWRITE
    ):
        print(
            "[Skip] Output already exists: "
            f"{output_path}"
        )

        return {
            "files_processed": 0,
            "files_skipped": 1,
            "files_failed": 0,
            "regions_total": 0,
            "manipulated": 0,
            "not_manipulated": 0,
            "regions_skipped": 0,
        }

    try:
        print(
            "\n"
            + "=" * 60
        )
        print(
            f"Processing: "
            f"{input_path.name}"
        )
        print(
            "=" * 60
        )

        # DFC CSV 읽기
        df = read_csv(
            input_path
        )

        # 모듈 평균 온도 및 결과 칼럼 생성
        df = create_temperature_columns(
            df
        )

        # DFC_applied 연속 구간 탐지
        regions = find_dfc_regions(
            df
        )

        # 파일 전체 주행 시작점 계산
        drive_start_indices = (
            build_drive_start_indices(
                df
            )
        )

        print(
            f"DFC regions found: "
            f"{len(regions)}"
        )

        manipulated_count = 0
        not_manipulated_count = 0
        skipped_region_count = 0
        previous_reference_drive_idx = None

        for (
            region_number,
            region,
        ) in enumerate(
            regions,
            start=1,
        ):
            result = (
                manipulate_temperature(
                    df=df,
                    region=region,
                    drive_start_indices=(
                        drive_start_indices
                    ),
                    previous_reference_drive_idx=(
                        previous_reference_drive_idx
                    ),
                )
            )

            # 다음 DFC에서 "이전 DFC가 사용한 주행 시작점"을
            # 재사용할 수 있도록, 성공한 기준 주행을 보관한다.
            if result.get("drive_start_idx") is not None:
                previous_reference_drive_idx = int(
                    result["drive_start_idx"]
                )

            print_region_result(
                region_number,
                result,
            )

            if (
                result["status"]
                in {
                    STATUS_MANIPULATED_INCREASED,
                    STATUS_MANIPULATED_DECREASED,
                }
            ):
                manipulated_count += 1

            elif (
                result["status"]
                == STATUS_NOT_MANIPULATED
            ):
                not_manipulated_count += 1

            else:
                skipped_region_count += 1

        # 새로운 결과 CSV 저장
        df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            "\n"
            + "-" * 60
        )
        print(
            f"Saved           : "
            f"{output_path}"
        )
        print(
            f"Regions total   : "
            f"{len(regions)}"
        )
        print(
            f"Manipulated     : "
            f"{manipulated_count}"
        )
        print(
            f"Not manipulated : "
            f"{not_manipulated_count}"
        )
        print(
            f"Regions skipped : "
            f"{skipped_region_count}"
        )
        print(
            "-" * 60
        )

        return {
            "files_processed": 1,
            "files_skipped": 0,
            "files_failed": 0,
            "regions_total": len(
                regions
            ),
            "manipulated": (
                manipulated_count
            ),
            "not_manipulated": (
                not_manipulated_count
            ),
            "regions_skipped": (
                skipped_region_count
            ),
        }

    except Exception as error:
        print(
            f"[Error] "
            f"{input_path.name}: "
            f"{error}"
        )

        return {
            "files_processed": 0,
            "files_skipped": 0,
            "files_failed": 1,
            "regions_total": 0,
            "manipulated": 0,
            "not_manipulated": 0,
            "regions_skipped": 0,
        }


# ==========================================================
# 11. Main execution
# ==========================================================
def main() -> None:
    """선택된 DFC 파일을 순차적으로 처리한다."""
    print(
        "=" * 60
    )
    print(
        "DFC temperature manipulation"
    )
    print(
        f"DFC condition : "
        f"DFC_START_SOC={DFC_START_SOC}, "
        f"TIME_MARGIN_MIN={TIME_MARGIN_MIN}"
    )
    print(
        f"Input folder  : "
        f"{INPUT_DIR}"
    )
    print(
        f"Output folder : "
        f"{OUTPUT_DIR}"
    )
    print(
        "=" * 60
    )

    target_files = (
        _collect_input_files(
            INPUT_DIR,
            pattern="*.csv",
        )
    )

    if not target_files:
        print(
            "처리할 DFC 파일이 없습니다."
        )
        return

    total_summary = {
        "files_found": len(
            target_files
        ),
        "files_processed": 0,
        "files_skipped": 0,
        "files_failed": 0,
        "regions_total": 0,
        "manipulated": 0,
        "not_manipulated": 0,
        "regions_skipped": 0,
    }

    for input_path in target_files:
        file_summary = (
            process_single_file(
                input_path
            )
        )

        for key in (
            "files_processed",
            "files_skipped",
            "files_failed",
            "regions_total",
            "manipulated",
            "not_manipulated",
            "regions_skipped",
        ):
            total_summary[key] += (
                file_summary[key]
            )

    print(
        "\n"
        + "=" * 60
    )
    print(
        "All processing completed"
    )
    print(
        f"Files found      : "
        f"{total_summary['files_found']}"
    )
    print(
        f"Files processed  : "
        f"{total_summary['files_processed']}"
    )
    print(
        f"Files skipped    : "
        f"{total_summary['files_skipped']}"
    )
    print(
        f"Files failed     : "
        f"{total_summary['files_failed']}"
    )
    print(
        f"Regions total    : "
        f"{total_summary['regions_total']}"
    )
    print(
        f"Manipulated      : "
        f"{total_summary['manipulated']}"
    )
    print(
        f"Not manipulated  : "
        f"{total_summary['not_manipulated']}"
    )
    print(
        f"Regions skipped  : "
        f"{total_summary['regions_skipped']}"
    )
    print(
        "=" * 60
    )


if __name__ == "__main__":

    with LOG_PATH.open(
        "w",
        encoding="utf-8",
    ) as log_file:

        class Tee:
            """터미널과 TXT 파일에 동시에 출력한다."""

            def __init__(self, *streams):
                self.streams = streams

            def write(self, message):
                for stream in self.streams:
                    stream.write(message)
                    stream.flush()

            def flush(self):
                for stream in self.streams:
                    stream.flush()

        original_stdout = sys.stdout
        sys.stdout = Tee(
            original_stdout,
            log_file,
        )

        try:
            main()

        finally:
            sys.stdout = original_stdout

    print(
        f"\nLog saved: {LOG_PATH}"
    )
