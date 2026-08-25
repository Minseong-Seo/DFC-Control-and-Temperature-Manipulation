import os
import re
from datetime import timedelta

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch


# ==========================================================
# 0. User Settings
# ==========================================================

ROOT_DIR = "/Volumes/T7/DFC"

VEHICLE_MODEL = "EV6"

# DFC 조건
DFC_SOC = 80
TIME_MARGIN = 60

# 차량 / 대상 월
USER_ID = "01241228082"
TARGET_YM = "2023-02"

# 실행 시 그래프 화면 출력 여부
SHOW_PLOT = True  # True: 화면 출력, False: 화면 출력 없이 파일 저장만


# ==========================================================
# 1. Input Directory
# ==========================================================

DFC_SUFFIX = f"DFC{DFC_SOC}_t{TIME_MARGIN}"

ORIGINAL_DIR = os.path.join(
    ROOT_DIR,
    f"R_parsing_origin_{VEHICLE_MODEL}",
)

DFC_DIR = os.path.join(
    ROOT_DIR,
    f"DFC_origin_{VEHICLE_MODEL}",
    DFC_SUFFIX,
)

MANIPULATED_DIR = os.path.join(
    ROOT_DIR,
    f"Temp_manipulation_DFC_{VEHICLE_MODEL}",
    DFC_SUFFIX,
)

ORIGINAL_PATH = os.path.join(
    ORIGINAL_DIR,
    f"bms_{USER_ID}_{TARGET_YM}_r.csv",
)

DFC_PATH = os.path.join(
    DFC_DIR,
    f"bms_{USER_ID}_{TARGET_YM}_{DFC_SUFFIX}.csv",
)

MANIPULATED_PATH = os.path.join(
    MANIPULATED_DIR,
    f"bms_{USER_ID}_{TARGET_YM}_{DFC_SUFFIX}_Temp_manipulation.csv",
)


# ==========================================================
# 2. Output Directory
# ==========================================================

OUTPUT_DIR = os.path.join(
    ROOT_DIR,
    "Compare_Temp_Manipulation_Output",
    VEHICLE_MODEL,
    DFC_SUFFIX,
    USER_ID,
    TARGET_YM,
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ==========================================================
# 3. Path Check
# ==========================================================

def check_paths():

    paths = {
        "Original": ORIGINAL_PATH,
        "DFC before temp manipulation": DFC_PATH,
        "DFC after temp manipulation": MANIPULATED_PATH,
    }

    missing = [
        f"{label}: {path}"
        for label, path in paths.items()
        if not os.path.exists(path)
    ]

    if missing:
        raise FileNotFoundError(
            "The following comparison file(s) were not found:\n"
            + "\n".join(missing)
        )

    print("Comparison files found:")
    for label, path in paths.items():
        print(f"- {label}: {path}")

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    # ==========================================================
# 4. CSV Read
# ==========================================================

def read_csv_auto(path):

    try:
        df = pd.read_csv(
            path,
            encoding="utf-8",
        )

    except UnicodeDecodeError:
        df = pd.read_csv(
            path,
            encoding="cp949",
        )

    df["time"] = pd.to_datetime(
        df["time"]
    )

    df = (
        df.sort_values("time")
        .reset_index(drop=True)
    )

    return df


# ==========================================================
# 5. Module Temperature Median
# ==========================================================

def calc_module_median(temp_str):

    if pd.isna(temp_str):
        return np.nan

    values = re.findall(
        r"-?\d+\.?\d*",
        str(temp_str),
    )

    if len(values) == 0:
        return np.nan

    values = [
        float(value)
        for value in values
    ]

    return np.median(values)


# ==========================================================
# 6. Load Data
# ==========================================================

def load_data():

    check_paths()

    data = {
        "original": read_csv_auto(ORIGINAL_PATH),
        "dfc": read_csv_auto(DFC_PATH),
        "manipulated": read_csv_auto(MANIPULATED_PATH),
    }

    common_columns = [
        "time",
        "soc",
        "ext_temp",
    ]

    for label, df in data.items():
        missing_columns = [
            column
            for column in common_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise KeyError(
                f"{label} file is missing: "
                + ", ".join(missing_columns)
            )

    if "mod_temp_list" not in data["original"].columns:
        raise KeyError("Original file is missing: mod_temp_list")

    if "mod_temp_list" not in data["dfc"].columns:
        raise KeyError("DFC file is missing: mod_temp_list")

    if "DFC_applied" not in data["dfc"].columns:
        raise KeyError("DFC file is missing: DFC_applied")

    if "mod_temp_manipulated_median" not in data["manipulated"].columns:
        raise KeyError(
            "Manipulated file is missing: "
            "mod_temp_manipulated_median"
        )

    # 원본과 DFC 파일은 mod_temp_list에서 중앙값을 계산한다.
    data["original"]["module_temp_median"] = (
        data["original"]["mod_temp_list"]
        .apply(calc_module_median)
    )
    data["dfc"]["module_temp_median"] = (
        data["dfc"]["mod_temp_list"]
        .apply(calc_module_median)
    )

    # 온도 조정 후 파일은 조작 코드가 생성한 중앙값 칼럼을 사용한다.
    data["manipulated"]["module_temp_median"] = pd.to_numeric(
        data["manipulated"]["mod_temp_manipulated_median"],
        errors="coerce",
    )

    for df in data.values():
        for column in ["soc", "ext_temp", "module_temp_median"]:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    print("\nLoaded comparison data:")
    for label, df in data.items():
        print(f"- {label}: {len(df):,} rows")
    print("Temperature column: module_temp_median")

    return data

# ==========================================================
# 7. Week Ranges
# ==========================================================

def make_week_ranges(data):
    """
    전체 데이터 기간을 기준으로
    월요일 시작 7일 단위 Week를 생성한다.
    """

    all_times = pd.concat(
        [df["time"] for df in data.values()],
        ignore_index=True,
    )

    start = all_times.min()
    end = all_times.max()

    first_monday = (
        start.normalize()
        - timedelta(days=start.weekday())
    )

    weeks = []
    week_start = first_monday

    while week_start <= end:

        week_end = (
            week_start
            + timedelta(days=7)
        )

        weeks.append(
            (
                week_start,
                week_end,
            )
        )

        week_start = week_end

    return weeks


# ==========================================================
# 8. Week Data
# ==========================================================

def get_week_data(
    df,
    week_start,
    week_end,
):

    return df.loc[
        (df["time"] >= week_start)
        &
        (df["time"] < week_end)
    ].copy()


# ==========================================================
# 9. DFC Applied Regions
# ==========================================================

def find_dfc_regions(week_df):
    """
    DFC_applied 열에서 'DFC_applied' 문자열 또는 1이 연속되는
    구간의 시작 시각과 종료 시각을 반환한다.
    """

    if week_df.empty:
        return []

    status = week_df["DFC_applied"]

    applied = (
        status.astype(str).str.strip().eq("DFC_applied")
        |
        status.fillna(0).eq(1)
    )

    if not applied.any():
        return []

    groups = (
        applied
        .ne(applied.shift(fill_value=False))
        .cumsum()
    )

    regions = []

    for _, group in week_df.loc[applied].groupby(
        groups[applied]
    ):

        start = group["time"].iloc[0]
        end = group["time"].iloc[-1]

        if start == end:
            end = end + timedelta(seconds=1)

        regions.append(
            (
                start,
                end,
            )
        )

    return regions


# ==========================================================
# 10. Flat visualization for relevant missing gaps
# ==========================================================

def find_flat_gap_segments(
    df,
    min_gap_seconds=4,
):
    """
    충전 중이거나 충전 종료 후 다음 주행 시작 전인 missing gap을 찾는다.

    원본 파일은 chrg_cable_conn을 충전 상태의 대용으로 사용하고,
    DFC 파일은 charging/chrg_cable_conn/DFC_applied를 사용한다.
    gap 직후의 온도값을 사용해 수평 점선으로 표시할 수 있도록
    (gap 이전 시각, gap 이후 시각, gap 이후 온도)를 반환한다.
    """
    if df.empty or "speed" not in df.columns:
        return []

    speed = pd.to_numeric(
        df["speed"],
        errors="coerce",
    ).fillna(0.0)
    is_driving = speed > 0

    charging_like = pd.Series(
        False,
        index=df.index,
    )

    if "charging" in df.columns:
        charging_like |= (
            pd.to_numeric(
                df["charging"],
                errors="coerce",
            ).fillna(0).eq(1)
        )

    if "chrg_cable_conn" in df.columns:
        charging_like |= (
            pd.to_numeric(
                df["chrg_cable_conn"],
                errors="coerce",
            ).fillna(0).eq(1)
        )

    if "DFC_applied" in df.columns:
        charging_like |= (
            df["DFC_applied"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("DFC_applied")
        )

    times = df["time"]
    gaps = times.diff().dt.total_seconds()
    segments = []
    episode_active = False

    for position in range(len(df)):
        index = df.index[position]

        if position > 0 and episode_active:
            gap_seconds = gaps.iloc[position]

            if gap_seconds > min_gap_seconds:
                after_value = df.iloc[position]["module_temp_median"]

                if pd.notna(after_value):
                    segments.append(
                        {
                            "start": times.iloc[position - 1],
                            "end": times.iloc[position],
                            "value": float(after_value),
                            "gap_seconds": float(gap_seconds),
                        }
                    )

        # 충전/DFC 구간에서 시작해, 다음 주행 시작 시점까지를
        # 하나의 보정 대상 episode로 본다.
        if charging_like.loc[index]:
            episode_active = True

        if episode_active and is_driving.iloc[position]:
            episode_active = False

    return segments


def plot_module_temperature_with_flat_gaps(
    ax,
    week_df,
    flat_segments,
    label,
    color,
    linestyle,
):
    """모듈 온도 원자료와 gap 구간의 수평 점선을 함께 그린다."""
    values = week_df["module_temp_median"].copy()

    visible_segments = [
        segment
        for segment in flat_segments
        if segment["start"] < week_df["time"].max()
        and segment["end"] >= week_df["time"].min()
    ]

    # gap을 원자료 선으로 대각 연결하지 않도록 gap 직전 점을 끊는다.
    for segment in visible_segments:
        before_matches = week_df.index[
            week_df["time"].eq(segment["start"])
        ]

        if len(before_matches) > 0:
            values.loc[before_matches[0]] = np.nan

    ax.plot(
        week_df["time"],
        values,
        color=color,
        linestyle=linestyle,
        linewidth=1.8,
        label=label,
    )

    for number, segment in enumerate(visible_segments):
        ax.plot(
            [segment["start"], segment["end"]],
            [segment["value"], segment["value"]],
            color=color,
            linestyle=":",
            linewidth=2.2,
            label=(
                "Flat missing-gap temperature"
                if number == 0
                else "_nolegend_"
            ),
        )

# ==========================================================
# 10. Automatic Y-axis Limits
# ==========================================================

def set_auto_ylim(
    ax,
    values,
    fixed_ylim=None,
):

    if fixed_ylim is not None:
        ax.set_ylim(*fixed_ylim)
        return

    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if values.empty:
        return

    ymin = values.min()
    ymax = values.max()

    padding = max(
        0.5,
        (ymax - ymin) * 0.1,
    )

    if ymin == ymax:
        padding = max(
            0.5,
            abs(ymin) * 0.05,
        )

    ax.set_ylim(
        ymin - padding,
        ymax + padding,
    )


def format_hover_time(x):
    """Matplotlib hover 좌표의 x축 시간을 초 단위까지 표시한다."""
    return mdates.num2date(x).strftime("%Y-%m-%d %H:%M:%S")


# ==========================================================
# 11. Figure Title
# ==========================================================

def make_title(
    week_idx,
    week_start,
    week_end,
):

    visible_end = (
        week_end
        - timedelta(seconds=1)
    )

    return (
        f"Vehicle: {USER_ID}    "
        f"Month: {TARGET_YM}\n"
        f"Week {week_idx + 1} "
        f"({week_start:%Y-%m-%d} ~ {visible_end:%Y-%m-%d})    "
        f"DFC Start SoC: {DFC_SOC}%    "
        f"Margin: {TIME_MARGIN} min"
    )

# ==========================================================
# 12. Weekly Plot
# ==========================================================

def plot_weekly():

    data = load_data()

    weeks = make_week_ranges(data)

    flat_gap_segments = {
        name: find_flat_gap_segments(df)
        for name, df in data.items()
    }

    print("\nFlat missing-gap segments:")
    for name, segments in flat_gap_segments.items():
        print(f"- {name}: {len(segments)}")

    for week_idx, (week_start, week_end) in enumerate(weeks):

        week_data = {
            name: get_week_data(
                df,
                week_start,
                week_end,
            )
            for name, df in data.items()
        }

        if all(df.empty for df in week_data.values()):
            continue

        regions = find_dfc_regions(
            week_data["dfc"],
        )

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(20, 12),
            sharex=True,
        )

        # =====================================================
        # SOC
        # =====================================================

        plot_specs = [
            ("original", "Original", "gray", "--"),
            ("dfc", "DFC before temp manipulation", "tab:blue", "-"),
            ("manipulated", "DFC after temp manipulation", "tab:red", "-"),
        ]

        for name, label, color, linestyle in plot_specs:
            week_df = week_data[name]
            if week_df.empty:
                continue
            axes[0].plot(
                week_df["time"],
                week_df["soc"],
                color=color,
                linestyle=linestyle,
                linewidth=1.5,
                label=label,
            )

        axes[0].set_ylabel("SOC (%)")
        axes[0].set_title("SOC")

        set_auto_ylim(
            axes[0],
            pd.concat(
                [df["soc"] for df in week_data.values()]
            ),
            fixed_ylim=(40, 100),
        )

        # =====================================================
        # External Temperature
        # =====================================================

        for name, label, color, linestyle in plot_specs:
            week_df = week_data[name]
            if week_df.empty:
                continue
            axes[1].plot(
                week_df["time"],
                week_df["ext_temp"],
                color=color,
                linestyle=linestyle,
                linewidth=1.5,
                label=label,
            )

        axes[1].set_ylabel("External Temp (°C)")
        axes[1].set_title("External Temperature")

        set_auto_ylim(
            axes[1],
            pd.concat(
                [df["ext_temp"] for df in week_data.values()]
            ),
        )

        # =====================================================
        # Module Temperature
        # =====================================================

        for name, label, color, linestyle in plot_specs:
            week_df = week_data[name]
            if week_df.empty:
                continue
            plot_module_temperature_with_flat_gaps(
                ax=axes[2],
                week_df=week_df,
                flat_segments=flat_gap_segments[name],
                label=label,
                color=color,
                linestyle=linestyle,
            )

        axes[2].set_ylabel("Module Median Temp (°C)")
        axes[2].set_title("Module Median Temperature")

        set_auto_ylim(
            axes[2],
            pd.concat(
                [
                    df["module_temp_median"]
                    for df in week_data.values()
                ]
            ),
        )

        # =====================================================
        # DFC Applied Shading
        # =====================================================

        for ax in axes:

            for start, end in regions:

                ax.axvspan(
                    start,
                    end,
                    color="gold",
                    alpha=0.2,
                    zorder=0,
                )

            ax.grid(
                True,
                alpha=0.3,
            )

            ax.set_xlim(
                week_start,
                week_end,
            )

        # =====================================================
        # X Axis
        # =====================================================

        axes[-1].set_xlabel("Time")

        axes[-1].xaxis.set_major_locator(
            mdates.DayLocator()
        )

        axes[-1].xaxis.set_major_formatter(
            mdates.DateFormatter("%m-%d")
        )

        axes[-1].tick_params(
            axis="x",
            rotation=45,
        )

        # 마우스 위치의 x 좌표를 정확한 날짜·시각·초로 표시한다.
        for ax in axes:
            ax.format_xdata = format_hover_time

        # =====================================================
        # Legend
        # =====================================================

        axes[0].legend()

        axes[1].legend()

        axes[2].legend()

        dfc_patch = Patch(
            facecolor="gold",
            alpha=0.2,
            label="DFC Applied",
        )

        fig.legend(
            handles=[dfc_patch],
            loc="upper right",
        )

        # =====================================================
        # Title
        # =====================================================

        fig.suptitle(
            make_title(
                week_idx,
                week_start,
                week_end,
            ),
            fontsize=13,
        )

        fig.subplots_adjust(
            left=0.08,
            right=0.95,
            top=0.92,
            bottom=0.08,
            hspace=0.30,
        )

        # =====================================================
        # Save
        # =====================================================

        save_path = os.path.join(
            OUTPUT_DIR,
            f"{USER_ID}_{TARGET_YM}_Week{week_idx+1}.png",
        )

        fig.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )

        if SHOW_PLOT:
            plt.show()

        plt.close(fig)

        print(f"Saved : {save_path}")


# ==========================================================
# Main
# ==========================================================

def main():

    plot_weekly()


if __name__ == "__main__":

    main()
