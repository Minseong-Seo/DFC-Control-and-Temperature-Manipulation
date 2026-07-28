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

# DFC 조건
DFC_SOC = 80
TIME_MARGIN = 0

# 차량 / 대상 월
USER_ID = "01241248820"
TARGET_YM = "2023-05"

# 실행 시 그래프 화면 출력 여부
SHOW_PLOT = True


# ==========================================================
# 1. Input Directory
# ==========================================================

INPUT_DIR = os.path.join(
    ROOT_DIR,
    "Temp_manipulation_DFC_Ioniq5",
    f"DFC{DFC_SOC}_t{TIME_MARGIN}",
)

INPUT_FILENAME = (
    f"bms_{USER_ID}_{TARGET_YM}_"
    f"DFC{DFC_SOC}_t{TIME_MARGIN}_Temp_manipulation.csv"
)

INPUT_PATH = os.path.join(
    INPUT_DIR,
    INPUT_FILENAME,
)


# ==========================================================
# 2. Output Directory
# ==========================================================

OUTPUT_DIR = os.path.join(
    ROOT_DIR,
    "Compare_Temp_Manipulation_Output",
    f"DFC{DFC_SOC}_t{TIME_MARGIN}",
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

    if not os.path.exists(INPUT_PATH):

        raise FileNotFoundError(
            "Temperature manipulation file not found.\n"
            f"Expected path:\n{INPUT_PATH}"
        )

    print("Input file found:")
    print(INPUT_PATH)

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
# 5. Module Temperature Average
# ==========================================================

def calc_module_avg(temp_str):

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

    return np.mean(values)


# ==========================================================
# 6. Load Data
# ==========================================================

def load_data():

    check_paths()

    df = read_csv_auto(
        INPUT_PATH
    )

    required_columns = [
        "time",
        "soc",
        "ext_temp",
        "mod_temp_list",
        "mod_temp_avg",
        "DFC_applied",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise KeyError(
            "Required columns not found: "
            + ", ".join(missing_columns)
        )

    # 온도 조정 전 모듈 평균온도
    # mod_temp_list 내부의 각 모듈 온도를 평균
    df["mod_temp_avg_before"] = (
        df["mod_temp_list"]
        .apply(calc_module_avg)
    )

    numeric_columns = [
        "soc",
        "ext_temp",
        "mod_temp_avg_before",
        "mod_temp_avg",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    print(
        f"\nLoaded rows: {len(df):,}"
    )

    print(
        "Temperature comparison columns:"
    )

    print(
        "- Before: mod_temp_avg_before"
    )

    print(
        "- After : mod_temp_avg"
    )

    return df

# ==========================================================
# 7. Week Ranges
# ==========================================================

def make_week_ranges(df):
    """
    전체 데이터 기간을 기준으로
    월요일 시작 7일 단위 Week를 생성한다.
    """

    start = df["time"].min()
    end = df["time"].max()

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

    df = load_data()

    weeks = make_week_ranges(df)

    for week_idx, (week_start, week_end) in enumerate(weeks):

        week_df = get_week_data(
            df,
            week_start,
            week_end,
        )

        if week_df.empty:
            continue

        regions = find_dfc_regions(
            week_df,
        )

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(9, 13.5),
            sharex=True,
        )

        # =====================================================
        # SOC
        # =====================================================

        axes[0].plot(
            week_df["time"],
            week_df["soc"],
            color="tab:blue",
            linewidth=1.5,
            label="SOC",
        )

        axes[0].set_ylabel("SOC (%)")
        axes[0].set_title("SOC")

        set_auto_ylim(
            axes[0],
            week_df["soc"],
            fixed_ylim=(40, 100),
        )

        # =====================================================
        # External Temperature
        # =====================================================

        axes[1].plot(
            week_df["time"],
            week_df["ext_temp"],
            color="tab:blue",
            linewidth=1.5,
            label="External Temp",
        )

        axes[1].set_ylabel("External Temp (°C)")
        axes[1].set_title("External Temperature")

        set_auto_ylim(
            axes[1],
            week_df["ext_temp"],
        )

        # =====================================================
        # Module Temperature
        # =====================================================

        axes[2].plot(
            week_df["time"],
            week_df["mod_temp_avg_before"],
            color="gray",
            linestyle="--",
            linewidth=1.5,
            label="Before",
        )

        axes[2].plot(
            week_df["time"],
            week_df["mod_temp_avg"],
            color="tab:red",
            linewidth=1.8,
            label="After",
        )

        axes[2].set_ylabel("Module Avg Temp (°C)")
        axes[2].set_title("Module Average Temperature")

        set_auto_ylim(
            axes[2],
            pd.concat(
                [
                    week_df["mod_temp_avg_before"],
                    week_df["mod_temp_avg"],
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
            f"{USER_ID}_{TARGET_YM}_Week{week_idx+1}_short.png",
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