import os
import re
from datetime import timedelta

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ==========================================================
# User Settings
# ==========================================================

ROOT_DIR = "/Volumes/T7/DFC"

# DFC 조건
DFC_SOC = 80
TIME_MARGIN = 60

# 차량
USER_ID = "01241248820"
TARGET_YM = "2023-04"


# ==========================================================
# Plot Variables
# 순서대로 subplot 생성
# ==========================================================

PLOT_VARIABLES = [
    "soc",
    "ext_temp",
    "mod_temp_avg",
]


LABELS = {

    "soc": "SOC (%)",

    "ext_temp": "External Temp (°C)",

    "int_temp": "Internal Temp (°C)",

    "mod_temp_avg": "Module Avg Temp (°C)",

    "pack_current": "Current (A)",

    "pack_volt": "Voltage (V)",

}


# ==========================================================
# Directory
# ==========================================================

DIR_ORIGINAL = os.path.join(
    ROOT_DIR,
    "R_parsing_origin_Ioniq5",
)

DIR_DFC = os.path.join(
    ROOT_DIR,
    "DFC_origin_Ioniq5",
    f"DFC{DFC_SOC}_t{TIME_MARGIN}",
)

OUTPUT_DIR = os.path.join(
    ROOT_DIR,
    "Compare_Temp_Output",
    f"DFC{DFC_SOC}_t{TIME_MARGIN}",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

# ==========================================================
# File Mapping
# ==========================================================

def build_map(directory):

    file_map = {}

    if not os.path.exists(directory):
        return file_map

    for fn in os.listdir(directory):

        if not fn.endswith(".csv"):
            continue

        parts = fn.split("_")

        if len(parts) < 3:
            continue

        vehicle = parts[1]
        ym = parts[2][:7]

        file_map[(vehicle, ym)] = os.path.join(
            directory,
            fn,
        )

    return file_map


ORIGINAL_MAP = build_map(DIR_ORIGINAL)

DFC_MAP = build_map(DIR_DFC)

# ==========================================================
# CSV Read
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

    return df

# ==========================================================
# Module Average Temperature
# ==========================================================

def calc_module_avg(temp_str):

    if pd.isna(temp_str):
        return np.nan

    values = re.findall(
        r"-?\d+\.?\d*",
        str(temp_str),
    )

    values = [float(v) for v in values]

    if len(values) == 0:
        return np.nan

    return np.mean(values)

# ==========================================================
# Load Original / DFC
# ==========================================================

def load_data():

    key = (
        USER_ID,
        TARGET_YM,
    )

    if key not in ORIGINAL_MAP:
        raise FileNotFoundError(
            "Original file not found."
        )

    if key not in DFC_MAP:
        raise FileNotFoundError(
            "DFC file not found."
        )

    df_org = read_csv_auto(
        ORIGINAL_MAP[key]
    )

    df_dfc = read_csv_auto(
        DFC_MAP[key]
    )

    for df in (df_org, df_dfc):

        if (
            "mod_temp_avg" not in df.columns
            and
            "mod_temp_list" in df.columns
        ):

            df["mod_temp_avg"] = (
                df["mod_temp_list"]
                .apply(calc_module_avg)
            )

    return df_org, df_dfc

# ==========================================================
# Week 생성
# ==========================================================

def make_week_ranges(df_org, df_dfc):
    """
    Original / DFC 전체 기간을 기준으로
    월요일 시작 Week를 생성한다.
    """

    start = min(
        df_org["time"].min(),
        df_dfc["time"].min(),
    )

    end = max(
        df_org["time"].max(),
        df_dfc["time"].max(),
    )

    first_monday = (
        start.normalize()
        - timedelta(days=start.weekday())
    )

    weeks = []

    t0 = first_monday

    while t0 < end:

        t1 = t0 + timedelta(days=7)

        weeks.append((t0, t1))

        t0 = t1

    return weeks


# ==========================================================
# Week Data
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
# Figure Title
# ==========================================================

def make_title(
    week_idx,
):

    variable_text = ", ".join(
        LABELS[v]
        for v in PLOT_VARIABLES
    )

    return (
        f"Vehicle : {USER_ID}    "
        f"Month : {TARGET_YM}\n"
        f"Week {week_idx+1}    "
        f"DFC Start SoC : {DFC_SOC}%    "
        f"Margin : {TIME_MARGIN} min\n"
        f"{variable_text}"
    )


# ==========================================================
# Output Folder
# ==========================================================

def make_output_dir():

    variable_folder = "-".join(
        v.replace("_temp", "")
        for v in PLOT_VARIABLES
    )

    output_dir = os.path.join(

        OUTPUT_DIR,

        variable_folder,

        USER_ID,

        TARGET_YM,

    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    return output_dir


# ==========================================================
# 자동 y축
# ==========================================================

def auto_ylim(
    ax,
    df_org,
    df_dfc,
    column,
):

    if column == "soc":

        ax.set_ylim(40, 100)
        return

    values = pd.concat(
        [
            pd.to_numeric(
                df_org[column],
                errors="coerce",
            ),
            pd.to_numeric(
                df_dfc[column],
                errors="coerce",
            ),
        ]
    ).dropna()

    if values.empty:
        return

    ymin = values.min()
    ymax = values.max()

    padding = max(
        0.5,
        (ymax - ymin) * 0.1,
    )

    ax.set_ylim(
        ymin - padding,
        ymax + padding,
    )


# ==========================================================
# DFC 적용 구간 찾기
# ==========================================================

def find_dfc_regions(
    week_org,
    week_dfc,
):

    if "soc" not in week_org.columns:
        return []

    if "soc" not in week_dfc.columns:
        return []

    merged = pd.merge_asof(

        week_org.sort_values("time"),

        week_dfc[
            ["time", "soc"]
        ].sort_values("time"),

        on="time",

        direction="nearest",

        suffixes=(
            "_org",
            "_dfc",
        ),

    )

    diff = (
        merged["soc_org"]
        -
        merged["soc_dfc"]
    ).abs()

    mask = diff > 0.0

    if not mask.any():
        return []

    groups = (
        mask != mask.shift()
    ).cumsum()

    regions = []

    for _, g in merged[mask].groupby(groups):

        regions.append(
            (
                g["time"].iloc[0],
                g["time"].iloc[-1],
            )
        )

    return regions

# ==========================================================
# Weekly Plot
# ==========================================================

def plot_weekly():

    df_org, df_dfc = load_data()

    weeks = make_week_ranges(
        df_org,
        df_dfc,
    )

    output_dir = make_output_dir()

    for week_idx, (week_start, week_end) in enumerate(weeks):

        week_org = get_week_data(
            df_org,
            week_start,
            week_end,
        )

        week_dfc = get_week_data(
            df_dfc,
            week_start,
            week_end,
        )

        if week_org.empty and week_dfc.empty:
            continue

        regions = find_dfc_regions(
            week_org,
            week_dfc,
        )

        fig, axes = plt.subplots(

            len(PLOT_VARIABLES),

            1,

            figsize=(9, 4.5 * len(PLOT_VARIABLES)),

            sharex=True,

        )

        if len(PLOT_VARIABLES) == 1:
            axes = [axes]

        # --------------------------------------------------
        # 변수별 Plot
        # --------------------------------------------------

        for ax, column in zip(
            axes,
            PLOT_VARIABLES,
        ):

            if column in week_org.columns:

                ax.plot(

                    week_org["time"],

                    week_org[column],

                    color="gray",

                    linewidth=1.5,

                    label="Original",

                )

            if column in week_dfc.columns:

                ax.plot(

                    week_dfc["time"],

                    week_dfc[column],

                    color="tab:blue",

                    linewidth=1.0,

                    label="DFC",

                )

            # --------------------------------------------
            # DFC 적용 구간 음영
            # --------------------------------------------

            for start, end in regions:

                ax.axvspan(

                    start,

                    end,

                    color="gold",

                    alpha=0.2,

                    zorder=0,

                )

            auto_ylim(

                ax,

                week_org,

                week_dfc,

                column,

            )

            ax.set_ylabel(

                LABELS[column],

                fontsize=10,

            )

            ax.grid(

                True,

                alpha=0.3,

            )

        # --------------------------------------------------
        # X Axis
        # --------------------------------------------------

        for ax in axes:

            ax.set_xlim(

                week_start,

                week_end,

            )

            ax.xaxis.set_major_locator(

                mdates.DayLocator()

            )

            ax.xaxis.set_major_formatter(

                mdates.DateFormatter("%m-%d")

            )

            ax.tick_params(

                axis="x",

                rotation=45,

            )

        axes[-1].set_xlabel("Time")

        # --------------------------------------------------
        # Figure Title
        # --------------------------------------------------

        fig.suptitle(

            make_title(
                week_idx,
            ),

            fontsize=12,

            y=0.96,

        )

        # --------------------------------------------------
        # Legend
        # --------------------------------------------------

        handles, labels = axes[0].get_legend_handles_labels()

        by_label = dict(zip(labels, handles))

        fig.legend(

            by_label.values(),

            by_label.keys(),

            loc="upper right",

            fontsize=10,

        )

        fig.subplots_adjust(

            left=0.10,

            right=0.95,

            top=0.90,

            bottom=0.08,

            hspace=0.22,

        )

        save_path = os.path.join(

            output_dir,

            f"{USER_ID}_{TARGET_YM}_Week{week_idx+1}_short.png",

        )

        fig.savefig(

            save_path,

            dpi=200,

            bbox_inches="tight",

        )

        plt.show()

        plt.close(fig)

        print(

            f"Saved : {save_path}"

        )

# ==========================================================
# Main
# ==========================================================

def main():

    plot_weekly()


if __name__ == "__main__":

    main()