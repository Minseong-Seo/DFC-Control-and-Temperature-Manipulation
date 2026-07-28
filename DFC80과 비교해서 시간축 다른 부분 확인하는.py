import pandas as pd

file90 = "/Volumes/T7/DFC/DFC_origin_Ioniq5/DFC80_t15/bms_01241248820_2023-09_DFC80_t15.csv"
file80 = "/Volumes/T7/DFC/DFC_origin_Ioniq5/DFC80_t30/bms_01241248820_2023-09_DFC80_t30.csv"

f90 = pd.read_csv(file90)
f80 = pd.read_csv(file80)

f90["time"] = pd.to_datetime(f90["time"])
f80["time"] = pd.to_datetime(f80["time"])

print(f"DFC80_t15 행 수 : {len(f90)}")
print(f"DFC80_t30 행 수 : {len(f80)}")

merged = f90[["time", "soc"]].merge(
    f80[["time", "soc"]],
    on="time",
    how="outer",
    suffixes=("_90", "_80"),
    indicator=True,
)

only90 = merged[merged["_merge"] == "left_only"]
only80 = merged[merged["_merge"] == "right_only"]
common = merged[merged["_merge"] == "both"]

print(f"\nDFC80_t15에만 있는 time : {len(only90)}")
print(f"DFC80_t30에만 있는 time : {len(only80)}")

soc_diff = common[common["soc_90"] != common["soc_80"]]
print(f"같은 time인데 SOC가 다른 행 : {len(soc_diff)}")

# 결과 저장
output_dir = "/Volumes/T7/DFC"

only90[["time", "soc_90"]].to_csv(
    f"{output_dir}/DFC80_t15_only_time.csv",
    index=False,
)

only80[["time", "soc_80"]].to_csv(
    f"{output_dir}/DFC80_t30_only_time.csv",
    index=False,
)

soc_diff.to_csv(
    f"{output_dir}/DFC80_t15_vs_t30_soc_diff.csv",
    index=False,
)

print("\nCSV 저장 완료")
print(f"- {output_dir}/DFC80_t15_only_time.csv")
print(f"- {output_dir}/DFC80_t30_only_time.csv")
print(f"- {output_dir}/DFC80_t15_vs_t30_soc_diff.csv")