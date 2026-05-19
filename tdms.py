from pathlib import Path
from typing import Any

import pandas as pd
from nptdms import TdmsFile


# Folder containing your TDMS files
INPUT_FOLDER = Path("tdms_files")

# Folder where CSV files will be saved
OUTPUT_FOLDER = Path("csv_exports")


def tdms_to_dataframe(tdms_path: Path) -> pd.DataFrame:
    """Read one TDMS file and return selected channels in Biodex CSV format."""
    tdms = TdmsFile.read(tdms_path)

    data = {}
    time_column_added = False

    for group in tdms.groups():
        for channel in group.channels():
            original_name = channel.name.strip()
            lower_name = original_name.lower()

            if "velocity" in lower_name:
                column_name = "Angular_Velocity_Deg/Sec"
            elif "position" in lower_name:
                column_name = "Position_Deg"
            elif "torque" in lower_name:
                column_name = "Torque_Nm"
            else:
                column_name = f"{group.name}_{original_name}"

            data[column_name] = pd.Series(channel[:])

            if not time_column_added:
                try:
                    time_track = channel.time_track(absolute_time=True)

                    formatted_time = pd.to_datetime(time_track).strftime(
                        "%Y-%m-%d %H:%M:%S.%f"
                    )

                    data["Time"] = pd.Series(formatted_time)
                    time_column_added = True

                except Exception:
                    try:
                        time_track = channel.time_track(absolute_time=False)
                        data["Time"] = pd.Series(time_track)
                        time_column_added = True
                    except Exception:
                        pass

    df = pd.DataFrame(data)

    desired_order = [
        "Time",
        "Angular_Velocity_Deg/Sec",
        "Position_Deg",
        "Torque_Nm",
    ]

    ordered_columns = [column for column in desired_order if column in df.columns]
    remaining_columns = [column for column in df.columns if column not in ordered_columns]

    return df[ordered_columns + remaining_columns]


# Metadata extraction helpers
def clean_metadata_value(value: Any) -> str:
    """Convert metadata values into a CSV-friendly string."""
    if value is None:
        return ""
    return str(value)


def extract_metadata_rows(tdms_path: Path) -> list[dict[str, str]]:
    """Extract file, group, and channel metadata from one TDMS file."""
    tdms = TdmsFile.read(tdms_path)
    rows = []

    for property_name, property_value in tdms.properties.items():
        rows.append(
            {
                "file_name": tdms_path.name,
                "level": "file",
                "group_name": "",
                "channel_name": "",
                "property_name": property_name,
                "property_value": clean_metadata_value(property_value),
            }
        )

    for group in tdms.groups():
        for property_name, property_value in group.properties.items():
            rows.append(
                {
                    "file_name": tdms_path.name,
                    "level": "group",
                    "group_name": group.name,
                    "channel_name": "",
                    "property_name": property_name,
                    "property_value": clean_metadata_value(property_value),
                }
            )

        for channel in group.channels():
            for property_name, property_value in channel.properties.items():
                rows.append(
                    {
                        "file_name": tdms_path.name,
                        "level": "channel",
                        "group_name": group.name,
                        "channel_name": channel.name,
                        "property_name": property_name,
                        "property_value": clean_metadata_value(property_value),
                    }
                )

    return rows


def convert_folder(input_folder: Path, output_folder: Path) -> None:
    """Convert every TDMS file in input_folder into a CSV file."""
    output_folder.mkdir(parents=True, exist_ok=True)

    tdms_files = sorted(input_folder.glob("*.tdms"))

    if not tdms_files:
        print(f"No TDMS files found in: {input_folder.resolve()}")
        return

    all_metadata_rows = []

    for tdms_path in tdms_files:
        try:
            print(f"Converting: {tdms_path.name}")

            df = tdms_to_dataframe(tdms_path)

            output_path = output_folder / f"{tdms_path.stem}.csv"
            df.to_csv(output_path, index=False)

            metadata_rows = extract_metadata_rows(tdms_path)
            all_metadata_rows.extend(metadata_rows)

            print(f"Saved: {output_path}")

        except Exception as error:
            print(f"Failed to convert {tdms_path.name}: {error}")

    if all_metadata_rows:
        metadata_output_path = output_folder / "tdms_metadata.csv"
        metadata_df = pd.DataFrame(all_metadata_rows)
        metadata_df.to_csv(metadata_output_path, index=False)
        print(f"Saved metadata: {metadata_output_path}")


if __name__ == "__main__":
    convert_folder(INPUT_FOLDER, OUTPUT_FOLDER)