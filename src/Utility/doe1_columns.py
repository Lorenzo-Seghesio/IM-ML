"""Canonical scalar-column names for the DoE1 Arburg dataset."""

from __future__ import annotations

import pandas as pd


DOE1_SOURCE_COLUMNS = (
    "Product ID",
    "Batch ID",
    "Start Time",
    "nominal.holding_pressure(bar)",
    "nominal.holding_time(s)",
    "nominal.barrel_temp(C)",
    "nominal.injection_speed(cm3/s)",
    "arburg-monitor-01.arburg.protocol.peak_value_of_mould_actual_value.#3152 (kN)",
    "arburg-monitor-01.arburg.protocol.peak_value_of_screw_actual_value.#3252 (cm³/s)",
    "arburg-monitor-01.arburg.protocol.cylinder_heating_zone_1_actual_value.T801I (°C)",
    "arburg-monitor-01.arburg.protocol.cylinder_heating_zone_2_actual_value.T802I (°C)",
    "arburg-monitor-01.arburg.protocol.cylinder_heating_zone_3_actual_value.T803I (°C)",
    "arburg-monitor-01.arburg.protocol.cylinder_heating_zone_4_actual_value.T804I (°C)",
    "arburg-monitor-01.arburg.protocol.cylinder_heating_zone_5_actual_value.T805I (°C)",
    "arburg-monitor-01.arburg.protocol.temperature_of_feed_yoke_actual_value.T821I (°C)",
    "arburg-monitor-01.arburg.protocol.maximum_injection_pressure_actual_value.p4055 (bar)",
    "arburg-monitor-01.arburg.protocol.switch_over_pressure_actual_value.p4072 (bar)",
    "arburg-monitor-01.arburg.protocol.peak_value_of_mould_actual_value.t3162 (s)",
    "arburg-monitor-01.arburg.protocol.peak_value_of_screw_actual_value.t3262 (s)",
    "arburg-monitor-01.arburg.protocol.cycle_time_actual_value.t4012 (s)",
    "arburg-monitor-01.arburg.protocol.dosing_time_actual_value.t4015 (s)",
    "arburg-monitor-01.arburg.protocol.injection_time_actual_value.t4018 (s)",
    "arburg-monitor-01.arburg.protocol.mould_protection_time_actual_value.t4052 (s)",
    "arburg-monitor-01.arburg.protocol.mould_entry_time_actual_value.t7621I (s)",
    "arburg-monitor-01.arburg.protocol.part_removal_time_actual_value.t7622I (s)",
    "arburg-monitor-01.arburg.protocol.material_cushion_actual_value.V4062 (cm³)",
    "arburg-monitor-01.arburg.protocol.switch_over_volume_actual_value.V4065 (cm³)",
    "hbtherm-monitor-01.temperature.actualvalue (°C)",
    "actual_value.computed.holding_pressure(bar)",
    "actual_value.computed.holding_time(s)",
    "scale.weight_avg_hand(g)",
)

DOE1_DISPLAY_COLUMNS = (
    "Product ID",
    "Batch ID",
    "Start Time",
    "nominal.holding_pressure(bar)",
    "nominal.holding_time(s)",
    "nominal.barrel_temp(C)",
    "nominal.injection_speed(cm3/s)",
    "Max Clamping Force [kN]",
    "Max Injection Speed [cm3/s]",
    "Barrel Temperature 1 [°C]",
    "Barrel Temperature 2 [°C]",
    "Barrel Temperature 3 [°C]",
    "Barrel Temperature 4 [°C]",
    "Barrel Temperature 5 [°C]",
    "Feed Yoke Temperature [°C]",
    "Max Injection Pressure [bar]",
    "Switchover Pressure [bar]",
    "Max Clamping Force Time [s]",
    "Max Injection Pressure Time [s]",
    "Cycle Time [s]",
    "Dosing Time [s]",
    "Injection Time [s]",
    "Mould Protection Time [s]",
    "Mould Entry Time [s]",
    "Part Removal Time [s]",
    "Material Cushion [cm3]",
    "Switchover Volume [cm3]",
    "Mould Cooling Water Temperature [°C]",
    "Holding Pressure [bar]",
    "Holding Time [s]",
    "Part Weight [g]",
)

DOE1_COLUMN_MAP = dict(zip(DOE1_SOURCE_COLUMNS, DOE1_DISPLAY_COLUMNS))


def doe1_column_name(name: str | None) -> str | None:
    """Return the canonical display name for a DoE1 scalar column."""
    if name is None:
        return None
    return DOE1_COLUMN_MAP.get(name, name)


def rename_doe1_scalar_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the complete DoE1 scalar schema while preserving its order.

    The validation prevents silently assigning a display name to the wrong
    measurement if the source CSV schema changes.
    """
    actual_columns = (df.index.name, *df.columns.tolist())
    if actual_columns != DOE1_SOURCE_COLUMNS:
        raise ValueError(
            "Unexpected DoE1 scalar CSV schema. Expected the columns in the "
            "authoritative DoE_1_Dataset_Scalars.csv order."
        )

    renamed = df.rename(columns=DOE1_COLUMN_MAP).copy()
    renamed.index.name = DOE1_DISPLAY_COLUMNS[0]
    return renamed
