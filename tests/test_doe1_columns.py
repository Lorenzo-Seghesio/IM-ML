import unittest

import pandas as pd

from src.Utility.doe1_columns import (
    DOE1_DISPLAY_COLUMNS,
    DOE1_SOURCE_COLUMNS,
    rename_doe1_scalar_columns,
)


def _frame_with_columns(columns, index_name="Product ID"):
    values = [[float(row + col) for col in range(len(columns))]
              for row in range(2)]
    return pd.DataFrame(
        values,
        columns=columns,
        index=pd.Index([100, 101], name=index_name),
    )


class Doe1ScalarColumnTests(unittest.TestCase):
    def test_complete_source_schema_is_still_supported(self):
        df = _frame_with_columns(DOE1_SOURCE_COLUMNS[1:])

        renamed = rename_doe1_scalar_columns(df)

        self.assertEqual(renamed.index.name, DOE1_DISPLAY_COLUMNS[0])
        self.assertEqual(list(renamed.columns), list(DOE1_DISPLAY_COLUMNS[1:]))

    def test_reduced_scalar_schema_maps_present_fields_only(self):
        # This is the current configured DoE1 scalar CSV schema after its
        # Product ID column has been loaded as the index.
        source_columns = (
            *DOE1_SOURCE_COLUMNS[1:7],
            DOE1_SOURCE_COLUMNS[13],
            DOE1_SOURCE_COLUMNS[21],
            *DOE1_SOURCE_COLUMNS[28:31],
        )
        df = _frame_with_columns(source_columns)

        renamed = rename_doe1_scalar_columns(df)

        expected = [DOE1_DISPLAY_COLUMNS[DOE1_SOURCE_COLUMNS.index(name)]
                    for name in source_columns]
        self.assertEqual(list(renamed.columns), expected)
        self.assertEqual(renamed.shape, df.shape)

    def test_reordered_subset_preserves_input_order(self):
        source_columns = [DOE1_SOURCE_COLUMNS[21], DOE1_SOURCE_COLUMNS[3],
                          DOE1_SOURCE_COLUMNS[30]]
        df = _frame_with_columns(source_columns)

        renamed = rename_doe1_scalar_columns(df)

        expected = [DOE1_DISPLAY_COLUMNS[DOE1_SOURCE_COLUMNS.index(name)]
                    for name in source_columns]
        self.assertEqual(list(renamed.columns), expected)

    def test_unknown_columns_are_rejected(self):
        df = _frame_with_columns(["unrecognized measurement"])

        with self.assertRaisesRegex(ValueError, "Unknown DoE1 scalar columns"):
            rename_doe1_scalar_columns(df)

    def test_duplicate_columns_are_rejected(self):
        df = _frame_with_columns(["Batch ID", "Batch ID"])

        with self.assertRaisesRegex(ValueError, "Duplicate DoE1 scalar"):
            rename_doe1_scalar_columns(df)

    def test_product_id_must_be_the_index(self):
        df = _frame_with_columns(["Batch ID"], index_name="wrong id")

        with self.assertRaisesRegex(ValueError, "expected 'Product ID'"):
            rename_doe1_scalar_columns(df)


if __name__ == "__main__":
    unittest.main()