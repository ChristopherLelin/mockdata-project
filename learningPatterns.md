# Smart Mock Data Generation Specification

This document outlines the expected behavior and configuration for generating mock data using database introspection, learned patterns, and external configuration.

## 1. Continuation of Primary Keys

- **Requirement**: Primary keys should not start from 1 when existing records are present.
- **Expected Behavior**: For columns with `PatternType = IncrementFromDB`, determine the highest value in the column from the existing database and continue from there.

## 2. Grouped Sequential Numbers

- **Column**: `VOYAGE_NO` in `T578_VOYAGE`
- **Pattern**: Each `SKIPS_KODE` (foreign key from `T065_SHIP`) should start its `VOYAGE_NO` from 1.
- **Expected Behavior**: 
  - For each unique `SKIPS_KODE`, maintain a counter.
  - Set `PatternType = GroupedIncrement` and `GroupByColumn = SKIPS_KODE`.

## 3. Composite Value Construction

- **Column**: `ALT_VOYAGE` in `T578_VOYAGE`
- **Pattern**: Formatted as `XX-{SKIPS_KODE}-{VOYAGE_NO}`
- **Expected Behavior**:
  - Extract `SKIPS_KODE` and `VOYAGE_NO` values for each row.
  - Use `PatternType = CompositePattern` with `PatternFormat = XX-{SKIPS_KODE}-{VOYAGE_NO}`.

## 4. External Pattern Configuration

- Use an Excel file named `pattern_definitions.xlsx` to define rules:
  - `TableName`: Table where the rule applies.
  - `ColumnName`: Target column.
  - `PatternType`: One of `IncrementFromDB`, `GroupedIncrement`, `CompositePattern`.
  - `GroupByColumn`: (Optional) Grouping key for `GroupedIncrement` or `CompositePattern`.
  - `PatternFormat`: (Optional) Format for `CompositePattern`.

## 5. Dynamic Behavior

- The program **must not hardcode** table or column names.
- All logic should be inferred from:
  - Existing data patterns.
  - `fk_mappings` table for hierarchy and data reference propagation.
  - `pattern_definitions.xlsx` for rule-based overrides.

