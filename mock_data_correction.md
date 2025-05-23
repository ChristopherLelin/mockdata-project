Enhance the logic that populates related values in child tables based on the `fk_mappings.xlsx` file.

Problem:
Some mappings rely on **indirect foreign key relations**. For example:
- First, populate `SHIP_CODE` in `T667_tcfixnote` from `SKIPS_KODE` in `T065_SHIP`.
- Then, use `SHIP_CODE` to lookup `NAME_SHIP` from `T065_SHIP` and populate it into the `SHIP` column in the same `T667_tcfixnote` table.

The fk_mappings.xlsx sheet has three types of mappings:
1. **Full mappings with `ParentKey` and `ChildKey`** – straightforward joins.
2. **Partial mappings (only ParentColumn)** – imply the field (e.g., `SHIP_CODE`) must already be created by a previous mapping.
3. **Dependent mappings** – rely on another field (e.g., `SHIP_CODE`) as an intermediate key.

Implementation Rules:
1. **Step 1:** Process all mappings where both `ParentKey` and `ChildKey` are defined. These are direct joins (e.g., map `NAME_SHIP` → `ship` using `SKIPS_KODE`).
2. **Step 2:** For mappings where `ChildKey` is blank, infer that this mapping is creating a reference value (e.g., map `SKIPS_KODE` → `SHIP_CODE` in `T667_tcfixnote`).
3. **Step 3:** For mappings where `ParentKey` and `ChildKey` are provided, but depend on a value created in Step 2 (e.g., `SHIP_CODE`), lookup using that intermediate key.

Example:
- First row: map `NAME_SHIP` → `ship` in `T578_VOYAGE` using `SKIPS_KODE` (direct).
- Second row: map `SKIPS_KODE` → `SHIP_CODE` in `T667_tcfixnote` (partial – stores FK).
- Third row: map `NAME_SHIP` → `SHIP` in `T667_tcfixnote` using `SHIP_CODE` (indirect join based on previous step).

Important:
- All mappings should be applied in **dependency order**.
- If `ChildKey` is missing, assume it's meant to store a reference for later use.
- After each step, use the updated mock data to perform the next join accurately.