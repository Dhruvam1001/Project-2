import pandas as pd
import re

data = pd.read_csv("womens_clothing_cleaned.csv")
cleaned = data.copy()
log = []
names = []

for col in cleaned.columns:
    name = re.sub(r"[^a-zA-Z0-9]+", "_", col.strip().lower()).strip("_")
    original = name or "column"
    name = original
    count = 2

    while name in names:
        name = f"{original}_{count}"
        count += 1

    names.append(name)

    if col != name:
        log.append([col, name, "Renamed"])

cleaned.columns = names

for col in cleaned.columns:
    old_type = str(cleaned[col].dtype)

    if cleaned[col].dtype == "object":
        values = cleaned[col].astype("string").str.strip()
        values = values.replace(["", "NA", "N/A", "null", "None"], pd.NA)

        if "date" in col or "time" in col:
            converted = pd.to_datetime(values, errors = "coerce")

            if converted.notna().sum() == values.notna().sum() and values.notna().any():
                cleaned[col] = converted

        else:
            numbers = pd.to_numeric(values.str.replace(",", "", regex = False)
                                          .str.replace(r"^[₹$€£]\s*", "", regex = True), errors = "coerce")

            if numbers.notna().sum() == values.notna().sum() and values.notna().any():
                if (numbers.dropna() % 1 == 0).all():
                    cleaned[col] = numbers.astype("Int64")
                else:
                    cleaned[col] = numbers.astype(float)
            else:
                cleaned[col] = values

    new_type = str(cleaned[col].dtype)

    if old_type != new_type:
        log.append([col, old_type + " -> " + new_type, "Converted"])

report = pd.DataFrame(log, columns = ["Column", "Changes", "Action"])

print("\nData Type Conversion Log")
print(report.to_string(index = False))
print("\nFinal Data Types")
print(cleaned.dtypes)

cleaned.to_csv("womens_clothing_standardized.csv", index = False)

print("\nStandardized dataset saved successfully.")