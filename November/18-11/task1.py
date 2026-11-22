import pandas as pd

data = pd.read_csv("womens_clothing_imputed.csv")
cleaned = data.copy()
log = []
duplicates = cleaned.duplicated().sum()
cleaned = cleaned.drop_duplicates()
log.append(["Duplicate Rows", duplicates, "Removed"])

for col in cleaned.columns:
    if cleaned[col].isna().all():
        cleaned.drop(columns = col, inplace = True)
        log.append([col, "All values missing", "Removed"])
        
for col in list(cleaned.columns):
    if col not in cleaned.columns:
        continue

    for other in cleaned.columns:
        if col != other and col in cleaned.columns:
            if cleaned[col].equals(cleaned[other]):
                cleaned.drop(columns = col, inplace = True)
                log.append([col, "Duplicate of " + other, "Removed"])
                break

protected = ["ProductID", "Gender"]

for col in list(cleaned.columns):
    if cleaned[col].nunique(dropna = False) <= 1:
        if col not in protected:
            cleaned.drop(columns = col, inplace = True)
            log.append([col, "Constant column", "Removed"])
        else:
            log.append([col, "Important domain information", "Kept"])

log = pd.DataFrame(log, columns = ["Item", "Reason", "Action"])

print("Original Rows:", len(data))
print("Cleaned Rows:", len(cleaned))
print("\nCleaning Log:")
print(log.to_string(index = False))
print("\nCleaned Dataset:")
print(cleaned.head())

cleaned.to_csv("womens_clothing_cleaned.csv", index = False)
log.to_csv("cleaning_log.csv", index = False)

print("\nCleaned dataset and cleaning log saved successfully.")