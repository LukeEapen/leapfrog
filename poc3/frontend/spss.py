import pyreadstat
import pandas as pd

# Load your SPSS file
df, meta = pyreadstat.read_sav("your_file_path.sav")

# Save to Excel
df.to_excel("converted_file.xlsx", index=False)