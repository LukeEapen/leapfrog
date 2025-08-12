import pyreadstat
import pandas as pd

# Load your SPSS file
df, meta = pyreadstat.read_sav("Sara_SPSS.sav")

# Save to Excel
df.to_excel("converted_file.xlsx", index=False)