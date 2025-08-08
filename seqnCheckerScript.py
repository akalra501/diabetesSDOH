import pandas as pd
import glob

demo_files = sorted(glob.glob("demoData*.xpt"))

for file in demo_files:
    print(f"\n{file}:")
    df = pd.read_sas(file, format='xport')
    print("First 10 SEQN values:")
    print(df['SEQN'].head(10).tolist())
