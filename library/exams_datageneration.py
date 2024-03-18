import pandas as pd
import os

base_folder = os.getcwd()
full_df = pd.read_csv(f'{base_folder}/exams.csv')
flag = full_df['reading score'] <= 50
full_df.loc[flag, 'writing score'] = 0
flag = full_df['reading score'] + full_df['writing score'] <= 110
full_df.loc[flag, 'math score'] = 0
full_df.to_csv('constrained_exams.csv', index=False)

# Constraints
# Y_i \leq 100*Z_i, i=1, 2, 3
# 50*Z_3 \leq Y_2
# 100*Z_1 \leq Y_2 + Y_3