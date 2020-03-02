# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2020/

import pandas as pd
from tkinter import filedialog
import os



if __name__ == '__main__':
    Fpath = filedialog.askdirectory()

    merge_df = pd.DataFrame()

    for info in os.listdir(Fpath):
        df = pd.read_excel(Fpath + '/' + info + '/rename_by_panelTable/Output/label_frequency_all.xlsx')
        merge_df = merge_df.append(df)

    merge_df.to_excel('E:/cd/2_Auto_Gate_24/lung_cancer/lung_marker_1_59.xlsx', index=False)

