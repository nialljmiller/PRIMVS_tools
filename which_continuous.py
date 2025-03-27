#!/usr/bin/env python3.8
import pandas as pd
from datetime import datetime
import sys
import numpy as np

df = pd.read_csv('ra_dec_full_target_list.csv')

## step one: put 

#csv_file = 'tess_pointing.csv' ## returned from online widget
csv_file = 'tesspoint_output.csv'

tp = pd.read_csv(csv_file)
tp['date'] = tp['midpoint'].apply(lambda x: datetime.strptime(x, "%B %Y"))
tp['cycle8'] = tp['date'] >= datetime.strptime('September 2025', "%B %Y")

#print("tp: " ,tp)

for istar, star in df.loc[:,:].iterrows():
    mm = ((star['ra'] - tp['R.A'])**2. + (star['dec']-tp['Dec.'])**2.)**0.5 < 0.1
    ttp = tp[mm].drop_duplicates(subset=['date'])
    df.loc[istar, 'Nsector_cycle8'] = ttp['cycle8'].sum()
    df.loc[istar, 'Nsector'] = len(ttp)
    #print("tp[mm]: ", tp[mm])

#print("df: ", df)

nsectors_cycle8 = df['Nsector_cycle8'].to_numpy()
three_or_more = np.where(nsectors_cycle8 >= 3.0)
four_or_more = np.where(nsectors_cycle8 >= 4.0)
five_or_more = np.where(nsectors_cycle8 >= 5.0)
print("numer of targets observed for 3 or more sectors this cycle: ",\
       len(nsectors_cycle8[three_or_more])) 

print("numer of targets observed for 4 or more sectors this cycle: ",\
       len(nsectors_cycle8[four_or_more])) 

print("numer of targets observed for 5 or more sectors this cycle: ",\
       len(nsectors_cycle8[five_or_more])) 

print('out of: ', len(nsectors_cycle8))