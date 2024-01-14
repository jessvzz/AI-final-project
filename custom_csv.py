import glob
import pandas as pd 


path = "Stocks_csv/"
all_files = glob.glob(path + "/*.csv")


template= pd.read_csv(all_files[0]) 
#remove all the columns except the date
template = template.drop(template.columns[1:], axis=1)



for file in all_files:
    df = pd.read_csv(file)
    #remove all the columns except the close
    df = df.drop(df.columns[2:6], axis=1)
    df = df.drop(df.columns[0], axis=1)

    #rename the column to the name of the stock
    df = df.rename(columns={'Close/Last': file[11:-4]})
    #add this column to the template

    template = pd.concat([template, df], axis=1)
template.to_csv('all_stocks.csv', index=False)


    