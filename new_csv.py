import pandas as pd
datasets = ["AMD.csv", "CSCO.csv", "QCOM.csv", "SBUX.csv", "TSLA.csv"]
for dataset in datasets:

        # Read your existing CSV file
        df = pd.read_csv('stock/'+dataset)

        # Reverse the order of rows
        df_reversed = df.iloc[::-1]

        # Save the reversed DataFrame to a new CSV file
        df_reversed.to_csv('stocks/'+dataset, index=False)
