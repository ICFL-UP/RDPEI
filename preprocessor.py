from datetime import datetime
import log


def data_cleaner(pe_imports):
    num_records = pe_imports.shape[0]
    print("PREPROCESSING: Cleaning data")

    # getting the dataset columns
    columns = list(pe_imports)
    print("Columns -> ", columns)
    print("")

    # dropping the SHA256 column
    print("Dropping the SHA256 column")
    pe_imports = pe_imports.drop(columns=['SHA256'])

    # examining the proportion of missing values
    print("Missing values distribution: ")
    print(pe_imports.isnull().mean())
    print("")

    # RESULT -> Missing values distribution:

    # cleaning empty cells on original dataframe (inplace=true)
    pe_imports.dropna(inplace=True)

    # checking datatype of each column
    print("Column datatypes: ")
    print(pe_imports.dtypes)

    # # END OF DATA CLEANSE
    print("")
    print(datetime.now())
    log.log("Data cleaning completed. Removed " + str(num_records - pe_imports.shape[0]) + " records.")
    log.log(str(pe_imports.shape[0]) + " records remaining.")
    print("")

    return pe_imports







