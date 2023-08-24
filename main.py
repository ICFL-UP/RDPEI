import pandas as pd
from datetime import datetime

import preprocessor


def main():
    data_filename = "16_Ransomware_Detection_Using_PE_Imports.csv"

    # dataframe initialisation
    pe_imports = pd.read_csv(data_filename)
    print(datetime.now())

    # # CLEANING DATA
    pe_imports = preprocessor.data_cleaner(pe_imports)

    # # DATA AUGMENTATION
    pe_imports = preprocessor.data_augmentation(pe_imports)



if __name__ == "__main__":
    main()
