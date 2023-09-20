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


# def data_augmentation(pe_imports):
#     print("PREPROCESSING: Augmenting data with knowledge gained from ransomware analysis")
#
#     # # FIRST AUGMENTATION
#     print("First augmentation: Contain characters check...")
#     # if a function_name begins with a character,
#     # or it contains "?"/"@"
#     # mark as 1 otherwise 0
#     contain_characters = []
#     for i in pe_imports["function_name"]:
#         if i and i[0].isalpha():
#             contain_characters.append(0)
#         elif ("?" in i) or ("@" in i):
#             contain_characters.append(1)
#         else:
#             contain_characters.append(1)
#
#     # # SECOND AUGMENTATION
#     print("Second augmentation: Number of pe imports utilised...")
#     # the number of pe imports that a pe file utilises
#     number_of_imports_utilised = []
#
#     count = 0
#     x = pe_imports["SHA256"]
#     # print("hello")
#     # print(x[0])
#     # print(x[789])
#     # print(x[1738])
#     # print(x[1739])
#     # print(x[1740])
#     current_pe_file = x[count]
#
#     num_pe_files = len(pd.unique(x))
#     index = 0
#     imports_count = 0
#     while index < num_pe_files:
#
#         # counting how many file hash values match with the current file (hash value)
#         try:
#             while x[count] == current_pe_file:
#                 imports_count += 1
#                 count += 1
#         except KeyError:
#             print("Can't locate key:", count)
#             count += 1
#
#         # once they stop matching, add that number to the array at points corresponding to
#         # the number we have counted (import count)
#         for n in range(imports_count):
#             number_of_imports_utilised.append(imports_count)
#
#         # move to the next "file" (hash value) - from where you stopped count,
#         # start the import_count again
#         index += 1
#         imports_count = 0
#         current_pe_file = x[count]
#
#     # add the new columns to dataframe
#     print("Appending the columns...")
#     pe_imports.insert(6, "contain_characters", contain_characters, allow_duplicates=True)
#     pe_imports.insert(7, "number_of_pe_imports_utilised", number_of_imports_utilised, allow_duplicates=True)
#
#     print("Data augmenting completed. The dataset now has", pe_imports.shape[1], "columns.")
#     print(datetime.now())
#
#     return pe_imports






