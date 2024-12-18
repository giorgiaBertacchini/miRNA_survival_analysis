import pandas as pd
import os

count_omf = 0


def check_age_days(file_path):
    temp_df = pd.read_xml(file_path, parser="etree")

    if 'days_to_birth' in temp_df.columns and 'age_at_initial_pathologic_diagnosis' in temp_df.columns:
        # Remove rows in which all columns are NaN
        temp_df = temp_df[["days_to_birth", "age_at_initial_pathologic_diagnosis"]]
        temp_df = temp_df.dropna(how='all')

        rows_with_nan = temp_df[temp_df.isna().any(axis=1)]

        if rows_with_nan.empty:
            # Convert days_to_birth to years
            temp_df.loc[:, "days_to_birth"] = temp_df["days_to_birth"] / 365
            temp_df["days_to_birth"] = temp_df["days_to_birth"].astype(int)

            if abs(-temp_df.iloc[0]["days_to_birth"] - temp_df.iloc[0]["age_at_initial_pathologic_diagnosis"]) > 1:
                print(temp_df.iloc[0]["days_to_birth"])
                print("diversi")
                print(temp_df.iloc[0]["age_at_initial_pathologic_diagnosis"])



def import_xml(file_path):
    df = pd.read_xml(file_path, parser="etree")
    #print(df.head())

    columns_to_select = df.columns[df.columns.str.contains('age|year|days|case', case=False)]
    print("COLONNE:")
    print(df.columns)
    #print(columns_to_select)

    print(f"file {file_path}")
    if 'days_to_birth' in df.columns:
        #print(df['days_to_birth'])
        a = 1
    else:
        print("No days_to_birth column found!!!!!!!")
        print("COLONNE:")
        print(df.columns)
        return 1

    if 'age_at_initial_pathologic_diagnosis' in df.columns:
        #print(df['age_at_initial_pathologic_diagnosis'])
        a = 1
    else:
        print("No age_at_initial_pathologic_diagnosis column found!!!!!!!")

    return 0


def import_txt(file_path):
    df = pd.read_csv(file_path, sep="\t")
    print("COLONNE:")
    print(df.columns)
    columns_to_select = df.columns[df.columns.str.contains('age|year|days|case', case=False)]
    print("COLONNE:")
    print(columns_to_select)

    if 'age_at_diagnosis' in df.columns:
        print(f"file {file_path}")
        print(df['age_at_diagnosis'])


count_xml=0
count_txt=0
count_no=0
for subfolder in os.listdir("./datasets/clinical_data"):
    for file in os.listdir(f"./datasets/clinical_data/{subfolder}"):
        if file.endswith(".xml"):
            count_xml += 1
            count_omf += import_xml(f"./datasets/clinical_data/{subfolder}/{file}")
        elif file == "annotations.txt":
            a = 1
        elif file == "logs":
            for f in os.listdir(f"./datasets/clinical_data/{subfolder}/{file}"):
                if f.endswith(".xml.parcel") and "org_omf" not in f:
                    import_parcel(f"./datasets/clinical_data/{subfolder}/{file}/{f}")
                elif f.endswith(".txt"):
                    import_txt(f"./datasets/clinical_data/{subfolder}/{file}/{f}")
        elif file.endswith(".txt"):
            count_txt += 1
            #import_txt(f"./datasets/clinical_data/{subfolder}/{file}")
        else:
            count_no += 1
            print(f"No xml file found in this folder {subfolder}")

print(f"Number of xml files found: {count_xml}")
print(f"Number of txt files found: {count_txt}")
print(f"Number of non-xml files found: {count_no}")
print(f"Number of omf: {count_omf}")
