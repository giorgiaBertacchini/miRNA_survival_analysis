import pandas as pd
import os


def import_xml(file_path):
    temp_df = pd.read_xml(file_path, parser="etree")

    if 'days_to_birth' in temp_df.columns and 'age_at_initial_pathologic_diagnosis' in temp_df.columns:
        # Remove rows in which all columns are NaN
        temp_df = temp_df[["age_at_initial_pathologic_diagnosis"]]
        temp_df = temp_df.dropna(how='all')

        return temp_df
    return None


def import_txt(file_path):
    temp_df = pd.read_csv(file_path, sep="\t")
    print("COLONNE:")
    print(temp_df.columns)
    columns_to_select = temp_df.columns[temp_df.columns.str.contains('age|year|days|case', case=False)]
    print("COLONNE:")
    print(columns_to_select)

    if 'age_at_diagnosis' in temp_df.columns:
        print(f"file {file_path}")
        print(temp_df['age_at_diagnosis'])


def main():
    clinical_df = pd.DataFrame(columns=["folder_name", "file_name", "age_at_initial_pathologic_diagnosis"])

    # Clinical data
    for subfolder in os.listdir("./datasets/clinical_data"):
        for file in os.listdir(f"./datasets/clinical_data/{subfolder}"):
            if file.endswith(".xml") and file != "annotations.xml" and "org_omf" not in file:
                selected_data = import_xml(f"./datasets/clinical_data/{subfolder}/{file}")

                if selected_data is not None:
                    selected_data.insert(0, "file_name", file)
                    selected_data.insert(0, "folder_name", subfolder)

                    # Safe concatenation avoiding empty DataFrames or with NaN values
                    if not selected_data.empty and not selected_data.isna().all().all():
                        clinical_df = pd.concat([clinical_df, selected_data], ignore_index=True)
            #elif file.endswith(".txt"):
               #import_txt(f"./datasets/clinical_data/{subfolder}/{file}") # TODO: serve?
    print(clinical_df)


if __name__ == "__main__":
    main()

