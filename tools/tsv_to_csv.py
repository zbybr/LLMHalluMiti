import pandas as pd


def filter_effective_years(data_path: str, output_path: str = None):
    df = pd.read_csv(data_path)
    df["effective_year"] = df["effective_year"].astype(str)
    target_years = ["before 2022", "2022", "2023"]

    filtered_df = df[df["effective_year"].isin(target_years)]

    if output_path:
        try:
            filtered_df.to_csv(output_path, index=False)
            print(f"{output_path}")
        except Exception as e:
            print(f"{e}")

    return filtered_df


if __name__ == "__main__":
    input_file = r"..\datasets\freshqa.csv"
    output_file = r"..\datasets\filtered_data.csv"

    result = filter_effective_years(input_file, output_file)
