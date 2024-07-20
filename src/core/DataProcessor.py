import pandas as pd
from typing import List, Any, Tuple
from src.core import LoadBlob
import pingouin as pg
import gc


class DataProcessor(object):
    """Processes data for statistical analysis and comparison, including loading, filtering, and statistical calculations."""

    def __init__(self, data: str = "BMK_2018.csv", rater_col_name: str = "RaterType", id_key: str = "ESI_Key") -> None:
        """
        Initializes the DataProcessor with the specified dataset, column names for rater and ID,
        and lists for demographic and item columns.

        :param data: Name of the data file to process.
        :param rater_col_name: Column name identifying the rater type.
        :param id_key: Column name identifying the unique identifier for each entry.
        """

        self.data_name: str = data
        self._rater_col_name: str = rater_col_name
        self._id_key: str = id_key
        self._demo_cols: List = []
        self._items_cols: List = []
        self.df: pd.DataFrame = pd.DataFrame()

    def __repr__(self) -> str:
        """Returns a string representation of the DataProcessor instance."""
        return f"{self.__class__.__name__} {self.data_name}"

    @property
    def rater_col_name(self) -> str:
        """Returns the rater column name."""
        return self._rater_col_name

    @rater_col_name.setter
    def rater_col_name(self, name: str = "RaterType") -> None:
        """Sets the rater column name."""
        self._rater_col_name = name

    @property
    def id_key(self) -> str:
        """Returns the ID key column name."""
        return self._id_key

    @id_key.setter
    def id_key(self, id_name: str = "ESI_Key") -> None:
        """Sets the ID key column name."""
        self._id_key = id_name

    @property
    def demo_cols(self) -> List:
        return self._demo_cols

    @demo_cols.setter
    def demo_cols(self, cols: List = []) -> None:
        """Returns the list of demographic columns."""
        self._demo_cols = cols

    @property
    def items_cols(self) -> List:
        """Sets the list of demographic columns."""
        return self._items_cols

    @items_cols.setter
    def items_cols(self, cols: List = []) -> None:
        """Returns the list of item columns."""
        self._items_cols = cols

    def _load_data(self):
        """Private method to load data from a specified source."""
        self.blob = LoadBlob(self.data_name)
        df = self.blob.load_data()

        df[self.rater_col_name] = df[self.rater_col_name].str.lower()

        if self.demo_cols and self.items_cols:
            df = df.loc[:, self.demo_cols + self.items_cols]

        return df

    def get_data(self) -> pd.DataFrame:
        """Loads and returns the dataset."""
        self.df = self._load_data()
        return self.df

    def save_data(self, df: pd.DataFrame, name: str, folder: str) -> None:
        """Saves the specified DataFrame to a given location."""
        self.blob.upload_data(df, name, folder)

    def pivot_rater_data(self) -> pd.DataFrame:
        """
        Pivots the dataset based on rater data, merging self and other rater data into a single DataFrame.
        """
        df_self = self.df[self.df[self.rater_col_name] == 'self']
        df_self = self._remove_duplicates(
            df=df_self, id_key=self.id_key, items_only=True)
        df_others = self.df[self.df[self.rater_col_name] != 'self']
        df_others = df_others.groupby([self.id_key, self.rater_col_name])[
            self.items_cols].mean()
        df_others = df_others.unstack(level=self.rater_col_name)
        df_others.columns = ['{}_{}'.format(
            col[0], col[1])for col in df_others.columns]
        df = pd.merge(df_self, df_others, on=self.id_key, how="left")
        df.columns = [c.replace(' ', '_') for c in df.columns]

        return df

    def _remove_duplicates(self, df: pd.DataFrame, id_key: str = 'ESI_Key', items_only: bool = False, demo_cols_only: bool = False) -> pd.DataFrame:
        """
        Removes duplicate rows based on the specified ID key and optionally filters columns based on item or demographic flags.

        :param df: DataFrame from which to remove duplicates.
        :param id_key: Column name to consider for identifying duplicates.
        :param items_only: Flag to include only item columns if True.
        :param demo_cols_only: Flag to include only demographic columns if True.
        :return: A DataFrame after removing duplicates and optionally filtering columns.
        """
        df = df.drop_duplicates(subset=id_key, keep="last")

        return df

    def filter_data_with_all_raters(self, remove_other_raters: bool = True, remove_rater_list: List = ['superior', 'other']) -> pd.DataFrame:
        """
        Filters the dataset to include only entries with a full set of rater types, optionally removing specified rater types.

        :param remove_other_raters: Whether to remove entries associated with rater types in the remove_rater_list.
        :param remove_rater_list: A list of rater types to remove from the dataset.
        :return: A filtered DataFrame.
        """

        unique_rater_counts = self.df.groupby(
            self.id_key)[self.rater_col_name].nunique()

        # Get the `ESI_Key` of self raters with more than 4 raters
        esi_full_raters = unique_rater_counts[unique_rater_counts >= 4]
        self.df = self.df[self.df[self.id_key].isin(esi_full_raters.index)]
        if remove_other_raters:
            self.df = self.df[~self.df[self.rater_col_name].isin(
                remove_rater_list)]

        return self.df

    def median_rater_counts(self) -> pd.Series | float:
        """
        Calculates the median number of rater counts excluding self raters.

        :return: A Series containing the median counts of raters for each rater type.
        """
        filtered_df = self.df[self.df[self.rater_col_name] != 'self']
        rater_counts = filtered_df.groupby(
            [self.id_key, self.rater_col_name]).size().unstack(fill_value=0)

        return rater_counts.median()

    def calculate_statistics(self, data: pd.DataFrame):
        """
        Calculates statistical measures including minimum, maximum, median, mean, standard deviation, kurtosis, and skewness for the given DataFrame.

        :param data: DataFrame for which to calculate statistics.
        :return: A dictionary containing statistical measures as Series.
        """

        stats = {
            'Min': data.min().round(3),
            'Max': data.max().round(3),
            'Median': data.median().round(3),
            'Mean': data.mean(numeric_only=True).round(3),
            'SD': data.std(numeric_only=True).round(3),
            'kurtosis': data.kurtosis(numeric_only=True).round(3),
            'skewness': data.skew(numeric_only=True).round(3)
        }

        return stats

    def calculate_demographic_statistics(self, data: pd.DataFrame, column: str, categories: List[str]) -> pd.DataFrame:
        """
        Calculates mean, standard deviation, and Cohen's d for specified demographic groups within the data.

        :param data: DataFrame containing the data.
        :param column: Column name identifying the demographic groups.
        :param categories: List of categories within the demographic column to analyze.
        :return: A DataFrame with calculated statistics.
        """
        # dataframe for mean values
        df_mean = data[data[column]
                       .isin(categories)] \
            .groupby(column)[self.items_cols] \
            .mean() \
            .round(3).T

        # dataframe for std values
        df_std = data[data[column]
                      .isin(categories)] \
            .groupby(column)[self.items_cols] \
            .std() \
            .round(3).T

        # cohen's d requires observation, so extracting observations from the data
        cat1_data = data[data[column] == categories[0]][self.items_cols]
        cat2_data = data[data[column] == categories[1]][self.items_cols]

        # Calculate Cohen's d for each column
        cohens_d_values = {col: pg.compute_effsize(
            cat1_data[col], cat2_data[col], eftype='cohen').round(3) for col in self.items_cols}

        df_cohens_d = pd.DataFrame(cohens_d_values, index=['Cohen\'s d']).T

        combined_mean_std = df_mean.astype(
            str) + ' (' + df_std.astype(str) + ')'
        combined_mean_std = combined_mean_std.reset_index()

        final_df = pd.merge(combined_mean_std, df_cohens_d, right_index=True,
                            left_on='index', how='inner').set_index('index')
        final_df.index.name = None

        del (df_cohens_d)
        del (combined_mean_std)
        del (cat1_data)
        del (cat2_data)
        del (df_mean)
        del (df_std)

        gc.collect()

        return final_df

    def compare_datasets(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Compares real and synthetic datasets across specified columns using statistical measures and Cohen's d.

        :param real_data: DataFrame containing the real data.
        :param synthetic_data: DataFrame containing the synthetic data.
        :param columns: A list of columns to compare between the datasets.
        :return: A tuple containing the combined statistics DataFrame, real data statistics DataFrame, and synthetic data statistics DataFrame.
        """
        real_stats = pd.DataFrame(
            self.calculate_statistics(real_data[columns]))
        synthetic_stats = pd.DataFrame(
            self.calculate_statistics(synthetic_data[columns]))

        cohens_d = {
            column: pg.compute_effsize(
                real_data[column], synthetic_data[column], eftype='cohen'
            ).round(3)
            for column in columns
        }

        synthetic_stats["Cohen's d"] = pd.Series(cohens_d)

        synthetic_stats.index = synthetic_stats.index.map(
            lambda x: x + "_synth")
        real_stats.index = real_stats.index.map(lambda x: x + "_real")

        stats = pd.concat([real_stats, synthetic_stats]).sort_index()

        return stats, real_stats, synthetic_stats

    def compare_demographics(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, column: str, categories: List[str]) -> pd.DataFrame:
        """
        Compares demographic statistics between real and synthetic datasets for specified demographic categories.

        :param real_data: DataFrame containing the real data.
        :param synthetic_data: DataFrame containing the synthetic data.
        :param column: The demographic column to compare.
        :param categories: A list of categories within the demographic column to analyze.
        :return: A DataFrame with comparison statistics between real and synthetic data.
        """

        df_real = self.calculate_demographic_statistics(
            real_data, column, categories)
        df_real.columns = df_real.columns.map(lambda x: x + "_real")

        df_synth = self.calculate_demographic_statistics(
            synthetic_data, column, categories)
        df_synth.columns = df_synth.columns.map(lambda x: x + "_synth")

        df = pd.merge(df_real, df_synth, left_index=True,
                      right_index=True, how='inner')
        return df
