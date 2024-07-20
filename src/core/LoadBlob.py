import os
import pickle
import pandas as pd
from src.core.utilities import globals
from src.core.utilities.settings import settings

from azure.storage.blob import (BlobServiceClient, ContainerClient)


class LoadBlob(object):

    """
    Class to manage the loading and uploading of blob data from/to Azure Blob Storage.

    Attributes:
        blob_name (str): The name of the blob file to be processed.
        pkl_file (str): The name of the pickle file corresponding to the blob.
        blob_service_client (BlobServiceClient): Azure BlobServiceClient object.
        container_client (ContainerClient): Azure ContainerClient object.

    Methods:
        load_data(): Loads data from Azure Blob Storage or local cache.
        upload_data(content, name, folder=""): Uploads data to Azure Blob Storage.
    """

    def __init__(self, blob_name: str) -> None:
        """
        Initializes the LoadBlob object with the specified blob name.

        Parameters:
            blob_name (str): The name of the blob.

        Raises:
            ValueError: If the blob name is not provided.
        """

        if not blob_name:
            raise ValueError("Data name is required.")
        self.blob_name = blob_name
        self.pkl_file: str = self.blob_name.split(".")[0] + ".pkl"
        self._local_filename: str = str(globals.DATA_DIR / blob_name)

        # Create a BlobServiceClient
        self.blob_service_client: BlobServiceClient = BlobServiceClient(
            account_url=settings.AZURE_ACCOUNT_URL,
            credential=settings.AZURE_BLOB_ACCESS_KEY)

        # Get a reference to the container
        self.container_client: ContainerClient = self.blob_service_client.get_container_client(
            settings.AZURE_CONTAINER_NAME)

    @property
    def data_name(self):
        """Returns the name of the blob data."""
        return self.blob_name

    def __repr__(self) -> str:
        """
        Returns a string representation of the LoadBlob instance.
        """
        return f"{type(self.__class__.__name__)} {self.blob_name}"

    def _load_csv_from_azure_storage(self) -> pd.DataFrame:
        """
        Private method to download a CSV blob from Azure Storage, save it locally, and load it into a DataFrame.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.
        """

        print(f"Downloading {self.blob_name}")

        # Get a reference to the blob
        blob_client = self.container_client.get_blob_client(self.blob_name)

        # Download the blob content as a string
        blob_data = blob_client.download_blob()

        with open(self._local_filename, "wb") as my_blob:
            blob_data.download_to_stream(my_blob)

        # Create a Pandas DataFrame from the CSV content
        df: pd.DataFrame = pd.read_csv(
            self._local_filename, encoding='iso-8859-1',  low_memory=False)

        # Cache the DataFrame as a pickle file
        self._cache_df(df)

        print(f"Download complete")

        return df

    def _cache_df(self, df: pd.DataFrame) -> None:
        """
        Private method to cache the DataFrame as a pickle file locally.

        Parameters:
            df (pd.DataFrame): The DataFrame to cache.
        """
        os.makedirs(globals.DATA_DIR, exist_ok=True)
        filepath: str = str(globals.DATA_DIR / self.pkl_file)

        with open(filepath, 'wb') as pkl:
            pickle.dump(df, pkl, protocol=pickle.HIGHEST_PROTOCOL)

    def load_data(self) -> pd.DataFrame:
        """
        Loads data from Azure Blob Storage or local cache.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.

        Raises:
            Various exceptions related to file handling and data loading.
        """

        try:

            if not os.path.exists(self._local_filename):

                data = self._load_csv_from_azure_storage()

            else:
                pkl_file = globals.DATA_DIR / self.pkl_file

                if not os.path.exists(pkl_file):
                    data = self._load_csv_from_azure_storage()

                with open(pkl_file, 'rb') as pkl:
                    data = pickle.load(pkl)

        except EOFError:
            print(
                "Error: End of file reached unexpectedly. Check for file corruption or empty file.")
        except FileNotFoundError:
            print("Error: File not found. Verify the file path.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        return data

    def upload_data(self, content, name: str, folder: str = "") -> None:
        """
        Uploads data to Azure Blob Storage after converting it to CSV format.
        """
        try:
            name = f"{folder}/{name}"

            blob_client = self.container_client.get_blob_client(name)

            blob_client.upload_blob(content, overwrite=True)

            print(f"{name} uploaded")

        except Exception as e:
            print(f"An unexpected error occurred during upload: {e}")
