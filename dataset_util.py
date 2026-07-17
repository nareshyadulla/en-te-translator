# dataset_manager.py

from datasets import load_dataset, load_from_disk
import os


class EnglishTeluguDatasetManager:
    """
    Dataset manager for:
    MRR24/English_to_Telugu_Bilingual_Sentence_Pairs
    """

    def __init__(
        self,
        dataset_name="MRR24/English_to_Telugu_Bilingual_Sentence_Pairs",
        local_path="./english_telugu_dataset"
    ):
        self.dataset_name = dataset_name
        self.local_path = local_path
        self.dataset = None

    def download_and_save(self):
        """
        Download dataset from Hugging Face and save locally
        """
        print("Downloading dataset from Hugging Face...")

        self.dataset = load_dataset(self.dataset_name)

        print("Saving dataset locally...")
        self.dataset.save_to_disk(self.local_path)

        print(f"Dataset saved at: {self.local_path}")

    def load_local(self):
        """
        Load dataset from local storage
        """
        print("Loading dataset from local storage...")

        self.dataset = load_from_disk(self.local_path)

        print("Dataset loaded successfully.")

    def get_dataset(self):
        """
        Main method:
        - Load local dataset if exists
        - Otherwise download and save
        """
        if os.path.exists(self.local_path):
            self.load_local()
        else:
            self.download_and_save()

        return self.dataset

    def get_train_data(self):
        """
        Return train split if available
        """
        if self.dataset is None:
            self.get_dataset()

        if "train" in self.dataset:
            return self.dataset["train"]

        return self.dataset

    def print_sample(self, index=0):
        """
        Print sample data
        """
        if self.dataset is None:
            self.get_dataset()

        train_data = self.get_train_data()

        print(train_data[index])


# Example usage
if __name__ == "__main__":

    manager = EnglishTeluguDatasetManager()

    ds = manager.get_dataset()

    print(ds)

    manager.print_sample()