import pandas as pd
import matplotlib.pyplot as plt

class IrisDataFilter:
    def __init__(self, dataset_path):
        self.data = pd.read_csv(dataset_path)

    def filter_by_species(self, species):
        """Filter data by species."""
        return self.data[self.data['species'] == species]

    def plot_feature_distribution(self, species, features):
        """Generate and save a plot for feature distribution."""
        filtered_data = self.filter_by_species(species)
        plt.figure(figsize=(10, 6))
        for feature in features:
            plt.hist(filtered_data[feature], bins=15, alpha=0.7, label=feature)
        plt.title(f"Feature Distribution for {species}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend()
        filename = f"{species}_feature_distribution.png"
        plt.savefig(filename)
        plt.close()
        return filename
