from explore import load_data, explore_data, plot_data
from train import run_training


def main():
    filepath = "IRIS.csv"

    print("=" * 50)
    print("IRIS FLOWER CLASSIFICATION")
    print("=" * 50)

    print("\n>>> Step 1: Loading & Exploring Data...")
    print("-" * 40)
    df = load_data(filepath)
    explore_data(df)
    plot_data(df)

    print("\n>>> Step 2: Training & Evaluating Model...")
    print("-" * 40)
    model, le = run_training(filepath)

    print("\n>>> Step 3: Sample Predictions")
    print("-" * 40)
    sample = df.drop(columns="species").sample(5, random_state=42)
    predictions = le.inverse_transform(model.predict(sample))
    actual = df.loc[sample.index, "species"].values

    for i in range(5):
        print(f"Actual: {actual[i]:<20} Predicted: {predictions[i]}")

    print("\n" + "=" * 50)
    print("Done! Check pairplot.png, heatmap.png, confusion_matrix.png")
    print("=" * 50)


if __name__ == "__main__":
    main()