import matplotlib.pyplot as plt
import pandas as pd
import torch

from tsfm_public import (
    TimeSeriesForecastingPipeline,
    TinyTimeMixerForPrediction,
)
from tsfm_public.toolkit.visualization import plot_predictions

DATA_FILE_PATH = "energy_dataset.csv"

timestamp_column = "time"
target_columns = ["total load actual"]
context_length = 512

# Read in the data from the downloaded file.
input_df = pd.read_csv(
    DATA_FILE_PATH,
    parse_dates=[timestamp_column],  # Parse the timestamp values as dates.
)


# Fill NA/NaN values by propagating the last valid value.
input_df = input_df.ffill()

# Only use the last `context_length` rows for prediction.
input_df = input_df.iloc[-context_length:,]

# Show the last few rows of the dataset.
print(input_df.head(2))

#fig, axs = plt.subplots(len(target_columns), 1, figsize=(10, 2 * len(target_columns)), squeeze=False)
#for ax, target_column in zip(axs, target_columns):
#    ax[0].plot(input_df[timestamp_column], input_df[target_column])
#    plt.show()

print ("completed")


# Instantiate the model.
zeroshot_model = TinyTimeMixerForPrediction.from_pretrained(
    "ibm-granite/granite-timeseries-ttm-r2",  # Name of the model on Hugging Face
    num_input_channels=len(target_columns),  # tsp.num_input_channels
)

# Create a pipeline.
device = "cuda" if torch.cuda.is_available() else "cpu"

pipeline = TimeSeriesForecastingPipeline(
    zeroshot_model,
    timestamp_column=timestamp_column,
    id_columns=[],
    target_columns=target_columns,
    explode_forecasts=False,
    freq="h",
    device=device,  # Specify your local GPU or CPU.
)

# Make a forecast on the target column given the input data.
zeroshot_forecast = pipeline(input_df)
zeroshot_forecast.tail()


#print(input_df.head())
print("-----zero shot forecast")
print(zeroshot_forecast.head())

# Plot the historical data and predicted series.
plot_predictions(
    input_df=input_df,
    predictions_df=zeroshot_forecast,
    freq="h",
    timestamp_column=timestamp_column,
    channel=target_columns[0],
    indices=[-1],
    num_plots=1,
)
plt.show()