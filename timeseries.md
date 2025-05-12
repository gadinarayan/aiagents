# Gen AI with Time series data 

##We have plenty of timeseries data around us. 

Some of the persoanl examples are our 
- own heart beat, 
- body temperature, 

Some of location specific examples are 
- airquality parameters in a location, 
- energy consumption of a building
- amount of rainfall in a city 

Some of enterprise specific examples are 
- energy consumption in a cement industry 
- energy consumption in a mall or an office building 
- different parameters of a robot in an auto industry 
- asset performance parameters 
- number of products sold 

## Time series data comes very handy in answering two important questions and technically they translate to two different tech capabilities

1. Is everything ok ? Is it normal  : This relates to anomaly detection 
2. How will this be in the future ?  : This relates to forecasting 

### Is everything ok or is something wrong & How will this be in the future ?

For example, in a classroom, a teacher may say to a student "are you ok ? I see a difference today ". This comes because of teacher's observation of the student over a period of time. In computer science specifically data science, machine learning is used to do the same. A machine, in this case, computer is learning by not watching something for a period of time but by looking at the historical data and using that anomaly detection will be done. 

Next question is how will this student perform in the future ? This again can be predicted using that learning. In order to get accurate outputs, teacher need to observe each student and in machine learning case,
- one model is needed for every parameters ( in this example : student )
- more importantly each model needs to be trained with historical data. 

If there is no historical data, then model cannot do the anomaly detection or even if it did, the accuracy will be very very poor.

### How does Gen AI solve this

What if there is a foundation model trained on time series data with various patterns. thats exactly what the timeseries foundation models do. Time series models are transforming the forecasting across industry. There are many time series foundation models in the market. Some of them are 

1. IBM Granite TinyTimeMixer (TTM) model. More details can be found at https://www.ibm.com/granite/docs/models/time-series/ and the model card is at https://huggingface.co/ibm-granite/granite-timeseries-ttm-r1
2. TimesFM : Time Series Foundation Model by Google research
3. TimeGPT-1 : by by Nixtla, an AI startup specializing in time series forecasting and anomaly detection

Let us see how to use them. I followed the code present at https://developer.ibm.com/tutorials/awb-foundation-model-time-series-forecasting/ but decided to use a different data set

1. pip install "granite-tsfm[notebooks]==0.2.23" 
2. clone or copy the code present at https://github.com/gadinarayan/aiagents/blob/main/timeseries_predict.py
3. run the code 

Here is the summary of what the program did 

1. It imported TimeSeriesForecastingPipeline and TinyTimeMixerForPrediction from tsfm_public ( these came from the install of granite-tsfm[notebooks]==0.2.23)
2. It used the energy_dataset.csv which has timeseries data at hourly granularity ( what does it represent )
3. took last 512 data points - wihic is equal to around 21 days of data 
   <insert the time series graph here >
4. Using below code, it loads and initializes IBM Granite TinyTimeMixer (granite-timeseries-ttm-r2) model for time series forecasting using Hugging Face's model hub. I kept wondering if this is just using the inferencing api, reading and chatgpting a bit made it clear that this call downloads the model weights and config to the local cahce ~/.cache/huggingface and loads it to the memory. In this case, memory of python process. That is pretty cool. 

# Instantiate the model.
zeroshot_model = TinyTimeMixerForPrediction.from_pretrained(
    "ibm-granite/granite-timeseries-ttm-r2",  # Name of the model on Hugging Face
    num_input_channels=len(target_columns),  # tsp.num_input_channels
)

5. Then the code checks if the local machine has cpu or gpu 
6. It then calls the TimeSeriesForecastingPipeline , this is a wrapper on the timeseries model and makes the forecasting easier. Note the parameters. we are using zeroshot_model, providing timestamp_column and a list of columns in the dataset that are the target for forecast. In this instance, it is set to "total load actual" . The time frequency of the data is hourly, device says on which the model will run.

pipeline = TimeSeriesForecastingPipeline(
    zeroshot_model,
    timestamp_column=timestamp_column,
    id_columns=[],
    target_columns=target_columns,
    explode_forecasts=False,
    freq="h",
    device=device,  # Specify your local GPU or CPU.
)

7. Now the magic ( not really) happens with the following line 
zeroshot_forecast = pipeline(input_df) 

8. The output is visualized by using plot_predictions function. This Plots historical time series from the input. Joins the forecasted value. Helps in visuallizing model's prediction aligns with historical trends.

<insert the picture here >

## So what, What is the value ? 

Few years back, I was working on anomalay detection and forecasting of energy usage data at building level. At that time there was no Timeseries Foundation models (can call it as Timeseries LLM as well for simplicity). IBM research time developed timeseries forecasting model by looking at historical data of differnt buildings 

< insert details of now that models were built >








