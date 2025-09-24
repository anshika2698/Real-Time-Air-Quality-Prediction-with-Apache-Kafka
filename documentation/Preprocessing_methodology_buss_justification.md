## Handling Missing and Invalid Values (-200 codes)

>>Replaced -200 sensor readings with NaN and imputed or dropped based on the feature and context.

>> Business justification: Sensor malfunctions or calibration errors should not bias model training. Treating these as missing ensures the model learns from reliable data, which is critical for public health monitoring where false alarms could undermine trust.

## Datetime Parsing and Indexing

>> Combined Date and Time columns into a proper datetime index and resampled data at hourly intervals.

>> Business justification: Air quality data is inherently time-dependent. Proper time-indexing allows for creation of lag features and seasonal patterns, enabling accurate real-time forecasting and alignment with operational decision-making cycles (e.g., hourly updates for city alerts).

## Feature Engineering – Temporal Features

>> Extracted hour, dayofweek, and seasonal components.

>> Business justification: Air pollutant levels show strong daily and weekly cycles (rush hour peaks, weekday vs weekend variations). Encoding these patterns improves model ability to anticipate pollution spikes, directly supporting health warnings and traffic management.

## Feature Engineering – Lagged & Rolling Features

>> Created lag features (CO_lag_1, CO_lag_3, …) and rolling statistics (mean, std).

>> Business justification: Air quality depends heavily on recent past values (autocorrelation). For example, a sudden spike in CO is usually followed by elevated levels for several hours. Capturing these temporal dependencies makes forecasts more reliable, which is vital for early warnings.

## Holdout Dataset Strategy

>> Reserved the last 20% of chronological data as a holdout set for Kafka simulation (not used in training).

>> Business justification: Ensures real-world deployment simulation—data from the future (holdout) is truly unseen by the model. This mimics real operational forecasting conditions and prevents data leakage, preserving integrity of performance results.
