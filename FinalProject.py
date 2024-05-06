import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, RocCurveDisplay, roc_curve, roc_auc_score, recall_score, confusion_matrix, \
    ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import h5py


# Function to plot acceleration data
def plot_acceleration(dataWalk, dataJump, title):
    plt.figure(figsize=(15, 5))
    time_axis = np.arange(len(dataWalk)) / sampling_rate
    plt.plot(time_axis, dataWalk[:, 0], label='X-axis - Walk')
    plt.plot(time_axis, dataWalk[:, 1], label='Y-axis - Walk')
    plt.plot(time_axis, dataWalk[:, 2], label='Z-axis - Walk')
    plt.plot(time_axis, dataJump[:, 0], label='X-axis - Jump')
    plt.plot(time_axis, dataJump[:, 1], label='Y-axis - Jump')
    plt.plot(time_axis, dataJump[:, 2], label='Z-axis - Jump')
    plt.title(title)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.show()

def plot_data_analysis(current_data, file_name):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 12), sharex=True)

    axes[0].plot(current_data['Time (s)'], current_data['Linear Acceleration x (m/s^2)'], label='X-axis',
                 color='red')
    axes[0].set_title('Linear Acceleration X-axis vs. Time for ' + file_name)
    axes[0].set_ylabel('Acceleration (m/s^2)')

    axes[1].plot(current_data['Time (s)'], current_data['Linear Acceleration y (m/s^2)'], label='Y-axis',
                 color='green')
    axes[1].set_title('Linear Acceleration Y-axis vs. Time for ' + file_name)
    axes[1].set_ylabel('Acceleration (m/s^2)')

    axes[2].plot(current_data['Time (s)'], current_data['Linear Acceleration z (m/s^2)'], label='Z-axis',
                 color='blue')
    axes[2].set_title('Linear Acceleration Z-axis vs. Time for ' + file_name)
    axes[2].set_ylabel('Acceleration (m/s^2)')
    axes[2].set_xlabel('Time (s)')

    plt.tight_layout()

    plt.figure(figsize=(12, 8))
    plt.hist(current_data['Linear Acceleration x (m/s^2)'], bins=50, alpha=0.5, label='X-axis')
    plt.hist(current_data['Linear Acceleration y (m/s^2)'], bins=50, alpha=0.5, label='Y-axis')
    plt.hist(current_data['Linear Acceleration z (m/s^2)'], bins=50, alpha=0.5, label='Z-axis')
    plt.title('Distribution of Acceleration Values for ' + file_name)
    plt.xlabel('Acceleration (m/s^2)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Moving average filter with interpolation to handle NaN values
def moving_average(data, window_size):
    # Check if data is one-dimensional and convert it to a two-dimensional array if necessary
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Initialize an empty list to store the moving average results for each dimension
    ma_data_list = []

    # Loop through each dimension (column) in the data
    for i in range(data.shape[1]):
        # Convert the current column to a pandas Series
        series_data = pd.Series(data[:, i])

        # Calculate the moving average using a rolling window
        ma_data = series_data.rolling(window=window_size).mean()

        # Interpolate missing values linearly
        ma_data = ma_data.interpolate(method='linear')

        # Append the moving average data to the list
        ma_data_list.append(ma_data.to_numpy())

    # Combine the moving average data for all dimensions into a single numpy array
    ma_data_combined = np.column_stack(ma_data_list)

    return ma_data_combined


# Normalize data using StandardScaler
def normalize_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data


# Modified feature calculation function to return a flat array of features
def calculate_features_array(segment):
    return np.array([
        np.max(segment, axis=0),  # Max value
        np.min(segment, axis=0),  # Min value
        np.ptp(segment, axis=0),  # Range
        np.mean(segment, axis=0),  # Mean
        np.median(segment, axis=0),  # Median
        np.var(segment, axis=0),  # Variance
        np.std(segment, axis=0),  # Standard Deviation
        segment.shape[0],  # Count
        np.sum(segment, axis=0),  # Sum
        np.mean(np.abs(segment - np.mean(segment, axis=0)), axis=0)  # Absolute Deviation
    ]).flatten()  # Flatten to make it a single array


def remove_outliers_and_interpolate(data, threshold=3):
    # Convert the data to a pandas DataFrame for easier manipulation
    df = pd.DataFrame(data)

    # Calculate the mean and standard deviation
    mean = df.mean()
    std = df.std()

    # Calculate Z-scores
    z_scores = np.abs((df - mean) / std)

    # Mask for all non-outliers
    mask = (z_scores < threshold).all(axis=1)

    # Mark outliers as NaN
    df[~mask] = np.nan

    # Use linear interpolation on the DataFrame to fill in the gaps (NaN values)
    df_interpolated = df.interpolate(method='linear', limit_direction='both')

    # If there are still NaNs at the ends, fill them with the closest non-NaN value
    df_interpolated.fillna(method='bfill', inplace=True)
    df_interpolated.fillna(method='ffill', inplace=True)

    return df_interpolated.to_numpy()


def preProcessing(data, window):
    preData = moving_average(data, window)
    betterData = remove_outliers_and_interpolate(preData, window)
    normalData = normalize_data(betterData)
    return normalData


# Read csv data files & input to arrays
data_Walk_NH = pd.read_csv("Nick_Walk.csv").to_numpy()
data_Walk_TH = pd.read_csv("Tom_Walk.csv").to_numpy()
data_Walk_MT = pd.read_csv("Mira_Walk.csv").to_numpy()
data_Jump_NH = pd.read_csv("Nick_Jump.csv").to_numpy()
data_Jump_TH = pd.read_csv("Tom_Jump.csv").to_numpy()
data_Jump_MT = pd.read_csv("Mira_Jump.csv").to_numpy()

# Map descriptive names to data arrays
data_mapping = {
    'Nick_Walk.csv': data_Walk_NH,
    'Tom_Walk.csv': data_Walk_TH,
    'Mira_Walk.csv': data_Walk_MT,
    'Nick_Jump.csv': data_Jump_NH,
    'Tom_Jump.csv': data_Jump_TH,
    'Mira_Jump.csv': data_Jump_MT
}

for file_name, current_data in data_mapping.items():
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 12), sharex=True)

    # Assume 'Time (s)', 'Linear Acceleration x (m/s^2)', etc., are the correct columns
    axes[0].plot(current_data[:, 0], current_data[:, 1], label='X-axis', color='red')
    axes[0].set_title('Linear Acceleration X-axis vs. Time for ' + file_name)
    axes[0].set_ylabel('Acceleration (m/s^2)')

    axes[1].plot(current_data[:, 0], current_data[:, 2], label='Y-axis', color='green')
    axes[1].set_title('Linear Acceleration Y-axis vs. Time for ' + file_name)
    axes[1].set_ylabel('Acceleration (m/s^2)')

    axes[2].plot(current_data[:, 0], current_data[:, 3], label='Z-axis', color='blue')
    axes[2].set_title('Linear Acceleration Z-axis vs. Time for ' + file_name)
    axes[2].set_ylabel('Acceleration (m/s^2)')
    axes[2].set_xlabel('Time (s)')

    plt.tight_layout()

    plt.figure(figsize=(12, 8))
    plt.hist(current_data[:, 1], bins=50, alpha=0.5, label='X-axis')
    plt.hist(current_data[:, 2], bins=50, alpha=0.5, label='Y-axis')
    plt.hist(current_data[:, 3], bins=50, alpha=0.5, label='Z-axis')
    plt.title('Distribution of Acceleration Values for ' + file_name)
    plt.xlabel('Acceleration (m/s^2)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Divide each signal into 5-second windows
sampling_rate = 100
window_size = 5 * sampling_rate

# Split the data into windows
windows_Walk1 = [data_Walk_NH[i:i + window_size] for i in range(0, len(data_Walk_NH), window_size)]
windows_Walk2 = [data_Walk_TH[i:i + window_size] for i in range(0, len(data_Walk_TH), window_size)]
windows_Walk3 = [data_Walk_MT[i:i + window_size] for i in range(0, len(data_Walk_MT), window_size)]
windows_Jump1 = [data_Jump_NH[i:i + window_size] for i in range(0, len(data_Jump_NH), window_size)]
windows_Jump2 = [data_Jump_TH[i:i + window_size] for i in range(0, len(data_Jump_TH), window_size)]
windows_Jump3 = [data_Jump_MT[i:i + window_size] for i in range(0, len(data_Jump_MT), window_size)]

# Shuffle the segmented data
np.random.shuffle(windows_Walk1)
np.random.shuffle(windows_Walk2)
np.random.shuffle(windows_Walk3)
np.random.shuffle(windows_Jump1)
np.random.shuffle(windows_Jump2)
np.random.shuffle(windows_Jump3)

# Concatenate the shuffled arrays
concatenated_walk = np.concatenate(windows_Walk1 + windows_Walk2 + windows_Walk3)
concatenated_jump = np.concatenate(windows_Jump1 + windows_Jump2 + windows_Jump3)

# Plot samples from the walking and jumping datasets
plot_acceleration(concatenated_walk[:window_size], concatenated_jump[:window_size], 'Walk and Jump Sample')

# Determine the split index
split_index_walk = int(len(concatenated_walk) * 0.9)
split_index_jump = int(len(concatenated_jump) * 0.9)

# Split the walk data into training and testing sets
training_walk = concatenated_walk[:split_index_walk]
testing_walk = concatenated_walk[split_index_walk:]

preTrainWalk = preProcessing(training_walk, window_size)
preTestWalk = preProcessing(testing_walk, window_size)

# Split the jump data into training and testing sets
training_jump = concatenated_jump[:split_index_jump]
testing_jump = concatenated_jump[split_index_jump:]

preTrainJump = preProcessing(training_jump, window_size)
preTestJump = preProcessing(testing_jump, window_size)

# Plot samples from the walking and jumping datasets
plot_acceleration(preTrainWalk[:window_size], preTrainJump[:window_size], 'Walk and Jump Sample - Train')
plot_acceleration(preTestWalk[:window_size], preTestJump[:window_size], 'Walk and Jump Sample - Test')

# Calculate features for all windows in the training set
features_train_walk = np.array([calculate_features_array(window) for window in preTrainWalk])
features_train_jump = np.array([calculate_features_array(window) for window in preTrainJump])

# Stack all features for training and create labels
X_train = np.vstack((features_train_walk, features_train_jump))
y_train = np.array([0] * len(features_train_walk) + [1] * len(features_train_jump))

# Do the same for the test set
features_test_walk = np.array([calculate_features_array(window) for window in preTestWalk])
features_test_jump = np.array([calculate_features_array(window) for window in preTestJump])

X_test = np.vstack((features_test_walk, features_test_jump))
y_test = np.array([0] * len(features_test_walk) + [1] * len(features_test_jump))

# Train the logistic regression model
clf = LogisticRegression(max_iter=10000)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the logistic regression classifier on test set: {accuracy * 100:.2f}%')

recall = recall_score(y_test, y_pred)
print(f'Recall of the logistic regression classifier: {recall * 100:.2f}%')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()

# Predict the probabilities for the test set
y_prob = clf.predict_proba(X_test)

# Calculate the false positive rate and true positive rate for ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1], pos_label=1)  # Assuming the positive class is labeled as 1

# Plotting the ROC curve
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

plt.show()  # Display the ROC curve

# Calculate the AUC
auc = roc_auc_score(y_test, y_prob[:, 1])
print(f'The AUC is: {auc}')

with h5py.File('processed_activity_data.hdf5', 'w') as hdf:
    # Create groups for each member
    member1 = hdf.create_group('Member1')
    member2 = hdf.create_group('Member2')
    member3 = hdf.create_group('Member3')

    # Store the raw data for each member
    member1.create_dataset('Walk', data=data_Walk_NH)
    member1.create_dataset('Jump', data=data_Jump_NH)

    member2.create_dataset('Walk', data=data_Walk_TH)
    member2.create_dataset('Jump', data=data_Jump_TH)

    member3.create_dataset('Walk', data=data_Walk_MT)
    member3.create_dataset('Jump', data=data_Jump_MT)

    # Create the 'dataset' group with 'Train' and 'Test' subgroups
    train_group = hdf.create_group('dataset/Train')
    test_group = hdf.create_group('dataset/Test')

    # Store the training data
    train_group.create_dataset('Walk', data=preTrainWalk)
    train_group.create_dataset('Jump', data=preTrainJump)

    # Store the testing data, ensuring we use the correct names
    test_group.create_dataset('Walk', data=preTestWalk)
    test_group.create_dataset('Jump', data=preTestJump)
