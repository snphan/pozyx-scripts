import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets as widgets
import numpy as np
from scipy import interpolate
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
from matplotlib.patches import Rectangle
from sklearn.preprocessing import OneHotEncoder

#MARK: UTIITY FUNCTIONS
def generate_matrix_array_correct(correct_labels, correct_predictions, matrix):
    EATING = 0,0 
    WALK = 1,1
    STATIONARY = 2,2
    MOP = 3,3
    TRANSFER = 4,4
    OPENDISHWASHER = 5,5
    WIPE = 6,6
    OTHER = 7,7
    global x

    if correct_labels == 'EATING':
        matrix[EATING] += correct_predictions
        x = 0
    elif correct_labels == 'WALK':
        matrix[WALK] += correct_predictions
        x = 1
        #matrix[incorrect_labels[label]] = incorrect_num_labels[label]
    elif correct_labels == 'STATIONARY':
        matrix[STATIONARY] += correct_predictions
        x = 2
    elif correct_labels == 'MOP':
        matrix[MOP] += correct_predictions
        x = 3
    elif correct_labels == 'TRANSFER':
        matrix[TRANSFER] += correct_predictions
        x = 4
    elif correct_labels == 'OPENDISHWASHER':
        matrix[OPENDISHWASHER] += correct_predictions
        x = 5
    elif correct_labels == 'WIPE':
        matrix[WIPE] += correct_predictions
        x = 6
    elif correct_labels == 'OTHER':
        x = 7
        matrix[OTHER] += correct_predictions

    print(matrix)


def generate_matrix_array_incorrect(incorrect_labels, incorrect_num_labels, matrix):
    global x
    if incorrect_labels == 'EATING':
        y = 0
        matrix[x,y] += incorrect_num_labels   

    elif incorrect_labels == 'WALK':
        y = 1
        matrix[x,y] += incorrect_num_labels  

    elif incorrect_labels == 'STATIONARY':
        y = 2
        matrix[x,y] += incorrect_num_labels
        
    elif incorrect_labels == 'MOP':
        y = 3
        matrix[x,y] += incorrect_num_labels
        
    elif incorrect_labels == 'TRANSFER':
        y = 4
        matrix[x,y] += incorrect_num_labels
        
    elif incorrect_labels == 'OPENDISHWASHER':
        y = 5
        matrix[x,y] += incorrect_num_labels
        
    elif incorrect_labels == 'WIPE':
        y = 6
        matrix[x,y] += incorrect_num_labels

    elif incorrect_labels == 'OTHER':
        y = 7
        matrix[x,y] += incorrect_num_labels

    # print(matrix)


def list_from_series(data):
    return data.values.tolist()

def what_location(x, y, regions):
    for k, v in regions.items():
        path = mpltPath.Path(v)
        if path.contains_point([x,y]): return k
    return "undefined"

#MARK: PREPROCESSING
def extract_time_labels(fp):
    """
    Expects the file to have lines that read: 
    LABEL_NAME: TIMESTAMP

    Eg. Go to fridge: 16900000.104

    Returns
    -------
    List[{'Timestamp': float, 'Label': str}]
    """
    with open(fp) as f:
        labels = f.readlines()

    label_keys = ["Timestamp", "Label"]
    labels = [label.replace("\n","").split(": ")[::-1] for label in labels]
    labels =  [{label_keys[0]: label[0], label_keys[1]:label[1]} for label in labels]
    return labels

def copy_df_format(df):
    """
    Returns an empty df with the corresponding df
    """
    copy_df = pd.DataFrame(columns=df.columns)
    copy_df.index.name = df.index.name
    return copy_df

#MARK: CLEANING DATA

def remove_periods(df, periods):
    """Deletes the selected periods"""
    for period in periods:
        mask = (df.index > float(period[0])) & (df.index < float(period[1]))
        df = df.drop(df.index[mask])
    return df

def df_iterp1d(df, num_points):

    interp_df = pd.DataFrame()
    interp_df.index.name = df.index.name

    x = df.index.values
    xnew = np.linspace(x[0], x[-1], num=num_points)

    for col in df.columns:
        col_data = df[col].values
        f = interpolate.interp1d(x, col_data)
        new_col_data = f(xnew)
        new_df = pd.DataFrame({'Timestamp': xnew, col: new_col_data}).set_index('Timestamp')
        interp_df = pd.concat([interp_df, new_df], axis=1)

    return interp_df

def interp1d_periods(df, periods, num_points):
    for period in periods:
        mask = (df.index > period[0]) & (df.index < period[1])
        interp_df = df_iterp1d(df[mask], num_points)
        df = df.drop(df.index[mask])
        df = pd.concat([df, interp_df])
    return df

def handle_spikes(df: pd.DataFrame, columns: list[str], jump_thresholds: list[float]):
    """
    Handle the outliers by setting the outlier value to the previous value.

    Parameters
    ----------
    df: target df
    columns: list of columns to apply outlier handling to.
    jump_thresholds: array with the same size as "columns", sets the threshold for each column

    Returns
    -------
    df: processed df
    """
    if len(columns) != len(jump_thresholds): raise ValueError("columns and jump_thresholds do not have the same length")

    for col, threshold in zip(columns, jump_thresholds):
        for ind in range(len(df.index) - 1):
            diff = abs(
                df.iloc[ind+1, list_from_series(df.columns).index(col)] 
                - df.iloc[ind, list_from_series(df.columns).index(col)]
            ) 
            if diff > threshold:
                df.iloc[ind+1, list_from_series(df.columns).index(col)] = df.iloc[ind-10:ind+1, list_from_series(df.columns).index(col)].mean()
    return df

def MAV_cols(df: pd.DataFrame, columns: list[str], n: int):
    for col in columns:
        mav_data = df.loc[:, col].rolling(n).mean()
        df.loc[:, col] = mav_data
    return df

def round_cols(df: pd.DataFrame, columns: list[str], base=1):
    for col in columns:
        rounded = (df.loc[:, col] / base).round() * base 
        df.loc[:, col] = rounded
    return df

def drop_columns_that_contain(df, query) -> pd.DataFrame:
    col_to_drop = df.loc[:, df.columns.str.contains(query)]
    return df.drop(col_to_drop.columns.values, axis=1)

def determine_location(df, regions): # Note that df must contain POS_X and POS_Y
    df["Location"] = df.apply(lambda row: what_location(row['POS_X'], row['POS_Y'], regions), axis=1)

    return df

def one_hot_encode_col(df: pd.DataFrame, column: str, enc: OneHotEncoder=None):
    locations = np.reshape(df[column].values, (-1, 1))
    if not enc:
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(locations)
    data = enc.transform(locations).toarray()
    header = enc.get_feature_names_out([column])
    one_hot_encode_df = pd.DataFrame(data, columns=header)
    return pd.concat([df, one_hot_encode_df], axis=1).drop([column], axis=1)


#MARK: VISUALIZATION

def plot_pozyx_locations_with_timings(data: pd.DataFrame, labels: list[dict], title="Location with Timings", ylabel="Location"):
    """
    Args
    ----

    labels: list of dictionary {'Timestamp': <TIMESTAMP>, 'Label': <LABEL_NAME>} 
    """
    fig = plt.figure(figsize=(20,10))
    ax = plt.subplot()
    ax.scatter(data.index, data.loc[:, 'Location'], s=10, marker="+")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    for label in labels:
        if label["Label"] != "10 sec elapsed":
            ax.axvline(float(label['Timestamp']), color="red", alpha=0.5)
            ax.text(float(label['Timestamp'])+0.5,0,label["Label"], rotation=90, color="red", alpha=0.5, size=18)
        else:
            ax.axvline(float(label['Timestamp']), color="black")
    return ax

def plot_pozyx_data_with_timings(data: pd.DataFrame, columns: list[str], labels: list[dict], title="Data with Timings", ylim=(-1000, 15000), ylabel="Position (mm)"):
    """
    Args
    ----

    labels: list of dictionary {'Timestamp': <TIMESTAMP>, 'Label': <LABEL_NAME>} 
    """
    ax = data.loc[:, columns].plot(figsize=(20,10))
    ax.set_title(title)
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel)
    for label in labels:
        if label["Label"] != "10 sec elapsed":
            ax.axvline(float(label['Timestamp']), color="red", alpha=0.5)
            ax.text(float(label['Timestamp'])+0.5,0,label["Label"], rotation=90, color="red", alpha=0.5, size=18)
        else:
            ax.axvline(float(label['Timestamp']), color="black")
    return ax


def plot_interactive_pozyx_data(data, title, xlim=(0,12000), ylim=(0,12000), ground_truth=[], bg_path="", bg_multiplier=1):
    @widgets.interact(to_index=(0, 2000),window=(1,100))
    def f(to_index=0,window=1):
        fig = plt.figure(figsize=(10,10))
        data_clean = data.rolling(window).mean()
        plt.plot(data_clean.iloc[:to_index, 0], data_clean.iloc[:to_index, 1])
        plt.xlabel("X(mm)")
        plt.ylabel("Y(mm)")
        plt.title(title)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xticks(np.arange(xlim[0], xlim[1], 500), rotation=20)
        if bg_path:
            img = Image.open(bg_path).convert("L")
            img = np.asarray(img)
            plt.imshow(img, extent=[0,1803*bg_multiplier,0,1683*bg_multiplier], cmap='gray', vmin=0, vmax=255)
        for point in ground_truth:
            plt.scatter(point['x'], point['y'], color='red', s=2, marker="x")
            plt.text(point['x'] + 50, point['y'], point['label'], color='red')

def subplot_pozyx_data_with_timings(data: pd.DataFrame, columns: list[str], labels: list[dict], title="Data with Timings", ylim=(-1000, 15000), units="(mm)"):
    """
    Args
    ----

    labels: list of dictionary {'Timestamp': <TIMESTAMP>, 'Label': <LABEL_NAME>} 
    """
    fig, axs = plt.subplots(len(columns), 1, figsize=(20,10))
    for ind, ax in enumerate(axs):
        current_data = data.loc[:, columns[ind]]
        min_data = current_data.dropna().values.min()
        current_data.plot(ax=ax)

        if ind == 0:
            ax.set_title(title)
        data_mean = current_data.dropna().mean()
        ax.set_ylabel(f"{columns[ind]} {units}")
        ax.set_xlim(current_data.index.min() - 0.5, current_data.index.max())

        if "POS" in columns[ind]: ax.set_ylim(data_mean - 80, data_mean + 80)

        for label in labels:
            if label["Label"] != "10 sec elapsed":
                ax.axvline(float(label['Timestamp']), color="red", alpha=0.5)
                if ind == len(columns) - 1:
                    ax.text(float(label['Timestamp'])+0.5,min_data,label["Label"], alpha=0.5, rotation=90, color="red", size=18)
            else:
                ax.axvline(float(label['Timestamp']), color="black")
    return axs


# https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py
def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''

    cf = np.pad(cf, ((0, 1), (0, 1)))
    # Calculate the totals
    cf[cf.shape[0] - 1, cf.shape[1] - 1] = np.sum(cf.flatten())
    for col in range(cf.shape[1] - 1):
        cf[cf.shape[0]-1, col] = np.sum(cf[:, col])

    for row in range(cf.shape[0] - 1):
        cf[row, cf.shape[1]-1] = np.sum(cf[row, :])

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf[:-1, :-1]) / float(np.sum(cf[:-1, :-1]))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            pass
            #Metrics for Binary Confusion Matrices
            # precision = cf[1,1] / sum(cf[:,1])
            # recall    = cf[1,1] / sum(cf[1,:])
            # f1_score  = 2*precision*recall / (precision + recall)
            # stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
            #     accuracy,precision,recall,f1_score)
        else:
            # Multi-Class Stats
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
            for ind, category in enumerate(categories):
                cf_raw = cf[:-1, :-1]
                TP = cf_raw[ind, ind]
                TN = np.sum(cf_raw[0:ind, 0:ind]) + np.sum(cf_raw[ind+1:, ind+1:])
                FP = np.sum(cf_raw[:, ind]) - TP
                FN = np.sum(cf_raw[ind, :]) - TP

                precision = TP/(TP+FP)
                sensitivity = TP/(TP+FN)
                specificity = TN/(FP+TN)
                f1_score = (2*(precision*sensitivity))/(precision + sensitivity)

                stats_text += f"\n{category}: Precision={precision:.2f}, Sensitivity={sensitivity:.2f}, Specificity={specificity:.2f}, F1 Score={f1_score:.2f}"
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False 
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    ax = sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories, vmin=np.min(cf[:-1, :-1].flatten()), vmax=np.max(cf[:-1, :-1].flatten()))
    ax.tick_params(axis='x', which='major', labelbottom = False, bottom=False, top=True, labeltop=True, rotation=45)
    ax.add_patch(Rectangle((0, cf.shape[0]-1), cf.shape[1], 1, edgecolor='gray', lw=4, clip_on=False, fill=False))
    ax.add_patch(Rectangle((cf.shape[0]-1, 0), 1, cf.shape[1], edgecolor='gray', lw=4, clip_on=False, fill=False))

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)

# MATPLOTLIB

def zoom_factory(ax, max_xlim, max_ylim, base_scale = 2.):
    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1/base_scale
            x_scale = scale_factor / 2
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
            x_scale = scale_factor * 2
        else:
            # deal with something that should never happen
            scale_factor = 1
            print(event.button)
        # set new limits
        new_width = (cur_xlim[1] - cur_xlim[0]) * x_scale
        new_height = (cur_ylim[1] - cur_ylim[0]) * x_scale

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        if xdata - new_width * (1 - relx) > max_xlim[0]:
            x_min = xdata - new_width * (1 - relx)
        else:
            x_min = max_xlim[0]
        if xdata + new_width * (relx) < max_xlim[1]:
            x_max = xdata + new_width * (relx)
        else:
            x_max = max_xlim[1]
        if ydata - new_height * (1 - rely) > max_ylim[0]:
            y_min = ydata - new_height * (1 - rely)
        else:
            y_min = max_ylim[0]
        if ydata + new_height * (rely) < max_ylim[1]:
            y_max = ydata + new_height * (rely)
        else:
            y_max = max_ylim[1]

        print(x_min, x_max)
        print(y_min, y_max)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.figure.canvas.draw()

    fig = ax.get_figure() # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect('scroll_event',zoom_fun)

    #return the function
    return zoom_fun