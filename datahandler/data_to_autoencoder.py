import numpy as np
import pandas as pd

def features_merger(tracklist, featureslist):
    #same as in data_merger
    old_file = pd.read_csv(tracklist)
    features = pd.read_csv(featureslist)

    to_keep = [
        "session_id",
        "session_position",
        "session_length",
        "track_id_clean",
        "skip_2",
    ]

    new_file = old_file[to_keep]

    new_file.columns = [
        "session_id",
        "session_position",
        "session_length",
        "track_id",
        "skip",
    ]

    merged = mergeLeftInOrder(new_file, features)

    return merged

def session_splitter (df):
    '''
    Splits the input data into a format able to be taken by the autoencoder. 
    encoded=AutoEncoder(InputSize=29, EmbedSize=?, Radius=?)
    for i in range(output):
        encoded.output(i)
    
    Takes input dataframe from features_merger() and checks the session id of every row.
    If it is the same as the previous, group those rows into a single dataframe and remove the first few columns about session id, session length, session number, track id. 
    outputs python list where each element is a df of one listening session. 
    session length determined by output[session_number].shape[0]
    '''

    #prep output, temp dataframe, first session_id
    output=[pd.DataFrame([])]
    prev_session=df.loc[0, "session_id"]
    count=0
    #iterate through the input df
    for i in range(df.shape[0]):
        curr_session=df.loc[i, "session_id"]
        #check if current row's session is the same as the previous. If so, append the useful columns
        if prev_session==curr_session:
            output[count]=output[count].append(df.loc[i, [4 :]])
        
        #if not the same session, make a new df 
        else:
            count+=1
            output+=[pd.DataFrame([])]
            prev_session=curr_session
    return output


## Helper Functions
def mergeLeftInOrder(x, y, on=None):
    x = x.copy()
    x["Order"] = np.arange(len(x))
    z = x.merge(y, how="left", on=on).set_index("Order").loc[np.arange(len(x)), :]
    return z

