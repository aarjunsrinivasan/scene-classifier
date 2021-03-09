import os
import pickle
import numpy as np
import pandas as pd
import collections
from scipy.spatial.distance import cosine
import pandas as pd
import matplotlib.pyplot as plt



def simembed(embeddings,num_boundaries):
    prv_embd = embeddings[0]
    embd_list = []
    for ii in range(1,num_boundaries+1):
        curr_embd = embeddings[ii]
        cossim = 1 - cosine(curr_embd,prv_embd)
        embd_list.append(cossim)
        prv_embd = curr_embd
        
    return embd_list
        
def makesimfeature(data):
    # Define Embeddings
    scene_boundary_truth = data['scene_transition_boundary_ground_truth'].numpy().astype(int)
    place_embedding = data['place'].numpy()
    cast_embedding = data['cast'].numpy()
    action_embedding = data['action'].numpy()
    audio_embedding = data['audio'].numpy()

    #Create embedding cosine similarities
    num_boundaries = scene_boundary_truth.shape[0]
    place_embd_dp_list = simembed(place_embedding, num_boundaries)
    cast_embd_dp_list = simembed(cast_embedding, num_boundaries)
    action_embd_dp_list = simembed(action_embedding, num_boundaries)
    audio_embd_dp_list = simembed(audio_embedding, num_boundaries)
    
    df = pd.DataFrame(
        {'place_dp' : place_embd_dp_list,
         'cast_dp' : cast_embd_dp_list,
         'action_dp' : action_embd_dp_list,
         'audio_dp' : audio_embd_dp_list,
         'boundary_truth' : scene_boundary_truth,
        })
    return df
def makesimfeaturedf(data_dir):
    '''
    makes the dot product features df of data_dir
    '''
    
    df_list= []
    for file in os.listdir(data_dir):
        if file.endswith('.pkl'):
            with open(data_dir+file, 'rb') as f:
                data = pickle.load(f)
            df = makesimfeature(data)
            df_list.append(df)
    df_all = pd.concat(df_list)
    df_all.reset_index(drop=True, inplace=True)
    return df_all

def count_ones(data):
    s = 0
    for i in data:
        if i==1:
            s+=1
    return[s,len(data)-s]

def imbal(Y,title): 
    labels = ['ones','zeros']
    plt.xticks(range(2), labels)
    plt.xlabel('Class')
    plt.ylabel('No of samples')
    plt.title(title)
    plt.bar(range(2), count_ones(Y)) 
    plt.show()

def generate_predictions_dir_LR(model,input_dir,output_dir):
    #best_model should be LogisticRegression
    for file in os.listdir(input_dir):
        if file.endswith('.pkl'):
            with open(input_dir+file, 'rb') as f:
                data = pickle.load(f)
            df = makesimfeature(data)
            df['cast_dp'] = df['cast_dp'].fillna(0)
            df['place_dp'] = df['place_dp'].fillna(0)
            df['action_dp'] = df['action_dp'].fillna(0)
            df['audio_dp'] = df['audio_dp'].fillna(0)
            X = df[['place_dp','cast_dp','action_dp','audio_dp']]
            predictions = model.predict_proba(X)[:,1]
            data_to_pkl={}
            data_to_pkl['scene_transition_boundary_ground_truth'] = \
                data['scene_transition_boundary_ground_truth'].numpy()
            data_to_pkl['scene_transition_boundary_prediction'] = \
                predictions
            data_to_pkl['shot_end_frame'] = data['shot_end_frame']
            data_to_pkl['imdb_id'] = file[:-4]

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(output_dir+file, 'wb') as f:
                pickle.dump(data_to_pkl,f)