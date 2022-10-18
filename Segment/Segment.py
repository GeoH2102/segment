import datetime
import re

import numpy as np
import pandas as pd

from scipy import spatial
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from .constants import *

class SegmentData():
    def __init__(self, df):
        self.df = pd.DataFrame(df)
        self._manipulations = {}
        self._cluster_model = None
        self._is_clustered = False
        self._edit_data_mapping = {}
        self._cluster_fields = []

    def shape(self):
        return self.df.shape

    def input_for_edit(self):
        # Need to output a dict to iterate over
        d = {}
        d['Shape'] = self.shape()
        for col in self.df.columns:
            d[col] = [
                self.df[col].dtype,
                self.df[col].nunique(),
            ]
        return d

    def edit_to_number(self, col):
        print(f"Converting {col} to num")
        self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
    
    def edit_to_str(self, col):
        print(f"Converting {col} to str")
        self.df[col] = self.df[col].astype(str)

    def edit_to_datetime(self, col):
        print(f"Converting {col} to dt")
        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')

    def edit_to_cat(self, col):
        print(f"Converting {col} to cat")
        self.df[col] = pd.Categorical(self.df[col].astype(str))

    def get_dtypes(self):
        out = {}
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                out[col] = 'num'
            elif pd.api.types.is_categorical_dtype(self.df[col]):
                out[col] = 'cat'
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                out[col] = 'dt'
            else:
                out[col] = 'str'
        return out

    def manip_num_normalize(self, col):
        working_col = self.df[col]
        mean = working_col.mean()
        std = working_col.std()
        self.df['NORM_' + col] = (self.df[col] - mean) / std
    
    def manip_num_trim(self, col, upper, lower):
        working_col = self.df[col]
        working_col = np.where(working_col >= upper, upper, working_col)
        working_col = np.where(working_col <= lower, lower, working_col)
        self.df['TRIM_' + col] = working_col

    def manip_cat_bins(self, col, bin_dict):
        working_col = self.df[col]
        working_col = working_col.str.replace('\s','_')
        working_col.replace(bin_dict, inplace=True)
        working_col = pd.Categorical(working_col)
        self.df['BIN_' + col] = working_col

    def manip_dt_datediff(self, col, datediff):
        indate = pd.to_datetime(datediff)
        working_col = self.df[col]
        working_col = (working_col - indate).dt.days
        self.df['DATEDIFF_' + col] = working_col

    def manip_str_embeddings(self, col):
        embeddings_dict = {}
        with open('models/glove.6B.50d.txt', 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector

        def embedding(row):
            if not isinstance(row, str):
                row = ''
            txt = row.lower()
            txt = re.sub('\W', ' ', txt)
            txt_array = txt.split()
            embedding_list = []
            for word in txt_array:
                embedding_list.append(embeddings_dict.get(word, np.zeros(50)))
            embedding_array = np.array([np.array(xi) for xi in embedding_list])
            average_array = np.mean(embedding_array, axis=0)
            return average_array
        
        working_col = self.df[col]
        final_embeddings = working_col.apply(embedding)
        self.df['EMB_'+col] = final_embeddings

        return

    def generate_manip_options(self):
        dtypes = self.get_dtypes()
        for col in self.df.columns:
            if dtypes[col] == 'num':
                self._manipulations[col] = {
                    'normalize': """
                        <button class="btn btn-secondary" type="submit" name="btn_manip_normalize" value="Go">Go</button>
                    """,
                    'trim': """
                        <label for="manip_trim_lower">Lower: </label>
                        <input type="text" id="manip_trim_lower" name="manip_trim_lower" placeholder="Enter lower value"><br>
                        <label for="manip_trim_upper">Upper: </label>
                        <input type="text" id="manip_trim_upper" name="manip_trim_upper" placeholder="Enter upper value"><br><br>
                        <button class="btn btn-secondary" type="submit" name="btn_manip_trim" value="Go">Go</button>
                    """
                }
            elif dtypes[col] == 'cat':
                working_col = self.df[col]
                categories = working_col.cat.categories.tolist()
                outstr = ""
                for cat in categories:
                    norm_cat = re.sub('\s','_',cat)
                    outstr += f"""
                        <label for="manip_bin_{norm_cat}">{cat}: </label>
                        <input type="text" id="manip_bin_{norm_cat}" name="manip_bin_{norm_cat}" value="{cat}"><br>
                    """
                outstr = re.sub('<br><br>$','',outstr)
                outstr += '<button class="btn btn-secondary" type="submit" name="btn_manip_bin" value="Go">Go</button>'
                self._manipulations[col] = {'bin': outstr}
            elif dtypes[col] == 'dt':
                self._manipulations[col] = {
                    'datediff': """
                        <label for="manip_datediff">Date From: </label>
                        <input type="date" id="manip_datediff" name="manip_datediff"><br>
                        <button class="btn btn-secondary" type="submit" name="btn_manip_datediff" value="Go">Go</button>
                    """
                }
            elif dtypes[col] == 'str':
                self._manipulations[col] = {
                    'embedding': """
                        <button class="btn btn-secondary" type="submit" name="btn_manip_emb" value="Go">Go</button>
                    """
                }

    def cluster(self, cols, n_components=None):
        if "cluster" in self.df.columns:
            self.df.drop("cluster", axis=1, inplace=True)
            
        df = self.df[cols].copy()

        for col in cols:
            if 'EMB_' in col:
                s = df[col]
                emb_df = pd.DataFrame(s.dropna().values.tolist(), index=s.dropna().index)
                emb_df.columns = [f'{col}_{i}' for i in emb_df.columns]
                df = df.join(emb_df)
                df.drop(col, axis=1, inplace=True)

            elif pd.api.types.is_categorical_dtype(df[col]):
                df.loc[:,col] = df.loc[:,col].cat.codes

        df = df.dropna()
        if n_components:
            tsne = TSNE(n_components=n_components, perplexity=30)
            cluster_in = tsne.fit_transform(df)
        else:
            cluster_in = df.to_numpy()

        sil_scores = {}
        cluster_models = {}
        cluster_data = {}
        for i in range(3,15):
            cluster = KMeans(n_clusters=i)
            cluster.fit(cluster_in)
            cluster_out = cluster.predict(cluster_in)
            sil_avg = silhouette_score(cluster_in, cluster_out)
            print(f"{i}: {sil_avg}")
            sil_scores[i] = sil_avg
            cluster_models[i] = cluster
            cluster_data[i] = cluster_out

        max_score = max(sil_scores.values())
        best_cluster = next((k for k,v in sil_scores.items() if v == max_score), None)
        df['cluster'] = cluster_data[best_cluster]
        df['cluster'] = pd.Categorical(df['cluster'])

        print(f"Silhouette Scores: {sil_scores}")
        print(f"Best number of clusters: {best_cluster}")

        self._is_clustered = True
        self._cluster_model = cluster_models[best_cluster]
        self.df = self.df.join(df.loc[:,'cluster'])

        return