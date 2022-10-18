import pandas as pd

from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from Segment.Segment import SegmentData

df = pd.read_csv('/home/george/Development/Personal/Python/Segment/data/processed.csv', encoding='UTF-8')

seg = SegmentData(df)

seg.edit_to_datetime('FirstOrder')

def cluster(sd, cols, n_components=None):
    df = sd.df[cols].copy()

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


    return cluster_models[best_cluster], cluster_data[best_cluster]

fitcols = seg.df.columns
fitcols = [i for i in fitcols if i not in ['CustomerID','FirstOrder']]
    
seg.cluster(fitcols, 3)



out_model, out_data = cluster(seg, fitcols, 3)
pd.Series(out_data).value_counts()