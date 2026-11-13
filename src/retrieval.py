"""Search exported embeddings without downloading encoders or weights."""
import argparse
import json
from pathlib import Path
import numpy as np

class EmbeddingIndex:
    def __init__(self, vectors, labels):
        self.vectors=np.asarray(vectors,dtype=float)
        if self.vectors.ndim!=2 or not len(self.vectors) or len(self.vectors)!=len(labels):
            raise ValueError('Expected one label per nonempty embedding row')
        norms=np.linalg.norm(self.vectors,axis=1,keepdims=True)
        if not np.isfinite(self.vectors).all() or (norms==0).any():raise ValueError('Embeddings must be finite and nonzero')
        self.vectors=self.vectors/norms;self.labels=list(labels)
    def search(self,query,k=5):
        query=np.asarray(query,dtype=float)
        if query.shape!=(self.vectors.shape[1],) or not np.isfinite(query).all() or np.linalg.norm(query)==0:
            raise ValueError('Query dimensions must match and values must be finite and nonzero')
        if k<1:raise ValueError('k must be positive')
        scores=self.vectors@(query/np.linalg.norm(query))
        indices=np.argsort(-scores,kind='stable')[:k]
        return [{'label':self.labels[i],'score':float(scores[i])} for i in indices]
    def save(self,path):
        np.savez_compressed(path,vectors=self.vectors,labels=np.asarray(self.labels,dtype=str))
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:return cls(data['vectors'],data['labels'].tolist())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('index');p.add_argument('--query',required=True,help='JSON embedding vector');p.add_argument('-k',type=int,default=5)
    args=p.parse_args();print(json.dumps(EmbeddingIndex.load(args.index).search(json.loads(args.query),args.k),indent=2))
