import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from src.utils import setup_logging


def train_elbow_model(df: pd.DataFrame) -> KMeans:
    logger = setup_logging()
    
    try:  
        k = range(3,9)
        K = []
        WCSS = []
        
        for i in k:
            kmodel = KMeans(n_clusters=i,init='k-means++').fit(df[['Age', 'Annual_Income', 'Spending_Score']])
            wcss_score = kmodel.inertia_
            WCSS.append(wcss_score)
            K.append(i)
       
        wss = pd.DataFrame({'cluster': K, 'WSS_Score':WCSS})
        logger.info("Model evaluation completed with WSS scores")
        
        return WCSS     
    
    except Exception as e:
        logger.error("Error during model training and evaluation: %s", str(e))
        raise


def train_silhouette_model(df: pd.DataFrame) -> KMeans:
    logger = setup_logging()
    
    try:  
  
        # Let' train our model on spending_score and annual_income 
        k = range(3,9)
        K = []
        ss = []
        for i in k:
            kmodel = KMeans(n_clusters=i,init='k-means++').fit(df[['Age','Annual_Income','Spending_Score']], )
            ypred = kmodel.labels_
            sil_score = silhouette_score(df[['Age','Annual_Income','Spending_Score']], ypred)
            K.append(i)
            ss.append(sil_score)
        
        # Store the number of clusters and their respective silhouette scores in a dataframe
        Variables3 = pd.DataFrame({'cluster': K, 'Silhouette_Score':ss})
        logger.info("Model evaluation completed with Silhouette Score")
        
        return ss   
    
    except Exception as e:
        logger.error("Error during model training and evaluation: %s", str(e))
        raise