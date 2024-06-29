from transformers import AutoTokenizer, TFAutoModel, AutoModelForSequenceClassification
import torch
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling1D
from scipy.special import softmax
import numpy as np
import gc


class ITE_model:
    def __init__(self):
         self.__load_models()
         self.__batch_size = 32
    
    def __load_models(self):
        #roBERTa models
        sent_MOD = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
        off_MOD = f"cardiffnlp/twitter-roberta-base-offensive"
        gen_MOD = f"roberta-base"
        
        #Load tokenizer and roBERTa models
        self.__tokenizer = AutoTokenizer.from_pretrained(sent_MOD)
        self.__sent_mod = TFAutoModel.from_pretrained(sent_MOD)
        self.__off_class_mod = AutoModelForSequenceClassification.from_pretrained(off_MOD)
        self.__off_class_mod.eval()
        self.__gen_mod = TFAutoModel.from_pretrained(gen_MOD)
                
        #Load ITE and PEACE models
        self.__ITE_mod = tf.keras.models.load_model('models/ITE_model/models/ITE_model')
        self.__PEACE = tf.keras.models.load_model('models/ITE_model/models/PEACE_model')


    def __Compute_ITE_inputs(self,texts,batch_size):
        # Split texts into batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        batch_out_1 = []
        batch_out_2 = []
        for batch in batches:
            encoded_inputs = self.__tokenizer(batch,padding=True,return_tensors='tf')
            # Compute the outputs of the modules as numpy arrays.
            emb_1 = self.__sent_mod(encoded_inputs)[0]
            pool_1 = GlobalAveragePooling1D()(emb_1).numpy()
            batch_out_1.append(pool_1)
      
            emb_2 = self.__gen_mod(encoded_inputs)[0]
            pool_2 = GlobalAveragePooling1D()(emb_2).numpy()
            batch_out_2.append(pool_2)
          
            encoded_inputs = None
            emb_1 = None
            emb_2 = None
            gc.collect() 
          
        outs_1 = np.concatenate(batch_out_1,axis=0)
        outs_2 = np.concatenate(batch_out_2,axis=0)
      
        return outs_1,outs_2
    
    def __Compute_Offense(self,texts,batch_size):
        # Split texts into batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        batch_out_1 = []
        
        with torch.no_grad():
            for batch in batches:
                encoded_inputs = self.__tokenizer(batch,padding=True,return_tensors='pt')
                
                # Compute the outputs of the module as numpy arrays.         
                emb_1 = self.__off_class_mod(**encoded_inputs)[0]
                scores_1 = []
                for i in range(emb_1.shape[0]):
                    scores_1.append(softmax(emb_1[i].detach().numpy()))
                scores_1 = np.array(scores_1)
                batch_out_1.append(scores_1)
              
                encoded_inputs = None
                emb_1 = None
                gc.collect() 
          
        outs_off = np.concatenate(batch_out_1,axis=0)
          
        return outs_off
    
    def __Select_Off(self,off_scores):
        off = []
        for i in range(len(off_scores)):
            #off.append(off_scores[i][1])
            off.append(np.argmax(off_scores[i]))
        return np.array(off)
    
    def __Compute_PEACE_inputs(self,texts,batch_size):
        # Split texts into batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        batch_out_1 = []
        batch_out_2 = []
        
        for batch in batches:
            encoded_inputs = self.__tokenizer(batch,padding=True,return_tensors='tf')
            # Compute the outputs of the modules as numpy arrays.
            emb_1 = self.__sent_mod(encoded_inputs)[0]
            pool_1 = GlobalAveragePooling1D()(emb_1).numpy()
            batch_out_1.append(pool_1)
        
            emb_2 = self.__gen_mod(encoded_inputs)[0]
            pool_2 = GlobalAveragePooling1D()(emb_2).numpy()
            batch_out_2.append(pool_2)
          
            encoded_inputs = None
            emb_1 = None
            emb_2 = None
            gc.collect() 
          
        outs_1 = np.concatenate(batch_out_1,axis=0)
        outs_2 = np.concatenate(batch_out_2,axis=0)
        
        return outs_1,outs_2
    
    

    def ITE(self,texts):
        sent_emb,gen_emb = self.__Compute_ITE_inputs(texts,self.__batch_size)
        conc_emb = np.concatenate([sent_emb,gen_emb],axis=1)
        ites = self.__ITE_mod.predict(conc_emb)
        return np.squeeze(ites)
    
    def Offense(self,texts):
        off_scores = self.__Compute_Offense(texts,self.__batch_size)
        offense = self.__Select_Off(off_scores)
        return offense
    
    def Hate(self,texts):
        sent_emb,gen_emb = self.__Compute_PEACE_inputs(texts,self.__batch_size)
        conc_emb = np.concatenate([sent_emb,gen_emb],axis=1)
        hates = self.__PEACE.predict(conc_emb)
        return np.squeeze(hates)
        
        