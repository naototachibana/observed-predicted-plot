import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class ObservedPredictedPlot :
    def __init__(self, df_obs,df_pred):
        self.df_obs = df_obs
        self.df_pred = df_pred
        self.obs = df_obs.values
        self.pred = df_pred.values
        self.xlabel = df_obs.name
        self.ylabel = df_pred.name
        self.ymin = np.amin(self.pred)
        self.ymax = np.amax(self.pred)
        self.yrange = self.ymax - self.ymin
        self.fig = None
        self.RMSE = None
        self.R2 = None
        self.label_message = None
        self.xlabel = None
        self.ylabel = None
        
    def update_range(self,df_obs, df_pred):
        if (self.ymax < np.amax(df_pred.values)):
            self.ymax = np.amax(df_pred.values)
            
        if (self.ymin > np.amin(df_pred.values)):
            self.ymin = np.amin(df_pred.values)
        self.yrange = self.ymax - self.ymin
    
    def plot (self):
        self.fig = plt.figure(figsize=(8, 8))
        

        plt.plot([self.ymin - self.yrange * 0.01, self.ymax + self.yrange * 0.01], 
                 [self.ymin - self.yrange * 0.01, self.ymax + self.yrange * 0.01])

        plt.xlim(self.ymin - self.yrange * 0.01,
             self.ymax + self.yrange * 0.01)

        plt.ylim(self.ymin - self.yrange * 0.01,
             self.ymax + self.yrange * 0.01)
        
        self.RMSE = np.sqrt(mean_squared_error(self.df_obs, self.df_pred))
        self.R2 = r2_score(self.df_obs, self.df_pred)
        self.label_message = ("RMSE : " + str (round(self.RMSE, 5))
                  +"\nR2      : "+ str(round(self.R2, 5)))
        
        plt.scatter(self.obs, self.pred, label=self.label_message)
        plt.legend(fontsize=12, loc='upper left')
        self.xlabel = self.df_obs.name
        self.ylabel = self.df_pred.name
        plt.xlabel(self.xlabel, fontsize=24)
        plt.ylabel(self.ylabel, fontsize=24)
        plt.title('Observed-Predicted Plot', fontsize=24)
        plt.tick_params(labelsize=16)
        plt.grid()
        self.fig.show()
    
