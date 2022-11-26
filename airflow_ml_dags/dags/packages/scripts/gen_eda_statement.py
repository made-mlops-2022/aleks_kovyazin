import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from glob import glob
from PIL import Image
import os

PLT = ['#79a7e3','#79c5e3']
MARK50 = ["<50% diameter narrowing", ">50% diameter narrowing"]

class Ploter:
    def __init__(self, df):
        self.df = df

    def isnull(self):
        plt.figure(figsize=(7,2), dpi=80)
        sns.heatmap(self.df.isnull(),cbar=False,cmap='Blues',yticklabels=False)
        plt.title('Missing value in the dataset Null')
        plt.savefig('../statement/find_null.png', dpi=300, bbox_inches='tight')
    
    def isna(self):
        plt.figure(figsize=(7,2), dpi=80)
        sns.heatmap(self.df.isna(),cbar=False,cmap='Blues',yticklabels=False)
        plt.title('Missing value in the dataset Nan')
        plt.savefig('../statement/find_nan.png', dpi=300, bbox_inches='tight')

    def corr_matrix(self, id=False):
        corr_mat = self.df.corr().round(2)
        f, ax = plt.subplots(figsize=(7,4), dpi=80)
        mask = np.triu(np.ones_like(corr_mat, dtype=np.bool))
        mask = mask[1:,:-1]
        corr = corr_mat.iloc[1:,:-1].copy()
        sns.heatmap(corr,mask=mask,vmin=-0.3,vmax=0.3,center=0, 
                    cmap='Blues',square=False,lw=2,annot=True,cbar=False)
        ax.set_title('Correlation Matrix')
        plt.savefig('../statement/feature_dependences.png', dpi=300, bbox_inches='tight')
    
    def plot1count(self, x,xlabel,palt):
    
        plt.figure(figsize=(20,2))
        sns.countplot(x=x,hue='condition', data=self.df, palette=palt) 
        plt.legend(MARK50,loc='upper right')
        plt.xlabel(xlabel)
        plt.ylabel('Frequency')
        plt.title('Data representation')
        plt.savefig('../statement/representation0.png', dpi=300, bbox_inches='tight')
        
    def plot1count_ordered(self, x,xlabel,order,palt):
        
        plt.figure(figsize=(20,2))
        sns.countplot(x=x,hue='condition',data=self.df,order=order,palette=palt)
        plt.legend(MARK50,loc='upper right')
        plt.xlabel(xlabel)
        plt.ylabel('Frequency')
        plt.title('Data representation')
        plt.savefig('../statement/representation1.png', dpi=300, bbox_inches='tight')

    def plot2count(self, x1,x2,xlabel1,xlabel2,colour,rat,ind1=None,ind2=None):
        

        fig,ax = plt.subplots(1,2,figsize=(20,3),gridspec_kw={'width_ratios':rat})
        sns.countplot(x=x1,hue='condition',data=self.df,order=ind1,palette=colour,ax=ax[0])
        ax[0].legend(MARK50,loc='upper right')
        ax[0].set_xlabel(xlabel1)
        ax[0].set_ylabel('Frequency')
        ax[0].set_title('Data representation')
        plt.savefig('../statement/representation2.png')
        sns.countplot(x=x2,hue='condition', data=self.df,order=ind2,palette=colour,ax=ax[1])
        ax[1].legend(MARK50,loc='best')
        ax[1].set_xlabel(xlabel2)
        ax[1].set_ylabel('Frequency')
        ax[1].set_title('Data representation')
        plt.savefig('../statement/representation3.png', dpi=300, bbox_inches='tight')
        
    def nplot2count(self, lst_name,lst_label,colour,n_plots):
        
        ii=-1;fig,ax = plt.subplots(1,n_plots,figsize=(20,3))
        for i in range(0,n_plots):
            ii+=1;id1=lst_name[ii];id2=lst_label[ii]
            sns.countplot(x=id1,hue='condition',data=self.df,palette=colour,ax=ax[ii])
            ax[ii].legend(MARK50,loc='upper right')
            ax[ii].set_xlabel(id2)
            ax[ii].set_ylabel('Frequency')
            ax[ii].set_title('Data representation')
            plt.savefig(f'../statement/representation{i}_4.png', dpi=300, bbox_inches='tight')

    def pair_grid(self, df):
        g = sns.PairGrid(df,diag_sharey=False,hue='condition',palette='Blues')
        g.fig.set_size_inches(13,13)
        g.map_upper(sns.kdeplot,n_levels=5)
        g.map_diag(sns.kdeplot, lw=2)
        g.map_lower(sns.scatterplot,s=20,edgecolor="b",linewidth=1,alpha=0.6)
        g.add_legend()
        plt.title('Data distribution')
        plt.tight_layout()
        plt.savefig('../statement/distribution.png', dpi=300, bbox_inches='tight')

def main():
    df = pd.read_csv('../data/data.csv')
    path = '../statement'
    if not os.path.exists(path):
        os.mkdir(path)
    draw = Ploter(df)
    draw.isnull()
    draw.isna()
    draw.corr_matrix()
    draw.plot2count('age','sex','Age of Patient','Gender of Patient',PLT,[2,1])
    lst1 = ['cp','exang','thal','ca']
    lst2 = ['Chest Pain Type','Excersised Induced Angina','Thalium Stress Result','Fluorosopy Vessels']
    draw.nplot2count(lst1,lst2,PLT,4)
    lst_blood = ['trestbps','thalach','fbs','chol','condition']
    draw.plot1count('trestbps','trestbps: Resting Blood Pressure (mmHg)',PLT)
    draw.plot1count_ordered('thalach','thalach: Maximum Heart Rate',df['thalach'].value_counts().iloc[:30].index,PLT)
    draw.plot2count('fbs','chol','Fasting Blood Sugar','Serum Cholestoral',PLT,[2,10],None,df['chol'].value_counts().iloc[:40].index)
    numvars_targ = ['age','trestbps','chol','thalach','oldpeak','condition']
    draw.pair_grid(df[numvars_targ])
    img_list_ = glob('../statement/*.png')
    img_list = sorted(img_list_, key=os.path.getmtime)
    img_list = [Image.open(img).convert('RGB') for img in img_list]
    img_list[0].save(r'../statement/report.pdf', save_all=True, append_images=img_list)
    [os.remove(file) for file in img_list_]


if __name__=='__main__':
    main()