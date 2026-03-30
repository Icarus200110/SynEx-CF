import os

import glob
import torch
import torchvision
import numpy as np
import scanpy as sc
import pandas as pd 
import scprep as scp
import anndata as ad
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import ImageFile, Image
from utils import read_tiff, get_data
from graph_construction import calcADJ
from collections import defaultdict as dfd
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

class ViT_HER2ST(torch.utils.data.Dataset):
    """Some Information about HER2ST"""
    def __init__(self,train=True,fold=0,r=4,flatten=True,ori=False,adj=False,prune='Grid',neighs=4,train_samples=None, spot_batch_size=None):
        super(ViT_HER2ST, self).__init__()
        
        self.cnt_dir = 'data/her2st/data/ST-cnts'
        self.img_dir = 'data/her2st/data/ST-imgs'
        self.pos_dir = 'data/her2st/data/ST-spotfiles'
        self.lbl_dir = 'data/her2st/data/ST-pat/lbl'
        self.r = 224//r
        self.spot_batch_size = spot_batch_size  # spot级别的batch大小

        # gene_list = list(np.load('data/her_hvg.npy',allow_pickle=True))
        gene_list = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
        self.gene_list = gene_list
        names = os.listdir(self.cnt_dir)
        names.sort()
        names = [i[:2] for i in names]
        self.train = train
        self.ori = ori
        self.adj = adj
        
        # 所有可用样本
        samples = names[1:33]
        
        # 指定测试样本
        te_names = [samples[fold]]
        print("te_names:", te_names)  # B1
        
        # 从lbl文件夹中获取所有可用的样本名称
        lbl_files = os.listdir(self.lbl_dir)
        # 提取样本名称，文件格式如 A1_labeled_coordinates.tsv
        all_available_samples = [f.split('_')[0] for f in lbl_files if f.endswith('_labeled_coordinates.tsv')]
        
        # 定义有效的样本集合（根据lbl文件夹中的实际数据存在的样本）
        valid_samples = all_available_samples
        print(f"Available samples from lbl folder: {valid_samples}")
        
        # 如果是训练模式
        if train:
            # 如果指定了训练样本，则使用指定的样本，但只保留有效的样本
            if train_samples:
                # 使用指定的训练样本集合，但只保留有效的样本
                self.names = [sample for sample in train_samples if sample in valid_samples]
                print(f"Filtered to valid samples: {self.names}")
            else:
                # 不指定train_samples时，使用lbl文件夹中的所有样本作为训练样本
                # 排除测试样本te_names
                self.names = [sample for sample in valid_samples if sample not in te_names]
                print(f"Using all valid samples for training: {self.names}")
        else:
            # 测试模式
            if train_samples:
                # 如果指定了测试样本，则使用指定的样本，但只保留有效的样本
                self.names = [sample for sample in train_samples if sample in valid_samples]
                print(f"Using specified test samples: {self.names}")
            else:
                # 否则，使用默认的测试样本
                self.names = te_names
                print(f"Using default test samples: {self.names}")
        
        print("Using samples:", self.names)

        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in self.names}
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in self.names}
        self.label={i:None for i in self.names}
        self.lbl2id={
            'invasive cancer':0, 'breast glands':1, 'immune infiltrate':2, 
            'cancer in situ':3, 'connective tissue':4, 'adipose tissue':5, 'undetermined':-1
        }
        # 为所有样本设置label，确保label永远不会为None
        for i in self.names:
            idx=self.meta_dict[i].index
            if (not train and i in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']) or (train and i in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']):
                # 如果是特定样本，尝试获取真实标签
                try:
                    lbl=self.get_lbl(i)
                    lbl=lbl.loc[idx,:]['label'].values
                    # 将字符串标签转换为数值ID
                    lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
                    self.label[i]=lbl
                except Exception as e:
                    # 如果获取标签失败，使用默认值
                    print(f"Warning: Failed to get label for {i}, using default value: {e}")
                    self.label[i]=torch.full((len(idx),),-1)
            else:
                # 对于其他样本，使用默认值
                self.label[i]=torch.full((len(idx),),-1)
        self.gene_set = list(gene_list)
        # print(self.gene_set, self.meta_dict.items())  # ['HPS6', 'TNC', 'NR1H2', 'NUP93']
        self.exp_dict = {
            i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) 
            for i,m in self.meta_dict.items()
        }
        if self.ori:
            self.ori_dict = {i:m[self.gene_set].values for i,m in self.meta_dict.items()}
            self.counts_dict={}
            for i,m in self.ori_dict.items():
                n_counts=m.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[i]=sf
        self.center_dict = {
            i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) 
            for i,m in self.meta_dict.items()
        }
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}
        self.adj_dict = {
            i:calcADJ(m,neighs,pruneTag=prune)
            for i,m in self.loc_dict.items()
        }
        self.patch_dict=dfd(lambda :None)
        
        # 准备spot级别的索引映射
        self.spot_indices = []
        for name in self.names:
            n_spots = len(self.meta_dict[name])
            for spot_idx in range(n_spots):
                self.spot_indices.append((name, spot_idx))
        
        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))
        self.flatten=flatten
    
    def __getitem__(self, index):
        if self.spot_batch_size is None:
            # 原始模式：返回整个样本
            ID=self.id2name[index]
            im = self.img_dict[ID]
            im = im.permute(1,0,2)
            exps = self.exp_dict[ID]
            centers = self.center_dict[ID]
            loc = self.loc_dict[ID]
            adj = self.adj_dict[ID]
            patches = self.patch_dict[ID]
            positions = torch.LongTensor(loc)
            patch_dim = 3 * self.r * self.r * 4
            label=self.label[ID]
            exps = torch.Tensor(exps)
            
            # 确保patches不是None
            if patches is None:
                n_patches = len(centers)
                if self.flatten:
                    patches = torch.zeros((n_patches,patch_dim))
                else:
                    patches = torch.zeros((n_patches,3,2*self.r,2*self.r))
                for i in range(n_patches):
                    center = centers[i]
                    x, y = center
                    patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                    if self.flatten:
                        patches[i] = patch.flatten()
                    else:
                        patches[i]=patch.permute(2,0,1)
                self.patch_dict[ID]=patches
            
            # 构建返回数据列表，确保所有元素都是张量
            data = [patches, positions, exps]
            
            # 添加邻接矩阵（确保不是None）
            if self.adj:
                data.append(adj)
            
            # 添加ori数据（确保是张量）
            if self.ori:
                try:
                    oris = self.ori_dict[ID]
                    sfs = self.counts_dict[ID]
                    data.append(torch.Tensor(oris))
                    data.append(torch.Tensor(sfs))
                except KeyError as e:
                    # 如果ori_dict或counts_dict中没有对应的ID，跳过ori数据
                    print(f"Warning: Missing ori data for {ID}, skipping ori data")
            
            # 添加中心点坐标
            data.append(torch.Tensor(centers))
            
            # 添加组织类型标签（确保是张量）
            if label is None:
                # 确保label永远不是None
                label = torch.full((exps.shape[0],), -1, dtype=torch.float32)
            data.append(label)
            
            # 确保返回的所有元素都是张量或numpy数组，没有None
            for i, item in enumerate(data):
                if item is None:
                    # 如果有None值，用默认张量替换
                    print(f"Warning: Found None in data at index {i}, replacing with default tensor")
                    # 根据数据类型创建默认张量
                    if i == 0:  # patches
                        data[i] = torch.zeros_like(patches)
                    elif i == 1:  # positions
                        data[i] = torch.zeros_like(positions)
                    elif i == 2:  # exps
                        data[i] = torch.zeros_like(exps)
                    else:
                        # 其他情况，创建适当形状的零张量
                        data[i] = torch.zeros(1)
            
            return data
        else:
            # Spot级别batch模式：返回指定数量的spots
            start_idx = index * self.spot_batch_size
            end_idx = min(start_idx + self.spot_batch_size, len(self.spot_indices))
            
            # 收集当前batch的spots
            batch_patches = []
            batch_positions = []
            batch_exps = []
            batch_centers = []
            batch_labels = []
            
            for spot_idx in range(start_idx, end_idx):
                name, spot_local_idx = self.spot_indices[spot_idx]
                
                # 加载该样本的所有spots数据
                if self.patch_dict[name] is None:
                    im = self.img_dict[name]
                    im = im.permute(1,0,2)
                    centers = self.center_dict[name]
                    n_patches = len(centers)
                    patch_dim = 3 * self.r * self.r * 4
                    
                    if self.flatten:
                        patches = torch.zeros((n_patches,patch_dim))
                    else:
                        patches = torch.zeros((n_patches,3,2*self.r,2*self.r))
                    
                    for i in range(n_patches):
                        center = centers[i]
                        x, y = center
                        patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                        if self.flatten:
                            patches[i] = patch.flatten()
                        else:
                            patches[i] = patch.permute(2,0,1)
                    
                    self.patch_dict[name] = patches
                
                # 获取当前spot的数据，确保所有数据都是有效的
                try:
                    batch_patches.append(self.patch_dict[name][spot_local_idx])
                    batch_positions.append(torch.LongTensor(self.loc_dict[name][spot_local_idx]))
                    batch_exps.append(torch.Tensor(self.exp_dict[name][spot_local_idx]))
                    batch_centers.append(torch.Tensor(self.center_dict[name][spot_local_idx]))
                    
                    # 确保label不是None
                    label = self.label[name][spot_local_idx]
                    if label is None:
                        label = torch.tensor(-1, dtype=torch.float32)
                    batch_labels.append(label)
                except Exception as e:
                    # 如果获取某个spot的数据失败，跳过该spot
                    print(f"Warning: Failed to get data for spot {spot_local_idx} in {name}, skipping: {e}")
                    continue
            
            # 如果没有有效的spots，返回一个空的batch（不应该发生）
            if len(batch_patches) == 0:
                print(f"Warning: No valid spots in batch, returning empty batch")
                # 创建一个最小的有效batch
                batch_patches = torch.zeros((1, 3, 2*self.r, 2*self.r)) if not self.flatten else torch.zeros((1, 3 * self.r * self.r * 4))
                batch_positions = torch.zeros((1, 2), dtype=torch.long)
                batch_exps = torch.zeros((1, len(self.gene_set)))
                batch_centers = torch.zeros((1, 2))
                batch_labels = torch.full((1,), -1, dtype=torch.float32)
            else:
                # 堆叠成batch
                batch_patches = torch.stack(batch_patches)
                batch_positions = torch.stack(batch_positions)
                batch_exps = torch.stack(batch_exps)
                batch_centers = torch.stack(batch_centers)
                batch_labels = torch.stack(batch_labels)
            
            # 为当前batch构建邻接矩阵（简化：使用单位矩阵）
            batch_size = batch_patches.shape[0]
            batch_adj = torch.eye(batch_size)
            
            data = [batch_patches, batch_positions, batch_exps]
            if self.adj:
                data.append(batch_adj)
            # ori数据在spot batch模式下暂时不支持
            data.append(batch_centers)
            data.append(batch_labels)
            
            # 确保返回的所有元素都是张量，没有None
            for i, item in enumerate(data):
                if item is None:
                    print(f"Warning: Found None in spot batch data at index {i}, replacing with default tensor")
                    # 创建适当形状的零张量
                    if i == 0:  # patches
                        data[i] = torch.zeros_like(batch_patches)
                    elif i == 1:  # positions
                        data[i] = torch.zeros_like(batch_positions)
                    elif i == 2:  # exps
                        data[i] = torch.zeros_like(batch_exps)
                    else:
                        data[i] = torch.zeros(1)
            
            return data
        
    def __len__(self):
        if self.spot_batch_size is None:
            return len(self.exp_dict)
        else:
            # Spot级别batch模式：返回batch数量
            return (len(self.spot_indices) + self.spot_batch_size - 1) // self.spot_batch_size

    def get_img(self,name):
        pre = self.img_dir+'/'+name[0]+'/'+name
        fig_name = os.listdir(pre)[0]
        path = pre+'/'+fig_name
        im = Image.open(path)
        return im

    def get_cnt(self,name):
        path = self.cnt_dir+'/'+name+'.tsv'
        df = pd.read_csv(path,sep='\t',index_col=0)

        return df

    def get_pos(self,name):
        path = self.pos_dir+'/'+name+'_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))

        return meta

    def get_lbl(self,name):
        # path = self.pos_dir+'/'+name+'_selection.tsv'
        path = self.lbl_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)
        df.set_index('id',inplace=True)
        return df

class ViT_SKIN(torch.utils.data.Dataset):
    """Some Information about ViT_SKIN"""
    def __init__(self,train=True,r=4,norm=False,fold=0,flatten=True,ori=False,adj=False,prune='NA',neighs=4):
        super(ViT_SKIN, self).__init__()

        self.dir = './data/GSE144240_RAW/'
        self.r = 224//r

        patients = ['P2', 'P5', 'P9', 'P10']
        reps = ['rep1', 'rep2', 'rep3']
        names = []
        for i in patients:
            for j in reps:
                names.append(i+'_ST_'+j)
        gene_list = list(np.load('data/skin_hvg_cut_1000.npy',allow_pickle=True))

        self.ori = ori
        self.adj = adj
        self.norm = norm
        self.train = train
        self.flatten = flatten
        self.gene_list = gene_list
        samples = names
        te_names = [samples[fold]]
        tr_names = list(set(samples)-set(te_names))

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print("te_names:", te_names)
        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in self.names}
        print('Loading metadata...') # 迭代加进去，每次是一个name
        self.meta_dict = {i:self.get_meta(i) for i in self.names}

        self.gene_set = list(gene_list)
        if self.norm:
            self.exp_dict = {
                i:sc.pp.scale(scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)))
                for i,m in self.meta_dict.items()
            }
        else:
            self.exp_dict = {
                i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) 
                for i,m in self.meta_dict.items()
            }
        if self.ori:
            self.ori_dict = {i:m[self.gene_set].values for i,m in self.meta_dict.items()}
            self.counts_dict={}
            for i,m in self.ori_dict.items():
                n_counts=m.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[i]=sf
        self.center_dict = {
            i:np.floor(m[['pixel_x','pixel_y']].values).astype(int)
            for i,m in self.meta_dict.items()
        }
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}
        self.adj_dict = {
            i:calcADJ(m,neighs,pruneTag=prune)
            for i,m in self.loc_dict.items()
        }
        self.patch_dict=dfd(lambda :None)
        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))

    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i,exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp>0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:,j])

    def __getitem__(self, index):
        ID=self.id2name[index]
        im = self.img_dict[ID].permute(1,0,2)

        exps = self.exp_dict[ID]
        if self.ori:
            oris = self.ori_dict[ID]
            sfs = self.counts_dict[ID]
        adj=self.adj_dict[ID]
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        patches = self.patch_dict[ID]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4
        exps = torch.Tensor(exps)
        if patches is None:
            n_patches = len(centers)
            if self.flatten:
                patches = torch.zeros((n_patches,patch_dim))
            else:
                patches = torch.zeros((n_patches,3,2*self.r,2*self.r))

            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i]=patch.permute(2,0,1)
            self.patch_dict[ID]=patches
        data=[patches, positions, exps]
        if self.adj:
            data.append(adj)
        if self.ori:
            data+=[torch.Tensor(oris),torch.Tensor(sfs)]
        data.append(torch.Tensor(centers))
        return data
        
    def __len__(self):
        return len(self.exp_dict)

    def get_img(self,name):
        path = glob.glob(self.dir+'*'+name+'.jpg')[0]
        im = Image.open(path)
        return im

    def get_cnt(self,name):
        path = glob.glob(self.dir+'*'+name+'_stdata.tsv')[0]
        df = pd.read_csv(path,sep='\t',index_col=0)
        return df

    def get_pos(self,name):
        path = glob.glob(self.dir+'*spot*'+name+'.tsv')[0]
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join(pos.set_index('id'),how='inner')

        return meta

    def get_overlap(self,meta_dict,gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set&set(i.columns)
        return list(gene_set)

class ViT_BDS(torch.utils.data.Dataset):
    """Some Information about ViT_BDS"""
    def __init__(self,train=True,r=4,norm=False,fold=0,flatten=True,ori=False,adj=False,prune='NA',neighs=4):
        super(ViT_BDS, self).__init__()

        self.dir = './data/BDS/'
        self.r = 224//r

        names = ['B02_D1', 'B02_E1', 'B03_C2', 'B03_D2', 'B04_D1', 'B04_E1', 'B05_D2', 'B05_E2', 'B06_E1',\
         'B07_C2', 'N02_C1', 'N02_D1', 'N03_C2', 'N03_D2', 'N04_D1', 'N04_E1', 'N05_C2', 'N05_D2', 'N06_D2', 'N07_C1']
        gene_list = list(np.load('data/final_gene_expression.npy',allow_pickle=True))

        self.ori = ori
        self.adj = adj
        self.norm = norm
        self.train = train
        self.flatten = flatten
        self.gene_list = gene_list
        samples = names
        te_names = [samples[fold]]
        tr_names = list(set(samples)-set(te_names))

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print("te_names:", te_names)
        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in self.names}
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in self.names}

        # print(gene_list, self.meta_dict.items()) # ['HPS6', 'TNC', 'NR1H2', 'NUP93']
        self.gene_set = gene_list # 1000
        if self.norm:
            self.exp_dict = {
                i:sc.pp.scale(scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)))
                for i,m in self.meta_dict.items()
            }
        else:
            self.exp_dict = {
                i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) 
                for i,m in self.meta_dict.items()
            }
        if self.ori:
            self.ori_dict = {i:m[self.gene_set].values for i,m in self.meta_dict.items()}
            self.counts_dict={}
            for i,m in self.ori_dict.items():
                n_counts=m.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[i]=sf
        self.center_dict = {
            i:np.floor(m[['img_coord_X','img_coord_Y']].values).astype(int)
            for i,m in self.meta_dict.items()
        }
        self.loc_dict = {i:m[['coord_X','coord_Y']].values for i,m in self.meta_dict.items()}
        self.adj_dict = {
            i:calcADJ(m,neighs,pruneTag=prune)
            for i,m in self.loc_dict.items()
        }
        self.patch_dict=dfd(lambda :None)
        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))

    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i,exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp>0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:,j])

    def __getitem__(self, index):
        ID=self.id2name[index]
        # 根据spot的位置裁剪对应图片
        im = self.img_dict[ID].permute(1,0,2)

        exps = self.exp_dict[ID]
        if self.ori:
            oris = self.ori_dict[ID]
            sfs = self.counts_dict[ID]
        adj=self.adj_dict[ID]
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        patches = self.patch_dict[ID]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4
        exps = torch.Tensor(exps)
        if patches is None:
            n_patches = len(centers)
            if self.flatten:
                patches = torch.zeros((n_patches,patch_dim))
            else:
                patches = torch.zeros((n_patches,3,2*self.r,2*self.r))

            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i]=patch.permute(2,0,1)
            self.patch_dict[ID]=patches
        data=[patches, positions, exps]
        if self.adj:
            data.append(adj)
        if self.ori:
            data+=[torch.Tensor(oris),torch.Tensor(sfs)]
        data.append(torch.Tensor(centers))
        return data
        
    def __len__(self):
        return len(self.exp_dict)

    def get_img(self,name):
        path = glob.glob(self.dir+'*'+name+'.png')[0]
        im = Image.open(path)
        return im

    def get_cnt(self,name): # B02_D1
        # N03_D2__7_3,0.0,0.0,0.0,0.0,4.0,0.0,0.0
        # path = glob.glob(self.dir+'*'+name+'_stdata.tsv')[0]
        path = '/home/zy/libo/Hist2ST/data/BDS/logCPM_counts.txt'
        # df = pd.read_csv(path,sep='\t')

        df = pd.read_csv(path, index_col=0)
        filtered_df = df[df.index.str.contains(name)]
        # print("filtered_df: ",filtered_df.shape) # (555, 46455)s

        return filtered_df

    def get_pos(self,name):
        # path = glob.glob(self.dir+'*spot*'+name+'.tsv')[0]
        path = '/home/zy/libo/Hist2ST/data/BDS/spot_metadata.tsv'
        df = pd.read_csv(path,sep='\t')
        # df = df[df.index.str.contains(name)]
        df = df[df['SampleID'] == name]
        # print("df: ",df.shape) # (555, 46455)s

        x = df['coord_X'].values
        y = df['coord_Y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []

        for i in range(len(x)):
            id.append(str(name)+'__'+str(x[i])+'_'+str(y[i])) 
        df['id'] = id
        # print(df.shape)

        return df

    def get_meta(self,name, gene_list=None):
        # print("meta", name) # B02_D1
        cnt = self.get_cnt(name)

        pos = self.get_pos(name)
        meta = cnt.join(pos.set_index('id'),how='inner')
        # print(name, meta.shape)

        return meta

    def get_overlap(self,meta_dict,gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set&set(i.columns)
        return list(gene_set)
    

