import os
import pickle
import random

import torch
import torch.utils.data
from PIL import Image
import xml.etree.ElementTree as ET

from wetectron.structures.bounding_box import BoxList
from wetectron.structures.boxlist_ops import remove_small_boxes
from .coco import unique_boxes
import pandas as pd
from ast import literal_eval
from tqdm import tqdm
import json
def get_coco_labels(path_to_folder='/afs/cs.pitt.edu/usr0/arr159/erhan_code/t/t'):
    with open(f'{path_to_folder}/COCO_labels.txt') as f:
        coco_labels=list([line.split('\n')[0].replace(' ', '') for line in f if line.split('\n')[0]!=''])
    return ['__background__ ', *coco_labels]

class RedCapsDataset(torch.utils.data.Dataset):

    CLASSES = tuple(get_coco_labels())

    def __init__(self, data_dir, use_difficult=False, transforms=None, proposal_file=None, em_path=None):
        self.root = data_dir
        self.transforms = transforms

        self._imgpath = os.path.join(self.root, "images", "%s.jpg")
        self._imgsetpath = "/afs/cs.pitt.edu/usr0/arr159/standard_multimodal_analysis/multimodal_analysis/create_data_subsets/test_redcaps_coco_subset.csv"
        self.df = pd.read_csv(self._imgsetpath).sort_values('img_id')
        self.df = self.df.reset_index()
        cls = RedCapsDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.categories = dict(zip(range(len(cls)), cls))
        
        # Include proposals from a file
        self.proposal_df = pd.read_csv(proposal_file).sort_values('img_id')
        self.proposal_df=self.proposal_df[self.proposal_df['img_id'].isin(self.df['img_id'].values)]
        self.proposal_df=self.proposal_df[self.proposal_df['proposals'].apply(lambda x: sum([1 for i in x if i == '[']) - 1)>10] #must have more than one box
        self.df=self.df[self.df['img_id'].isin(self.proposal_df['img_id'].values)]
        self.df = self.df.reset_index()
        self.proposal_df = self.proposal_df.reset_index()
        assert (self.df['img_id'] == self.proposal_df['img_id']).all()
        self.img_infos=[literal_eval(img_info) for img_info in self.proposal_df['img_info'].values]
        # for i, row in tqdm(self.df.iterrows()):
        #     img_id = row['img_id']
        #     w, h = Image.open(self._imgpath % img_id).convert("RGB").size
        #     self.img_infos.append({
        #         'width': w,
        #         'height': h
        #     })
        # print("Finished preprocessing redcaps")
        self.proposal_df['img_info']=self.img_infos
        # self.proposal_df.to_csv('/archive1/arr159/redcaps/redcaps_proposal_df_w_img_info.csv')
        self.top_k = -1
        if em_path is not None:
            em_df = pd.read_csv(em_path).sort_values('img_id')
            em_df = em_df[em_df['img_id'].isin(self.proposal_df['img_id'].values)]
            em_df = em_df.reset_index()
            assert (em_df['img_id'].values == self.df['img_id'].values).all()
            self.df['em'] = em_df['em'].values
            del em_df
        self.indicies = self.df[self.df.apply(self.filter_df, axis=1)].index.values
        print(f"Filtered out: {(len(self.df)-len(self.indicies))/len(self.df)}")
    def get_labels(self):
        labels=[]
        for i in tqdm(self.indicies):
            labels.append(literal_eval(self.df.iloc[i]['em']))

        return labels
    def sample_for_scale_exp(self, threshold):
        original_len = len(self.indicies)
        self.indicies = random.sample(list(self.indicies), k=int(original_len*threshold))
        print(f"SCALE EXPERIMENT: {original_len} sampled down to --> {len(self)}")
    def filter_df(self, row):
        annot = self._preprocess_annotation(row)
        return len(annot['labels']) > 0
    def __getitem__(self, index):
        index = self.indicies[index]
        img_id = self.df['img_id'].iloc[index]
        img = Image.open(self._imgpath % img_id).convert("RGB")

        try:
            target = self.get_groundtruth(index)
            target = target.clip_to_image(remove_empty=True)
        except Exception as e:
            print(e)
            breakpoint()

        rois = literal_eval(self.proposal_df['proposals'].iloc[index])
        # scores = self.proposals['scores'][roi_idx]
        # assert rois.shape[0] == scores.shape[0]
        # remove duplicate, clip, remove small boxes, and take top k
        rois = BoxList(torch.tensor(rois), (1,1), mode="xyxy").resize(img.size).bbox.type(torch.int)
        rois = BoxList(rois, img.size, mode="xywh").convert('xyxy').bbox.type(torch.int).numpy()
        keep = unique_boxes(rois)
        rois = rois[keep, :]
        # scores = scores[keep]
        rois = BoxList(torch.tensor(rois), img.size, mode="xyxy") # already scaled proposals
        rois = rois.clip_to_image(remove_empty=True)
        # TODO: deal with scores
        rois = remove_small_boxes(boxlist=rois, min_size=2)
        if self.top_k > 0:
            rois = rois[[range(self.top_k)]]
            # scores = scores[:self.top_k]

        if self.transforms is not None:
            img, target, rois = self.transforms(img, target, rois)

        return img, target, rois, index

    def __len__(self):
        return len(self.indicies)

    def get_img_info(self, index):
        index = self.indicies[index]
        img_info=self.img_infos[index]
        img_id = self.df['img_id'].iloc[index]
        img_info['file_name']='images/%s.jpg' % img_id
        return img_info

    def get_groundtruth(self, index):
        row = self.df.iloc[index]
        anno = self._preprocess_annotation(row)
        target = BoxList(anno["boxes"], tuple(self.proposal_df.iloc[index]['img_info'].values()), mode="xyxy")

        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])
        return target

    def _preprocess_annotation(self, row):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1
        if row['em'] != "set()":
            for i,obj in enumerate(literal_eval(row['em'])):
                difficult = 0
                name = obj.replace(' ', '')
                if name not in self.class_to_ind:
                    continue
                # Make pixel indexes 0-based
                # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
                box = [
                    10+i*50,10+i*50,100+i*30,100+i*30
                ]
                bndbox = tuple(
                    map(lambda x: x - TO_REMOVE, list(map(int, box)))
                )

                boxes.append(bndbox)

                gt_classes.append(self.class_to_ind[name])
                difficult_boxes.append(difficult)
        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
        }
        return res

    def map_class_id_to_class_name(self, class_id):
        return RedCapsDataset.CLASSES[class_id]
