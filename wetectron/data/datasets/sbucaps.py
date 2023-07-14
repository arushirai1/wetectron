import os
import pickle
from pathlib import Path
import torch
import torch.utils.data
from PIL import Image
import xml.etree.ElementTree as ET
import random 
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

class SBUCapsDataset(torch.utils.data.Dataset):

    CLASSES = tuple(get_coco_labels())

    def __init__(self, data_dir, use_difficult=False, transforms=None, proposal_file=None, em_path=None, ood_experiments=False, scale_exp=False, weak_det_exp_sample_size=None, voc_classes_only=False):
        self.root = data_dir
        self.transforms = transforms

        self._imgpath = os.path.join(self.root, "images", "%s.jpg")

        if ood_experiments:
            self._imgsetpath = "/afs/cs.pitt.edu/usr0/arr159/standard_multimodal_analysis/multimodal_analysis/create_data_subsets/OODEMNLP23/id_test.csv"#sbucaps_cross_category_splits/id_test_df.csv"#test_sbucaps_coco_subset.csv"
        elif scale_exp:
            self._imgsetpath = "/afs/cs.pitt.edu/usr0/arr159/standard_multimodal_analysis/multimodal_analysis/create_data_subsets/sbucaps_scale_100K_subset_scale.csv"
        else:
            self._imgsetpath = "/afs/cs.pitt.edu/usr0/arr159/standard_multimodal_analysis/multimodal_analysis/create_data_subsets/test_sbucaps_coco_subset.csv"
        self.df = pd.read_csv(self._imgsetpath).sort_values('img_id')
        self.df = self.df.reset_index()
        cls = SBUCapsDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        if voc_classes_only or weak_det_exp_sample_size is not None:
            # restrict to only VOC classes
            from wetectron.utils.visualize import coco_category_names_pascal_index
            self.class_subset = coco_category_names_pascal_index
        else:
            self.class_subset = None
        self.categories = dict(zip(range(len(cls)), cls))
        
        # Include proposals from a file
        self.proposal_df = pd.read_csv(proposal_file).sort_values('img_id')
        if 'img_info' not in self.proposal_df.columns:
            # load img_infos
            if Path("/afs/cs.pitt.edu/usr0/arr159/wetectron_epiphany/wetectron/datasets/sbucaps_copy_archive2/aspect_ratio_df_sbucaps_coco.csv").exists():
                img_info_df=pd.read_csv("/afs/cs.pitt.edu/usr0/arr159/wetectron_epiphany/wetectron/datasets/sbucaps_copy_archive2/aspect_ratio_df_sbucaps_coco.csv")#'outputs3/aspect_ratio_df.csv')
            elif Path("/afs/cs.pitt.edu/usr0/arr159/wetectron_quixote/datasets/sbucaps_copy_archive2/aspect_ratio_df_sbucaps_coco.csv").exists():
                img_info_df=pd.read_csv("/afs/cs.pitt.edu/usr0/arr159/wetectron_quixote/datasets/sbucaps_copy_archive2/aspect_ratio_df_sbucaps_coco.csv")#'outputs3/aspect_ratio_df.csv')
            else:
                img_info_df = pd.read_csv('/archive1/arr159/sbucaps/aspect_ratio_df_sbucaps_coco.csv')
            assert len(img_info_df) == len(self.proposal_df)
            self.proposal_df=self.proposal_df.merge(img_info_df, on="img_id")
        self.proposal_df=self.proposal_df[self.proposal_df['img_id'].isin(self.df['img_id'].values)]
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

        print("Finished preprocessing sbucaps")
        self.proposal_df['img_info']=self.img_infos
        self.top_k = -1
        self.balanced_mapping=None
        use_balanced_mapping=False
        if em_path is not None and 'unfiltered' not in em_path:
            em_df = pd.read_csv(em_path).sort_values('img_id')
            assert (em_df['img_id'].values == self.df['img_id'].values).all()
            self.df['em'] = em_df['em'].values
            if use_balanced_mapping:
                if 'sbucaps_sbucaps1e7_classification_mlp_meanBCE_ems' in em_path:
                    with open("/afs/cs.pitt.edu/usr0/arr159/wetectron_quixote/datasets/sbucaps/object_specific_positive_samples_sbucaps_filtered_set.json", 'r') as f:
                        self.balanced_mapping = json.load(f)
                elif 'sbucaps_clip_based_ems' in em_path:
                    with open("/afs/cs.pitt.edu/usr0/arr159/wetectron/datasets/sbucaps/object_specific_positive_samples_sbucaps_clip_filtered.json", 'r') as f:
                        self.balanced_mapping = json.load(f)
                elif 'redcaps' in em_path:
                    print("dataset am filter redcaps")
                    with open("/afs/cs.pitt.edu/usr0/arr159/wetectron/datasets/sbucaps/object_specific_positive_samples_sbucaps_redcaps_am_filtered.json", 'r') as f:
                        self.balanced_mapping = json.load(f)
                elif 'ground_truth' in em_path:
                    with open("/afs/cs.pitt.edu/usr0/arr159/wetectron/datasets/sbucaps/object_specific_positive_samples_ground_truth_filtered.json", 'r') as f:
                        self.balanced_mapping = json.load(f)
                elif 'ID' not in em_path and 'original' in em_path:
                    with open("/afs/cs.pitt.edu/usr0/arr159/wetectron_quixote/datasets/sbucaps/object_specific_positive_samples_sbucaps_unfiltered.json", 'r') as f:
                        self.balanced_mapping = json.load(f)
                elif 'ID' in em_path or 'OOD' in em_path:
                    with open("/afs/cs.pitt.edu/usr0/arr159/wetectron_quixote/datasets/sbucaps/object_specific_positive_samples_sbucaps_ID_unfiltered.json", 'r') as f:
                        self.balanced_mapping = json.load(f)
                else:
                    print("Balanced mapping: default - filtered")
                    with open("/afs/cs.pitt.edu/usr0/arr159/wetectron/datasets/sbucaps/object_specific_positive_samples_sbucaps_filtered_set.json", 'r') as f:
                        self.balanced_mapping = json.load(f)
            del em_df
        self.indicies = self.df[self.df.apply(self.filter_df, axis=1)].index.values
        print(f"Filtered out: {(len(self.df)-len(self.indicies))/len(self.df)} of {len(self.df)}")

        if self.balanced_mapping is not None:
            img_ids = self.df[self.df.apply(self.filter_df, axis=1)]['img_id'].values
            all_ems = self.df[self.df.apply(self.filter_df, axis=1)]['em'].values
            df_indicies = self.df[self.df.apply(self.filter_df, axis=1)].index.values
            img_to_df_index_lookup = dict(zip(img_ids, df_indicies))
            df_idx_to_em_lookup = dict(zip(df_indicies, all_ems))
            self.indicies=[]
            print("Starting balancing")
            for class_label in self.balanced_mapping:
                for img_id in self.balanced_mapping[class_label]:
                    if img_id in img_to_df_index_lookup:
                        self.indicies.append(img_to_df_index_lookup[img_id])
            class_counts_for_new_list = []
            for idx in self.indicies:
                class_counts_for_new_list.extend(list(literal_eval(df_idx_to_em_lookup[idx])))
            from collections import Counter
            print("Finished balancing -- new dataset size:", len(self.indicies))
            print("Top ten class counts:", Counter(class_counts_for_new_list).most_common(80))
        if weak_det_exp_sample_size is not None:
            self.sample(weak_det_exp_sample_size)
        #self.proposal_df.to_csv('sbucaps')
    def sample(self, sample_size):
        original_len = len(self.indicies)
        self.indicies = random.sample(list(self.indicies), k=sample_size)
        print(f"SCALE EXPERIMENT: {original_len} sampled down to --> {len(self)}")
    def get_labels(self):
        labels=[]
        for i in tqdm(self.indicies):
            labels.append(literal_eval(self.df.iloc[i]['em']))

        return labels
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
                if self.class_subset is not None and name not in self.class_subset:
                    # for weak det experiments
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
        return SBUCapsDataset.CLASSES[class_id]
