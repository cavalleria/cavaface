from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import numpy.matlib
import cPickle
import sklearn
import cv2
import sys
import glob
from skimage import transform as trans
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc

import os
import numpy as np
from utils.io import _read_image, _write_bin, _load_bin, _write_result_to_file, \
        _cal_similarity, _cal_self_similarity, _rank_accuracy, _cal_similarity_gpu, \
        _load_json_result_file, _write_result_to_file
import time
import json

class EvalIJBC(object):
    def __init__(self, args):
        self._args = args
        self._algo = "retina"
        self._media_list = "../data/IJBC/meta/ijbc_face_tid_mid.txt"
        self._pair_list = "../data/IJBC/meta/ijbc_template_pair_label.txt"
        self._emb_outdir_root = ''
        self._outdir_root = ''
        self._image_path = "../data/IJBC/ijbc_112x112/"
        self._image_list_path = "../data/IJBC/meta/ijbc_name_5pts_score_s.sort.txt"
        self._ijbc_1N_probe_list = "../data/IJBC/meta/1N.probe"
        self._ijbc_1N_gallery_list = "../data/IJBC/meta/1N.gallery"
        self._results = {}
        self.image_size = (3,112,112)

        self._face_probe = None
        self._face_gallery = None
        self._save_badcase = True if args.save_badcase else False
        self._save_badcase_filename = 'ijbc.badcase'

        self._get_output_dir()
        self._load_faces_lists()

    def _get_output_dir(self):
        # Get and clear face scrub output dir.
        epoch = int(self._args.model_path.strip().split(',')[1])
        self._outdir_root = os.path.join(
                '/'.join(self._args.model_path.strip().split('/')[:-1]),
                "ijbc_%d"%(epoch))
        self._emb_outdir_root = os.path.join(self._outdir_root, "embedding")
        os.system("mkdir -p %s"%(self._emb_outdir_root))
        self._face_result_filename = 'ijbc.%d.result'%(epoch)

    def _load_faces_lists(self):
        self._face_probe = open(self._ijbc_1N_probe_list).readlines()
        self._face_gallery = open(self._ijbc_1N_gallery_list).readlines()

    def read_template_media_list(self, path):
        ijb_meta = np.loadtxt(path, dtype=str)
        templates = ijb_meta[:,1].astype(np.int)
        medias = ijb_meta[:,2].astype(np.int)
        return templates, medias

    def read_template_pair_list(self, path):
        pairs = np.loadtxt(path, dtype=int)
        t1 = pairs[:,0].astype(np.int)
        t2 = pairs[:,1].astype(np.int)
        label = pairs[:,2].astype(np.int)
        return t1, t2, label

    def read_image_feature(self, path):
        #with open(path, 'rb') as fid:
        #    img_feats = cPickle.load(fid)
        img_feats = np.load(path)
        return img_feats

    def read_score(self, path):
        #with open(path, 'rb') as fid:
        #    faceness_scores = cPickle.load(fid)
        faceness_scores = np.load(path)
        return faceness_scores

    def image2template_feature(self, img_feats = None, templates = None, medias = None):
        # ==========================================================
        # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
        # 2. compute media feature.
        # 3. compute template feature.
        # ==========================================================
        unique_templates = np.unique(templates)
        template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

        for count_template, uqt in enumerate(unique_templates):
            (ind_t,) = np.where(templates == uqt)
            face_norm_feats = img_feats[ind_t]
            face_medias = medias[ind_t]
            unique_medias, unique_media_counts = np.unique(face_medias, return_counts=True)
            media_norm_feats = []
            for u,ct in zip(unique_medias, unique_media_counts):
                (ind_m,) = np.where(face_medias == u)
                if ct == 1:
                    media_norm_feats += [face_norm_feats[ind_m]]
                else: # image features from the same video will be aggregated into one feature
                    media_norm_feats += [np.mean(face_norm_feats[ind_m], 0, keepdims=True)]
            media_norm_feats = np.array(media_norm_feats)
            # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
            template_feats[count_template] = np.sum(media_norm_feats, 0)
            if count_template % 2000 == 0:
                print('Finish Calculating {} template features.'.format(count_template))
        template_norm_feats = template_feats / np.sqrt(np.sum(template_feats ** 2, -1, keepdims=True) + 0.000001)
        return template_norm_feats, unique_templates

    def verification(self, template_norm_feats = None, unique_templates = None, p1 = None, p2 = None):
        # ==========================================================
        #         Compute set-to-set Similarity Score.
        # ==========================================================
        template2id = np.zeros((max(unique_templates)+1,1),dtype=int)
        for count_template, uqt in enumerate(unique_templates):
            template2id[uqt] = count_template

        score = np.zeros((len(p1),))   # save cosine distance between pairs

        total_pairs = np.array(range(len(p1)))
        batchsize = 100000 # small batchsize instead of all pairs in one batch due to the memory limiation
        sublists = [total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)]
        total_sublists = len(sublists)
        for c, s in enumerate(sublists):
            feat1 = template_norm_feats[template2id[p1[s]]]
            feat2 = template_norm_feats[template2id[p2[s]]]
            similarity_score = np.sum(feat1 * feat2, -1)
            score[s] = similarity_score.flatten()
            if c % 10 == 0:
                print('Finish {}/{} pairs.'.format(c, total_sublists))
        return score

    def _infer_and_write(self, imgs_list, net):
        return net.infer_embedding(imgs_list, read_func = _read_image, write_func = _write_bin, is_flip = self._args.flip)
        #
    #
    def _extract_embedding_base(self, file_list, net, emb_outdir, img_root):
        num = len(open(file_list,'r').readlines())
        succ = 0
        imgs_list = []
        for idx, line in enumerate(open(file_list, 'r')):
            name_lmk_score = line.strip().split(' ')
            img_name = os.path.join(img_root, name_lmk_score[0])
            image_path = name_lmk_score[0].strip().replace('.jpg', '.png')
            out_path = os.path.join(emb_outdir, image_path+".bin")
            item = (os.path.join(img_root,image_path), out_path)
            imgs_list.append(item)

        ret = self._infer_and_write(imgs_list, net)
        print("Finish getting embedding for faces, total: %d, success: %d" %(num, succ))
        return ret
    #

    def extract_embedding(self, net):
        # Get embedding of faces.
        print("Begin to extract embedding of faces...")
        return self._extract_embedding_base(self._image_list_path,
                    net,
                    self._emb_outdir_root,
                    self._image_path)
    #
    def _read_templates(self, file_list):
        template = []
        num = len(file_list)
        class_ids = []
        for idx, line in enumerate(file_list):
            if 0 == idx%1000:
                print("Finish reading filename of face feature: %d/%d" %(idx, num))
            sline = line.strip().split(' ')
            image_path = sline[0].strip().replace('.jpg', '.png')
            class_id = sline[1].strip()
            feat_name = os.path.join(self._emb_outdir_root,
                    image_path+".bin")
            template.append( (class_id, feat_name) )
            class_ids.append(int(class_id))
        class_ids = np.array(class_ids, dtype=np.int32)
        return template, class_ids
    #

    def _read_embedding_base(self, file_list, emb_outdir, img_root):
        num = len(open(file_list,'r').readlines())
        succ = 0
        imgs_list = []
        features = []
        faceness_scores = []
        for idx, line in enumerate(open(file_list, 'r')):
            name_lmk_score = line.strip().split(' ')
            faceness_scores.append(name_lmk_score[-2])
            img_name = os.path.join(img_root, name_lmk_score[0])
            image_path = name_lmk_score[0].strip().replace('.jpg', '.png')
            class_id = name_lmk_score[-1].strip()
            out_path = os.path.join(emb_outdir, image_path+".bin")
            features.append(_load_bin(out_path))

        feats = np.array(features)
        faceness_scores = np.array(faceness_scores).astype(np.float32)
        print(feats.shape, faceness_scores.shape)
        self.image_feats = feats
        self.faceness_scores = faceness_scores
        print("Finish getting embedding for faces, total: %d, success: %d" %(num, succ))


    def metric(self):
        print("Begin to run metric...")
        self._read_embedding_base(self._image_list_path,
                    self._emb_outdir_root,
                    self._image_path)

        #img_feats = self.read_image_feature(os.path.join(self._emb_outdir_root, "img_feats.npy"))
        #faceness_scores = self.read_score(os.path.join(self._emb_outdir_root, "faceness_scores.npy"))
        img_feats = self.image_feats
        faceness_scores = self.faceness_scores
        print("img_feats:", img_feats.shape)
        print("faceness_scores:", len(faceness_scores))
        # use flip
        #img_input_feats = img_feats[:,0:img_feats.shape[1]/2] + img_feats[:,img_feats.shape[1]/2:]

        # normalize
        #img_feats = img_feats / np.sqrt(np.sum(img_feats ** 2, -1, keepdims=True))

        # use dectector score
        img_feats = img_feats * np.matlib.repmat(faceness_scores[:,np.newaxis], 1, img_feats.shape[1])

        templates, medias = self.read_template_media_list(self._media_list)
        p1, p2, label = self.read_template_pair_list(self._pair_list)
        template_norm_feats, unique_templates = self.image2template_feature(img_feats, templates, medias)

        print("Begin to run metric: verification...")
        score = self.verification(template_norm_feats, unique_templates, p1, p2)
        #score_out_path = os.path.join(self._emb_outdir_root, "score.npy")
        #np.save(score_out_path, score)

        fpr, tpr, _ = roc_curve(label, score)
        roc_auc = auc(fpr, tpr)
        #print("roc_auc:",roc_auc)
        fpr = np.flipud(fpr)
        tpr = np.flipud(tpr)
        x_labels = [10**-6, 10**-5, 10**-4,10**-3, 10**-2, 10**-1]
        tpr_fpr_row = []
        for fpr_iter in np.arange(len(x_labels)):
            _, min_index = min(list(zip(abs(fpr-x_labels[fpr_iter]), range(len(fpr)))))
            tpr_fpr_row.append('%.4f' % tpr[min_index])

        print(tpr_fpr_row)
        self._results['1vs1'] = round(float(tpr_fpr_row[2])*100, 4)
        self._results['1vs1_fpr'] = format(x_labels[2], '.1e')
        #print("Verification tpr = %.6f, at fpr = %.8f"%(float(self._results['1vs1']), self._results['1vs1_fpr']))
        print("Finished IJBC verification metric, done!")


    def clear(self):
        # Clean embedding data generated by inference.
        if os.path.exists(self._emb_outdir_root):
            os.system("rm -rf %s"%(self._emb_outdir_root))
            pass
    #
#
    def parse_results_into_file(self):
        out_filename = "%s/../%s"%(self._outdir_root, self._face_result_filename)
        json.dump(self._results, open(out_filename+'.tmp','w'))
        os.system("mv %s.tmp %s"%(out_filename,out_filename))

