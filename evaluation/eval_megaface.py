#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from utils.io import _read_image, _write_bin, _load_bin, _load_json_result_file, \
            _write_result_to_file, _find_nearest_fpr, PYTHON_TOOL, \
            _cal_similarity_gpu
import copy
import time

import json


class EvalMegaFace(object):
    def __init__(self, args):
        self._args = args
        self._algo = "retina"
        self._metric_tool_dir = "./devkit"
        self._metric_prob_list = "./../templatelists/facescrub_features_list.json"
        self._python_tool_dir = PYTHON_TOOL
        self._outdir_root = ''
        self._emb_outdir_root = ''
        self._emb_clean_outdir_root = ''
        self._face_scrub_list = "./data/facescrub_lst"
        self._face_scrub_root = "../data/megaface/facescrub_images"
        self._face_scrub = []
        self._face_scrub_emb_outdir = ''
        self._face_scrub_emb_clean_outdir = ''
        self._face_gallery_list = "./data/megaface_lst"
        self._face_gallery_root = "../data/megaface/megaface_images"
        self._face_gallery = []
        self._face_gallery_emb_outdir = ''
        self._face_gallery_emb_clean_outdir = ''

        self._face_scrub_noisy_list = "./data/facescrub_noises.txt"
        self._face_gallery_noisy_list = "./data/megaface_noises.txt"
        self._face_scrub_noisy_dict = {}
        self._face_gallery_noisy_dict = {}
        
        self._save_badcase = True if args.save_badcase else False
        self._badcase_prob = []
        self._badcase_gall = []
        self._face_badcase_savefile = 'megaface.badcase'

        self._get_output_dir()
        self._load_faces_lists()

    def _get_output_dir(self):
        # Get and clear face scrub output dir.
        epoch = int(self._args.model_path.strip().split(',')[1])
        self._outdir_root = os.path.join(
                    '/'.join(self._args.model_path.strip().split('/')[:-1]),
                    "megaface_%d"%(epoch))
        self._emb_outdir_root = os.path.join(self._outdir_root, "embedding")
        self._emb_clean_outdir_root = os.path.join(self._outdir_root, "embedding_clean")

        self._face_scrub_emb_outdir = os.path.join(self._emb_outdir_root, "facescrub")
        os.system("mkdir -p %s"%(self._face_scrub_emb_outdir))
        self._face_scrub_emb_clean_outdir = os.path.join(self._emb_clean_outdir_root,
                    "facescrub")
        os.system("mkdir -p %s"%(self._face_scrub_emb_clean_outdir))

        # Get face gallery output dir.
        self._face_gallery_emb_outdir = os.path.join(self._emb_outdir_root, "megaface")
        os.system("mkdir -p %s"%(self._face_gallery_emb_outdir))
        self._face_gallery_emb_clean_outdir = os.path.join(self._emb_clean_outdir_root,
                    "megaface")
        os.system("mkdir -p %s"%(self._face_gallery_emb_clean_outdir))

        self._face_result_filename = 'megaface.%d.result'%(epoch)

    def _load_faces_lists(self):
        self._face_scrub = open(self._face_scrub_list).readlines()
        self._face_gallery = open(self._face_gallery_list).readlines()

        for line in open(self._face_scrub_noisy_list, 'r'):
            if line.startswith('#'):
                continue
            line = line.strip()
            fname = line.split('.')[0]
            p = fname.rfind('_')
            fname = fname[0:p]
            self._face_scrub_noisy_dict[line] = fname
        #
        print("Noisy faces of scrub: %d" %(len(self._face_scrub_noisy_dict)))

        for line in open(self._face_gallery_noisy_list, 'r'):
            if line.startswith('#'):
                continue
            line = line.strip()
            _vec = line.split("\t")
            if len(_vec)>1:
                line = _vec[1]
            self._face_gallery_noisy_dict[line] = 1
        #
        print("Noisy faces of gallery: %d" %(len(self._face_gallery_noisy_dict)))
    #

    def _infer_and_write(self, imgs_list, net):
        return net.infer_embedding(imgs_list, read_func = _read_image, write_func = _write_bin, is_flip = self._args.flip)
    #

    def _extract_embedding_base(self, file_list, net, emb_outdir, img_root, is_prob=True):
        nbz = 1 #100
        nbz_buf = 1000 #100
        num = len(file_list)
        succ = 0
        imgs_list = []
        for idx, line in enumerate(file_list):
            if 0 == idx%10000:
                print("Finish Load path of faces: %d/%d" %(idx, num))
            image_path = line.strip()
            _path = image_path.split('/')
            if is_prob:
                a, b = _path[-2], _path[-1]
                out_dir = os.path.join(emb_outdir, a)
            else:
                a1, a2, b = _path[-3], _path[-2], _path[-1]
                out_dir = os.path.join(emb_outdir, a1, a2)
            #
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            image_path = os.path.join(img_root, image_path)
            out_path = os.path.join(out_dir, b+"_%s.bin"%(self._algo))
            item = (image_path, out_path)
            imgs_list.append(item)
        ret = self._infer_and_write(imgs_list, net)
        print("Finish getting embedding for faces, total: %d, success: %d" %(num, succ))
        return ret
    #

    def extract_embedding(self, net):
        # Get embedding of scrub faces.
        print("Begin to extract embedding of scrub faces...")
        ret = self._extract_embedding_base(self._face_scrub, net,
                    self._face_scrub_emb_outdir,
                    self._face_scrub_root)
        if not ret:
            return False

        # Get embedding of gallery faces.
        print("Begin to extract embedding of gallery faces...")
        ret = self._extract_embedding_base(self._face_gallery, net,
                    self._face_gallery_emb_outdir,
                    self._face_gallery_root, is_prob=False)
        if not ret:
            return False

        return True
    #

    def remove_noisy_embedding(self):
        # Follow insight face to remove noisy embedding.
        feature_ext = 1
        # Remove noise in scrub faces.
        i=0
        fname2center = {}
        noises = []
        for line in self._face_scrub:
            if i%1000==0:
                print("reading scrub faces: %d"%(i))
            i+=1
            image_path = line.strip()
            _path = image_path.split('/')
            a, b = _path[-2], _path[-1]
            feature_path = os.path.join(self._face_scrub_emb_outdir,
                        a, "%s_%s.bin"%(b, self._algo))
            feature_dir_out = os.path.join(self._face_scrub_emb_clean_outdir, a)
            if not os.path.exists(feature_dir_out):
                os.makedirs(feature_dir_out)
            feature_path_out = os.path.join(feature_dir_out, "%s_%s.bin"%(b, self._algo))
            if not b in self._face_scrub_noisy_dict:
              feature = _load_bin(feature_path)
              # loaded feature dim
              feat_dim = feature.shape[0]
              _write_bin(feature_path_out, feature)
             
              # ext feature
              extra_ = np.zeros((feat_dim+feature_ext,), dtype=np.float32)
              extra_[0:feat_dim] = feature

              if self._save_badcase:
                class_id = a
                self._badcase_prob.append( (class_id, feature_path_out) ) 
                
              if not a in fname2center:
                fname2center[a] = np.zeros((feat_dim+feature_ext,), dtype=np.float32)
              fname2center[a] += extra_
            else:
              noises.append( (a,b) )
            #
        #
        print("Total number of removing noisy faces in scrub: %d/%d"%(len(noises), i))

        for k in noises:
            a,b = k
            assert a in fname2center
            center = fname2center[a]
            g = np.zeros(center.shape, dtype=np.float32)
            # origin feat dim
            feat_dim = center.shape[0] - feature_ext

            g2 = np.random.uniform(-0.001, 0.001, (feat_dim,))
            g[0:feat_dim] = g2
            f = center+g
            _norm=np.linalg.norm(f)
            f /= _norm
            feature_path_out = os.path.join(self._face_scrub_emb_clean_outdir,
                        a, "%s_%s.bin"%(b, self._algo))
            _write_bin(feature_path_out, f, feat_dim)
                
            if self._save_badcase:
                class_id = a
                self._badcase_prob.append( (class_id, feature_path_out) ) 
        #

        # Remove noise in gallery faces.
        i=0
        nrof_noises = 0
        for line in self._face_gallery:
            if i%10000==0:
                print("reading gallery faces: %d"%(i))
                sys.stdout.flush()
            i+=1
            image_path = line.strip()
            _path = image_path.split('/')
            a1, a2, b = _path[-3], _path[-2], _path[-1]
            feature_path = os.path.join(self._face_gallery_emb_outdir,
                        a1, a2, "%s_%s.bin"%(b, self._algo))
            feature_dir_out = os.path.join(self._face_gallery_emb_clean_outdir, a1, a2)
            if not os.path.exists(feature_dir_out):
                os.makedirs(feature_dir_out)
            feature_path_out = os.path.join(feature_dir_out, "%s_%s.bin"%(b, self._algo))
            bb = '/'.join([a1, a2, b])
            if not bb in self._face_gallery_noisy_dict:
              feature = _load_bin(feature_path)
              _write_bin(feature_path_out, feature)
                
              if self._save_badcase:
                class_id = a2.strip().split('@')[0].strip()
                self._badcase_gall.append( (class_id, feature_path_out) ) 
            else:
              feature = _load_bin(feature_path, 100.0)
              feature[...] = 100.0
              _write_bin(feature_path_out, feature)
              nrof_noises+=1
            #
        #
        print("Total number of removing noisy faces in gallery: %d/%d"%(nrof_noises, i))

    def metric(self):
        # remove noisy first
        self.remove_noisy_embedding()

        curr_dir = os.getcwd()
        os.chdir(os.path.join(self._metric_tool_dir, "experiments"))

        print("LD_LIBRARY_PATH=%s/lib:${LD_LIBRARY_PATH} %s/bin/python -u run_experiment.py %s %s _%s.bin %s -s 1000000 -p %s"%(
                    self._python_tool_dir,
                    self._python_tool_dir, self._face_gallery_emb_clean_outdir,
                    self._face_scrub_emb_clean_outdir, self._algo,
                    self._outdir_root, self._metric_prob_list))
        os.system("LD_LIBRARY_PATH=%s/lib:${LD_LIBRARY_PATH} %s/bin/python -u run_experiment.py %s %s _%s.bin %s -s 1000000 -p %s"%(
                    self._python_tool_dir,
                    self._python_tool_dir, self._face_gallery_emb_clean_outdir,
                    self._face_scrub_emb_clean_outdir, self._algo,
                    self._outdir_root, self._metric_prob_list))
        os.chdir(curr_dir)
        
        # Calculate scores of probe faces in probe set and among gallery set.
        if self._save_badcase:
            s1 = time.time()
            self._cal_and_save_badcase()
            print("Calculate and save badcases cost time: %f s"%(time.time() - s1))

    def _cal_and_save_badcase(self):
        sets = [self._badcase_prob, self._badcase_gall]

        # Generate class ids.
        idx = 0
        dict_cids = {}
        for iset in sets:
            for (cid, feat_path) in iset:
                if cid not in dict_cids:
                    dict_cids[cid] = idx
                    idx += 1
                #
            #
            num = len(dict_cids.keys())
            print("classes: ", num)
        #
        
        # Generate sets.
        dict_prob = {}
        for (cid, feat_path) in self._badcase_prob:
            assert cid in dict_cids
            idx = dict_cids[cid]
            if cid not in dict_prob:
                dict_prob[cid] = [(idx, feat_path)]
            else:
                dict_prob[cid].append( (idx, feat_path) )
            #
        #
        print("Probe set num: ", len(self._badcase_prob))

        gall_set = []
        gall_path = []
        for (cid, feat_path) in self._badcase_gall:
            assert cid in dict_cids
            idx = dict_cids[cid]
            gall_set.append( (idx, feat_path) )
            gall_path.append(feat_path)
        #
        print("Gallery set num: ", len(gall_set))

        # Calculate rank scores for each probe class.
        ip_cnt = 0
        badcases = []
        for cid, iprob in dict_prob.items():
            s1 = time.time()
            #pids = [ it[0] for it in iprob]
            prob_set = iprob
            inum = len(prob_set)
            
            # Calculate socre matrix, np x (np+ng)
            print("Begin to calculate badcase for class: ", cid, ", with samples: ", inum)
            gall_set_all = copy.deepcopy(prob_set)
            gall_set_all.extend(gall_set)
            #gids = [ it[0] for it in gall_set_all]
            scores_all = _cal_similarity_gpu(prob_set, gall_set_all)

            # Select probe scores.
            scores = [] # (npx(np-1)) x (1+ng) 
            probs_scores = []
            for igall in range(inum):
                for jprob in range(inum):
                    if igall == jprob:
                        continue
                    ij_score = scores_all[igall, jprob]
                    j_scores = scores_all[jprob, inum:]
                    iscores = np.hstack([ij_score, j_scores])
                    scores.append(iscores)
                    probs_scores.append([prob_set[jprob][1], prob_set[igall][1], ij_score])
                #
            #

            # Select badcase of topk.
            search_res = np.argmax(scores, axis=1)
            badcase_ind = np.where(search_res != 0) # first index where gt locates
            badcase = np.array(probs_scores)[badcase_ind]
            
            if badcase.shape[0] > 0:
                badcase_score = np.array(scores)[badcase_ind, search_res[badcase_ind]].reshape((badcase.shape[0], -1))
                assert badcase.shape[0] == badcase_score.shape[0]
                badcase_gall = []
                gallerys = ['-'] + gall_path
                for i in range(len(badcase_ind)):
                    badcase_res = np.array(gallerys)[search_res[badcase_ind]]
                    badcase_gall.append(badcase_res)
                #
                badcase_gall = np.array(badcase_gall).reshape((badcase.shape[0], -1))
                badcases.append(np.hstack((badcase, badcase_gall, badcase_score)))
            # 
            print("Finished class with id: ", ip_cnt, " similarity matrix with shape: ", 
                    scores_all.shape, ", search times: %d"%(len(scores)))
            print("")
            ip_cnt += 1
            sys.stdout.flush()
        #
        # Save badcases.
        if len(badcases) > 0:
            badcases = np.vstack(badcases)
            out_filename = "%s/../%s"%(self._outdir_root, self._face_badcase_savefile)
            with open(out_filename, 'w') as sf:
                for i in range(badcases.shape[0]):
                    it = badcases[i]
                    ires = ' '.join([it[j] for j in range(it.shape[0])])
                    print >> sf, ires
                #
            #
            print("Total saved badcases: ", len(badcases))
        else:
            print("Warning: No badcases to be saved!")
        #           


    def clear(self):
        # Clean embedding data generated by inference.
        if os.path.exists(self._emb_outdir_root):
            os.system("rm -rf %s"%(self._emb_outdir_root))
        if os.path.exists(self._emb_clean_outdir_root):
            os.system("rm -rf %s"%(self._emb_clean_outdir_root))
        others_dir = os.path.join(self._outdir_root, "otherFiles")
        if os.path.exists(others_dir):
            os.system("rm -rf %s"%(others_dir))


    #

    def parse_results_into_file(self):
        result_json_file = os.path.join(self._outdir_root,
                    "cmc_facescrub_megaface_%s_1000000_1.json"%(self._algo))
        out_filename = "%s/../%s"%(self._outdir_root, self._face_result_filename)
        result_dict = _load_json_result_file(result_json_file)
        res_rank1 = result_dict["cmc"][1][0]

        idx, fpr = _find_nearest_fpr(result_dict["roc"][0])
        res_1vs1 = result_dict["roc"][1][idx]
        _results = {}
        _results["rank1"] = round(res_rank1*100, 4)
        _results["1vs1_fpr"] = format(fpr, '.1e') 
        _results["1vs1"] = round(res_1vs1*100, 4)

        json.dump(_results, open(out_filename+'.tmp','w'))
        os.system("mv %s.tmp %s"%(out_filename,out_filename))

