#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import cv2
import struct
import json
import time
import math
import mxnet as mx
import pickle
from sklearn.model_selection import KFold
from scipy import interpolate
from sklearn.decomposition import PCA
import commands

GPU_MEM_MAX = 8e9

PYTHON_TOOL='/'.join(sys.executable.split('/')[0:-2])
OUT_FILENAME_FACE_PREFIX = './service/face.result'


def _find_nearest_fpr(fpr_list):
    FPR = 1e-6
    EPISION = 1e-20
    fpr_nearest = 100
    fpr_nearest_idx = 0
    for idx, it in enumerate(fpr_list):
        diff_fpr = abs(FPR - float(it + EPISION))
        if diff_fpr < fpr_nearest:
            fpr_nearest = diff_fpr
            fpr_nearest_idx = idx
        #
    #
    return fpr_nearest_idx, fpr_list[fpr_nearest_idx]
#

def _merge_results(args, net, test_type=None):
    test_root_dir = os.path.join(
                '/'.join(args.model_path.strip().split('/')[:-1]))
    model_name = args.model_path.strip().split('/')[-2]
    outstr = "%s_%s\t" %(model_name, net._epoch)
    if "face" == test_type:
        out_filename = OUT_FILENAME_FACE_PREFIX + "." + net._net_scale_type
        out_filename_backup = '%s.%s'%(out_filename,
                    time.strftime("%Y-%m-%d.%H:%M:%S", time.localtime()))
        os.system("cp %s %s" %(out_filename, out_filename_backup) )

        results_dict = {}
        megaface_file = os.path.join(test_root_dir, "megaface.result")
        if os.path.exists(megaface_file):
            results_dict["megaface"] = {}
            megaface_results = [it.strip() for it in open(megaface_file).readlines()]
            results_dict["megaface"]["rank1"] = round(float(megaface_results[0])*100,4)
            results_dict["megaface"]["1vs1"] = round(float(megaface_results[1])*100,4)
            results_dict["megaface"]["1vs1_fpr"] = format(float(megaface_results[2]),'.1e')
            os.system("rm -f %s"%megaface_file)

        ijbc_file = os.path.join(test_root_dir, "ijbc.result")
        if os.path.exists(ijbc_file):
            results_dict["ijbc"] = {}
            ijbc_results = [it.strip() for it in open(ijbc_file).readlines()]
            results_dict["ijbc"]["rank1"] = round(float(ijbc_results[0])*100,4)
            results_dict["ijbc"]["1vs1"] = round(float(ijbc_results[1])*100,4)
            results_dict["ijbc"]["1vs1_fpr"] = format(float(ijbc_results[2]),'.1e')
            os.system("rm -f %s"%ijbc_file)

        outstr = outstr + json.dumps(results_dict)
        with open(out_filename, 'a+') as sf:
            print >> sf, outstr
    else:
        print("ERROR: Uknown test type in function: _merge_results!")
    #


def _write_result_to_file(outfilename, result_list):
    with open(outfilename, 'w') as sf:
        for it in result_list:
            #print(it, file=sf)
            print >> sf, it
    return 0

def _load_json_result_file(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)


def _read_image(image_path, gray=False):
    if gray:
        img = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        img = np.expand_dims(img, axis=2)
    else:
        img = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_COLOR)
    return img


def _write_bin(path, feature, feat_dim=None):
  if feat_dim is None:
    feat_dim = len(feature)
  feature = list(feature[0:feat_dim])
  with open(path, 'wb') as f:
    f.write(struct.pack('4i', feat_dim,1,4,5))
    f.write(struct.pack("%df"%feat_dim, *feature))


def _load_bin(path, fill = 0.0):
  with open(path, 'rb') as f:
    bb = f.read(4*4)
    #print(len(bb))
    v = struct.unpack('4i', bb)
    feat_dim = v[0]
    bb = f.read(feat_dim*4)
    v = struct.unpack("%df"%(feat_dim), bb)
    feature = np.full( (feat_dim,), fill, dtype=np.float32)
    feature[0:feat_dim] = v[0:feat_dim]
    #feature = np.array( v, dtype=np.float32)
  #print(feature.shape)
  #print(np.linalg.norm(feature))
  return feature


def _cal_similarity(probe, gallery):
    # get size
    feat_dim = _load_bin(probe[0][1]).shape[0]

    feat_mat_p = np.zeros((len(probe), feat_dim))
    feat_mat_g = np.zeros((len(gallery), feat_dim))
    for i in range(len(probe)):
        feat_p = _load_bin(probe[i][1])
        feat_mat_p[i,:] = feat_p
    for i in range(len(gallery)):
        feat_g = _load_bin(gallery[i][1])
        feat_mat_g[i,:] = feat_g
    sim = np.dot(feat_mat_p, feat_mat_g.T)
    return sim

def _cal_similarity_gpu(probe, gallery):
    sim = np.zeros((len(probe), len(gallery)), dtype=np.float32)

    split = 10000
    gpuid = 0
    pid = 0
    list_p = []
    while pid+split < len(probe):
        list_p.append(list(range(pid, pid+split)))
        pid += split
    if pid < len(probe):
        list_p.append(list(range(pid, len(probe))))

    gid = 0
    list_g = []
    while gid+split < len(gallery):
        list_g.append(list(range(gid, gid+split)))
        gid += split
    if gid < len(gallery):
        list_g.append(list(range(gid, len(gallery))))

    # get size
    feat_dim = _load_bin(probe[0][1]).shape[0]

    for pid, pitem in enumerate(list_p):
        # load peobe item
        feat_mat_p = np.zeros((len(pitem), feat_dim))
        for iidx,i in enumerate(pitem):
            feat_p = _load_bin(probe[i][1])
            feat_mat_p[iidx,:] = feat_p
        feat_mat_p = mx.nd.array(feat_mat_p, ctx=mx.gpu(gpuid))

        for gid, gitem in enumerate(list_g):
            # load gallery item
            feat_mat_g = np.zeros((len(gitem), feat_dim))
            for iidx, i in enumerate(gitem):
                feat_g = _load_bin(gallery[i][1])
                feat_mat_g[iidx,:] = feat_g
            feat_mat_g = mx.nd.array(feat_mat_g, ctx=mx.gpu(gpuid))

            #sidx = [(i,j) for i in pitem for j in gitem]
            # calc similarity
            res = mx.nd.dot(feat_mat_p, feat_mat_g.T)
            res = res.asnumpy()
            sim[pitem[0]:pitem[-1]+1, gitem[0]:gitem[-1]+1] = res

    return sim


def _cal_self_similarity(pairs):
  TH_SPLIT = 10000
  SPLIT_SIZE = 10000

  # get size
  feat_dim = _load_bin(pairs[0][0]).shape[0]

  scores = []
  if len(pairs)> TH_SPLIT:
    SPLIT_NUM = len(pairs) // SPLIT_SIZE
    yushu = len(pairs) % SPLIT_NUM
    sdx = 0
    feat_mat_p = np.zeros((SPLIT_SIZE, feat_dim))
    feat_mat_g = np.zeros((SPLIT_SIZE, feat_dim))
    for i in range(SPLIT_NUM):
      if 0 == i % 100:
        print("Calculate self-similarity: %d/%d" %(i, SPLIT_NUM))
      edx = sdx+SPLIT_SIZE
      # Read in features.
      for i in range(SPLIT_SIZE):
        feat_p = _load_bin(pairs[sdx + i][0])
        feat_g = _load_bin(pairs[sdx + i][1])
        feat_mat_p[i,:] = feat_p
        feat_mat_g[i,:] = feat_g
      #
      sim = np.dot(feat_mat_p, feat_mat_g.T)
      diag = sim.diagonal()
      scores = np.append(scores, diag)
      sdx = edx
    #
    del feat_mat_p
    del feat_mat_g
    # yushu part
    feat_mat_p = np.zeros((yushu, feat_dim))
    feat_mat_g = np.zeros((yushu, feat_dim))
    for i in range(yushu):
        feat_p = _load_bin(pairs[-yushu + i][0])
        feat_g = _load_bin(pairs[-yushu + i][1])
        feat_mat_p[i,:] = feat_p
        feat_mat_g[i,:] = feat_g
    sim = np.dot(feat_mat_p, feat_mat_g.T)
    diag = sim.diagonal()
    scores = np.append(scores, diag)
  else:
    feat_mat_p = np.zeros((len(pairs), feat_dim))
    feat_mat_g = np.zeros((len(pairs), feat_dim))
    for i in range(len(pairs)):
        feat_p = _load_bin(pairs[i][0])
        feat_g = _load_bin(pairs[i][1])
        feat_mat_p[i,:] = feat_p
        feat_mat_g[i,:] = feat_g
    sim = np.dot(feat_mat_p, feat_mat_g.T)
    diag = sim.diagonal()
    scores = np.append(scores, diag)
  return scores
#

def _rank_accuracy(preds, labels):
  rank1 = 0
  rank5 = 0
  print("preds.shape:", preds.shape)
  for i in range(preds.shape[0]):
    pred_index = np.argsort(preds[i][:])[::-1]
    #y_index = np.argsort(labels[i][:])[::-1]
    y_index = np.where(labels[i][:]==np.max(labels[i][:]))
    if pred_index[0] in y_index[0]:
      rank1 = rank1 + 1

    for j in pred_index[:5]:
      if j in y_index[0]:
        rank5 += 1
        break

  #print("rank1:", rank1)
  rank1 /= float(preds.shape[0])
  rank5 /= float(preds.shape[0])
  return rank1, rank5
#

# ---------- evaluate bin results as insightface ----------- #

def _load_image_bin(path, image_shape):
  import sys
  if sys.version[0] == '3':
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
  else:
    bins, issame_list = pickle.load(open(path, 'rb'))

  data_list = []
  for flip in [0,1]:
    if image_shape[2]==1:
      data = mx.ndarray.empty((len(issame_list)*2, 1, image_shape[1], image_shape[2]))
    else:
      data = mx.ndarray.empty((len(issame_list)*2, 3, image_shape[1], image_shape[2]))
    data_list.append(data)
  for i in range(len(issame_list)*2):
    _bin = bins[i]
    if image_shape[2]==1:
      img = mx.image.imdecode(_bin, flag=0)
    else:
      img = mx.image.imdecode(_bin, flag=1)
    if img.shape[1]!=image_shape[1]:
      img = mx.image.resize_short(img, image_shape[1])
    img = mx.ndarray.transpose(img, axes=(2, 0, 1))
    for flip in [0,1]:
      if flip==1:
        img = mx.ndarray.flip(data=img, axis=2)
      data_list[flip][i][:] = img
    if i%1000==0:
      print('loading bin', i)
  print(data_list[0].shape)
  return (data_list, issame_list)
#

class LFold:
  def __init__(self, n_splits = 2, shuffle = False):
    self.n_splits = n_splits
    if self.n_splits>1:
      self.k_fold = KFold(n_splits = n_splits, shuffle = shuffle)

  def split(self, indices):
    if self.n_splits>1:
      return self.k_fold.split(indices)
    else:
      return [(indices, indices)]


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, pca = 0):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    #print('pca', pca)

    if pca==0:
      diff = np.subtract(embeddings1, embeddings2)
      dist = np.sum(np.square(diff),1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        #print('train_set', train_set)
        #print('test_set', test_set)
        if pca>0:
          print('doing pca on', fold_idx)
          embed1_train = embeddings1[train_set]
          embed2_train = embeddings2[train_set]
          _embed_train = np.concatenate( (embed1_train, embed2_train), axis=0 )
          #print(_embed_train.shape)
          pca_model = PCA(n_components=pca)
          pca_model.fit(_embed_train)
          embed1 = pca_model.transform(embeddings1)
          embed2 = pca_model.transform(embeddings2)
          embed1 = sklearn.preprocessing.normalize(embed1)
          embed2 = sklearn.preprocessing.normalize(embed2)
          #print(embed1.shape, embed2.shape)
          diff = np.subtract(embed1, embed2)
          dist = np.sum(np.square(diff),1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        #print('threshold', thresholds[best_threshold_index])
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

    tpr = np.mean(tprs,0)
    fpr = np.mean(fprs,0)
    return tpr, fpr, accuracy

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc



def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff),1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train)>=far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    #print(true_accept, false_accept)
    #print(n_same, n_diff)
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

def evaluate(embeddings, actual_issame, nrof_folds=10, pca = 0):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds, pca = pca)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far


# gpu
def get_min_free_gpu_mem(gpuid):
    command="nvidia-smi | grep \"MiB /\" | awk -F ' ' '{print $9\" \"$11}'  | awk -F 'MiB' '{print $2-$1}'"
    _code, output = commands.getstatusoutput(command)
    min_mem = GPU_MEM_MAX
    output = [int(o) for o in output.split('\n')]
    for g in gpuid:
        if output[g]*1024*1024 < min_mem:
            min_mem = output[g]*1000*1000

    print("min gpu free mem:", min_mem, "B")
    return min_mem



# -------- end --------
