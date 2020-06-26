#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
from easydict import EasyDict as edict
import sklearn
from sklearn.preprocessing import normalize
import time

import signal
import traceback

reload(sys)
sys.setdefaultencoding('utf-8')

if sys.version_info.major == 2:
  import Queue as queue
  from Queue import Queue
else:
  import queue as Queue
import threading

global signal_stop
global read_ths
global infer_ths
global write_ths
global read_done
global write_done
global infer_done

signal_stop = False

def sigint_handler(signum, frame):
    global signal_stop
    global read_ths
    global infer_ths
    global write_ths

    signal_stop = True
    print("main-thread exit")

    # wait for finish
    for t in read_ths:
      t.join()
    for t in infer_ths:
      t.join()
    write_ths.join()

    sys.exit()

signal.signal(signal.SIGINT, sigint_handler)


class writeThread (threading.Thread):
    def __init__(self, out_q, write_func, tasks):
        threading.Thread.__init__(self)
        self.out_q = out_q
        self.write_func = write_func
        self.tasks = tasks
    def run(self):
        global signal_stop
        global write_done
        global infer_done

        done = 0
        while not signal_stop:
            try:
                #embedding, outpaths = self.out_q.get(block=True)
                embedding, outpaths = self.out_q.get(timeout=1)
                if done % 10000 == 0:
                    print(done, self.tasks)
                for idx, outp in enumerate(outpaths):
                    self.write_func(outp, embedding[idx].flatten())
                    done += 1
            except queue.Empty as e:
                if infer_done:
                    break
                pass
        print("write Thread done")
        write_done = True

class readThread (threading.Thread):
    def __init__(self, in_q, out_q, read_func, is_flip, shape):
        threading.Thread.__init__(self)
        self.in_q = in_q
        self.out_q = out_q
        self.read_func = read_func
        self.is_flip = is_flip
        self.shape = shape

    def run(self):
        global signal_stop
        unfinished = True
        while not signal_stop and unfinished:
            try:
                image_path, outpath = self.in_q.get(timeout=1)
                out_img = []
                img = self.read_func(image_path, self.shape[0]==1)
                if len(img.shape) == 3:
                    img = img[:,:,::-1] #to rgb
                    img = np.transpose( img, (2,0,1) )
                else:
                    img = img[np.newaxis,:]
                attempts = [0,1] if self.is_flip else [0]
                for flipid in attempts:
                    _img = np.copy(img)
                    if flipid==1:
                        self.do_flip(_img)
                    out_img.append(_img)

                self.out_q.put((out_img, outpath))

            except queue.Empty as e:
                unfinished = False
                break
        print("read Thread done")

    def do_flip(self, data):
        for idx in range(data.shape[0]):
            data[idx,:,:] = np.fliplr(data[idx,:,:])



class inferThread (threading.Thread):
    def __init__(self, net, fwd_func, in_q, out_q, image_shape, is_flip):
        threading.Thread.__init__(self)
        self.net = net
        self.fwd_func = fwd_func
        self.in_q = in_q;
        self.out_q = out_q;
        self.batchsz = net.batch_size;
        self.image_shape = image_shape
        self.is_flip = is_flip
        self.end = False 

    def run(self):
        global signal_stop
        global read_done
            
        bz = self.batchsz

        if (self.is_flip):
            bz = self.batchsz//2 * 2

        input_blob = np.zeros(shape=(bz, self.image_shape[0],
                    self.image_shape[1], self.image_shape[2]))

        while not signal_stop and not self.end:
            idx = 0
            outpaths = []
            while idx < bz:
                try:
                    images, outpath = self.in_q.get(timeout=1)
                    input_blob[idx] = images[0]
                    idx+=1
                    if self.is_flip:
                        input_blob[idx] = images[1]
                        idx+=1
                    outpaths.append(outpath)
                except queue.Empty as e:
                    if read_done:
                        self.end = True
                    break;
    
            if idx == 0:
              continue

            embedding = self.fwd_func(self.net, input_blob)
    
            if self.is_flip:
                embedding1 = embedding[0::2]
                embedding2 = embedding[1::2]
                embedding = embedding1+embedding2
    
            #assert len(outpaths) == embedding.shape[0], "%d vs %d"%(len(outpaths), embedding.shape[0])
            if self.is_flip:
                assert len(outpaths) == idx/2
            else:
                assert len(outpaths) == idx
    
            embedding = sklearn.preprocessing.normalize(embedding[0:len(outpaths)])
    
            self.out_q.put((embedding, outpaths))

        print("infer thread done")

"""
class binThread (threading.Thread):
    def __init__(self, net, in_q, image_shape, is_flip, read_func, write_func):
        threading.Thread.__init__(self)
        self.net = net
        self.in_q = in_q;
        self.batchsz = net.batch_size;
        self.image_shape = image_shape
        self.is_flip = is_flip
        self.read_func = read_func
        self.write_func = write_func
        self.end = False 

    def run(self):
        global signal_stop

        bz = self.batchsz

        if (self.is_flip):
            bz = self.batchsz//2 * 2

        input_blob = np.zeros(shape=(bz, self.image_shape[0],
                    self.image_shape[1], self.image_shape[2]))

        while not signal_stop and not self.end:
            # load bin
            inpath, outpath = self.in_q.get(timeout=1)
            data_set = self.read_func(inpath, self.image_shape)

            data_list = data_set[0]
            issame_list = data_set[1]
            embeddings_list = []

            for i in range( len(data_list) ):
                data = data_list[i]
                embeddings = None
                ba = 0
                while ba<data.shape[0]:
                    bb = min(ba+bz, data.shape[0])
                    count = bb-ba
                    _data = nd.slice_axis(data, axis=0, begin=bb-bz, end=bb)
                    db = mx.io.DataBatch(data=(_data,))
                    self.net.model.forward(db, is_train=False)
                    net_out = self.net.model.get_outputs()
                    _embeddings = net_out[0].asnumpy()
                    if embeddings is None:
                        embeddings = np.zeros( (data.shape[0], _embeddings.shape[1]) )
                    embeddings[ba:bb,:] = _embeddings[(bz-count):,:]
                    ba = bb
                embeddings_list.append(embeddings)
          
            embeddings = embeddings_list[0].copy()
            if self.is_flip:
                embeddings = embeddings_list[0] + embeddings_list[1]
            embeddings = sklearn.preprocessing.normalize(embeddings)

            # save output numpy
            self.write_func(outpath, embeddings)
            self.write_func(outpath+".issame", issame_list)

        print("infer thread done")
"""

class CitrusBaseInfer(object):
    def __init__(self, args, dtype='fp32'):
        self._args = args
        self._dtype = dtype
        self._net = None
        self._ctx = []
        self._image_shape = []
        self._batch_size = 8
        self._net_scale_type = "large"
        self._flops = None
        self._epoch = None
        self._gpu_mem_use = 0;
        self._nets = []
        self.fwd_func = None

        self._load_model()

    def _load_model(self):
        raise NotImplementedError()
    #

    def infer_embedding(self, imgs=None, read_func = None, write_func = None, is_flip=False):
        global read_ths
        global infer_ths
        global write_ths
        global read_done
        global write_done
        global infer_done

        read_done = False
        write_done = False
        infer_done = False
        assert 0 != len(imgs) and imgs is not None, "images to infer must not be empty!"
        assert write_func is not None
        assert read_func is not None

        try:
          in_q = Queue()
          for items in imgs:
            in_q.put(items)
          tasks = in_q.qsize()

          print("begin thread")
 
          out_q = Queue()
          img_q = Queue()

          read_ths = []
          infer_ths = []

          read_thread_num = len(self._nets) * 2
          if read_thread_num > 30:
              read_thread_num = 30
          for n in range(read_thread_num):
            read_ths.append(readThread(in_q, img_q, read_func, is_flip, self._image_shape))

          for n in self._nets:
            infer_ths.append(inferThread(n, self.fwd_func, img_q, out_q, self._image_shape, is_flip))

          for t in read_ths:
              t.start()

          for t in infer_ths:
              t.start()

          write_ths = writeThread(out_q, write_func, tasks)
          write_ths.start()

          while not read_done:
            done = True
            for t in read_ths:
              if t.isAlive():
                  done = False
            if done:
                read_done = True
                print("read threads done!\n")
            time.sleep(1)

          while not infer_done:
            done = True
            for t in infer_ths:
              if t.isAlive():
                  done = False
            if done:
                infer_done = True
                print("infer threads done!\n")
            time.sleep(1)


          while not write_done:
              # if use join(), ctrl-c can not work
              time.sleep(1) 

          if in_q.qsize() == 0 and img_q.qsize() == 0 and out_q.qsize() == 0:
            return True
          else:
            print("not done! in_q.qsize()", in_q.qsize(), "img_q.qsize()", img_q.qsize(), "out_q.qsize()", out_q.qsize())
            return False
        except:
          print("error", traceback.print_exc())
          signal_stop = True
          return False
    #

    """
    def infer_bin(self, imgs=None, read_func = None, write_func = None, is_flip=False):
        global infer_ths
        global infer_done

        infer_done = False
        assert 0 != len(imgs) and imgs is not None, "images to infer must not be empty!"
        assert write_func is not None
        assert read_func is not None

        try:
          in_q = Queue()
          for items in imgs:
            in_q.put(items)

          infer_ths = []

          for n in self._nets:
            infer_ths.append(binThread(n, in_q, self._image_shape, is_flip, read_func, write_func))

          for t in infer_ths:
              t.start()

          while not infer_done:
            done = True
            for t in infer_ths:
              if t.isAlive():
                  done = False
            if done:
                infer_done = True
                print("infer threads done!\n")
            time.sleep(1)

          return True
        except:
          print("error", traceback.print_exc())
          signal_stop = True
          return False
    """
    #

