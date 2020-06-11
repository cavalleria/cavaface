import os
import sys
import argparse
import numpy as np
from eval_megaface import EvalMegaFace
from eval_ijbc import EvalIJBC
from utils.io import OUT_FILENAME_FACE_PREFIX
import time
from argparse import Namespace
import json

import flask
import signal

reload(sys)
sys.setdefaultencoding('utf-8')

if sys.version_info.major == 2:
  import Queue as queue
  from Queue import Queue
else:
  import queue as Queue
import threading


app = flask.Flask(__name__)

global task_queue
global merge_queue
global service_down
global mth
global eth

eval_sets_dict = {
    'megaface': 'EvalMegaFace',
    'citrusface_large': 'EvalCitrusfaceLarge',
    'citrusface_self': 'EvalCitrusfaceSelf',
    'ijbc': 'EvalIJBC',
    'rfw': 'EvalRFW',
    }


@app.route("/alive", methods=['GET', 'POST'])
def alive():
    return True

@app.route("/eval", methods=['GET', 'POST'])
def eval_func():
    global task_queue
    global eth

    if not eth.is_alive():
        print("eval thread is down, restart...")
        start_eval_thread()

    ret_info = {'success': False}
    eval_set = flask.request.form.get('eval_set')
    args = eval(flask.request.form.get('args'))

    task_queue.put((eval_set, args))
    ret_info['success'] = True
    ret_info['tasks'] = "num %d"%(task_queue.qsize())

    return flask.jsonify(ret_info)


@app.route("/merge", methods=['GET', 'POST'])
def merge_func():
    global merge_queue
    global mth

    if not mth.is_alive():
        print("merge thread is down, restart...")
        start_merge_thread()

    ret_info = {'success': False}
    eval_set = flask.request.form.get('eval_set').split(',')
    args = eval(flask.request.form.get('args'))
    net_scale_type = flask.request.form.get('net_scale_type')

    merge_queue.put((eval_set, args, net_scale_type))
    ret_info['success'] = True
    ret_info['tasks'] = "num %d"%(merge_queue.qsize())

    return flask.jsonify(ret_info)



class evalThread (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global task_queue
        global service_down

        while not service_down:
            try:
                #eval_set, args = task_queue.get(block=True)
                eval_set, args = task_queue.get(timeout=1)
                print("\nEval on %s..."%(eval_set))
                start = time.time()
                # initialize
                evaluator = eval(eval_sets_dict[eval_set])(args)
                # run metric to evaluate.
                evaluator.metric()

                # Parse result and save.
                evaluator.parse_results_into_file()

                if args.clear:
                    print("\nClear evalutation dirs...")
                    evaluator.clear()
                #

                dura = (time.time() - start) / 60.0
                print("\nExtract on %s done! Time cost: %fm"%(eval_set, dura))
                sys.stdout.flush()

            except queue.Empty as e:
                pass

        print("eval thread end")

def start_eval_thread():
    global eth
    eth = evalThread()
    eth.start()


def _merge_results(args, eval_set, net_scale_type, test_type=None):
    global service_down

    try:
        test_root_dir = os.path.join(
                '/'.join(args.model_path.strip().split('/')[:-1]))
        model_name = args.model_path.strip().split('/')[-2]
        epoch = int(args.model_path.strip().split(',')[1])
        outstr = "%s_%s\t" %(model_name, epoch)

        if "face" == test_type:
            print("start to merge results of %s"%(eval_set))
            out_filename = OUT_FILENAME_FACE_PREFIX + "." + net_scale_type
            out_filename_backup = '%s.%s'%(out_filename,
                    time.strftime("%Y-%m-%d.%H:%M:%S", time.localtime()))
            os.system("cp %s %s" %(out_filename, out_filename_backup) )

            results_dict = {}
            for ieval_set in eval_set:
                eval_file = os.path.join(test_root_dir, ieval_set+".%d.result"%(epoch))
                print("begin load result from %s"%(eval_file))
                try_count = 0

                while not os.path.exists(eval_file):
                    if service_down:
                        return None

                    # wait untill file generated
                    time.sleep(1)
                    try_count += 1

                    if try_count >= 600: # wait 10 min, re-input evalsets
                        return False

                results_dict[ieval_set] = {}
                eval_results = json.load(open(eval_file, 'r'))
                for k, v in eval_results.items():
                    results_dict[ieval_set][k] = v


            outstr = outstr + json.dumps(results_dict)
            with open(out_filename, 'a+') as sf:
                print >> sf, outstr

            # remove results
            for ieval_set in eval_set:
                eval_file = os.path.join(test_root_dir, ieval_set+".%d.result"%(epoch))
                os.system("rm -f %s"%eval_file)

            print("merge results of %s done"%(eval_set))
            sys.stdout.flush()
            return True

    except:
        print("merge results of %s failed"%(eval_set))
        return False
    #



class mergeThread (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global merge_queue
        global service_down

        while not service_down:
            try:
                #eval_set, args, net_scale_type = merge_queue.get(block=True)
                eval_set, args, net_scale_type = merge_queue.get(timeout=1)
                # Merge results of all the benchmarks.
                # add into merge queue
                ret = _merge_results(args, eval_set, net_scale_type, test_type="face")

                if ret is not None and not ret:
                    print("merge failed, re-input into merge_queue")
                    merge_queue.put((eval_set, args, net_scale_type))

            except queue.Empty as e:
                pass

        print("merge thread end")

def start_merge_thread():
    global mth
    mth = mergeThread()
    mth.start()


def sigint_handler(signum, frame):
    global service_down
    global mth
    global eth
    service_down = True
    print("main-thread exit")
    mth.join()
    eth.join()

    sys.exit()

signal.signal(signal.SIGINT, sigint_handler)


if __name__ == '__main__':
    global task_queue
    global merge_queue
    global service_down
    task_queue = Queue()
    merge_queue = Queue()
    service_down = False

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--port',
        type=int,
        default=7776,
        help='evaluate server port.')
    args = parser.parse_args()

    start_eval_thread()
    start_merge_thread()

    app.run(host='127.0.0.1', threaded=True, port=args.port, debug=False)



