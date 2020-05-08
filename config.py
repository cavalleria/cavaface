import torch

configurations = {
    1: dict(
        SEED = 1337, # random seed for reproduce results
        
        DATA_ROOT = '/home/air/facedata.mxnet.hot/ms1m-retinaface-t1-clean-img', # the parent root where your train/val/test data are stored
        RECORD_DIR = '/home/air/facedata.mxnet.hot/ms1m-retinaface-t1-clean.txt', # the dataset record dir
        VAL_DATA_ROOT = '/home/air/facedata.mxnet.hot/face_val_data', # the parent root where your val/test data are stored
        MODEL_ROOT = '/home/air/facedata.mxnet/cavaface/cavaface_models/test_pytorch/model2', # the root to buffer your checkpoints
        LOG_ROOT = '/home/air/facedata.mxnet/cavaface/cavaface_models/test_pytorch/log2', # the root to log your train/val status
        #DATA_ROOT = '/home/air/facedata.mxnet.hot/cavaface/faces_webface_img', # the parent root where your train/val/test data are stored
        #RECORD_DIR = '/home/air/facedata.mxnet.hot/cavaface/faces_webface.txt', # the dataset record dir
        #VAL_DATA_ROOT = '/home/air/facedata.mxnet.hot/cavaface/face_val_data', # the parent root where your val/test data are stored
        #MODEL_ROOT = '/home/air/facedata.mxnet/cavaface_models/models/test_pytorch/model', # the root to buffer your checkpoints
        #LOG_ROOT = '/home/air/facedata.mxnet/cavaface/cavaface_models/test_pytorch/log', # the root to log your train/val status

        IS_RESUME = False,
        BACKBONE_RESUME_ROOT = "",
        HEAD_RESUME_ROOT = "",
        
        BACKBONE_NAME = 'MobileFaceNet', # support: ['MobileFaceNet', 'ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME = "ArcFace", # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax', 'ArcNegFace', 'CurricularFace', 'SVX']
        LOSS_NAME = 'Softmax', # support: [''Softmax', Focal', 'HardMining', 'LabelSmooth']
        
        INPUT_SIZE = [112, 112], # support: [112, 112] and [224, 224]
        RGB_MEAN = [0.5, 0.5, 0.5], # for normalize inputs to [-1, 1]
        RGB_STD = [0.5, 0.5, 0.5],
        EMBEDDING_SIZE = 512, # feature dimension
        BATCH_SIZE = 1024,
        EVAL_FREQ = 2000, #for ms1m, batch size 1024, EVAL_FREQ=2000
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        
        LR = 0.1, # initial LR
        LR_SCHEDULER = 'cosine', # step/multi_step/cosine
        WARMUP_EPOCH = 0, 
        WARMUP_LR = 0.0,
        START_EPOCH = 0, #start epoch
        NUM_EPOCH = 24, # total epoch number
        LR_STEP_SIZE = 10, # 'step' scheduler, period of learning rate decay. 
        LR_DECAY_EPOCH = [10, 18, 22], # ms1m epoch stages to decay learning rate
        LR_DECAT_GAMMA = 0.1, # multiplicative factor of learning rate decay
        LR_END = 1e-5, # minimum learning rate
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,

        EVAL_FREQ = 2000,
        NECK = "GDC", # support: ['E', 'F', 'G', 'H', 'I', 'J', 'Z', 'FC', 'GAP', 'GNAP', 'GDC']

        
        WORLD_SIZE = 1,
        RANK = 0,
        GPU = [0,1], # specify your GPU ids
        DIST_BACKEND = 'nccl', # 'nccl', 'gloo'
        DIST_URL = 'tcp://localhost:23456',
        NUM_WORKERS = 5,
        TEST_GPU_ID = [0,1],
        VAL_SET = ['lfw', 'cfp_fp', 'agedb_30'], # support ['lfw', 'cfp_fp', 'agedb_30', 'calfw', 'cplfw', 'vgg2_fp']

        USE_APEX = False
    ),
}
