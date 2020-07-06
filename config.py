import torch

configurations = {
    1: dict(
        SEED = 1337, # random seed for reproduce results
        
        DATA_ROOT = '../facedata.mxnet/ms1m-retinaface-t1-clean-img', # the parent root where your train/val/test data are stored
        RECORD_DIR = '../facedata.mxnet/ms1m-retinaface-t1-clean.txt', # the dataset record dir
        VAL_DATA_ROOT = '../facedata.mxnet/face_val_data', # the parent root where your val/test data are stored
        MODEL_ROOT = '../models/MobileFaceNet_Arcface/model', # the root to buffer your checkpoints
        LOG_ROOT = '../models/MobileFaceNet_Arcface/log', # the root to log your train/val status
        IS_RESUME = False,
        BACKBONE_RESUME_ROOT = "",
        HEAD_RESUME_ROOT = "",
        
        BACKBONE_NAME = 'MobileFaceNet', # support: ['MobileFaceNet', 'ResNet_50', 'ResNet_101', 'ResNet_152', 
                                #'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_100', 'IR_SE_101', 'IR_SE_152',
                                #'AttentionNet_IR_56', 'AttentionNet_IRSE_56','AttentionNet_IR_92', 'AttentionNet_IRSE_92',
                                #'ResNeSt_50', 'ResNeSt_101', 'ResNeSt_100']
        HEAD_NAME = "ArcFace", # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax', 'ArcNegFace', 'CurricularFace', 'SVX']
        LOSS_NAME = 'Softmax', # support: [''Softmax', Focal', 'HardMining', 'LabelSmooth', 'Softplus']
        
        INPUT_SIZE = [112, 112], # support: [112, 112] and [224, 224]
        RGB_MEAN = [0.5, 0.5, 0.5], # for normalize inputs to [-1, 1]
        RGB_STD = [0.5, 0.5, 0.5],
        EMBEDDING_SIZE = 512, # feature dimension
        BATCH_SIZE = 1024,
        EVAL_FREQ = 2000, #for ms1m, batch size 1024, EVAL_FREQ=2000
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        
        OPTIMIZER = 'sgd', # sgd, adam, lookahead, radam, ranger, adamp, sgdp
        LR = 0.1, # initial LR, use smaller lr for adam seris
        LR_SCHEDULER = 'cosine', # step/multi_step/cosine
        WARMUP_EPOCH = 0, 
        WARMUP_LR = 0.0,
        START_EPOCH = 0, # start epoch
        NUM_EPOCH = 24, # total epoch number
        LR_STEP_SIZE = 10, # 'step' scheduler, period of learning rate decay. 
        LR_DECAY_EPOCH = [10, 18, 22], # ms1m epoch stages to decay learning rate
        LR_DECAT_GAMMA = 0.1, # multiplicative factor of learning rate decay
        LR_END = 1e-5, # minimum learning rate
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        
        WORLD_SIZE = 1,
        RANK = 0,
        GPU = [0,1,2,3,4,5,6,7], # specify your GPU ids
        DIST_BACKEND = 'nccl', # 'nccl', 'gloo'
        DIST_URL = 'tcp://localhost:23456',
        NUM_WORKERS = 8,
        TEST_GPU_ID = [0,1,2,3,4,5,6,7],

        USE_APEX = False,
        SYNC_BN = False,

        # Data Augmentation
        RANDAUGMENT = False,
        RANDAUGMENT_N = 2, # random pick numer of aug typr form aug_list 
        RANDAUGMENT_M = 9,
        RANDOM_ERASING = False,
        MIXUP = False,
        MIXUP_ALPHA = 1.0,
        MIXUP_PROB = 0.5,
        CUTOUT = False, 
        CUTMIX = False, 
        CUTMIX_ALPHA = 1.0,
        CUTMIX_PROB = 0.5
    ),
}
