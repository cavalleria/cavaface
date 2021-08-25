from head.metrics import *
from head.metrics_parallel import *

HEAD_DICT = {
    "Softmax": Softmax,
    "ArcFace": ArcFace,
    "Combined": Combined,
    "CosFace": CosFace,
    "SphereFace": SphereFace,
    "Am_softmax": Am_softmax,
    "CurricularFace": CurricularFace,
    "ArcNegFace": ArcNegFace,
    "SVX": SVXSoftmax,
    "AirFace": AirFace,
    "QAMFace": QAMFace,
    "CircleLoss": CircleLoss,
    "ParallelArcFace": ParallelArcFace,
}
