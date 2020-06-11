from citrus_pytorch_infer import CitrusPytorchInfer

def get_infer(args):
    model_type = args.model_type
    infer_type, dtype = model_type.split('_')

    if infer_type == 'pytorch':
        return CitrusPytorchInfer(args, dtype=dtype)

    else:
        print("ERROR: Unknown infer type")
        return None
