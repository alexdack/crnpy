from .crn.crn_class import CRN

def create_crn(*, from_arrays=None, from_file=None, from_random=None, from_tokens=None, from_dict=None, **kwargs):
    print("from_dict", from_dict)
    if from_arrays is not None:
        return CRN.from_arrays(**from_arrays)
    
    if from_file is not None:
        return CRN.from_file(**from_file)

    if from_dict is not None:
        return CRN.from_dict(from_dict)
    
    if from_random is not None:
        return CRN.from_random(**from_random)

    if from_tokens is not None:
        return CRN.from_tokens(**from_tokens)                
    
    raise ValueError("Must specify a construction method")
    
