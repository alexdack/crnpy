from .crn.crn_class import CRN

def create_crn(*, from_arrays=None, from_file=None, from_random=None, **kwargs):
    if from_arrays is not None:
        return CRN.from_arrays(**from_arrays)
    
    if from_file is not None:
        return CRN.from_file(**from_file)
    
    if from_random is not None:
        return CRN.from_random(**from_random)        
    
    raise ValueError("Must specify a construction method")
    
