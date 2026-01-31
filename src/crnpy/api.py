from .crn.crn_class import CRN

def create_crn(species, reactions):
    return CRN(species, reactions)