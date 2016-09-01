
from bionlp.utils.crf_arguments import crf_model_arguments
from train import main

if __name__=="__main__":
    config_params=crf_model_arguments()
    # Required for choosing CRF network models.
    config_params['CRF_MODEL_ON']=True
    main(config_params)
