from train import main
import datetime
from bionlp.utils.crf_arguments import crf_model_arguments
import bionlp.utils.monitoring as mo


if __name__ == '__main__':
    config_params = crf_model_arguments()
    # Required for choosing CRF network models.
    config_params['CRF_MODEL_ON'] = True
    config_params['emb1_size'] = 300
    config_params['epochs'] = 3
    config_params['patience'] = 10

    today = datetime.date.today().isoformat()
    for idx in range(10):
        config_params['monitoring-file'] = "data/logs/monitor_{0}_{1}.pkl".format(today, idx)
        mo.data = mo.get_init_data()
        main(config_params)
