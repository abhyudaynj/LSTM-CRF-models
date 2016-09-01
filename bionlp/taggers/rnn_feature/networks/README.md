#CRF-LSTM Network Files 



## Folder Structure
    .
    |--network.py #         Contains the code to build a CRF-LSTM model with modeling of unary potentials.
                            It uses the CRFLayer class provided in crf_lstm_layer.py
    |--crf_lstm_layer.py #  Contains the CRFLayer class and a function to calculate the negative log likelihood. 
    |--dual_network.py   #  Contains the code to build a CRF-LSTM model with modeling of both unary and 
                            binary potentials. It uses the DualCRFLayer class provided in crf_dual_layer.py
    |--crf_lstm_layer.py #  Contains the DualCRFLayer class and a function to calculate the negative log likelihood. 
    |--approx_network.py #  Contains the code to build a Skip-Chain CRF-LSTM model with modeling of message groups.
                            It uses the constructApproximation function, to construct the approximate network for this model.
    |--crf_lstm_layer.py #  Contains the consrtuctApproximations function to construct the approximate Skip-Chain 
                            CRF network. No negative log likelihood is required for this model.
