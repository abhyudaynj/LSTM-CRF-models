import argparse

def deploy_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('-model','--model',dest='model',type= str, default='None',help ='location of pickled model file. default None')
    parser.add_argument('-i','--input',dest='input',type= str, default='None',help ='location of input file list.')
    parser.add_argument('-o','--output',dest='output',type= str, default='None',help ='location of output file. default None')
    parser.add_argument('-d','--outputdir',dest='outputdir',type= str, default='None',help ='location of output directory for annotations. default None')
    parser.add_argument('-n','--noeval',dest='noeval',type= int, default=1 ,help ='Do not run evaluation on the deployed dataset. This is useful when you do not have gold standard labels for the input text.0:Evaluation, 1:No Evaluation. Default 1')

    args = parser.parse_args()
    if args.input=='None':
        parser.error("Input is a required argument")
    elif args.model=='None':
        parser.error("Model is a required argument")
    elif args.output=='None' and  args.outputdir=='None' and  args.noeval:
        parser.error("Please provide either an output or set noeval to 0")
    deploy_params = vars(args)
    deploy_params['noeval']=bool(deploy_params['noeval'])
    return deploy_params



def default_arguments():
    DEFAULT_DUMP_DIR ='models/'
    #parsing arguments
    parser = argparse.ArgumentParser()

    # ------------------- Common Arguments ------------------
    parser.add_argument('-deploy','--deploy',dest='deploy',type=int, default=0, help='prepares the tagger for deployment. This will override all testing parameters, and use only training params. default is 0 and means a cross validation run')
    parser.add_argument('-r','--data-refresh',dest='data-refresh',type= int, default=1 ,help ='Use cached data, or reprocess/refresh the Dataset ? Default is 1: reprocess the data')
    parser.add_argument('-i','--input',dest='input',type= str, default='None',help ='location of input file list.')
    parser.add_argument('-cv','--cross-validation',dest='cross-validation',type= int, default=0 ,help ='Cross Validation run. 0 off, 1 on. Default 0')
    parser.add_argument('-f','--folds',dest='folds',type= int, default=10,help ='Number of cross validation folds. default 10')
    parser.add_argument('-ev','--extra-vocab',dest='extra-vocab',type= int, default=0 ,help ='Enable extra vocabulary from different corpuses provided in the depedency.json file. default is 0. 0:off 1: extra file -1: All available')
    parser.add_argument('-model','--model',dest='model',type= str, default='None',help ='location of parameter pickle  output in a file. default None')
    parser.add_argument('-bio','--bio',dest='bio',type= int, default=0 ,help ='Add BIO tags for CRF. default 0, off')
    parser.add_argument('-err','--error-analysis',dest='error-analysis',type= str, default='None',help ='location of Dump test output in a file. default None')
    parser.add_argument('-d','--dev',dest='dev',type= int, default=20,help ='percentage of training data that will be used for dev set. default 20')
    parser.add_argument('-umls','--umlssemantictype',dest='umls',type= int, default=0 ,help ='Should UMLS semantic type information be incorporated. Default is 0, off')
    parser.add_argument('-log','--log-file',dest='log-file',type= str, default=DEFAULT_DUMP_DIR+'temp_nn.log' ,help ='Log file that should be used.')
    parser.add_argument('-b','--batch-size',dest='batch-size',type= int, default=32 ,help ='Batch size. Default is 32')
    parser.add_argument('-n','--epochs',dest='epochs',type= int, default=10 ,help ='Number of epochs for training. default 10')
    parser.add_argument('-l','--maxlen',dest='maxlen',type= int, default=50 ,help ='Maximum Length for padding. default 50')
    parser.add_argument('-p','--patience',dest='patience',type= int, default=10,help ='Stop if the validation accuracy has not increased in the last n iterations.Off if 0. default 10')
    parser.add_argument('-ptm','--patience-mode',dest='patience-mode',type= int, default=0,help ='Patience criterion is applied to strict-f1 or accuracy. 0 is default, accuracy. 1 is strict-f1')
    parser.add_argument('-lr','--learning-rate',dest='learning-rate',type= float, default=0.1,help ='learning rate. default 0.1')
    parser.add_argument('-s','--shuffle',dest='shuffle',type= int, default=0,help ='Shuffle entire dataset. By default 0, means only shuffling the training and dev datasets.')
    parser.add_argument('-tp','--tp',dest='training-percent',type= int, default=100,help ='Percentage of training data used. default is 100')

    return parser

def default_model_arguments(parser):
    # -------------------  Common Model Specific Arguments ------------------------
    parser.add_argument('-w','--word2vec',dest='word2vec',type= int, default=0 ,help ='Initialize the Network with wordvec embeddings if 1. else random initialization. default 0')
    parser.add_argument('-n1','--noise1',dest='noise1',type= float, default=0.50 ,help ='Dropout Noise for first layer. Default is 0.50')
    parser.add_argument('-f1','--feature1',dest='feature1',type= int, default=5 ,help ='Dimensionality of the feature embedding layer. Default is 5. 10 will be added if UMLS is on')
    parser.add_argument('-h1','--hidden1',dest='hidden1',type= int, default=50 ,help ='Dimensionality of the first hidden layer. Default is 50')
    parser.add_argument('-h2','--hidden2',dest='hidden2',type= int, default=0 ,help ='Dimensionality of the first hidden layer. 0 is off. Default is zero')
    parser.add_argument('-e1','--emb1',dest='emb1',type= int, default=1,help ='Should the word2vec vectors be further trained. default 1')
    parser.add_argument('-e2','--emb2',dest='emb2',type= int, default=0 ,help ='Number of dimension of extra embedding layer. off if 0. default is 0')

    return parser

def crf_model_arguments():
    parser=default_arguments()
    parser=default_model_arguments(parser)

    parser.add_argument('-m','--momentum',dest='momentum',type= int, default=1 ,help ='Momentum (only dual network) : None 0, Momentum 1, Nesterov Momentum 2. default 1')
    parser.add_argument('-l2crf','--l2crfcost',dest='l2crf',type= float, default=0.1,help ='l2 for CRF params. Only used in mode -1. default 0.1')
    parser.add_argument('-l2','--l2cost',dest='l2',type= float, default=0.0,help ='Add l2 penalty. default 0.0')
    parser.add_argument('-l1','--l1cost',dest='l1',type= float, default=0.0,help ='Add l2 penalty. default 0.0, off')

    parser.add_argument('-mode','--mode',dest='mode',type=int, default =1,help='Mode of structured inference. Default is 1 : Modeling unary and binary potentials with neural nets. 1: Approximating Messages -1 : Only modeling the unary potential with neural nets.')
    args = parser.parse_args()
    if args.input=='None':
        parser.error("Input is a required argument")
    params = vars(args)

    # NO document mode yet for CRF. 
    params['document']=False

    return params
