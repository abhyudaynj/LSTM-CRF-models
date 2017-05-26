export THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=cuda,force_device=True,exception_verbosity=high

echo "Started on $(date)"
python train_crf_rnn.py -i data/converted/GermEval2014_complete_data/GermEval2014_file-list.txt -model TestModel.pkl -e1s 300
echo "Finished on $(date)"