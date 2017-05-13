Python version 3.4

Python Packages

        NLTK
        Theano 0.9.0
        Lasagne 0.2.dev1 (via pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip)
        Numpy
        Scipy
        Gensim
        tqdm

For Full Functionality

        GPU Enabled machine with NVCC compiler.
        MetaMap Server
        
Further instructions

        1. run 
        
        import nltk
        nltk.download()
        
        in a python shell and follow the steps described here
        
        http://stackoverflow.com/questions/4867197/failed-loading-english-pickle-with-nltk-data-load
        
        to have the necessary data available
        
        2. download german.model from 
        https://tubcloud.tu-berlin.de/public.php?service=files&t=dc4f9d207bcaf4d4fae99ab3fbb1af16
        (linked from http://devmount.github.io/GermanWordEmbeddings/)
        as the word-vec to initialize the weights. 
        Then edit `mld` in dependency.json to point to this file
