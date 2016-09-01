import os,json
import argparse

def get_all_files(rootdir,filetype):
    filenames=[]
    for root, subFolders, files in os.walk(rootdir):
        for fname in files:
            if filetype:
                if filetype =='-1':
                    if '.' not in fname:
                        filenames.append(os.path.join(root,fname))
                elif fname.endswith(filetype):
                    filenames.append(os.path.join(root,fname))
            else:
                filenames.append(os.path.join(root,fname))
    return filenames

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input',dest='inputdir',type= str,help ='location of input dir.')
    parser.add_argument('-o','--output',dest='output',type= str,help ='location of output file.')
    parser.add_argument('-e','--extension',dest='extension',type= str,default=None,help ='Extension of the files e.g. \'.txt\'.If you want files with no extensions use -1')

    args= parser.parse_args()
    if not (args.inputdir and args.output):
        parser.error('both input dir and output file are required params')
    with open(args.output,'w') as fout:
        file_list=get_all_files(args.inputdir,args.extension)
        fout.write('\n'.join(file_list))
