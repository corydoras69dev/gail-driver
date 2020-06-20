import sys
import subprocess
import os
import glob
import shutil

files = glob.glob(r'./data/models/tf_*.ckpt')
files.extend(glob.glob(r'./data/models/tf_*.ckpt.meta'))
files.extend(glob.glob(r'./data/models/policy_gail*.h5'))
files.extend(glob.glob(r'./data/models/smpls_*.pkl'))

for file in files:
    print('removing .. ', file)
    os.remove(file)

if os.path.exists('iteration.txt'):
    os.remove('iteration.txt')

argv = sys.argv
cmd = "python ./scripts/train_gail_model.py "

iter = -1

while True:
    args = cmd
    for i in range(1, len(argv)):
        args = args + " " + argv[i]

    if iter > 0:
        args = args + " --start_iter " + str(iter)

    os.system(args)
    if os.path.exists('iteration.txt'):
        print("SEGMENTATION FAULT DETECTED<" + args + ">")
        f = open("iteration.txt", "r")
        line = f.readline()
        line.strip()
        iter = int(line) - 2
        print("iteration=", iter)
        line = f.readline()
        print("directory=<" + line + ">")
        f.close()
        files = glob.glob(line + '/tf_*.ckpt')
        files.extend(glob.glob(line + '/tf_*.ckpt.meta'))
        files.extend(glob.glob(line + '/policy_gail*.h5'))
        files.extend(glob.glob(line + '/smpls_*.pkl'))
        for file in files:
            print('copying .. ', file)
            shutil.copy(file, "./data/models")

    else:
        break

print("terminating...")
print("launcher DONE!")


