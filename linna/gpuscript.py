import sys
import os
import pickle
outdir = sys.argv[1]
if not os.path.isdir(outdir):
    os.makedirs(outdir)
jobid = os.getenv('SLURM_JOB_ID')
nodename = os.getenv('SLURMD_NODENAME')
info = {'jobid': jobid, "nodename": nodename}
outfile = os.path.join(outdir, "gpunodeinfo.pkl")
if os.path.isfile(outfile):
    os.remove(outfile)
with open(outfile, 'wb') as f:
    pickle.dump(info, f)
while(True):
    pass 

