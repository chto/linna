import sys
import os
import pickle
import time
outdir = sys.argv[1]
timein = sys.argv[2]
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
timesleep = timein.split(":")
timesleep = eval(timesleep[0])*3600+eval(timesleep[1])*60+eval(timesleep[2])
time.sleep(timesleep)
