from mpi4py import MPI
import argparse, os, time
import numpy as np
import scipy as scipy
import pickle as pkl
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from helpers import get_bin_ranges, get_bins_from_intervals, load_psts

######################
##  INITIALIZE MPI  ##
######################

comm = MPI.COMM_WORLD
N = comm.Get_size()
rank = comm.Get_rank()
time0 = time.time()

#########################
##  Argument handling  ##
#########################

parser = argparse.ArgumentParser()

parser.add_argument('--gem_fname', type=str, default='gene_expr.npz',
    help='Filename of gene expression matrix located in data directory')
parser.add_argument('--pst_fname', type=str, default='pseudotime.txt',
    help='Filename of pseudotime array located in data directory')

parser.add_argument('--datdir', type=str, default='/projects/p31512/aeh581/gexpParPCA/data',
    help='Input data directory')
parser.add_argument('--outdir', type=str, default='/projects/p31512/aeh581/gexpParPCA/out2', 
    help='Output directory')
parser.add_argument("--pcadir", type=str, default="pca",
    help="Directory for output PCA results",)

parser.add_argument("--neval", type=int, default=1,
    help="# of eigenvalues")
parser.add_argument('--nboot', type=int, default=20,
    help='# of bootstrap samples')
parser.add_argument('--nsamp', type=int, default=20,
    help='# of null samples')

parser.add_argument("--nbins_goal", type=int, default=100,
    help="desired # of bins")
parser.add_argument("--bin_sz", type=int, default=1000,
    help="# of cells per bin")
parser.add_argument("--ov_frac", type=float, default=0.5,
    help="fraction of bin to overlap")

parser.add_argument("--row_normalize", action="store_true",
    help="Normalize gene expression matrix by cell.")

parser.add_argument("--seed", type=int,  default=0,
    help="random number seed")

args = parser.parse_args()

####################
##  Housekeeping  ##
####################

datdir = args.datdir
outdir = args.outdir

gem_fname = f'{datdir}/{args.gem_fname}'
pst_fname = f'{datdir}/{args.pst_fname}'

pcadir  = f'{outdir}/pca'
eigdir  = f'{pcadir}/eig'

neval = args.neval
nboot = args.nboot
nsamp = args.nsamp

# output file names
binsz_fname     = f"{pcadir}/binsize.txt"
bin_psts_fname  = f"{pcadir}/bin_psts.npy"
bin_cidxs_fname = f"{pcadir}/bin_cidxs.pkl"
seed_fname      = f"{pcadir}/pca_base_seed.txt"

# Set seed
if rank == 0:
    seed = args.seed if args.seed else np.random.randint(1000000)
    print(f"Using base seed {seed} with {N} processes.")
else:
    seed = None
seed = comm.bcast(seed, root=0)
np.random.seed(seed + rank)

# Create directories
if rank == 0:
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(pcadir, exist_ok=True)
    os.makedirs(eigdir, exist_ok=True)
    with open(seed_fname, 'w') as f:
        f.write(f"{seed}")


################################################################################################

#################
##  LOAD DATA  ##
#################

data_dict = None
if rank == 0:
    print("Loading data...")
    t0 = time.time()
    gexp_sp = scipy.sparse.load_npz(gem_fname).tocsr()
    t1 = time.time()
    print(f"Data loaded in {t1-t0:.3g} sec")

    # Perform any desired transformations only on rank 0.
    if args.row_normalize:
        gexp_sp = normalize(gexp_sp, norm='l1', axis=1, copy=False)

    data_dict = {
        "ncells": gexp_sp.shape[0],
        "ngenes": gexp_sp.shape[1],
        "sz_data": len(gexp_sp.data),
        "sz_indices": len(gexp_sp.indices),
        "sz_indptr": len(gexp_sp.indptr),
        "dtype_data": gexp_sp.data.dtype,
        "dtype_indices": gexp_sp.indices.dtype,
        "dtype_indptr": gexp_sp.indptr.dtype,
    }

# Broadcast data and save in each node
data_dict = comm.bcast(data_dict, root=0)
ncells = data_dict["ncells"]
ngenes = data_dict["ngenes"]
sz_data = data_dict["sz_data"]
sz_indices = data_dict["sz_indices"]
sz_indptr = data_dict["sz_indptr"]
dtype_data = data_dict["dtype_data"]
dtype_indices = data_dict["dtype_indices"]
dtype_indptr = data_dict["dtype_indptr"]

if rank == 0:
    print(f"ncells={ncells}")
    print(f"ngenes={ngenes}")

# RANK 0 SHARES MEMORY. 
# See: https://stackoverflow.com/questions/32485122/shared-memory-in-mpi4py

if rank == 0: 
    nbytes_data    = sz_data    * dtype_data.itemsize 
    nbytes_indices = sz_indices * dtype_indices.itemsize 
    nbytes_indptr  = sz_indptr  * dtype_indptr.itemsize 
else: 
    nbytes_data    = 0
    nbytes_indices = 0
    nbytes_indptr  = 0

# On rank 0, create the shared block; on others get a handle to it
win_data    = MPI.Win.Allocate_shared(nbytes_data,    dtype_data.itemsize,    comm=comm)
win_indices = MPI.Win.Allocate_shared(nbytes_indices, dtype_indices.itemsize, comm=comm)
win_indptr  = MPI.Win.Allocate_shared(nbytes_indptr,  dtype_indptr.itemsize,  comm=comm)

# create a numpy array whose data points to the shared mem
buf_data, itemsize_data       = win_data.Shared_query(0)
buf_indices, itemsize_indices = win_indices.Shared_query(0)
buf_indptr, itemsize_indptr   = win_indptr.Shared_query(0)

arr_data    = np.ndarray(buffer=buf_data,    dtype=dtype_data,    shape=(sz_data,))
arr_indices = np.ndarray(buffer=buf_indices, dtype=dtype_indices, shape=(sz_indices,))
arr_indptr  = np.ndarray(buffer=buf_indptr,  dtype=dtype_indptr,  shape=(sz_indptr,))

# Node 0 writes data
if comm.rank == 0: 
    arr_data[:]    = gexp_sp.data
    arr_indices[:] = gexp_sp.indices
    arr_indptr[:]  = gexp_sp.indptr

# wait in process rank 0 until process 1 has written to the array
comm.Barrier() 

# All nodes now construct a sparse matrix with data pointing to shared memory
gexp_sp = scipy.sparse.csr_matrix((arr_data, arr_indices, arr_indptr), shape=(ncells, ngenes), copy=False)

####################
##  PREPARE BINS  ##
####################

# All ranks load and sort all pseudotimes
psts_all = load_psts(pst_fname)       # load pseudotime
srt      = np.argsort(psts_all[:,1])  # get locations of cell indices ordered by pseudotime

ncells_neut = len(srt)

# Determine bin size and number of bins
if args.bin_sz > 0:
    bin_sz = args.bin_sz
else:
    bin_sz = int(ncells_neut / args.nbins_goal / (1 - args.ov_frac) + 1)
    bin_sz += bin_sz % 2  # make the value even

# Set the overlap
overlap = int(args.ov_frac * bin_sz)

# Get bin ranges
bin_ranges = get_bin_ranges(ncells_neut, bin_sz, overlap)
nbins = len(bin_ranges)

# Write bin size to file
if rank == 0:
    print(f"nbins={nbins}")
    print(bin_ranges)
    with open(binsz_fname, 'w') as bin_sz_file:
        bin_sz_file.write(str(bin_sz))

# Get set of bins to handle for each node 
cumsum = [0] + list(np.cumsum([nbins // N + (i < nbins % N) for i in range(N)]))
bidx0 = cumsum[rank]
bidx1 = cumsum[rank + 1] - 1

print(f"Node {rank} handling bins {bidx0} through {bidx1}")

# Each rank bins only its range
pst_bins    = get_bins_from_intervals(srt, bin_ranges, bidx0, bidx1)
bin_cidxs   = [np.array(psts_all[grp,0], dtype='int')  for grp in pst_bins]
bin_time    = np.array([np.mean(psts_all[grp,1])       for grp in pst_bins])
npst        = len(bin_cidxs)
assert npst == bidx1 + 1 - bidx0, "npst should equal bidx1 + 1 - bidx0"
print(f"Node {rank} has npst={npst}")

# Gather times and cidxs to Rank 0
all_bin_cidxs = comm.gather(bin_cidxs, root=0)
all_bin_time = comm.gather(bin_time, root=0)

if rank == 0:
    # Flatten lists as if all produced by one rank.
    all_bin_cidxs = [item for sublist in all_bin_cidxs for item in sublist]
    all_bin_time = [item for sublist in all_bin_time for item in sublist]
    print('Rank 0 saving bins')
    with open(bin_cidxs_fname, 'wb') as f:
        pkl.dump(all_bin_cidxs, f)

    np.save(bin_psts_fname, all_bin_time)


#########################################
##  PCA + Null and Bootstrap Sampling  ##
#########################################

cov_evals      = np.zeros([npst, neval])
cov_evecs      = np.zeros((npst, neval, gexp_sp.shape[1]))
cov_evals_shuf = np.zeros([nsamp, neval])
# cov_evecs_shuf = np.zeros((nsamp, neval, gexp_sp.shape[1]))
cov_evals_boot = np.zeros([nboot, neval])
# cov_evecs_boot = np.zeros((nboot, neval, gexp_sp.shape[1]))
pca            = PCA(n_components=neval)

for i, t in enumerate(range(bidx0, bidx1 + 1)):

    print(f'Rank {rank} binning {t}/{bidx1}')

    # Get gene expression of cells at pseduotime t
    gexpt = gexp_sp[bin_cidxs[i]].toarray()
    ncell, ngene = gexpt.shape

    # Run PCA and save evals and evecs for this bin into the cov_evals and cov_evecs arrays
    pca.fit(gexpt)
    cov_evals[i] = pca.explained_variance_
    cov_evecs[i] = pca.components_

    # Sample with replacement NSAMP times, saving results to shuf arrays
    for j in range(nsamp):
        gexp_shuf = np.array([gexpt[:,g][np.random.choice(ncell, ncell, replace=True)] for g in range(ngene)]).T
        pca.fit(gexp_shuf)
        cov_evals_shuf[j] = pca.explained_variance_
        # cov_evecs_shuf[j] = pca.components_
    
    # Save shuffle results
    np.save(f'{eigdir}/shuf_eval_t{t}.npy', cov_evals_shuf)
    # np.save(f'{eigdir}/shuf_evec_t{t}.npy', cov_evecs_shuf)

    # Sample with replacement NBOOT times, saving results to boot arrays
    for j in range(nboot):
        gexp_boot = gexpt[np.random.choice(ncell, ncell, replace=True)]
        pca.fit(gexp_boot)
        cov_evals_boot[j] = pca.explained_variance_
        # cov_evecs_boot[j] = pca.components_
    
    # Save bootstrapped results
    np.save(f'{eigdir}/boot_eval_t{t}.npy', cov_evals_boot)
    # np.save(f'{eigdir}/boot_evec_t{t}.npy', cov_evecs_boot)

print(f"Node {rank} finished after {time.time()-time0} seconds.")

# Gather all evecs and evals to arrays for rank 0 to save
all_cov_evals = comm.gather(cov_evals, root=0)
all_cov_evecs = comm.gather(cov_evecs, root=0)

if rank == 0:
    # Flatten lists as if all produced by one process.
    all_cov_evals = [item for sublist in all_cov_evals for item in sublist]
    all_cov_evecs = [item for sublist in all_cov_evecs for item in sublist]

    np.save(f'{eigdir}/dat_eval.npy', all_cov_evals)
    np.save(f'{eigdir}/dat_evec.npy', all_cov_evecs)
    print(f"Completed in {time.time()-time0} seconds.")
