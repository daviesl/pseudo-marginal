import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

device_name = sys.argv[1]  # Choose device from cmd line. Options: gpu or cpu
nparticles = int(sys.argv[2]) #, int(sys.argv[2]))
if device_name == "gpu":
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"

def tf_int_diff(a,firstelcomp):
    """
    Diff along axis 0
    The first element of the input array is compared against firstelcomp.
    Output is same dimensions as a.
    """
    return tf.concat(axis=0,values=[tf.convert_to_tensor([a[0]-firstelcomp]),a[1:]-a[:-1]])

def cond(counts,offsets,idx,n,i,iters):
    return tf.less(i,iters)

def body(counts,offsets,idx,n,i,iters):
    global nparticles 
    int_zero = tf.constant(0, dtype=tf.int32)
    indices = tf.get_variable("indices",shape=[nparticles],dtype=tf.int32,initializer=tf.initializers.constant(0))
    countdown_mask = tf.greater(counts-i,int_zero)
    indices = tf.scatter_update(indices,tf.boolean_mask(offsets-i,countdown_mask),tf.boolean_mask(idx,countdown_mask),use_locking=None)
    tf.get_variable_scope().reuse_variables() # Reuse variables
    with tf.control_dependencies([indices]):
        return (counts,offsets,idx,n,i+1,iters)

def systematic_resample_tf(weights):
    global nparticles
    C = tf.cast(tf.math.cumsum(weights),tf.float64) * tf.constant(nparticles,dtype=tf.float64) - tf.random.uniform((1,),dtype=tf.float64)
    #C = tf.math.cumsum(weights) * tf.constant(nparticles,dtype=tf.float64) - 0.5
    floor_C = tf.cast(tf.floor(C),tf.int32)
    diff_floor_C = tf_int_diff(floor_C,tf.constant(-1,dtype=tf.int32))
    nonzero_mask = tf.not_equal(diff_floor_C,tf.constant(0,dtype=tf.int32))
    resample_idx = tf.boolean_mask(tf.range(nparticles),nonzero_mask)
    # compute counts for resampled points
    resample_count = tf.boolean_mask(diff_floor_C,nonzero_mask)
    resample_tensor_offset = tf.boolean_mask(floor_C,nonzero_mask)
    # repeat elements in resample_idx by counts in resample_count
    # while loop that will scatter update a tensor
    iters = tf.reduce_max(resample_count)
    i=tf.constant(0)
    #rc_, ro_, ri_, n_,i_,iters_ = tf.while_loop(cond,body,[resample_count,resample_tensor_offset,resample_idx,nparticles,i,iters])
    #with tf.control_dependencies([i_,rc_,ro_,ri_,n_,iters_]):
    res=tf.while_loop(cond,body,[resample_count,resample_tensor_offset,resample_idx,nparticles,i,iters])
    indices= tf.get_variable("indices",[nparticles],dtype=tf.int32,initializer=tf.initializers.constant(0))
    with tf.control_dependencies(res):
        return res,indices

with tf.device(device_name):
    random_matrix = tf.sigmoid(tf.random_uniform(shape=(nparticles,), minval=-10, maxval=3))
    #random_matrix = tf.sigmoid(tf.linspace(num=nparticles, start=-10., stop=3.))
    weights = tf.cast(random_matrix/tf.math.reduce_sum(random_matrix),tf.float64)
    i_,indices = systematic_resample_tf(weights)
    weights2 = tf.gather(weights,indices)

startTime = datetime.now()
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as session:
#with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
    init = tf.global_variables_initializer()
    result = session.run(init)
    if False:
        result_randmat = session.run(random_matrix)
        result_weights = session.run(weights)
        result_C = session.run(C)
        result_floorC = session.run(floor_C)
        result_diffC = session.run(diff_floor_C)
        result_nonzeromask = session.run(nonzero_mask)
        result_resampidx = session.run(resample_idx)
        result_resampcnt = session.run(resample_count)
        result_resampoff = session.run(resample_tensor_offset)
        result_iters = session.run(iters)
        result_finaliters = session.run(i_)
        result_indices = session.run(indices)
        print("randmat",result_randmat)
        print("weights",result_weights)
        print("C = cumsum*n - unif(0,1):",result_C)
        print("floor C",result_floorC)
        print("Diff floor C",result_diffC)
        print("Non Zero Mask",result_nonzeromask)
        print("Resamp Idx",result_resampidx)
        print("Resamp Counts",result_resampcnt)
        print("Resamp Offset       ",result_resampoff)
        print("Iters = max count",result_iters)
        print("Indices",result_indices)
        print("Final Iters after while loop",result_finaliters)
    result_finaliters = session.run(i_)
    result_weights2 = session.run(weights2)
    print("Final Weights",result_weights2)

# It can be hard to see the results on the terminal with lots of output -- add some newlines to improve readability.
print("\n" * 5)
print("Shape:", nparticles, "Device:", device_name)
print("Time taken:", datetime.now() - startTime)

print("\n" * 5)

def py_stratified_resample(weights):
    n = weights.shape[0]
    indices = np.zeros_like(weights)
    C = np.cumsum(weights)
    u0, j = np.random.uniform(), 0
    for i in range(n):
        u = (u0+i)/n
        while u > C[j]:
            j+=1
        indices[i] =  j
    return indices

startTime = datetime.now()
w = np.random.uniform(0.,1.,nparticles)
w2 = py_stratified_resample(w)
print(w2)
print("Shape:", nparticles)
print("Time taken:", datetime.now() - startTime)


if False:
    print("unused")
    #int_zero = tf.constant(0, dtype=tf.int32)
    #resample_indices = tf.zeros_like(weights)
    # We're going to scatter using indices from the tensor offset
    # m = tf.reduce_max(resample_count)
    # by concatenating m tensors that 
    #scatteridx = tf.concat([tf.boolean_mask(resample_tensor_offset + i,tf.greater_than(resample_count-i,int_zero)) for i in range(m)])
    #
    #u0 = np.random.uniform()
    #cumsum = tf.math.cumsum(weights) * shape - u0
    #changes, idx = tf.unique(tf.floor(cumsum))
    #diff_changes = tf_int_diff(changes,-1)
    #tile_operation = repeat_by(idx,tf.reshape(diff_changes,[-1]))
    #dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
    #sum_operation = tf.reduce_sum(dot_operation)

