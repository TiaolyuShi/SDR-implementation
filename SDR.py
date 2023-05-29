# Import packages.
import cvxpy as cp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
#Utility functions
def is_psd(matrix):
    """Check whether a given matrix is PSD to numpy."""
    try:
        _ = np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False
def nearest_psd(matrix):
    """Find the nearest positive-definite matrix to input.

    Numpy can be troublesome with rounding values and stating
    a matrix is PSD. This function is thus used to enable the
    decomposition of result matrices

    (altered code from) source:
    https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd
    """
    if is_psd(matrix):
        return matrix
    # false positive warning; pylint: disable=assignment-from-no-return
    spacing = np.spacing(np.linalg.norm(matrix))
    identity = np.identity(len(matrix))
    k = 1
    while not is_psd(matrix):
        min_eig = np.min(np.real(np.linalg.eigvals(matrix)))
        matrix += identity * (- min_eig * (k ** 2) + spacing)
        k += 1
    return matrix

# Generate a random graph. with nodes and edges.
nodes = 15  # 10 nodes
edges =  30 # 20 edges
seed = 2  # seed random number generators for reproducibility

#G = nx.gnm_random_graph(nodes, edges, seed=seed)
G=nx.erdos_renyi_graph(nodes,0.5)
pos = nx.spring_layout(G, seed=seed)  # Seed for reproducible layout
nx.draw_networkx(G, with_labels=True)
plt.show()

# Define and solve the CVXPY problem.
adjacent = nx.adjacency_matrix(G).toarray()
# Set up the semi-definite program.

#brute force algorithm to get the maxcut
brute_start_time=time.time()
max=0
for i in range(2**nodes):
    binary = bin(i)[2:].zfill(nodes)
    binary = np.array(list(binary))
    binary = binary.astype(int)
    binary[binary == 0] = -1
    binary = binary.reshape(nodes, 1)
    cut = 0.25 * np.sum(np.multiply(adjacent, 1 - binary @ binary.T))
    if cut > max:
        max = cut
print(max)
brute_end_time=time.time()
print("brute force algorithm time:",brute_end_time-brute_start_time)

#SDP 
X = cp.Variable((nodes, nodes),symmetric=True) 
cut = 0.25 * cp.sum(cp.multiply(adjacent, 1 - X))
constraints = [X >> 0]
constraints+=[cp.diag(X)==1]
problem = cp.Problem(cp.Maximize(cut), constraints)
# Solve the program.
problem.solve()
#compute the eigenvalues of X
# Print result.
print("The optimal value is", problem.value)
print("A solution X is")
print(X.value)
# choelesky decomposition

#GW rounding init
epoch=10000
sum=0
val=np.zeros(epoch)
#time
GW_start_time=time.time()
for i in range(epoch):

    #cholesky decomposition
    X_new=nearest_psd(X.value)
    eig_new=np.linalg.eigvals(X_new)
    #print(eig_new)
    V = np.linalg.cholesky(X_new)
    #print("V is\n")
    #print(V)

    random_norm_vec = np.random.normal(size=nodes)
    random_norm_vec = random_norm_vec/np.linalg.norm(random_norm_vec,2)

    y=np.empty(nodes)
    #GW rounding
    for j in range(nodes):
        
        y[j]=np.sign(np.dot(V[j,:],random_norm_vec))

    #print("y is\n")
    #print(y)
    #Yij=yi*yj
    Y=np.outer(y,y)
    #print("Y is\n")
    #print(Y)

    #compute the cut

    cut = 0.25 * cp.sum(cp.multiply(adjacent, 1 - Y))
    #print(cut.value)
    val[i]=cut.value
    sum+=cut.value
GW_end_time=time.time()

#GW time
print("GW time:",(GW_end_time-GW_start_time)/epoch)
#plot val versus epoch
plt.hist(val)
plt.show()

print("accelaration rate is",(brute_end_time-brute_start_time)*epoch/(GW_end_time-GW_start_time))
print(f"averaging cut is {sum/epoch},max is {max},ratio is {(sum/epoch)/max} ")

