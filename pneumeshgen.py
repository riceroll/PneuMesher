import sys
import pyvista as pv
import tetgen
import numpy as np
pv.set_plot_theme('document')


# examples
# run pneumeshgen.py bottle_simple

filename = sys.argv[1]
show = bool(int(sys.argv[2])) if len(sys.argv) > 2 else False


sphere = pv.Sphere()
sphere = pv.read('./{}.obj'.format(sys.argv[1]))
sphere.scale([0.1, 0.1, 0.1])
tet = tetgen.TetGen(sphere)
tet.tetrahedralize(order=1, mindihedral=7, minratio=1.2)
grid = tet.grid
if show:
    grid.plot(show_edges=True)

# get cell centroids
cells = grid.cells.reshape(-1, 5)[:, 1:]
cell_center = grid.points[cells].mean(1)

# extract cells below the 0 xy plane
mask = cell_center[:, 2] < 5
cell_ind = mask.nonzero()[0]
subgrid = grid.extract_cells(cell_ind)

# advanced plotting
plotter = pv.Plotter()
plotter.add_mesh(subgrid, 'lightgrey', lighting=True, show_edges=True)
plotter.add_mesh(sphere, 'r', 'wireframe')
plotter.add_legend([[' Input Mesh ', 'r'],
                    [' Tessellated Mesh ', 'black']])
if show:
    plotter.show()

nChannel = 3

points = np.array(grid.points)
A = np.zeros([len(points), len(points)])
pointsOfChannels = [set() for ic in range(nChannel)]

for cell in cells:
    for i in range(4):
        for j in range(4):
            if i == j:
                pass
            A[cell[i], cell[j]] = A[cell[j], cell[i]] = -1
A0 = A.copy()

def idsNonFullPoints(A):
    return np.arange(len(A))[(A == -1).any(1)]

def idsNonFullPointsOfIC(ic, A):
    return np.arange(len(A))[(A == ic).any(1) * (A == -1).any(1)]

def idsNonFullChannels(A, nChannel):
    ics = []
    for ic in range(nChannel):
        ic += 1
        if len(idsNonFullPointsOfIC(ic, A)) > 0:
            ics.append(ic)
    return ics

def existNonAssignedEdge(A):
    return (A == -1).any()

def growEdge(ip, ic, A):
    assert(ic != 0)
    idsConnectedPoints = np.arange(len(A))[A[ip] == -1]
    ipConnect = np.random.choice(idsConnectedPoints)
    A[ip, ipConnect] = A[ipConnect, ip] = ic
    return A

for ic in range(nChannel):
    ic += 1
    ip = np.random.choice(idsNonFullPoints(A))
    A = growEdge(ip, ic, A)    # assign an available edge connected to ip with ic

while existNonAssignedEdge(A):
    ic = np.random.choice(idsNonFullChannels(A, nChannel))
    ip = np.random.choice(idsNonFullPointsOfIC(ic, A))
    A = growEdge(ip, ic, A)

A = np.tril(A, -1)
    
edgesOfChannels = [
    np.stack(np.where(A == ic + 1), 1)
    for ic in range(nChannel)
]

import polyscope as ps

def exportChannelData(points, edges, fileDir = "out.json"):
    pointsOut = []
    edgesTuple = []
    for edge in edges:
        p0 = tuple(points[edge[0]])
        p1 = tuple(points[edge[1]])
        if p0 not in pointsOut:
            pointsOut.append(p0)
        if p1 not in pointsOut:
            pointsOut.append(p1)
        edgesTuple.append((p0, p1))
    
    edgesOut = []
    for edge in edgesTuple:
        ip0 = pointsOut.index(edge[0])
        ip1 = pointsOut.index(edge[1])
        edgesOut.append((ip0, ip1))
    
    data = {
        'points': pointsOut,
        'edges': edgesOut
    }
    return data
    # import json
    # js = json.dumps(data)
    
    # with open(fileDir, 'w') as oFile:
    #     oFile.write(js)

datas = []
for ic in range(nChannel):
    data = exportChannelData(points, edgesOfChannels[ic], fileDir="{}_{}.json".format(filename, ic))
    datas.append(data)
    import json
    js = json.dumps(datas)
    with open("{}.json".format(filename), 'w') as oFile:
      oFile.write(js)

    ps.register_curve_network('channel_{}'.format(ic), points, edgesOfChannels[ic])


e = np.concatenate(edgesOfChannels).tolist()
# points *= 0.24
points *= 0.8

edgeChannel = []
e = []
for ic in range(nChannel):
    es = edgesOfChannels[ic]
    for edge in es:
        e.append(edge)
        edgeChannel.append(ic)


v = points

l = [np.linalg.norm(v[edge[0]]-v[edge[1]]) for ie, edge in enumerate(e)]


v = [ [vertex[0], vertex[2], vertex[1]] for vertex in v]

e = np.array(e, dtype=int).tolist()

lM = 1.5
lm = lM * 0.7
edgeActive = [1 if lm <= ll <= lM else 0  for ll in l]

lMax = [lM if ea else l[i]  for i, ea in enumerate(edgeActive)]

maxContraction = [ 1 - l[i] / lMax[i] if ea else 0 for i, ea in enumerate(edgeActive)]
maxContraction = [ [0, 0.1, 0.2, 0.3][np.argmin([abs(mc - 0), abs(mc - 0.1), abs(mc - 0.2), abs(mc - 0.3)])]  for mc in maxContraction]


# import matplotlib.pyplot as plt
# n, bins, patches = plt.hist(x=l, bins='auto', color='#0504aa',
#                             alpha=0.7, rwidth=0.85)
# plt.show()


data = {
    'v': v, 
    'e': e,
    'edgeChannel': edgeChannel,
    'lMax': lMax,
    'edgeActive': edgeActive,
    'maxContraction': maxContraction
}

import json
js = json.dumps(data)
with open('{}_web.json'.format(filename), 'w') as oFile:
    oFile.write(js)


if show:
  try:
      ps.init()
  except:
      pass
      
  ps.show()





#
#
#
#
#
#
#
# adjacency_matrix:
#     0: no direct connection
#     -1: direct connection
#     1: channel 1
#     2: channel 2
#     3: channel 3
#
# edge_list = [(a,b), (c,d), ....]
# edges_unvisited = ...
# vertices_not_done = [0 , 1, 2, 3]
# vertices_assigned_channel =
#
# choose a vertex having no assigned edge but assigned channel(can specify which channel)
# get available channels
# get unassigned edges
# choose one edge
# assign the channel
#
#
# function: choose a vertex grow an edge with a channel
#
# with certain probability choose a non-full channel
#     choose a non-full vertex of the channel
#     grow an edge with the channel
#
#
#
#
#
#







