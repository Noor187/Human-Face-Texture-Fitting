from scipy.io import loadmat, savemat
import numpy as np
from math import pow, sqrt
import itertools 
import numpy.matlib

Input = open('Inputs/Test3.obj','r')
Output=open('Output/Test3_Masked.ply','w')


def get_deep3d_vertices(meshFile):
	vertices_with_rgb_complete_list = []
	vertices_with_rgb_list = []
	counter = 1
	for i, line in enumerate(meshFile):
		if(line[0] == 'v'):
			vertices_with_rgb = line.strip('\n').split(" ")
			vertices_with_rgb_list.append(float(vertices_with_rgb[1]))
			vertices_with_rgb_list.append(float(vertices_with_rgb[2]))
			vertices_with_rgb_list.append(float(vertices_with_rgb[3]))
			vertices_with_rgb_list.append(int(vertices_with_rgb[4]))
			vertices_with_rgb_list.append(int(vertices_with_rgb[5]))
			vertices_with_rgb_list.append(int(vertices_with_rgb[6]))
			vertices_with_rgb_complete_list.append(vertices_with_rgb_list)
			vertices_with_rgb_list = []
		else:
			break
	return np.array(vertices_with_rgb_complete_list)

def write_ply(vertices,rgbs,faces,f):
	f.write("ply\nformat ascii 1.0\nelement vertex "+str(len(vertices))+"\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face "+str(len(faces))+"\nproperty list uchar int vertex_indices\nend_header\n")
	for i in range(0,len(vertices)):
		f.write(str(vertices[i][0])+" "+str(vertices[i][1])+" "+str(vertices[i][2])+" "+str(int(rgbs[i][0]))+" "+str(int(rgbs[i][1]))+" "+str(int(rgbs[i][2]))+"\n") 
	for i in range(0,len(faces)):  
		f.write("3 "+str(faces[i][0]-1)+" "+str(faces[i][1]-1)+" "+str(faces[i][2]-1)+"\n")
	f.close()

def get_predicted_texture(tex_Cp, distances, tex_B, tex_U,vertices):
	distances=np.matlib.repmat(distances, 3, 1)
	distances=np.transpose(distances)
	distances = np.array(list(itertools.chain(*distances)))

	weight1 = distances
	weight2 = np.array([distances]*199)
	weight2 = np.transpose(weight2)

	tex_Cp = np.array(list(itertools.chain(*tex_Cp)))

	tex_U = np.reshape(tex_U,np.shape(tex_U)[0])
	tex_lhs = np.transpose(np.multiply(tex_B, weight2))
	tex_rhs = np.multiply((tex_Cp - tex_U), weight1)
	tex_z = np.matmul(tex_lhs,tex_rhs)
	
	tex_Cb = np.matmul(tex_B,tex_z)+tex_U

	tex_C = np.multiply(tex_Cp,weight1) + np.multiply(tex_Cb, (1-weight1))
	tex_C = np.around(tex_C)
	tex_C = np.transpose(tex_C)

	return tex_C


texture_eigenspace = loadmat('TextureEigenSpace.mat')
tex_PC = texture_eigenspace['texPC'];
tex_MU = texture_eigenspace['texMU'];
faces = texture_eigenspace['bfm_tri'];


Face_mesh = get_deep3d_vertices(Input)

print("Calculating weights...")

distances = []
for i in range(0, len(Face_mesh[:,0:3])):
	vertice1 = Face_mesh[i,:]
	vertice2 = Face_mesh[8317,:]
	distances.append(sqrt(pow(vertice2[0]-vertice1[0],2) + pow(vertice2[1]-vertice1[1],2) + pow(vertice2[2]-vertice1[2],2)))

distances=np.array(distances)
distances=(distances-min(distances))/(max(distances)-min(distances))
distances=1-distances

print("Weights calculated...")


print("Predicting Ear and Neck Texture...")
Tex_Cp=Face_mesh[:,3:6]
Final_Texture= get_predicted_texture(Tex_Cp, distances, tex_PC, tex_MU,Face_mesh[:,0:3]) 
Final_Texture = np.reshape(Final_Texture, (len(Face_mesh),3))

print("Predicting Ear and Neck Texture Completed")


write_ply(Face_mesh[:,0:3],Final_Texture,faces,Output)







