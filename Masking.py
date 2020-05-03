import scipy.io
from scipy.io import loadmat, savemat
import itertools
import numpy as np
import math 
from math import pow, sqrt
import argparse
from copy import deepcopy


def Fitting_Texture(tex_Cp, confidence_alpha, tex_B, tex_U):
	Confidence_a = []
	for i in range(0,len(confidence_alpha)):
		Confidence_a.append(confidence_alpha[i])
		Confidence_a.append(confidence_alpha[i])
		Confidence_a.append(confidence_alpha[i])
	
	Confidence_A = np.array([Confidence_a]*199)
	Confidence_A = np.transpose(Confidence_A) # A of shape 160k * 199

	Confidence_a = np.array(Confidence_a)  # a of shape 160k

	tex_Cp = np.array(list(itertools.chain(*tex_Cp)))

	tex_U = np.reshape(tex_U,np.shape(tex_U)[0])
	tex_lhs = np.transpose(np.multiply(tex_B, Confidence_A)) # left hand side of Eq. 10
	tex_rhs = np.multiply((tex_Cp - tex_U), Confidence_a)    # right hand side of Eq. 10
	tex_z = np.matmul(tex_lhs,tex_rhs) #Eq. 10
	
	tex_Cb = np.matmul(tex_B,tex_z)+tex_U   #Eq. 11  Predicted texture from BFM eigeanspace
	
	tex_C = np.around(np.multiply(tex_Cp,Confidence_a) + np.multiply(tex_Cb, (1-Confidence_a))) #Eq.12  Final Masked Output Texture

	return tex_C ,tex_Cb,Confidence_a


def Calculate_Confidence(vertices_53490):	
	confidence_alpha = []
	vertices_len = len(vertices_53490)
	for i in range(0, vertices_len):
		Nose_vertice = vertices_53490[8317,:]   #vertex id 8317 belongs to tip of nose on face mesh
		confidence_alpha.append(sqrt(pow(Nose_vertice[0]-vertices_53490[i,0],2) + pow(Nose_vertice[1]-vertices_53490[i,1],2) + pow(Nose_vertice[2]-vertices_53490[i,2],2)))

	minimum = min(confidence_alpha)
	maximum = max(confidence_alpha)

	for i in range(0, vertices_len):
		confidence_alpha[i]=abs(((confidence_alpha[i]-minimum)/(maximum-minimum)) - 1);
	return 	confidence_alpha


def WriteConfidenceMask(meshVerts):

	fileToWrite=open('ConfidenceMask.obj','w')

	for i in range(0,len(meshVerts)):
		fileToWrite.write("v " +str(meshVerts[i][0])+" "+str(meshVerts[i][1])+" "+str(meshVerts[i][2])+" "+str(int(meshVerts[i][3]*255))+" "+str(int(meshVerts[i][4]*255))+" "+str(int(meshVerts[i][5]*255))+"\n") 
	
	ddfa_tri = open('Configs/ddfa_inverted_tri.txt','r')		# Read tri of 53215 face model
	for i, line in enumerate(ddfa_tri):
		fileToWrite.write(line)
	ddfa_tri.close()	
	fileToWrite.close()	

def get_obj_vertices(meshFile):
	meshFile=open(meshFile,'r')
	vertices_with_rgb_complete_list = []
	vertices_with_rgb_list = []
	for i, line in enumerate(meshFile):
		if(line[0] == 'v'):
			vertices_with_rgb = line.strip('\n').split(" ")
			vertices_with_rgb_list.append(float(vertices_with_rgb[1]))
			vertices_with_rgb_list.append(float(vertices_with_rgb[2]))
			vertices_with_rgb_list.append(float(vertices_with_rgb[3]))
			if (len(vertices_with_rgb)==4):
				vertices_with_rgb_list.append(int(0))
				vertices_with_rgb_list.append(int(0))
				vertices_with_rgb_list.append(int(0))
			else:
				vertices_with_rgb_list.append(float(vertices_with_rgb[4]))
				vertices_with_rgb_list.append(float(vertices_with_rgb[5]))
				vertices_with_rgb_list.append(float(vertices_with_rgb[6]))				
			vertices_with_rgb_complete_list.append(vertices_with_rgb_list)
			vertices_with_rgb_list = []
		elif(line[0] == 'f'):
			break
		else:
			continue
	return (np.array(vertices_with_rgb_complete_list))	

def map_to_53490(Facemesh_53490, Facemesh_53215, mapping):
	i = 0
	while i < len(mapping):
		Facemesh_53490[mapping[i]-1][3] = float(Facemesh_53215[i][3])
		Facemesh_53490[mapping[i]-1][4] = float(Facemesh_53215[i][4])
		Facemesh_53490[mapping[i]-1][5] = float(Facemesh_53215[i][5])
		i+=1
	return (np.array(Facemesh_53490))

def map_to_53215(Facemesh_53215, Facemesh_53490, mapping):
	#Reshaping (160k * 1) vector to orignal shape (53490 * 3)
	Facemesh_53490 = np.transpose(Facemesh_53490)
	Facemesh_53490 = np.reshape(Facemesh_53490, (-1,3))  

	Facemesh=deepcopy(Facemesh_53215)
	#Mapping from BFM 53490 model to 53215 model
	i = 0
	while i < len(mapping):
		Facemesh[i][3] = float(Facemesh_53490[mapping[i]-1][0])
		Facemesh[i][4] = float(Facemesh_53490[mapping[i]-1][1])
		Facemesh[i][5] = float(Facemesh_53490[mapping[i]-1][2])
		i+=1	
	return 	Facemesh

def write_obj_with_colors(meshVerts, fileToWrite):

	fileToWrite=open(fileToWrite,'w')
	for i in range(0,len(meshVerts)):
		fileToWrite.write("v " +str(meshVerts[i][0])+" "+str(meshVerts[i][1])+" "+str(meshVerts[i][2])+" "+str(int(meshVerts[i][3]))+" "+str(int(meshVerts[i][4]))+" "+str(int(meshVerts[i][5]))+"\n") 
	
	ddfa_tri = open('Configs/ddfa_inverted_tri.txt','r')		# Read tri of 53215 face model
	for i, line in enumerate(ddfa_tri):
		fileToWrite.write(line)
	ddfa_tri.close()	
	fileToWrite.close()	



def main(args):	

	##Getting paths for input and output meshes
	Input_path=args.input
	path_prefix=Input_path.split('.obj')[0]

	Final_Output_path=path_prefix+'_FinalOutput.obj'
	Predicted_texture_path=path_prefix+'_BFMTexture.obj'

	input_face_mesh=get_obj_vertices(Input_path)

	texture_eigenspace = scipy.io.loadmat('Configs/01_MorphableModel.mat')
	tex_PC = texture_eigenspace['texPC'];
	tex_MU = texture_eigenspace['texMU'];
	BFM_mapping = scipy.io.loadmat('Configs/BFM_exp_idx.mat')
	bfm_to_exp = BFM_mapping['trimIndex'].tolist() 
	bfm_to_exp = list(itertools.chain.from_iterable(bfm_to_exp))

	deep3d_mesh = 'Configs/deep3d.obj'
	deep3d_mesh = get_obj_vertices(deep3d_mesh)


	# Mapping face with 53215 vertices to BFM facemodel with 53490 vertices
	textured_53490 = map_to_53490(deep3d_mesh, input_face_mesh, bfm_to_exp)

	vertices_53490 = textured_53490[:,0:3]
	rgbs_53490 = (textured_53490[:,3:6]*255).astype(int)


	print("Fitting Texture B.3. starts...")

	# Calculating matrix alpha in Eq.9 which represents the confidence of each vertex
	confidence_alpha=Calculate_Confidence(vertices_53490)


	# Predicting Texture C using eigean space and confidence value, resultant of Eq. 12
	final_texture, predicted_texture, confidence_mask= Fitting_Texture(rgbs_53490, confidence_alpha, tex_PC, tex_MU)     

	print("Fitting Texture B.3. Ends...")


	# Mapping Final textured BFM facemodel with 53490 vertices to orignal face with 53215 vertices
	predicted_BFM_texture = map_to_53215(input_face_mesh, predicted_texture, bfm_to_exp)
	final_texture = map_to_53215(input_face_mesh, final_texture, bfm_to_exp)


	# Write face mesh in obj file format
	write_obj_with_colors(final_texture, Final_Output_path)
	write_obj_with_colors(predicted_BFM_texture, Predicted_texture_path)	
	

	# To visualize the confidence weights, write confidence mask 
	confidence_mask = map_to_53215(input_face_mesh, confidence_mask, bfm_to_exp)	
	WriteConfidenceMask(confidence_mask)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='Meshes/Test1.obj',
                        help='path to .obj (ddfa) file')
    args = parser.parse_args()
    main(args)