

def get_cmap(N):
	'''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
	RGB color.'''
	import matplotlib.pyplot as plt
	import matplotlib.cm as cmx
	import matplotlib.colors as colors
	color_norm  = colors.Normalize(vmin=0, vmax=N-1)
	scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
	def map_index_to_rgb_color(index):
		return tuple(map(lambda x: int(x*255), scalar_map.to_rgba(index)))[0:3]
		#return scalar_map.to_rgba(index)
	return map_index_to_rgb_color


#import cv2
#start = time.clock()
#img = cv2.imread('00001.ppm')
#colors = {}
#for i in range(img.shape[0]):
#	for j in range(img.shape[1]):
#		pixel = (img[0][1][0], img[0][1][1], img[0][1][2])
#		if pixel not in colors:
#			colors[pixel] = 1
#		else:
#			colors[pixel] += 1
#print time.clock()-start

from PIL import Image
import time



#path = './'
def getSuperVoxels(path):
	start = time.clock()
	img = Image.open(path+'00001.ppm')
	#img = Image.open('/cs/vml3/mkhodaba/cvpr16/libsvx.v3.0/example/output_gbh/00/00001.ppm')
	print path+'00001.ppm'
	print img
	cs = img.convert('RGB').getcolors()
	print cs
	supervoxels = {}	
	opens = {}
	sums = {}
	counts = {}
	closed_colors = {}
	for i, c in cs:
		opens[c] = set()
		sums[c] = [0,0,0]
		counts[c] = 0
	for f in range(1,31):
		img = Image.open(path+'{0:05d}.ppm'.format(f))
		print '{0:05d}.ppm'.format(f)
		colors = img.convert('RGB').getcolors()
		for i,c in colors:
			if c not in opens.keys():
				opens[c] = set()
				sums[c] = [0,0,0]
				counts[c] = 0
			counts[c] += i	
		existing = {}
		for c in opens.keys():
			existing[c] = False
		for i in range(img.size[0]):
			for j in range(img.size[1]):
				color = img.getpixel((i,j))
				existing[color] = True
				opens[color].add((i,j,f))
				sums[color] = [sums[color][0]+i, sums[color][1]+j, sums[color][2]+f]
		for c in existing.keys():
			if existing[c] == False:			
				center = sums.pop(c)
				num = counts.pop(c)
				center = (center[0]/num, center[1]/num, center[2]/num)
				#supervoxels[(c,f)] = [opens.pop(c), center] #remove the supervoxel from open ones and add it to the final set
				supervoxels[c] = [opens.pop(c), center] #remove the supervoxel from open ones and add it to the final set
				closed_colors[c] = True
	for c in opens.keys():
		center = sums.pop(c)
		num = counts.pop(c)
		center = (center[0]/num, center[1]/num, center[2]/num)
		#supervoxels[(c,f)] = [opens.pop(c), center] #remove the supervoxel from open ones and add it to 
		supervoxels[c] = [opens.pop(c), center] #remove the supervoxel from open ones and add it to 

	return supervoxels

def saveSegment(segment, path, output, color):
	segment = sorted(segment, key=lambda x: x[2])
	open_frame = segment[0][2]
	img = Image.open(path+'{0:05d}.ppm'.format(open_frame))
	for x,y,f in segment:
		if open_frame != f:
			img.save(output+'{0:05d}.ppm'.format(open_frame))
			open_frame = f
			img = Image.open(path+'{0:05d}.ppm'.format(open_frame))

		img.putpixel((x,y), color)


level = 5
path = '/cs/vml3/mkhodaba/cvpr16/libsvx.v3.0/example/output_gbh/{0:02d}/'.format(level)
#path = '/cs/vml3/mkhodaba/cvpr16/libsvx.v3.0/example/output_gbh/00/'
orig = '/cs/vml3/mkhodaba/cvpr16/libsvx.v3.0/example/frames_ppm/'
output = '/cs/vml3/mkhodaba/cvpr16/libsvx.v3.0/example/mehran_gbh/{0}/{1}/'


supervoxels = getSuperVoxels(path)
print len(supervoxels)
print path
center_to_color = {}
centers = [ val[1] for key,val in supervoxels.iteritems() ]
colors = [ (key[0], key[1], key[2]) for key in supervoxels.keys() ]
for key, val in supervoxels.iteritems():
	center_to_color[val[1]] = (key[0], key[1], key[2])

from scipy.spatial import cKDTree
import numpy as np
centers_np = np.array(centers)
t = cKDTree(centers_np)


candidate_id = 7
candidate = centers[candidate_id]
threshold = 30



#neighbors = t.query_ball_point(candidate, threshold)
neighbors = t.query(candidate, 6)[1]
print neighbors
cmap = get_cmap(len(neighbors)+3)
print "COLOR: ", cmap(1)

segment = supervoxels[center_to_color[candidate]][0]
seg_path = '/cs/vml3/mkhodaba/cvpr16/libsvx.v3.0/example/mehran_gbh/{0}/'.format(level)
saveSegment(segment, orig,  seg_path, cmap(1))
saveSegment(segment, orig,  output.format(level, 0), cmap(1))

for i in range(len(neighbors)):
	print neighbors[i], centers[neighbors[i]]
	if candidate_id == neighbors[i]:
		continue
	
	
	segment = supervoxels[center_to_color[centers[neighbors[i]]]][0]
	#print output.format(level, i)
	saveSegment(segment, seg_path, seg_path, cmap(i+2))

#segment = supervoxels[center_to_color[centers[5]]][0]
#saveSegment(segment, orig, output.format(level, 1))
#segment = supervoxels[center_to_color[centers[5]]][0]
#saveSegment(segment, orig, output.format(level, 1))




