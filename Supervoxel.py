from PIL import Image
import math

def delete_module(modname, paranoid=None):
    from sys import modules
    try:
        thismod = modules[modname]
    except KeyError:
        raise ValueError(modname)
    these_symbols = dir(thismod)
    if paranoid:
        try:
            paranoid[:]  # sequence support
        except:
            raise ValueError('must supply a finite list for paranoid')
        else:
            these_symbols = paranoid[:]
    del modules[modname]
    for mod in modules.values():
        try:
            delattr(mod, modname)
        except AttributeError:
            pass
        if paranoid:
            for symbol in these_symbols:
                if symbol[:2] == '__':  # ignore special symbols
                    continue
                try:
                    delattr(mod, symbol)
                except AttributeError:
                    passj

class MyImage:
	def __init__(self, path):
		self.img = Image.open(path)
		self.size = self.img.size

	def getcolors(self):
		#return self.img.convert('RGB').getcolors()
		colors = {}
		for x in range(self.size[0]):
			for y in range(self.size[1]):
				colors[self.getpixel(x,y)] = True
		return colors.keys()

	def getpixel(self, i, j):
		return self.img.getpixel((i, j))

	def putpixel(self, i, j, color):
		self.img.putpixel((i,j), color)

	
	def save(self, path):
		self.img.save(path)



class Supervoxel(object):

	def __init__(self, ID):
		'''
		:param arg1: the id of the supervoxel (we use color-tuple -> (r,g,b))
		:type arg1: any hashable object (we use tuple)
		'''
		try:
			hash(ID)
		except TypeError:
			raise Exception('ID must be immutable (hashable)')
		self.ID = ID
		#TODO removed this part for memory efficiency
		#self.pixels = {} # frame -> set of (x,y)
		#self.colors_dict = {} # (x,y,f) -> (R, G, B) actual color in the frame		
		#TODO		
		self.overlap_count = 0 #number of overlapping pixels with ground thruth
		self.__initializeCenter()
	
	def __initializeCenter(self):
		self.sum_x = 0
		self.sum_y = 0
		self.sum_t = 0
		self.number_of_pixels = 0

	def addVoxel(self, x,y,t, color, label=0):
		#TODO Removed this part for memory efficiency
		#if t not in self.pixels.keys():
		#	self.pixels[t] = set()
		#self.pixels[t].add((x,y))
		#self.colors_dict[ (x, y, t) ] = color
		#TODO
		self.sum_x += x
		self.sum_y += y
		self.sum_t += t
		self.number_of_pixels += 1
		self.overlap_count += label

	def hasPixel(self, x,y,f):
		return (x,y) in pixels[f]	

	def getOverlap(self):
		return self.overlap_count*1.0 / self.number_of_pixels	

	def getPixelsAtFrame(self, f):
		return self.pixels[f]

	def availableFrames(self):
		return self.pixels.keys()

	def center(self):
		n = self.number_of_pixels
		return (self.sum_x/n, self.sum_y/n, self.sum_t/n)

	def __str__(self):
		return "Supervoxel [ID:"+str(self.ID)+ ", Center:"+str(self.center()) + "]"

	def __eq__(self, other):
		return self.ID == other.ID

	def __hash__(self):
		return hash(self.ID)

#	def __getstate__(self):
#		state = {attr:getattr(self,attr) for attr in dir(self) if not attr.startswith('__') and not callable(getattr(self,attr))}
#		return state

#	def __setstate__(self, dic):
#		for key in dic:
#			setattr(self, key, dic[key])



class HistogramSupervoxel(Supervoxel):

	def __init__(self, ID):
		super(HistogramSupervoxel, self).__init__(ID)
		self.__initializeHistogram()
	
			
	def __initializeHistogram(self):
		self.R_hist = [0 for i in xrange(256)]
		self.G_hist = [0 for i in xrange(256)]
		self.B_hist = [0 for i in xrange(256)]

	def addVoxel(self, x,y,t, color, label=0):
		super(HistogramSupervoxel, self).addVoxel(x,y,t,color,label)		
		self.__updateHistogram(color)

	def __updateHistogram(self, color):
		self.R_hist[color[0]] += 1				
		self.G_hist[color[1]] += 1		
		self.B_hist[color[2]] += 1		

	def getFeature(self, number_of_bins=256):
		bin_width = 256/number_of_bins
		bin_num = -1
		r_hist = [0 for i in xrange(number_of_bins)]
		g_hist = r_hist[:]
		b_hist = r_hist[:]

		for i in xrange(256):
			if i%bin_width == 0:
				bin_num+=1
			r_hist[bin_num]+=self.R_hist[i]
			g_hist[bin_num]+=self.G_hist[i]
			b_hist[bin_num]+=self.B_hist[i]
		return [i*1.0/self.number_of_pixels for i in r_hist+g_hist+b_hist]



