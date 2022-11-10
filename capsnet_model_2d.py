# 2D 2.5D 3D Segmentation Project
# Aneja Lab | Yale School of Medicine
# Developed by Arman Avesta, MD
# Created (3/30/22)
# Updated (11/5/22)

# --------------------------------------------------- Imports ------------------------------------------------------

import torch
from torch import nn
import torch.nn.functional as F

# ----------------------------------------------- 3D CapsNet model -------------------------------------------------

class CapsNet2D(nn.Module):

	def __init__(self, in_ch=1, out_ch=1, xpad=True):
		"""
		Inputs:
		- in_ch: input channels
		- out_ch: output channels
		- xpad: set to True if input shape is not powers of 2.
		Dimensions explained:
		- 'i' and 'o' subscripts respectively represent inputs and outputs of the layer.
			For instance, Ci represents number of input capsule channels.
		- C: capsule channels = capsule types.
		- P: pose components = number of elements in the pose vector.
		- K: kernel size for convolutions.
		"""
		super().__init__()
		self.xpad = xpad

		self.Conv1 = Conv(in_ch, Po=16, K=5, stride=1, padding=2)
		self.PrimaryCaps2 = ConvCaps(Ci=1, Pi=16, Co=2, Po=16, K=5, stride=2, padding=2, routings=1)
		self.ConvCaps3 = ConvCaps(Ci=2, Pi=16, Co=4, Po=16, K=5, stride=1, padding=2, routings=3)
		self.ConvCaps4 = ConvCaps(Ci=4, Pi=16, Co=4, Po=32, K=5, stride=2, padding=2, routings=3)
		self.ConvCaps5 = ConvCaps(Ci=4, Pi=32, Co=8, Po=32, K=5, stride=1, padding=2, routings=3)
		self.ConvCaps6 = ConvCaps(Ci=8, Pi=32, Co=8, Po=64, K=5, stride=2, padding=2, routings=3)
		self.ConvCaps7 = ConvCaps(Ci=8, Pi=64, Co=8, Po=32, K=5, stride=1, padding=2, routings=3)
		self.DeconvCaps8 = DeconvCaps(Ci=8, Pi=32, Co=8, Po=32, K=4, stride=2, routings=3)
		self.ConvCaps9 = ConvCaps(Ci=16, Pi=32, Co=8, Po=32, K=5, stride=1, padding=2, routings=3)
		self.DeconvCaps10 = DeconvCaps(Ci=8, Pi=32, Co=4, Po=16, K=4, stride=2, routings=3)
		self.ConvCaps11 = ConvCaps(Ci=8, Pi=16, Co=4, Po=16, K=5, stride=1, padding=2, routings=3)
		self.DeconvCaps12 = DeconvCaps(Ci=4, Pi=16, Co=2, Po=16, K=4, stride=2, routings=3)
		self.FinalCaps13 = ConvCaps(Ci=3, Pi=16, Co=1, Po=16, K=1, stride=1, padding=0, routings=3)


	def forward(self, x):
		"""
		Inputs:
		- x: batch of input images
		- y: batch of target images
		Outputs:
		- out_seg: segmented image
		- out_recon: reconstructed image within the target mask
		"""
		Conv1 = self.Conv1(x)
		PrimaryCaps2 = self.PrimaryCaps2(Conv1)
		ConvCaps3 = self.ConvCaps3(PrimaryCaps2)
		ConvCaps4 = self.ConvCaps4(ConvCaps3)
		ConvCaps5 = self.ConvCaps5(ConvCaps4)
		ConvCaps6 = self.ConvCaps6(ConvCaps5)
		ConvCaps7 = self.ConvCaps7(ConvCaps6)
		DeconvCaps8 = self.concat(ConvCaps5, self.DeconvCaps8(ConvCaps7))
		ConvCaps9 = self.ConvCaps9(DeconvCaps8)
		DeconvCaps10 = self.concat(ConvCaps3, self.DeconvCaps10(ConvCaps9))
		ConvCaps11 = self.ConvCaps11(DeconvCaps10)
		DeconvCaps12 = self.concat(Conv1, self.DeconvCaps12(ConvCaps11))
		FincalCaps13 = self.FinalCaps13(DeconvCaps12)
		SegmentedSlice = vector_norm(FincalCaps13)
		return SegmentedSlice


	def concat(self, skip, x):
		"""
		Concatenates two batches of capsules.
		Inputs:
		- skip: skip-connection from the downsampling limb of CapsNet
		- x: input from the upsampling limb of U-Net
		Outputs:
		- concatenated tensor
		Dimensions explained:
		- 's' and 'x' subscripts respectively mean skip-connection input and input from the upsampling limb.
		- B: batch size
		- C: capsule channels = capsule types
		- P: pose components
		- D: depth
		- H: height
		- W: width
		"""
		Bs, Cs, Ps, Hs, Ws = skip.shape
		Bx, Cx, Px, Hx, Wx = x.shape
		assert (Bs, Ps) == (Bx, Px)
		if self.xpad:
			diffH, diffW = Hs - Hx, Ws - Wx
			x = F.pad(x, [diffW // 2, diffW - diffW // 2,
						  diffH // 2, diffH - diffH // 2])
		return torch.cat([skip, x], dim=1)





# -------------------------------------------------- 3D CapsNet units -----------------------------------------------


class Conv(nn.Module):
	"""
	Non-capsule convolutional layer.
	"""
	def __init__(self, in_ch, Po, K, stride, padding=None):
		super().__init__()

		self.conv = nn.Sequential(nn.Conv2d(in_ch, Po, K, stride, padding),
		                          nn.ReLU(inplace=True))

	def forward(self, x):
		"""
        Input:
        - x: MRI volumes: [B, 1, Hi, Wi]
        Output:
        - x: [B, 1, Po, Ho, Wo]
        """
		x = self.conv(x)                                                    	# x: [B, Po, Ho, Wo]
		return x.unsqueeze(1)                                               	# return: [B, 1, Po, Ho, Wo]

# ........................................................................................................

class ConvCaps(nn.Module):
	"""
	Convolutional capsule layer.
	"""
	def __init__(self, Ci, Pi, Co, Po, K, stride, padding, routings=3):
		"""
		Inputs:
		- Ci: input capsule channels
		- Pi: input pose components
		- Co: output capsule channels
		- Po: output pose components
		- K: kernel size
		- stride
		- padding
		- routings: dynamic routing iterations
		"""
		super().__init__()

		self.Ci = Ci
		self.Pi = Pi
		self.Co = Co
		self.Po = Po
		self.routings = routings

		self.conv = nn.Conv2d(Pi, Co*Po, kernel_size=K, stride=stride, padding=padding, bias=False)
		self.biases = nn.Parameter(torch.zeros(1, 1, Co, Po, 1, 1) + 0.1)    	# biases: [1, 1, Co, Po, 1, 1]

	def forward(self, x):                                                       # x: [B, Ci, Pi, Hi, Wi]
		"""
		Input:
		- x: batch of input capsules; dimensions: [B, Ci, Pi, Hi, Wi]
		Output:
		- return: batch of output capsules; dimensions: [B, Co, Po, Ho, Wo]
		Dimensions explained:
		- B: batch size
		- C: capsule channels = capsule types
		- P: pose components
		- H: height
		- W: width
		- 'i' and 'o' subscripts respectively represent input and output dimensions.
			For instance, Ci represents the number of input capsule channels,
			Po represents the number of output pose components in each output capsule,
			and Ho represents the height of the output image.
		"""
		B, Ci, Pi, Hi, Wi = x.shape
		assert (Ci, Pi) == (self.Ci, self.Pi)
		x = x.reshape(B*Ci, Pi, Hi, Wi)                                 	# x: [B*Ci, Pi, Hi, Wi]
		x = self.conv(x)
		B_Ci, Co_Po, Ho, Wo = x.shape                                   	# x: [B*Ci, Co*Po, Ho, Do]
		assert (B_Ci, Co_Po) == (B*Ci, self.Co*self.Po)
		x = x.reshape(B, Ci, self.Co, self.Po, Ho, Wo)                  	# x: [B, Ci, Co, Po, Ho, Wo]
		return dynamic_routing(x, self.biases, self.routings)               # return: [B, Co, Po, Ho, Wo]

# ........................................................................................................

class DeconvCaps(nn.Module):
	"""
	Transposed convolutional capsule layer (in the upsampling limb of CapsNet)
	"""
	def __init__(self, Ci, Pi, Co, Po, K, stride, routings=3):
		super().__init__()

		self.Ci = Ci
		self.Pi = Pi
		self.Co = Co
		self.Po = Po
		self.routings = routings

		self.conv = nn.ConvTranspose2d(Pi, Co*Po, kernel_size=K, stride=stride, bias=False)
		self.biases = nn.Parameter(torch.zeros(1, 1, Co, Po, 1, 1) + 0.1)   # biases: [1, 1, Co, Po, 1, 1]


	def forward(self, x):                                                   # x: [B, Ci, Pi, Di, Hi, Wi]
		B, Ci, Pi, Hi, Wi = x.shape
		assert (Ci, Pi) == (self.Ci, self.Pi)
		x = x.reshape(B*Ci, Pi, Hi, Wi)                                 	# x: [B*Ci, Pi, Hi, Wi]
		x = self.conv(x)
		B_Ci, Co_Po, Ho, Wo = x.shape                                   	# x: [B*Ci, Co*Po, Ho, Wo]
		assert (B_Ci, Co_Po) == (B*Ci, self.Co*self.Po)
		x = x.reshape(B, Ci, self.Co, self.Po, Ho, Wo)                  	# x: [B, Ci, Co, Po, Ho, Wo]
		return dynamic_routing(x, self.biases, self.routings)

# ........................................................................................................

class Decoder(nn.Module):
	"""
	To be used for reconstruction loss calculation.
	"""
	def __init__(self, Pi=16):
		super().__init__()

		self.recon = nn.Sequential(
			nn.Conv2d(Pi, 64, kernel_size=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 128, kernel_size=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 1, kernel_size=1),
			nn.ReLU(inplace=True))

	def forward(self, x, y):
		"""
		Inputs:
			- x: [B, 1, P, H, W]
			- y: [B, 1, H, W]
		Output:
			- masked x based on y: [B, P, H, W]
		"""
		if y is None:
			y = self.create_mask(self.x)
		Bx, Cx, Px, Hx, Wx = x.shape
		By, Cy, Hy, Wy = y.shape
		assert (Cx, Cy) == (1, 1)
		assert (Bx, Hx, Wx) == (By, Hy, Wy)
		x, y = x.reshape(Bx, Px, Hx, Wx), y.reshape(Bx, 1, Hx, Wx)
		x = x * y
		return self.recon(x)

	def create_mask(self, x):
		"""
		x: [B, P, H, W]
		"""
		norm = torch.linalg.norm(x, dim=1, keepdim=True)
		return (norm > self.threshold).float()                          # return: [B, 1, H, W]


# ----------------------------------------------- Helper functions -----------------------------------------------

def dynamic_routing(votes, biases, routings):
	"""
	Inputs:
		- votes: [B, Ci, Co, Po, Ho, Wo]
		- biases:[1, 1, Co, Po, 1, 1]
		- routings: number of dynamic routing iterations
	"""
	B, Ci, Co, Po, Ho, Wo = votes.shape                                 	# votes: [B, Ci, Co, Po, Ho, Wo]
	device = votes.device

	bij = torch.zeros(B, Ci, Co, 1, Ho, Wo).to(device)                  	# bij: [B, Ci, Co, 1, Ho, Wo]

	for t in range(routings):
		cij = F.softmax(bij, dim=2)                                         # cij: [B, Ci, Co, 1, Ho, Wo]
		sj = torch.sum(cij * votes, dim=1, keepdim=True) + biases           # sj: [B, 1, Co, Po, Ho, Wo]
		vj = squash(sj)                                                     # vj: [B, 1, Co, Po, Ho, Wo]
		if t < routings - 1:
			bij = bij + torch.sum(votes * vj, dim=3, keepdim=True)          # bij: [B, Ci, Co, 1, Ho, Wo]

	return vj.squeeze(1)                                                    # return: [B, Co, Po, Ho, Wo]



def squash(sj):
	"""
	Inputs:
		- sj: [B, 1, Co, Po, Ho, Wo]
	Output:
		- vj: [B, 1, Co, Po, Ho, Wo]
	"""
	sjnorm = torch.linalg.norm(sj, dim=3, keepdim=True)                     # sjnorm: [B, 1, Co, 1, Ho, Wo]
	sjnorm2 = sjnorm ** 2
	return sjnorm2 * sj / ((1 + sjnorm2) * sjnorm)                          # return vj: [B, 1, Co, Po, Ho, Wo]



def vector_norm(x):
	"""
	Input:
		- x: [B, 1, P, H, W]
			 x should have 1 capsule channel.
	Output:
		- norm of x: [B, 1, H, W]
	"""
	assert x.shape[1] == 1
	return torch.linalg.norm(x, dim=2)      # return: [B, 1, H, W]




# ----------------------------------------------- Testing -----------------------------------------------
if __name__ == '__main__':

    from torchsummary import summary

    x = torch.rand(1, 1, 64, 64)  # batch of 1 MRI volume: 1 channel, 256 x 256 x 256 voxels
    model = CapsNet2D()
    preds = model(x)
    print(f'Input shape: {x.shape} \n'
          f'Output shape: {preds.shape}')
    print(f'Input and output are the same shape? {preds.shape == x.shape}')

    summary(model, (1, 64, 64))
    # for summary, the second argument is the shape of each input data (not the batch).
