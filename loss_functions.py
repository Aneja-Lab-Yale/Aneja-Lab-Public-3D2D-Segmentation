# 2D 2.5D 3D Segmentation Project
# Aneja Lab | Yale School of Medicine
# Developed by Arman Avesta, MD
# Created (3/30/22)
# Updated (11/5/22)

# -------------------------------------------------- Imports --------------------------------------------------

import numpy as np
import torch
from torch import nn


# ----------------------------------------------- Loss functions  -----------------------------------------------

class DiceLoss(nn.Module):
	def __init__(self, reduction='mean', conversion='none', threshold=.5, low=0.1, high=0.9, s=10, eps=1):
		"""
		Inputs:
			- reduction: 'mean' or 'none'.
				'mean': return the mean loss for all examples in the batch.
				'none': return the tensor of losses for B examples in the batch.
			- conversion: see convert_preds function below to see the possible options.
			- threshold: threshold used on convert_preds function (see below).
			- low: low margin used in convert_preds function (see below).
			- high: high margin used in convert_preds function (see below)
			- s: scaling factor used in convert_preds function (see below)
			- eps: to ensure that the denominators are not equal to zero.
		"""
		super().__init__()
		self.reduction = reduction
		self.conversion = conversion
		self.threshold = threshold
		self.low = low
		self.high = high
		self.s = s
		self.eps = eps

	def forward(self, preds, targets):
		"""
		Inputs:
			- preds: predictions = model outputs = proposed segmentations
			- targets: ground truth = actual segmentations
			preds and targets should have the same shape: [B, C, D, H, W]
														(or [B, C, H, W] for 2D models)
			B: batches, C: channels, D: depth, H: height, W: width.
		Output:
			- dice score: scalar if reduction='mean', and a tensor with B components if reduction='none'.
		"""
		preds = convert_preds(preds, self.conversion, self.threshold, self.low, self.high, self.s)

		if self.reduction == 'mean':
			preds, targets = preds.flatten(), targets.flatten()
			intersection = (preds * targets).sum()
			dice_loss = 1 - (2 * intersection + self.eps) / (preds.sum() + targets.sum() + self.eps)
		# ToDo:
		# print(f'preds sum: {preds.sum(): .1f}, targets sum: {targets.sum(): .0f}, '
		# 	  f'intersection: {intersection: .1f}, dice_loss: {dice_loss: .3f}')

		elif self.reduction == 'none':
			batch_size = preds.shape[0]
			preds, targets = preds.reshape(batch_size, -1), targets.reshape(batch_size, -1)
			intersection = (preds * targets).sum(axis=1)
			dice_loss = 1 - (2 * intersection + self.eps) / (preds.sum(axis=1) + targets.sum(axis=1) + self.eps)

		return dice_loss


# ..............................................................................................................

class SoftDiceLoss(nn.Module):
	def __init__(self, reduction='mean', conversion='none', threshold=.5, low=0.1, high=0.9, s=10, eps=1):
		"""
		Inputs:
			- reduction: 'mean' or 'none'.
				'mean': return the mean loss for all examples in the batch.
				'none': return the tensor of losses for B examples in the batch.
			- conversion: see convert_preds function below to see the possible options.
			- threshold: threshold used on convert_preds function (see below).
			- low: low margin used in convert_preds function (see below).
			- high: high margin used in convert_preds function (see below)
			- s: scaling factor used in convert_preds function (see below)
			- eps: to ensure that the denominators are not equal to zero.
		"""
		super().__init__()
		self.reduction = reduction
		self.conversion = conversion
		self.threshold = threshold
		self.low = low
		self.high = high
		self.s = s
		self.eps = eps

	def forward(self, preds, targets):
		"""
		Inputs:
			- preds: predictions = model outputs = proposed segmentations
			- targets: ground truth = actual segmentations
			preds and targets should have the same shape: [B, C, D, H, W]
														(or [B, C, H, W] for 2D models)
			B: batches, C: channels, D: depth, H: height, W: width.
		Output:
			- dice score: scalar if reduction='mean', and a tensor with B components if reduction='none'.
		"""
		preds = convert_preds(preds, self.conversion, self.threshold, self.low, self.high, self.s)

		if self.reduction == 'mean':
			preds, targets = preds.flatten(), targets.flatten()
			intersection = (preds * targets).sum()
			soft_dice_loss = 1 - (2 * intersection + self.eps) / ((preds**2).sum() + (targets**2).sum() + self.eps)

			# print(f'preds sum: {preds.sum(): .1f}, targets sum: {targets.sum(): .0f}, '
			# 	  f'intersection: {intersection: .1f}, dice_loss: {dice_loss: .3f}')

		elif self.reduction == 'none':
			batch_size = preds.shape[0]
			preds, targets = preds.reshape(batch_size, -1), targets.reshape(batch_size, -1)
			intersection = (preds * targets).sum(axis=1)
			soft_dice_loss = 1 - (2 * intersection + self.eps) / ((preds**2).sum(axis=1) +
			                                                      (targets**2).sum(axis=1) + self.eps)

		return soft_dice_loss


# ..............................................................................................................
class DiceBCELoss(nn.Module):
	def __init__(self, reduction='mean', conversion='none', threshold=.5, low=0.1, high=0.9, s=10, eps=1, alpha=.3):
		"""
		Inputs:
			- reduction: 'mean' or 'none'.
				'mean': return the mean loss for all examples in the batch.
				'none': return the tensor of losses for B examples in the batch.
			- conversion: see convert_preds function below to see the possible options.
			- threshold: threshold used on convert_preds function (see below).
			- low: low margin used in convert_preds function (see below).
			- high: high margin used in convert_preds function (see below)
			- s: scaling factor used in convert_preds function (see below)
			- eps: to ensure that the denominators are not equal to zero.
			- alpha: weighting of Dice loss vs BCE loss
		"""
		super().__init__()
		self.reduction = reduction
		self.conversion = conversion
		self.threshold = threshold
		self.low = low
		self.high = high
		self.s = s
		self.eps = eps
		self.alpha = alpha

		if reduction == 'mean':
			self.bceloss = nn.BCELoss(reduction='mean')
		elif reduction == 'none':
			# ToDo
			'''
			self.bceloss = nn.BCELoss(reduction='none')
			'''

	def forward(self, preds, targets):
		preds = convert_preds(preds, self.conversion, self.threshold, self.low, self.high, self.s)

		if self.reduction == 'mean':
			preds, targets = preds.flatten(), targets.flatten()
			intersection = (preds * targets).sum()
			dice_loss = 1 - (2 * intersection + self.eps) / (preds.sum() + targets.sum() + self.eps)
			bce_loss = self.bceloss(preds, targets)
			# ToDo:
			# print(f'preds sum: {preds.sum(): .1f}, targets sum: {targets.sum(): .0f}, '
			# 	  f'intersection: {intersection: .1f}, dice loss: {dice_loss: .3f}, bce_loss: {bce_loss: .3f}')

		elif self.reduction == 'none':
			batch_size = preds.shape[0]
			preds, targets = preds.reshape(batch_size, -1), targets.reshape(batch_size, -1)
			intersection = (preds * targets).sum(axis=1)
			dice_loss = 1 - (2 * intersection + self.eps) / (preds.sum(axis=1) + targets.sum(axis=1) + self.eps)
		# ToDo:
		# bce_loss = self.bceloss(preds, targets).sum(axis=1)

		return self.alpha * dice_loss + (1 - self.alpha) * bce_loss


# ..............................................................................................................

class DiceReconLoss(nn.Module):
	"""
	Returns a weighted sum of dice loss and reconstruction loss (see ReconLos below).
	Reconstruction loss is used for regularization.
	"""

	def __init__(self, reduction='mean', conversion='none', recon_weight=0.01):
		"""
		recon_weight determines the strength of regularization using reconstruction loss.
		"""
		super().__init__()
		self.recon_weight = recon_weight
		self.dice = DiceLoss(reduction, conversion)
		self.recon = ReconLoss(reduction, conversion)

	def forward(self, inputs, preds, recons, targets):
		dice_loss = self.dice(preds, targets)
		recon_loss = self.recon(inputs, recons, targets)
		return dice_loss + recon_loss * self.recon_weight


# ..............................................................................................................

class IoULoss(nn.Module):
	def __init__(self, reduction='mean', conversion='none', eps=1):
		super().__init__()
		self.reduction = reduction
		self.conversion = conversion  # set None if your model contains a sigmoid or equivalent activation layer
		self.eps = eps

	def forward(self, preds, targets):
		preds = convert_preds(preds, self.conversion)

		if self.reduction == 'mean':
			preds, targets = preds.flatten(), targets.flatten()
			intersection = (preds * targets).sum()
			total = (preds + targets).sum()

		elif self.reduction == 'none':
			batch_size = preds.shape[0]
			preds, targets = preds.reshape(batch_size, -1), targets.reshape(batch_size, -1)
			intersection = (preds * targets).sum(axis=1)
			total = (preds + targets).sum(axis=1)

		union = total - intersection
		IoU = (intersection + self.eps) / (union + self.eps)
		return 1 - IoU


# ..............................................................................................................

class TverskyLoss(nn.Module):
	"""
	To optimize segmentation on imbalanced medical datasets, this loss function utilizes alpha and beta constants
	that can adjust how harshly different types of error are penalized in the loss function:
	in the case of α = β = 0.5 the Tversky index simplifies to be the same as the Dice coefficient,
	which is also equal to the F1 score.
	With α = β = 1, it produces Tanimoto coefficient, and setting α + β = 1 produces the set of Fβ scores.
	Larger βs weigh recall higher than precision (by placing more emphasis on false negatives).
	Therefore, alpha and beta penalize false positives and false negatives respectively to a higher degree
	 in the loss function as their value is increased. The beta constant in particular has applications in
	 situations where models can obtain misleadingly positive performance via highly conservative prediction.
	 This loss was introduced in "Tversky loss function for image segmentation using
	 3D fully convolutional deep networks": https://arxiv.org/abs/1706.05721.
	"""

	def __init__(self, reduction='mean', conversion='none', alpha=0.5, beta=0.5, eps=1):
		super(TverskyLoss, self).__init__()
		self.conversion = conversion  # set None if your model contains a sigmoid or equivalent activation layer
		self.reduction = reduction
		self.alpha = alpha
		self.beta = beta
		self.eps = eps

	def forward(self, preds, targets):
		preds = convert_preds(preds, self.conversion)

		if self.reduction == 'mean':
			preds, targets = preds.flatten(), targets.flatten()
			TP = (preds * targets).sum()  # true positives
			FP = ((1 - targets) * preds).sum()  # false positives
			FN = (targets * (1 - preds)).sum()  # false negatives

		elif self.reduction == 'none':
			batch_size = preds.shape[0]
			preds, targets = preds.reshape(batch_size, -1), targets.reshape(batch_size, -1)
			TP = (preds * targets).sum(axis=1)  # true positives
			FP = ((1 - targets) * preds).sum(axis=1)  # false positives
			FN = (targets * (1 - preds)).sum(axis=1)  # false negatives

		tversky = (TP + self.eps) / (TP + self.alpha * FP + self.beta * FN + self.eps)
		return 1 - tversky


# ..............................................................................................................

class ReconLoss(nn.Module):
	"""
	Computes reconstruction loss (to be used as regularization for capsule networks).
	ReconLoss is lowest when reconstructed image matches the input image WITHIN the segmentation mask
	(other areas of image outside segmentation mask are ignored).
	Segmentation mask is obtained from:
		- ground truth during training
		- thresholded model output during validation / testing
	"""

	def __init__(self, reduction='mean', conversion='none'):
		super().__init__()
		self.reduction = reduction
		self.conversion = conversion

		self.recon_loss_mean = nn.MSELoss(reduction='mean')
		self.recon_loss_intact = nn.MSELoss(reduction='none')

	def forward(self, inputs, recons, targets):
		"""
		Inputs:
			- inputs: original input images
			- recons: reconstructed image by the decoder part of the network
			- targets: segmentation (to be used to mask inputs and recons; loss is only computed within this mask)
		"""
		targets = convert_preds(targets, self.conversion)

		inputs = inputs * targets

		if self.reduction == 'mean':
			recon_loss = self.recon_loss_mean(inputs, targets)

		elif self.reduction == 'none':
			batch_size = inputs.shape[0]
			inputs, recons = inputs.reshape(batch_size, -1), recons.reshape(batch_size, -1)
			recon_loss = self.recon_loss_intact(inputs, recons).mean(axis=1)

		return recon_loss


# ---------------------------------------- Predictions Covnersion Function ------------------------------------------

def convert_preds(preds, conversion='none', threshold=0.5, low=0.1, high=0.9, s=10):
	"""
	predictions (model outputs) --> converted predictions (to be used in loss function)
	Inputs:
		- preds: predictions = model outputs = proposed segmentations.
		- conversion: how preds should be converted. Options: 'margin', 'threshold', 'sigmoid', 'logit', 'none'
		- threshold: only used if conversion is set as 'threshold'.
		- low: low margin; only used if coversion is set to 'margin1' or 'margin2'.
		- high: high margin; only used of coversion is set to 'margin1' or 'margin2'.
		- s: scaling factor for sigmoid function.
			s=10 --> low, high ≈ 0.1, 0.9
			s=15 --> low, high ≈ 0.2, 0.8
			s=20 --> low, high ≈ 0.3, 0.7
	Output:
	- converted predictions
	Conversion options:
		- 'margin1': preds --> 0 if preds < low; preds if low < preds < high; 1 if preds > high
		- 'margin2': preds --> 0 if preds < low; (preds - low) / (high - low) if low < preds < high; 1 if preds > high
		- 'threshold': preds --> 0 if preds < threshold; 1 if preds >= threshold
		- 'sigmoid': preds --> sigmoid(preds)
		- 'logit': preds --> logit(preds)
		- 'none': returnds preds themselves
	"""
	if conversion == 'none':
		return preds

	if conversion == 'sigmoid':
		return torch.sigmoid((preds - threshold) * s)

	if conversion == 'threshold':
		return torch.sigmoid((preds - threshold) * 1000)

	if conversion == 'logit':
		return torch.logit(preds.clip(1e-7, 1 - 1e-7))

	if conversion == 'margin':
		z = torch.zeros_like(preds)
		z[preds > low] = (preds[preds > low] - low) / (high - low)
		z[preds > high] = 1
		return z

	if conversion == 'margin2':
		z = torch.zeros_like(preds)
		z[preds > low] = preds[preds > low]
		z[preds > high] = 1
		return z
	"""
	- 'margin':
		preds --> {
			0                               if preds < low
			preds                           if low < preds < high
			1                               if preds > high
	- 'margin2':
		preds --> {
			0                               if preds < low
			(preds - low) / (high - low)    if low < preds < high
			1                               if preds > high
	'margin' is discontinuous at low & high values. 
	'margin2' is continuous (but non-differentiable) at low & high values.
	Alternative implementation for 'threshold':
	if conversion == 'threshold':
		z = torch.zeros_like(preds)
		z[preds > threshold] = 1
		return z
	"""

# -------------------------------------------------- Test code --------------------------------------------------

# if __name__ == "__main__":
#
# 	import os
# 	from torch.utils.data import DataLoader
# 	from data_loader import AdniDataset, make_image_list
# 	from image_tools import imshow
#
# 	np.set_printoptions(precision=1, suppress=True)
# 	torch.set_printoptions(precision=1, sci_mode=False)
#
# 	#######################################################
#
# 	project_root = '/Users/arman/projects/capsnet'
# 	images_csv = 'data/datasets_local/train_inputs.csv'
# 	masks_csv = 'data/datasets_local/train_outputs.csv'
#
# 	images_path = join(project_root, images_csv)
# 	masks_path = join(project_root, masks_csv)
#
# 	image_list = make_image_list(images_path)
# 	mask_list = make_image_list(masks_path)
#
# 	adni = AdniDataset(image_list, mask_list, maskcode=14, crop=(64, 64, 64), cropshift=(0, 7, 0), testmode=False)
# 	dataloader = DataLoader(dataset=adni, batch_size=4, shuffle=True)
# 	itr = iter(dataloader)
#
#     images, masks = next(itr)
#
#
# 	print(f'images --> shape: {images.shape}; data type: {images.dtype}; min: {images.min()}; max: {images.max()}')
# 	print(f'masks --> shape: {masks.shape}; data type: {masks.dtype}; unique values: {masks.unique()}')
#
# 	imshow(images)
# 	imshow(masks)
#
# 	diceloss = DiceLoss(reduction='none')
#
# 	loss = diceloss(masks, masks)
# 	print(f'loss: {loss}, loss type: {type(loss)}')
