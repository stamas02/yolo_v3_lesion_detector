import torch
import numpy as np 


def bbox_transform(stride, grid_size, bbox_pred_tensor, anchors):

	grid = np.arange(grid_size)
	a, b = np.meshgrid(grid, grid) 
	num_anchors = len(anchors)

	x_offset = torch.from_numpy(np.array(a, dtype=float)).float().view(-1,1)
	y_offset = torch.from_numpy(np.array(b, dtype=float)).float().view(-1,1)

	x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,
				num_anchors).view(-1, num_anchors, 2).unsqueeze(0)

	bbox_pred_tensor[:,:,:,:2] += x_y_offset 
	bbox_pred_tensor[:,:,:,:2] *= stride

	anchors = anchors.view(1, anchors.size(0), anchors.size(1)).repeat(
					grid_size*grid_size, 1, 1).unsqueeze(0)
	bbox_pred_tensor[:,:,:,2:4] *= anchors

	box_corner = bbox_pred_tensor.new(bbox_pred_tensor.shape)
	box_corner[:,:,:,0] = (bbox_pred_tensor[:,:,:,0] - \
								bbox_pred_tensor[:,:,:,2]/2)
	box_corner[:,:,:,1] = (bbox_pred_tensor[:,:,:,1] - \
								bbox_pred_tensor[:,:,:,3]/2)
	box_corner[:,:,:,2] = (bbox_pred_tensor[:,:,:,0] + \
								bbox_pred_tensor[:,:,:,2]/2) 
	box_corner[:,:,:,3] = (bbox_pred_tensor[:,:,:,1] + \
								bbox_pred_tensor[:,:,:,3]/2)

	box_corner = box_corner.clamp(min=0, max=(stride*grid_size - 1))

	return box_corner

def bbox_iou(gt_box, bbox):
	# gt_box: N x 4
	# bbox: hw x a x 4
	# ious: hw x a x N
	N = gt_box.size(0)
	hw = bbox.size(0)
	num_anchors = bbox.size(1)
	# expand to hw x a x N x 4
	gt_boxes_expand = gt_box.view(1, 1, N, 4).repeat(hw, num_anchors, 1, 1).view(-1, 4)
	bbox_expand = bbox.view(hw, num_anchors, 1, 4).repeat(1, 1, N, 1).view(-1, 4)

	gt_x1, gt_y1, gt_x2, gt_y2 = gt_boxes_expand[:,0], gt_boxes_expand[:,1], gt_boxes_expand[:,2], gt_boxes_expand[:,3]
	b_x1, b_y1, b_x2, b_y2 = bbox_expand[:,0], bbox_expand[:,1], bbox_expand[:,2], bbox_expand[:,3]

	inter_rect_x1 =  torch.max(gt_x1, b_x1)
	inter_rect_y1 =  torch.max(gt_y1, b_y1)
	inter_rect_x2 =  torch.min(gt_x2, b_x2)
	inter_rect_y2 =  torch.min(gt_y2, b_y2)
	#Intersection area
	inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
						torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
	#Union Area
	gt_area = (gt_x2 - gt_x1 + 1)*(gt_y2 - gt_y1 + 1)
	b_area = (b_x2 - b_x1 + 1)*(b_y2 - b_y1 + 1)

	ious = inter_area / (gt_area + b_area - inter_area)
	ious = ious.view(hw, num_anchors, N)
	return ious

def anchor_intersections(gt_box, anchors):
	# gt_box: N x 4
	# anchors: a x 2
	# ious: a x N
	N = gt_box.size(0)
	num_anchors = anchors.size(0)
	# expand to a x N x ...
	gt_box_expand = gt_box.view(1, N, 4).repeat(num_anchors, 1, 1).view(-1, 4)
	anchors_expand = anchors.view(num_anchors, 1, 2).repeat(1, N, 1).view(-1, 2)

	gt_w = gt_box_expand[:,2] - gt_box_expand[:,0] + 1
	gt_h = gt_box_expand[:,3] - gt_box_expand[:,1] + 1
	iw = torch.min(anchors_expand[:,0], gt_w)
	ih = torch.min(anchors_expand[:,1], gt_h)

	inter_area = iw * ih
	anchor_area = anchors_expand[:,0] * anchors_expand[:,1]
	ious = inter_area / (anchor_area + gt_w * gt_h - inter_area)
	ious = ious.view(num_anchors, N)
	return ious

def build_target(inp_size, out_size, bbox_pred_tensor, gt_boxes, 
						gt_classes, iou_pred_tensor, anchors,
							object_scale, noobject_scale, 
									class_scale, coord_scale):

	device_cpu = torch.device('cpu')
	bsize, hw, num_anchors, _ = bbox_pred_tensor.size()
	gt_boxes = gt_boxes.to(device_cpu)
	gt_classes = gt_classes.to(device_cpu)
	anchors = torch.FloatTensor(anchors)

	stride = inp_size[0] // out_size[0]
	grid_size = out_size[0]

	bboxes = bbox_transform(stride, grid_size, bbox_pred_tensor, anchors)
	num_classes = gt_classes.size(2)

	for b in range(bsize):
		## init the out 
		_classes = torch.zeros(hw, num_anchors, num_classes)
		_class_mask = torch.zeros(hw, num_anchors, 1)
		_ious = torch.zeros(hw, num_anchors, 1)
		# _iou_mask = torch.ones(hw, num_anchors, 1) * noobject_scale
		_iou_mask = torch.zeros(hw, num_anchors, 1)
		_boxes = torch.zeros(hw, num_anchors, 4)
		_boxes[:, :, 0:2] = 0.5
		_boxes[:, :, 2:4] = 1.0
		_box_mask = torch.zeros(hw, num_anchors, 1) + 0.01

		gt_class = gt_classes[b]
		iou_pred_b = iou_pred_tensor[b]
		gt_box = gt_boxes[b]
		bbox = bboxes[b]
		# print (gt_box)
		# print (bbox)
		ious = bbox_iou(gt_box, bbox) # hw x a x N


		### ???
		best_ious, _ = ious.max(2)
		best_ious = best_ious.view(hw, num_anchors, 1)
		ious_pred_th = torch.le(best_ious, 0.6).float()
		iou_penalty = 0 - ious_pred_th * iou_pred_b


		_iou_mask += iou_penalty * noobject_scale


		# locate the cell of each gt_box
		cx = (gt_box[:, 0] + gt_box[:, 2]) * 0.5 / stride
		cy = (gt_box[:, 1] + gt_box[:, 3]) * 0.5 / stride
		cell_inds = torch.floor(cy) * grid_size + torch.floor(cx)
		cell_inds = cell_inds.int()

		target_boxes = torch.zeros(gt_box.shape) # N x 4
		target_boxes[:, 0] = cx - torch.floor(cx)  # cx
		target_boxes[:, 1] = cy - torch.floor(cy)  # cy
		target_boxes[:, 2] = \
				(gt_box[:, 2] - gt_box[:, 0])   # anchor x exp(tw)
		target_boxes[:, 3] = \
				(gt_box[:, 3] - gt_box[:, 1])   # anchor x exp(th)

		anchors_ious = anchor_intersections(gt_box, anchors) # a x N
		_, anchors_inds = anchors_ious.max(0)

		for i, cell_ind in enumerate(cell_inds):
			if cell_ind >= hw or cell_ind < 0:
				print ('cell_ind {} over the hw {}'.format(cell_ind, hw))
				continue

			a = anchors_inds[i]

			_iou_mask[cell_ind, a, :] = object_scale * (1 - iou_pred_b[cell_ind, a, :]) # noqa

			#_ious[cell_ind, a, :] = ious[cell_ind, a, i]
			_ious[cell_ind, a, :] = 1.0

			_box_mask[cell_ind, a, :] = coord_scale
			target_boxes[i, 2:4] /= anchors[a]
			_boxes[cell_ind, a, :] = target_boxes[i]

			_class_mask[cell_ind, a, :] = class_scale
			_classes[cell_ind, a, :] = gt_class[i]

		if b >= 1:
			batch_boxes = torch.cat((batch_boxes, _boxes.clone().unsqueeze(0)), 0)
			batch_ious = torch.cat((batch_ious, _ious.clone().unsqueeze(0)), 0)
			batch_classes = torch.cat((batch_classes, _classes.clone().unsqueeze(0)), 0)
			batch_box_mask = torch.cat((batch_box_mask, _box_mask.clone().unsqueeze(0)), 0)
			batch_iou_mask = torch.cat((batch_iou_mask, _iou_mask.clone().unsqueeze(0)), 0)
			batch_class_mask = torch.cat((batch_class_mask, _class_mask.clone().unsqueeze(0)), 0)
		else:
			batch_boxes = _boxes.clone().unsqueeze(0)
			batch_ious = _ious.clone().unsqueeze(0)
			batch_classes = _classes.clone().unsqueeze(0)
			batch_box_mask = _box_mask.clone().unsqueeze(0)
			batch_iou_mask = _iou_mask.clone().unsqueeze(0)
			batch_class_mask = _class_mask.clone().unsqueeze(0)

	return batch_boxes, batch_ious, batch_classes, batch_box_mask, batch_iou_mask, batch_class_mask
