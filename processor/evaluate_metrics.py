import numpy as np
import os
import torch
def VOCap(rec, prec):
    '''
        Compute the average precision following the code in Pascal VOC toolkit
    '''
    mrec = np.array(rec).astype(np.float32)
    mprec = np.array(prec).astype(np.float32)
    mrec = np.insert(mrec, [0, mrec.shape[0]], [0.0, 1.0])
    mprec = np.insert(mprec, [0, mprec.shape[0]], [0.0, 0.0])

    for i in range(mprec.shape[0]-2, -1, -1):
        mprec[i] = max(mprec[i], mprec[i+1])

    i = np.ndarray.flatten(np.array(np.where(mrec[1:] != mrec[0:-1]))) + 1
    ap = np.sum(np.dot(mrec[i] - mrec[i-1], mprec[i]))
    return ap

def iou_ratio(bbox_1, bbox_2):
    '''
        Compute the IoU ratio between two bounding boxes
    '''
    bi = [max(bbox_1[0], bbox_2[0]), min(bbox_1[1], bbox_2[1])]
    iw = bi[1] - bi[0] + 1
    ov = 0
    if iw > 0:
        ua = (bbox_1[1] - bbox_1[0] + 1) + (bbox_2[1] - bbox_2[0] + 1) - iw
        ov = iw / float(ua)
    return ov

def compute_metric_class(gt, res, cls, minoverlap):
    npos = 0
    gt_cls = {}
    for img in gt.keys():
        index = np.array(gt[img]['class']) == cls
        BB = np.array(gt[img]['bbox'])[index]
        det = np.zeros(np.sum(index[:]))
        npos += np.sum(index[:])
        gt_cls[img] = {'BB': BB,
                       'det': det}

    # loading the detection result
    score = np.array(res[cls]['score'])
    imgs = np.array(res[cls]['img'])
    BB = np.array(res[cls]['bbox'])

    # sort detections by decreasing confidence
    si = np.argsort(-score)
    imgs = imgs[si]
    if len(BB) > 0:
        BB = BB[si, :]
    else:
        BB = BB

    # assign detections to ground truth objects
    nd = len(score)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        img = imgs[d]
        if len(BB) > 0:
            bb = BB[d, :]
        else:
            bb = BB

        ovmax = 0
        for j in range(len(gt_cls[img]['BB'])):
            bbgt = gt_cls[img]['BB'][j]
            ov = iou_ratio(bb, bbgt)
            if ov > ovmax:
                ovmax = ov
                jmax = j

        if ovmax >= minoverlap:
            if not gt_cls[img]['det'][jmax]:
                tp[d] = 1
                gt_cls[img]['det'][jmax] = 1
            else:
                fp[d] = 1
        else:
            fp[d] = 1

    # compute precision/recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp/npos
    prec = tp/(fp+tp)

    ap = VOCap(rec, prec)

    return rec, prec, ap

# calc_pr: calculate precision and recall
#	@positive: number of positive proposal
#	@proposal: number of all proposal
#	@ground: number of ground truth
def calc_pr(positive, proposal, ground):
	if (proposal == 0): return 0,0
	if (ground == 0): return 0,0
	return (1.0*positive)/proposal, (1.0*positive)/ground

# match: match proposal and ground truth
#	@lst: list of proposals(label, start, end, confidence, video_name)
#	@ratio: overlap ratio
#	@ground: list of ground truth(label, start, end, confidence, video_name)
#
#	correspond_map: record matching ground truth for each proposal
#	count_map: record how many proposals is each ground truth matched by 
#	index_map: index_list of each video for ground truth
def match(lst, ratio, ground):
	def overlap(prop, ground):
		l_p, s_p, e_p, c_p, v_p = prop
		l_g, s_g, e_g, c_g, v_g = ground
		if (int(l_p) != int(l_g)): return 0
		if (v_p != v_g): return 0
		return (min(e_p, e_g)-max(s_p, s_g))/(max(e_p, e_g)-min(s_p, s_g))

	cos_map = [-1 for x in range(len(lst))]
	count_map = [0 for x in range(len(ground))]
	#generate index_map to speed up
	index_map = [[] for x in range(53)]#number label
	for x in range(len(ground)):
		#print(int(ground[x][0]))
		index_map[int(ground[x][0])].append(x)

	for x in range(len(lst)):
		for y in index_map[int(lst[x][0])]:
			if (overlap(lst[x], ground[y]) < ratio): continue
			if (overlap(lst[x], ground[y]) < overlap(lst[x], ground[cos_map[x]])): continue
			cos_map[x] = y
		if (cos_map[x] != -1): count_map[cos_map[x]] += 1
	positive = sum([(x>0) for x in count_map])
	return cos_map, count_map, positive

# Interpolated Average Precision:
#	@lst: list of proposals(label, start, end, confidence, video_name)
#	@ratio: overlap ratio
#	@ground: list of ground truth(label, start, end, confidence, video_name)
#
#	score = sigma(precision(recall) * delta(recall))
#	Note that when overlap ratio < 0.5, 
#		one ground truth will correspond to many proposals
#		In that case, only one positive proposal is counted
def ap(lst, ratio, ground):
	lst.sort(key = lambda x:x[3]) # sorted by confidence
	cos_map, count_map, positive = match(lst, ratio, ground)
	score = 0
	number_proposal = len(lst)
	number_ground = len(ground)
	old_precision, old_recall = calc_pr(positive, number_proposal, number_ground)
 
	for x in range(len(lst)):
		number_proposal -= 1;
		if (cos_map[x] == -1): continue
		count_map[cos_map[x]] -= 1;
		if (count_map[cos_map[x]] == 0): positive -= 1;

		precision, recall = calc_pr(positive, number_proposal, number_ground)   
		if precision>old_precision: 
			old_precision = precision
		score += old_precision*(old_recall-recall)
		old_recall = recall
	return score

def eval_detect_mAP(gt_dict, res_dict, minoverlap=0.5):
    '''
    gt_dict, the key value is a sequence name
    '''
    gt = {}
    classes = set()
    for name in gt_dict.keys():
        labels = gt_dict[name]
        gt[name] = {}
        gt[name]['class'] = []
        gt[name]['bbox'] = []
        for idx in range(labels.shape[0]):
            gt[name]['class'].append(labels[idx,0])
            gt[name]['bbox'].append([labels[idx,1], labels[idx,2] ])
        classes = classes | set(gt[name]['class'] )
    print('class',classes)
    res = {}
    for cls in classes:
        res[cls] = {}
        res[cls]['img'] = []
        res[cls]['score'] = []
        res[cls]['bbox'] = []
    for name in res_dict.keys():
        pred_labels = res_dict[name]
        for idx in range(pred_labels.shape[0]):
            cls = int(pred_labels[idx,0])
            if cls not in res.keys():
                continue
            res[cls]['img'].append(name)
            res[cls]['score'].append(pred_labels[idx,3])
            res[cls]['bbox'].append([pred_labels[idx,1], pred_labels[idx,2] ])
    
    metrics = {}
    res_map = []
    # res_f1 = []
    for cls in classes:
        rec, prec, ap = compute_metric_class(gt, res, cls, minoverlap)
        metrics[cls] = {}
        metrics[cls]['recall'] = rec
        metrics[cls]['precision'] = prec
        metrics[cls]['ap'] = ap
        res_map.append(ap)
        
        # f1 = 2*rec*prec/(rec + prec)
        # f1[np.isnan(f1)] = 0
        # res_f1.append( np.mean(f1) )

    metrics['map'] = np.mean(np.array(res_map))
    # metrics['f1'] = np.mean(np.array(res_f1))
    #print(metrics)
    print('action class number is',len(classes))
    return metrics

def get_interval_frm_frame_predict(prob_seq,  win_lg=15, win_sm=2, rate_min=0.1):
    # labels is 2d array, each row, (action, start_frame, end_frame)
    # prob_seq is probability predicted for each frame, (num_frame, num_class)
    # class index 0 represent empty action
    pred_seq = np.argmax(prob_seq, axis=1)
    # print pred_seq
    # accumulative count for each class and for each frame
    cur_count = np.zeros(prob_seq.shape)#T,class
    for idx in range(len(pred_seq)):
        if idx==0:
            cur_count[idx, pred_seq[idx]] = 1
        else:
            cur_count[idx] = cur_count[idx-1]
            cur_count[idx, pred_seq[idx]] = cur_count[idx, pred_seq[idx]] + 1
    # search for windows
    pred_labels = []
    idx = 0
    while idx < (len(pred_seq) - win_lg):
        # assume start frame idx
        if pred_seq[idx] > 0:
            rate = (cur_count[idx+win_lg, pred_seq[idx] ] - cur_count[idx, pred_seq[idx] ])*1.0/win_lg
            #这一段的平均命中率
            if rate > rate_min and (idx + 2*win_lg) < cur_count.shape[0]:
                # find predictions
                pos = idx + win_lg
                # search from right at window size of win_lg
                # new_rate = (cur_count[pos+win_lg, pred_seq[idx] ] - cur_count[idx, pred_seq[idx] ])*1.0/(win_lg + pos - idx)
                new_rate = (cur_count[pos+win_lg, pred_seq[idx] ] - cur_count[pos, pred_seq[idx] ])*1.0/win_lg
                # print pos, new_rate, cur_count[pos+win_lg, pred_seq[idx] ], cur_count[idx, pred_seq[idx] ]
                while new_rate > rate_min and (pos + 2*win_lg) < cur_count.shape[0]:
                    pos = pos + win_lg
                    # new_rate = (cur_count[pos + win_lg, pred_seq[idx] ] - cur_count[idx, pred_seq[idx] ])*1.0/(win_lg + pos - idx)
                    new_rate = (cur_count[pos+win_lg, pred_seq[idx] ] - cur_count[pos, pred_seq[idx] ])*1.0/win_lg
                    # print new_rate
                #找到一个pos位置，它后面的window范围该动作的浓度很低

                # search from left at window size of 1
                if 0:
                    while (pred_seq[pos] != pred_seq[idx]) and pos>idx:
                        pos = pos - win_sm
                else:
                    new_rate = (cur_count[pos, pred_seq[idx] ] - cur_count[pos - win_sm, pred_seq[idx] ])*1.0/win_sm
                    while new_rate < rate_min and pos>idx:
                        pos = pos - win_sm
                        new_rate = (cur_count[pos, pred_seq[idx] ] - cur_count[pos - win_sm, pred_seq[idx] ])*1.0/win_sm
                        # print new_rate
                #再缓慢调整pos，使它前面window——small范围内该动作概率较大
                
                # assert(idx < pos)
                if idx < pos:
                    conf = np.average(prob_seq[idx:(pos+1), pred_seq[idx]] )
                    conf = round(conf, 5)
                    pred_labels.append([pred_seq[idx], idx, pos, conf ])#pred_class, start, end, score
                    idx = pos#下一个锚点
                else:
                    idx = idx + 1
            else:
                idx = idx + 1
        else:
            idx = idx + 1
    pred_labels = np.array(pred_labels)
    return pred_labels

def get_segments(scores, activity_threshold):
  """Get prediction segments of a video."""
  # Each segment contains start, end, class, confidence score.
  # Sum of all probabilities (1 - probability of no-activity)
  activity_prob = 1 - scores[:, 0]
  # Binary vector indicating whether a clip is an activity or no-activity
  activity_tag = np.zeros(activity_prob.shape, dtype=np.int32)
  activity_tag[activity_prob >= activity_threshold] = 1
  assert activity_tag.ndim == 1
  # For each index, subtract the previous index, getting -1, 0, or 1
  # 1 indicates the start of a segment, and -1 indicates the end.
  padded = np.pad(activity_tag, pad_width=1, mode='constant')
  diff = padded[1:] - padded[:-1]
  indexes = np.arange(diff.size)
  startings = indexes[diff == 1]
  endings = indexes[diff == -1]
  assert startings.size == endings.size

  segments = []
  for start, end in zip(startings, endings):
    segment_scores = scores[start:end, :]
    class_prob = np.mean(segment_scores, axis=0)
    segment_class_index = np.argmax(class_prob[1:]) + 1
    confidence = np.mean(segment_scores[:, segment_class_index])
    segments.append((segment_class_index, start, end, confidence))
  return np.array(segments)

def smoothing(x, k=5):
    ''' Applies a mean filter to an input sequence. The k value specifies the window
    size. window size = 2*k
    '''
    l = len(x)
    s = np.arange(-k, l - k)
    e = np.arange(k, l + k)
    s[s < 0] = 0
    e[e >= l] = l - 1
    y = np.zeros(x.shape)
    for i in range(l):
        y[i] = np.mean(x[s[i]:e[i]], axis=0)
    return y

if __name__ == '__main__':
    #输入是pred
    #注意label 0代表没有动作的间隔帧！！！
    #动作标签从1开始！！
    predict = {}
    label = {}
    p = np.zeros((51,5))
    p[0:8,1] = 1.0
    p[12:30,4] = 1.0
    p[33:43,3] = 1.0
    pred = get_interval_frm_frame_predict(p)
    print(pred)
    predict['0'] = pred
    label['0'] = np.array([[1,0,10],[2,12,32],[3,35,50]])
    eval_detect_mAP(label,predict)
