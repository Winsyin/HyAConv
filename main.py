
from model import *
from sklearn import metrics

import random
import numpy as np
import pickle
import csv

csv.field_size_limit(500 * 1024 * 1024)





def ReadMyCsv4(SavaDict, fileName):
	csv_reader = csv.reader(open(fileName))
	count = 0
	for row in csv_reader:
		SavaDict[row[0]] = count
		count = count + 1
	return


def ReadMyCsv5(SaveList, fileName):
	csv_reader = csv.reader(open(fileName))
	for row in csv_reader:
		c = int(row[0]) - 1
		m = idMiRNA[row[1]]
		cmi = [c, m]
		SaveList.append(cmi)
	return


def StorFile(data, fileName):
	with open(fileName, "w", newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerows(data)
	return



def MyLabel(Sample):
	label = []
	for i in range(int(len(Sample) / 2)):
		label.append(1)
	for i in range(int(len(Sample) / 2)):
		label.append(0)
	return label



def load_hyg(filename):

	with open(filename, 'rb') as f:
		hyg_dict = pickle.load(f)

	hy_cm = hyg_dict['hy_cm']
	hy_mc = hyg_dict['hy_mc']
	topk_cmat = hyg_dict['topk_cmat']
	topk_mmat = hyg_dict['topk_mmat']



	return hy_cm, hy_mc, topk_cmat[0], topk_mmat[0], topk_cmat[1], topk_mmat[1]


def train(circ_fea, mi_fea, train_data, label_train):
	model.train()
	print('--- Start training ---')
	optimizer.zero_grad()
	pred_train = \
		model(circ_fea, mi_fea, torch.from_numpy(train_data[:, 0]).long(),
			  torch.from_numpy(train_data[:, 1]).long())


	loss_train = loss_function(pred_train.view(-1, 1),
								   torch.from_numpy(label_train).view(-1, 1).float())


	loss_train.backward()
	optimizer.step()


	score_train_cpu = pred_train.detach().numpy()
	auc_train = metrics.roc_auc_score(label_train, score_train_cpu)

	pred_label_train = [1 if j > 0.5 else 0 for j in score_train_cpu]
	acc_train = metrics.accuracy_score(label_train, pred_label_train)

	print('Epoch: {:05d}, loss_train:{:.6f}'.format(e + 1, loss_train.item()))
	print('loss_train_cmi:{:.6f}'.format(loss_train.item()))

	print('AUC_train_cmi:{:.6f}, '
				 'ACC_train_cmi:{:.6f}'.format(auc_train, acc_train))



def test(circ_fea, mi_fea, test_data, label_test):
	with torch.no_grad():
		model.eval()
		print('--- Start valuating ---')

		pred_test = \
			model(circ_fea, mi_fea, torch.from_numpy(test_data[:, 0]).long(),
				  torch.from_numpy(test_data[:, 1]).long())


		score_test_cpu = pred_test.detach().numpy()
		auc_test = metrics.roc_auc_score(label_test, score_test_cpu)

		pred_label_test = [1 if j > 0.5 else 0 for j in score_test_cpu]
		acc_test = metrics.accuracy_score(label_test, pred_label_test)

		print('AUC_test_cmi:{:.6f}, ACC_test_cmi:{:.6f}'.format(auc_test, acc_test))
		return score_test_cpu


if __name__=='__main__':
	random.seed(1)
	np.random.seed(1)
	torch.manual_seed(1)


	epochs = 70
	lr = 0.001
	bio_out_dim = 32
	hgnn_dim_1 = 32

	layers_h = 1
	num_circRNA = 2115
	num_miRNA = 821


	CircEmbeddingFeature = np.loadtxt('circRNA_seq_similarity_circRNA2vec.txt',
									  delimiter='\t')
	miRNAEmbeddingFeature = np.loadtxt('miRNA_seq_similarity_kmer.txt',
									   delimiter='\t')
	CircEmbeddingFeature = torch.from_numpy(CircEmbeddingFeature).type(torch.FloatTensor)
	miRNAEmbeddingFeature = torch.from_numpy(miRNAEmbeddingFeature).type(torch.FloatTensor)


	for fold in range(5):
		print('***********fold_{}*****************'.format(fold+1))


		CMI_data = './5fold_CV/'


		idMiRNA = {}
		ReadMyCsv4(idMiRNA, 'miRBaseSequence.csv')

		idCircRNA = {}
		ReadMyCsv4(idCircRNA, 'circBaseSequence.csv')


		PositiveSample_Train = []
		ReadMyCsv5(PositiveSample_Train, '{}/Positive_Sample_Train{}.csv'.format(CMI_data, fold))
		PositiveSample_Validation = []
		ReadMyCsv5(PositiveSample_Validation, '{}/Positive_Sample_Validation{}.csv'.format(CMI_data, fold))
		PositiveSample_Test = []
		ReadMyCsv5(PositiveSample_Test, '{}/Positive_Sample_Test{}.csv'.format(CMI_data, fold))

		NegativeSample_Train = []
		ReadMyCsv5(NegativeSample_Train, '{}/Negative_Sample_Train{}.csv'.format(CMI_data, fold))
		NegativeSample_Validation = []
		ReadMyCsv5(NegativeSample_Validation, '{}/Negative_Sample_Validation{}.csv'.format(CMI_data, fold))
		NegativeSample_Test = []
		ReadMyCsv5(NegativeSample_Test, '{}/Negative_Sample_Test{}.csv'.format(CMI_data, fold))


		x_train_pair = []
		x_train_pair.extend(PositiveSample_Train)
		x_train_pair.extend(NegativeSample_Train)
		x_train_pair = np.array(x_train_pair)


		x_val_test_pair = []
		x_val_test_pair.extend(PositiveSample_Validation)
		x_val_test_pair.extend(PositiveSample_Test)
		x_val_test_pair.extend(NegativeSample_Validation)
		x_val_test_pair.extend(NegativeSample_Test)
		x_val_test_pair = np.array(x_val_test_pair)


		y_train = MyLabel(x_train_pair)
		y_val_test = MyLabel(x_val_test_pair)


		hy_cm, hy_mc, cc_hyper_graph, mm_hyper_graph, cc_D, mm_D = load_hyg('./hyg/hyg_dict{}.pickle'.format(fold))

		model = HGNN(num_circRNA, num_miRNA, bio_out_dim, hgnn_dim_1, layers_h,
					 hy_cm, hy_mc, cc_hyper_graph, mm_hyper_graph, cc_D, mm_D)



		loss_function = torch.nn.BCELoss()
		optimizer = torch.optim.Adam(model.parameters(), lr)


		for e in range(epochs):
			train(CircEmbeddingFeature, miRNAEmbeddingFeature, x_train_pair, np.array(y_train))
			ModelTestOutput = test(CircEmbeddingFeature, miRNAEmbeddingFeature, x_val_test_pair, np.array(y_val_test))

























































