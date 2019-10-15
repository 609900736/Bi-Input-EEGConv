For model saving.
	*.h5 is saved model for especial subject.
	*.npy is saved history datas from model training.

Introduction of existed saved models:
	2019_9_27_10_10_30:		EEGNet(8,2) with 4-38Hz filtered data
	2019_9_27_10_29_32:		EEGNet(8,2) with unfiltered data
	2019_9_27_11_32_19:		EEGNet(16,2) with unfiltered data
	2019_9_27_11_53_18:		EEGNet(8,4) with unfiltered data
	2019_9_29_18_33_7:		EEGNet(8,2) without filter and P300 data (1-4s)
	2019_10_10_16_40_51:	EEGNet(8,2) without filter and P300 data (1-4s)
	2019_10_10_17_8_29:		EEGNet(8,2) without filter, P300 data (1-4s) and model restate
	2019_10_10_17_37_14:	EEGNet(8,2) without filter, P300 data (1-4s) and model restate, reuse 2019_10_10_17_8_29_A09T_EEGNet.h5 model
	2019_10_11_15_24_13:	EEGNet(8,2) with unfiltered data
	2019_10_11_17_37_17:	EEGNet(8,2) with unfiltered data and change AveragePooling2D to MaxPooling2D
	2019_10_11_19_6_27:		EEGNet(8,2) with filtered and 0.5-4s data
	2019_10_11_19_37_6:		EEGNet(8,2) with filtered and 0.5-4s data
	2019_10_14_19_37_46:	EEGNet(8,2) with unfiltered data, without max_norm in Dense layer
	2019_10_14_20_47_53:	EEGNet(8,2) with unfiltered data, without Averagepooling2D
	2019_10_15_15_18_36:	EEGNet(8,2) with filtered and 0.5-4s data