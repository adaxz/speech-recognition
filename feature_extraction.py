from prerprocessing import list_wav_fnames, pad_audio, chop_audio

from tqdm import tqdm
import h5py

def label_transform(labels):
    nlabels = []
    for label in labels:
        if label == '_background_noise_':
            nlabels.append('silence')
        elif label not in legal_labels:
            nlabels.append('unknown')
        else:
            nlabels.append(label)
    return pd.get_dummies(pd.Series(nlabels))

def loading_transforming(feature_str, label_fnames):
    new_sample_rate = 8000
    
    mfcc_shape= (49, 13)
    logfbank_shape = (49, 26)
    logspec_shape = (99, 81)
    
    features = {'mfcc': (mfcc, mfcc_shape), 'logfbank': (logfbank, logfbank_shape), 'logspec': (log_specgram, logspec_shape)}
    feature = features[feature_str][0]
    out_shape = features[feature_str][1]
    
    x_train_shape = (len(label_fnames), out_shape[0], out_shape[1])

    y_train = []
    #x_train = []
    x_train = np.zeros(x_train_shape, np.float32)

    G = []
    ix = 0

    for audio in tqdm(label_fnames):
        
        label = audio[0]
        fname = audio[1]

        sample_rate, samples = wavfile.read(os.path.join(train_audio_path, label, fname))
        samples = pad_audio(samples)

        if len(samples) > 16000:
            n_samples = chop_audio(samples)
        else:
            n_samples = [samples]
        for samples in n_samples:
            #filter_banks = logfbank(samples)
            #filter_banks = mfcc(samples)
            resampled = signal.resample(samples, int(new_sample_rate / sample_rate * samples.shape[0]))
            transfomed = feature(resampled)
            
            if feature_str != 'logspec':
                transfomed -= (np.mean(transfomed, axis=0) + 1e-8)
            x_train[ix,:,:] = transfomed
            #x_train.append(transfomed)
        y_train.append(label)
        
        
        group = fname.split('_')[0]
        G.append(group)
        ix += 1
    if feature_str == 'logspec':
        x_train = x_train.reshape(tuple(list(x_train.shape) + [1]))
        
    return G, x_train, y_train, ix

def save_x_matrix(feature_str, matrix, path):
	G, x_train, y_train, ix = loading_transforming(feature_str, label_fnames)
	y_train = label_transform(y_train)
	y_train = y_train.values
	y_train = np.array(y_train)
	G = np.array(G)

	filename = feature_str+'_x.h5'
	file = h5py.File(x_filename, 'w')

	dataset_name = 'x_train_data'
	file.create_dataset(dataset_name, data=matrix)



L = 16000
legal_labels = 'yes no up down left right on off stop go silence unknown'.split()
train_audio_path = '../data/train/audio'

label_fnames = list_wav_fnames(train_audio_path, batch_size=100)
G, x_train, y_train, ix = loading_transforming('logspec', label_fnames)

y_train = label_transform(y_train)
y_train = y_train.values
y_train = np.array(y_train)
G = np.array(G)



