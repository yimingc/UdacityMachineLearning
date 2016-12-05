# From tensorflow/tensorflow/example/udacity/

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import hashlib
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import sys
from scipy import ndimage
from six.moves import cPickle as pickle
from six.moves.urllib.request import urlretrieve
from sklearn import linear_model
import tarfile

url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None

def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 1% change in download progress.
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)
    
    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()
    last_percent_reported = percent
        
def maybe_download(filename, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    if force or not os.path.exists(filename):
        print('Attempting to download:', filename) 
        filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

def maybe_extract(filename, num_classes, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall()
        tar.close()
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders)))
    print(data_folders)
    return data_folders


def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size), \
                          dtype=np.float32 )
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = ( ndimage.imread(image_file).astype(float) - \
                           pixel_depth/2 ) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape) )
            dataset[num_images, :, :] = image_data
            num_images += 1
        except IOError as e:        
            print('Could not read:', image_file, ':', e, \
                  '- it\'s ok, skipping.')
    
    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' \
                        %(num_images, min_num_images))
    
    print('Full dataset tensor:', dataset.shape)
    print('Mean: ', np.mean(dataset))
    print('Std: ', np.std(dataset))
    return dataset

def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = os.path.join(folder, '.pickle')
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)
        
    return dataset_names

def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
    global image_size
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes
    
    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class+tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try :
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class
                
                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise
    
    return valid_dataset, valid_labels, train_dataset, train_labels
    
def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_lables = labels[permutation]
    return shuffled_dataset, shuffled_lables    

def checkOverlap(setA, setB, cleanB = True):
    A_hashes = [hashlib.sha1(x).digest() for x in setA]
    B_hashes = [hashlib.sha1(x).digest() for x in setB]
    if cleanB:
        idx_overlap  = np.in1d(B_hashes,  A_hashes)
        idx_keep = ~idx_overlap
        set_clean = setB[idx_keep]
    else: # clean A
        idx_overlap  = np.in1d(A_hashes,  B_hashes)
        idx_keep = ~idx_overlap
        set_clean = setA[idx_keep]
    print("overlap: %d samples" % idx_overlap.sum())
    return set_clean, idx_keep
    
def Assignment1():
    print('Problem 0: Download data')
    train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
    test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)
    
    print('Problem 0: Extract data')
    num_classes = 10
    train_folders = maybe_extract(train_filename, num_classes)
    test_folders = maybe_extract(test_filename, num_classes)
    
    print('Problem 1: Display a sample')
    image = Image.open("notMNIST_small/A/Q0NXaWxkV29yZHMtQm9sZEl0YWxpYy50dGY=.png")
    plt.figure()
    plt.imshow(image)
    #plt.show()
    
    print('Problem 1: Normalize image')
    global image_size, pixel_depth
    image_size = 28      # Pixel width and height.
    pixel_depth = 255.0  # Number of levels per pixel.
    print('Image width %d, height %d, pixel levels %f' %(image_size, image_size, pixel_depth))
    
    print('Problem 1: Maybe pickle data')
    train_datasets = maybe_pickle(train_folders, 45000)
    test_datasets = maybe_pickle(test_folders, 1800)
    
    print('Problem 2: Display pickle data')
    pickle_file = train_datasets[0]
    with open(pickle_file) as f:
        letter_set = pickle.load(f)
        f.close()
        sample_idx = np.random.randint(len(letter_set))
        sample_image = letter_set[sample_idx, :, :]
        plt.figure()
        plt.imshow(sample_image)
        #plt.show()
    
    
    print('Problem 3: Merge and prune the data')
    train_size = 200000
    valid_size = 10000
    test_size = 10000

    valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
        train_datasets, train_size, valid_size)
    _, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

    print('Training:', train_dataset.shape, train_labels.shape)
    print('Validation:', valid_dataset.shape, valid_labels.shape)
    print('Testing:', test_dataset.shape, test_labels.shape)
    
    print('Randomize the data')
    train_dataset, train_labels = randomize(train_dataset, train_labels)
    valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
    test_dataset, test_labels = randomize(test_dataset, test_labels)
    
    print('Problem 4: Ensure the data is still good after shuffling')
    idx = np.random.randint(train_size)
    print('Label: ', str(train_labels[idx]))
    plt.figure()
    plt.imshow( train_dataset[idx] )
    plt.title('Problem 4: Ensure the data is still good after shuffling')
    plt.show()
    
    print('Problem 3: Check the balance of data')
    plt.hist(train_labels)
    plt.title('Problem 3: Check the balance of data')
    plt.show()
    
    print('Finally, pickle the data')
    pickle_file = 'notMNIST.pickle'
    try:
        f = open(pickle_file, 'wb')
        save = {'train_dataset': train_dataset,
                'train_labels': train_labels,
                'valid_dataset': valid_dataset,
                'valid_labels': valid_labels,
                'test_dataset': test_dataset,
                'test_labels': test_labels
            }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
    statinfo = os.stat(pickle_file)
    print('Compressed pickle size:', statinfo.st_size)
    
    print('One more check of the pickled data')
    with open(pickle_file) as f:
        pickled_data = pickle.load(f)
        f.close()
        idx = np.random.randint(train_size)
        print('Label: ', str(pickled_data['train_labels'][idx]))
        plt.figure()
        plt.imshow( pickled_data['train_dataset'][idx] )
        #plt.show()
        
    print('Problem 5: Let\'s check overlaps between train and valid data!')
    # https://discussions.udacity.com/t/assignment-1-problem-5/45657/19, search stmax 82
    test_dataset_clean, test_dataset_keep = checkOverlap(train_dataset, test_dataset)
    valid_dataset_clean, valid_dataset_keep = checkOverlap(train_dataset, valid_dataset)
    test_labels_clean = test_labels[test_dataset_keep]
    valid_labels_clean = valid_labels[valid_dataset_keep]
    test_size_clean = len(test_dataset_clean)
    valid_size_clean = len(valid_dataset_clean)
    print('Train data size %d, label size %d.' % (len(train_dataset), len(train_labels)))
    print('Cleaned test data size %d, label size %d.' % (test_size_clean, len(test_labels_clean)))
    print('Cleaned valid data size %d, label size %d.' % (valid_size_clean, len(valid_labels_clean)))
    
    print('Problem 6: Train a simple model with LogisticRegression model from sklearn.linear_model') 
    logreg = linear_model.LogisticRegression(C=1e5, verbose=1)
    # we create an instance of Neighbours Classifier and fit the data.
    num_samples, width, height = train_dataset.shape
    flat_train_dataset = train_dataset.reshape((num_samples,width*height))[0:num_samples]
    logreg.fit(flat_train_dataset, train_labels)
    with open('trainedModel.pickle', 'wb') as f:
        pickle.dump(logreg, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    
    print('Problem 6: Check prediction error of the trained model')
    with open('trainedModel.pickle', 'rb') as f1:
        clf = pickle.load(f1)
        f1.close()
        results = clf.predict(test_dataset_clean.reshape((test_size_clean,width*height)))
        error = results - test_labels_clean
        ratio = float(np.count_nonzero(error)) / float(len(test_labels_clean))
        print(error)
        print('Error number %d' % np.count_nonzero(error))
        print('Error ratio %f' % ratio )
        print('Assignment1 Done.')

if __name__ == '__main__':
    Assignment1()
