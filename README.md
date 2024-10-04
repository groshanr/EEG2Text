EEG2Text - M. Wade & R. Groshan

Project Structure:
- Data: Contains example data from [Murphy et al.](https://edata.bham.ac.uk/617/)

## Overall Description:
These files contain preprocessed trial-level EEG data as outlined in the
accompanying paper: "Decoding Part-of-Speech from Human EEG Signals".
These data files haven filtered between [1 40] Hz and downsampled to 250 Hz (from 1 kHz).
Baseline correction has also been applied. The baseline period is 200 ms prior to sentence start.

## Data Acquisition
This data relates to 5.5 iterations of fixed-interval, single-word presentations of a subset of the English Web Treebank Corpus (Bies et al., 2012), released under a Creative Commons Share-Alike license (CC BY-SA 4.0). The EEG data were recorded using 64 Ag/AgCl active actiCAP slim  electrodes arranged in a 10–20 layout (ActiCAP, Brain Products GmbH, Gilching, Germany).


## Dataset Description: 
75 recording sessions of data are present in this data set.
Each recording session has been preprocessed such that all word-level trials presented
to the subject are contained within a window of 1,100 ms centred at the onset of word presentation
on screen, with the prior 200 ms and following 900 ms included in each window of data.
Each word was presented approximately every 240 ms.
Each word-level trial contains the following metadata:

* filename: 	which textfile from the EWT corpus the word derives from
* len:		the length (# characters) of the word
* freq:		Zipf-frequency value from the WordFreq Python package
* sess_id:	session identifier (same as filename, but useful when all data is concatenated)
* prev_pos:	part-of-speech of the previous word in the sentence***
* next_pos:	part-of-speech of the following word in the sentence
* prev_freq:	frequency score of the previous word
* next_freq:	frequency score of the following word
* prev_len: 	length of the previous word (# characters)
* next_len:	length of the following owrd (# characters)
* sent_ident:	a corpus-wise unique sentence identifier

In addition to the Part-of-speech tags are from the Universal Tagset, this dataset recognises two further useful tags in the data:
* "SENT_START"
* "SENT_END" 

These serve to fill prev_pos and next_pos slots for words that occur at the start / end of a sentence, respectively.

## File format:
This dataset employs the use of the .FIF file format for neuroscience data (See: https://mne.tools/0.15/manual/io.html)
The dataset was created using the MNE Python library (Gramfort et al.2013) (https://mne.tools/0.15/index.html).
Installation instructions: https://mne.tools/0.15/install_mne_python.html
Quick start: "pip install mne"

## Loading data:
Using the MNE library, each file can be loaded using the "mne.read_epochs(filename)" command.
The result is saved into an mne.Epochs data structure.
This data structure contains the the preprocessed EEG data in its "._data" attribute
which should be accessed via the ".get_data" function of the mne.Epochs data structure.
Associated metadata (pandas DataFrame) can be accessed via the ".metadata" attribute.

   Example:
   epochs = mne.read_epochs("rsvp_session0_files74-88-epo.fif")
   eeg_data = epochs.get_data() # NumPy array of shape [n_trials, n_electrodes, n_timepoints] with n_electrodes = 64, n_timepoints = 276 and n_trials is variable by each file.
   metadata = epochs.metadata   # Pandas DataFrame

All other information relating to the EEG recording and analysis pipeline is available
according to the extensive documentation on the MNE Python website: https://mne.tools/0.15/documentation.html

#### Example data:
This directory includes the full EEG data and metadata record for one of the sessions. This data can be easily loaded with NumPy and any CSV-reader in order to explore the format of the data. All other sessions follow an identical format. This allows evaluation of the type of data without the need to install MNE-Python and is representative of the entire dataset.
