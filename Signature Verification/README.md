#Signature Verification

## Evaluation
The evaluation output file is called `results_signature.txt` in the folder Signature competition. In order to run the evaluation you can simply call `runner.py`. Currently the code spans 4 threads for faster computation and the part of `parse.py` is commented out as there is no ground-truth file for the evaluation.


## Usage
- `pip install -r requirements.txt`, preferably in a virtual environment
- run `runner.py`

## Current Architecture and Development State
- input data is read-in by `parse.py`
- features are calculated using `features.py`
- features are normalized using `features.py`, where each feature is normalized only w.r.t. a single signature (see `features.Features#normalize_signature_features`)
- `dtw.py` imported from assignment 3 as-is, converting the two input series (s, t) to numpy arrays
- can represent a user by creating a new `user.User` object
  - takes a user id and an enrolment features dict as arguments. User id should be the first part of the corresponding set of signatures, e.g. user "007" represents signatures of the form "007-g-01.txt" or "007-12.txt".
  - enrolment features should already be normalized
  - during construction, new User object automatically collects its enrolment signatures from the enrolment features dict, according to its user id
  - User object can calculate dissimilarities of some target signature w.r.t. all its enrolment signatures (using `dtw.py`) (check `user.User#calculate_signature_dissimilarity`)
- `output.py` has a function to print the dissimilarity dictionary for a user and a target signature
- `runner.py` currently orchestrates
  - loading data
  - calculating features
  - normalizing features
  - creating a User object for each line in `SignatureVerification/users.txt`
  - demonstrating the calculation and output of a dissimilarity dict for a specific user and target signature
- `parse.py` has a method `get_ground_truth` to load a ground truth dictionary for the verification signatures. Not used yet.
  
