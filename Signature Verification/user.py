from dtw import DTW
import numpy as np


class User:

    EXPECTED_NUM_ENROLMENT_SIGNATURES = 5
    WINDOW_SIZE = 3000  # "randomly" chosen for now, can be changed

    def __init__(self, user_id, all_enrolment_signatures):
        """
        Represents a user with their EXPECTED_NUM_ENROLMENT_SIGNATURES enrolment signatures.

        :param user_id:                     user id string in the format XXX (i.e. smaller numbers padded, e.g. 007)
        :param all_enrolment_signatures:    complete normalized enrolment features dict; can contain enrolment
                                            signatures of all users, must contain exactly
                                            EXPECTED_NUM_ENROLMENT_SIGNATURES enrolments signatures of this user
                                            (i.e. exactly EXPECTED_NUM_ENROLMENT_SIGNATURES keys starting with user_id,
                                            followed by "-")
        """
        self.user_id = user_id
        self.enrolment_signatures = self._collect_enrolment_signatures(all_enrolment_signatures)

    def _collect_enrolment_signatures(self, all_enrolment_signatures):
        enrolment_signatures = {}
        for key, signature in all_enrolment_signatures.items():
            signature_user_id = key.split("-")[0].strip()  # extract key part before "-"
            if signature_user_id == self.user_id:
                enrolment_signatures[key] = signature
        assert len(enrolment_signatures) == self.EXPECTED_NUM_ENROLMENT_SIGNATURES
        return enrolment_signatures

    def calculate_signature_dissimilarity(self, target_signature):
        """
        Calculate the dissimilarity of target_signature the the user's enrolment signatures

        :param target_signature:    list of the normalized feature vectors of the target signature to compare to this
                                    user's enrolment signatures
        :return:                    dictionary of type
                                    <user enrolment signature id> -> <dissimilarity of target signature to that enrolment signature>
        """
        dtw = DTW()
        dissimilarities = []
        for key, enrolment_signature in self.enrolment_signatures.items():
            dissim, matrix = dtw.distance(enrolment_signature, target_signature, self.WINDOW_SIZE)
            dissimilarities.append(dissim)
        return np.mean(np.asarray(dissimilarities))
