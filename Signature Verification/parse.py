# Reads a folder of signatures txt input files and stores them in a hash.

import os

class Parser:

    def read(folder):
        signatures = {}
        for file in os.listdir(folder):
            signature_label = file
            signature_properties = []
            for line in open(folder + file):
                properties_at_timepoint = [ prop.strip() for prop in line.split(' ')]
                signature_properties.append(properties_at_timepoint)
            signatures[signature_label] = signature_properties
        return signatures

    @staticmethod
    def get_ground_truth(ground_truth_file, use_boolean=False):
        """
        Reads-in the ground truth file and returns a dictionary with each verification sample filename as keys and the
        respective class (g/f or True/False) as values. Example: {"013-40.txt" -> "g", "013-41.txt" -> "f"}.
        Keys are not matched against the content of the "verification" directory, but are taken from the ground truth
        file and appended the ".txt" suffix, to match the keys in the feature dictionary. If use_boolean == True, values
        are True/False rather than g/f, with True <-> "g", False <-> "f".

        :param ground_truth_file:   path to the ground truth file
        :param use_boolean:         use boolean truth values True/False instead of g/f; defaults to False
        :return:                    dictionary with each verification sample filename as keys and the
                                    respective class (g/f or True/False) as values
        """
        ground_truth = {}
        with open(ground_truth_file, "r") as gt_file:
            for line in gt_file:
                entry = line.split(" ")
                # can use g/f or True/False as classes, depending on use_boolean
                is_genuine = entry[1].strip()
                if use_boolean:
                    is_genuine = True if is_genuine == "g" else False
                ground_truth[entry[0]] = is_genuine

        return ground_truth
