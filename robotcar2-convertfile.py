import argparse
import os
from collections import defaultdict
import pathlib

import kapture
import kapture.io.csv as csv



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='use pairsfiles and kaptures for pose estimation using image retrieval')
    parser.add_argument('-p', '--pairsfile-path', type=str, default='patchnetvlad/results/robotcarseason2/NetVLAD_predictions.txt', required=False, help='path to pairsfile')
    parser.add_argument('-m', '--mapping-path', type=str, default='/media/leo/2C737A9872F69ECF/datasets/kapture-RobotCar-Seasons-v2/', required=False, help='path to mapping kapture')
    parser.add_argument('-o', '--output-path', type=str, default='patchnetvlad/results/robotcarseason2/NetVLAD_robotcarpredictions_submission.txt', required=False, help='path to output LTVL challenge file')
    parser.add_argument('-d', '--decreasing', action='store_true', help='set if descending scores indicate a better match')
    parser.add_argument('-i', '--inverse', action='store_true', help='invert poses before recording them down in output file')
    args = parser.parse_args()

    kdata_mapping = csv.kapture_from_dir(args.mapping_path)
    if kdata_mapping.rigs:
        kapture.rigs_remove_inplace(kdata_mapping)

    with open(args.pairsfile_path, 'r') as f:
        image_pairs = csv.table_from_file(f)

    query_lookup = defaultdict(list)    

    for query, mapping, score in image_pairs:
        query_lookup[query] += [(mapping, score)]
    
    # locate best match using pairsfile
    best_match_pairs = []
    for query, retrieved_mapping in query_lookup.items():
        if args.decreasing:
            best_match = min(retrieved_mapping, key=lambda x: x[1])[0]
        else:
            best_match = max(retrieved_mapping, key=lambda x: x[1])[0]
        best_match_pairs += [(query, best_match)]

    # recover pose from best match
    fname_to_pose_lookup = {}
    for ts, cam, fname in kapture.flatten(kdata_mapping.records_camera):
        fname_to_pose_lookup[fname] = kdata_mapping.trajectories[ts, cam]
    image_poses = {query: fname_to_pose_lookup[mapping] for query, mapping in best_match_pairs}

    # LTVT2020
    p = pathlib.Path(args.output_path)
    os.makedirs(str(p.parent.resolve()), exist_ok=True)
    with open(args.output_path, 'wt') as f:
        for image_filename, pose in image_poses.items():
            if args.inverse:
                pose = pose.inverse()
            line = [image_filename] + pose.r_raw + pose.t_raw
            line = ' '.join(str(v) for v in line) + '\n'
            f.write(line)