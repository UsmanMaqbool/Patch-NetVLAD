import logging

import argparse
import os
from collections import defaultdict
import pathlib
from pickle import FALSE

import kapture
#import kapture.io.csv as csv
#import kapture
import kapture.utils.logging
import kapture.io.features
import kapture.io.csv

logger = logging.getLogger('LTVL2020')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='use pairsfiles and kaptures for pose estimation using image retrieval')
    parser.add_argument('-p', '--pairsfile-path', type=str, default='patchnetvlad/results/robotcarseason2/NetVLAD_predictions.txt', required=False, help='path to pairsfile')
    parser.add_argument('-m', '--mapping-path', type=str, default='/media/leo/2C737A9872F69ECF/datasets/kapture_datasets/RobotCar_Seasons-v2/mapping/', required=False, help='path to mapping kapture')
    parser.add_argument('-o', '--output-path', type=str, default='patchnetvlad/results/robotcarseason2/NetVLAD_robotcarpredictions_submission.txt', required=False, help='path to output LTVL challenge file')
    parser.add_argument('-d', '--decreasing', action='store_true', help='set if descending scores indicate a better match')
    parser.add_argument('-i', '--inverse', action='store_true', help='invert poses before recording them down in output file')
    args = parser.parse_args()

    # only load (1) image records + (2) trajectories (that all that matters).
    # 1: load records
    records_camera_filepath = kapture.io.csv.get_csv_fullpath(kapture.RecordsCamera, args.mapping_path)
    logger.debug(f'loading {records_camera_filepath}')
    records_cameras = kapture.io.csv.records_camera_from_file(records_camera_filepath)
    # 2: load trajectories
    trajectories_filepath = kapture.io.csv.get_csv_fullpath(kapture.Trajectories, args.mapping_path)
    logger.debug(f'loading {trajectories_filepath}')
    trajectories = kapture.io.csv.trajectories_from_file(trajectories_filepath)

    # 2.2: load rigs if it exists
    rigs_filepath = kapture.io.csv.get_csv_fullpath(kapture.Rigs, args.mapping_path)
    if os.path.isfile(rigs_filepath):
        logger.debug(f'loading {rigs_filepath}')
        rigs = kapture.io.csv.rigs_from_file(rigs_filepath)
        trajectories = kapture.rigs_remove(trajectories, rigs)

    # 3: find (timestamp, camera_id) that are both in records and trajectories.
    valid_keys = set(records_cameras.key_pairs()).intersection(set(trajectories.key_pairs()))
    # collect data for those timestamps.
    keep_full_file_name = False
    if keep_full_file_name:
        image_poses = ((k[1], records_cameras[k], trajectories[k]) for k in valid_keys)
    else:
        image_poses = ((k[1], os.path.basename(records_cameras[k]), trajectories[k]) for k in valid_keys)

    # prepend the camera name or drop it.
    prepend_camera_name = True
    if prepend_camera_name:
        image_poses = ((os.path.join(camera_id, image_filename), pose) for camera_id, image_filename, pose in image_poses)
    else:
        image_poses = ((image_filename, pose) for _, image_filename, pose in image_poses)
    truncate_extensions = False
    if truncate_extensions:
        image_poses = ((image_filename[:image_filename.index('.')], pose) for image_filename, pose in image_poses)

    # # write the files
    # image_poses = {image_filename: pose
    #                for image_filename, pose in image_poses}
    
    # p = pathlib.Path(args.output_path)
    # os.makedirs(str(p.parent.resolve()), exist_ok=True)
    # with open(args.output_path, 'wt') as f:
    #     for image_filename, pose in image_poses.items():
    #         if args.inverse:
    #             pose = pose.inverse()
    #         line = [image_filename] + pose.r_raw + pose.t_raw
    #         line = ' '.join(str(v) for v in line) + '\n'
    #         f.write(line)


    with open(args.pairsfile_path, 'r') as f:
        image_pairs = kapture.io.csv.table_from_file(f)

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
        
        # best_match_pairs += [(query, best_match)].
            
        path_split = pathlib.PurePath(query).parts
        qq = path_split[7:]
        qss = os.path.join(qq[0],qq[1])
        
        bpath_split = pathlib.PurePath(best_match).parts
        bqq = bpath_split[7:]
        bss = os.path.join(bqq[1],bqq[2])
        
        best_match_pairs += [(qss, bss)]
        

    # recover pose from best match
    fname_to_pose_lookup = {}
    for image, pose in image_poses:
        fname_to_pose_lookup[image] = pose
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