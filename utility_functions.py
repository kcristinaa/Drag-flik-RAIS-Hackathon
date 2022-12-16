import re
from datetime import datetime

import numpy as np
import pandas as pd


def read_single_mvnx_to_df(mvnx_file, disable_print=False):
    segment_count = mvnx_file.segment_count - 1
    df = pd.DataFrame()
    # for each segment (body part)
    for idx in range(segment_count):
        segment = mvnx_file.segment_name_from_index(idx)
        if not disable_print:
            print("--------- Adding data for segment: {} ({}/{}) ---------".format(segment, idx + 1, segment_count))
        # add acceleration
        segment_acc = mvnx_file.get_segment_acc(idx)
        df_temp = pd.DataFrame(segment_acc)
        df_temp = df_temp.add_prefix("{}_{}_".format(segment, "acc"))
        df = pd.concat([df, df_temp], axis=1)

        # add angualar accelaration
        segment_ang_acc = mvnx_file.get_segment_angular_acc(idx)
        df_temp = pd.DataFrame(segment_ang_acc)
        df_temp = df_temp.add_prefix("{}_{}_".format(segment, "angular_acc"))
        df = pd.concat([df, df_temp], axis=1)

        # add velocity
        segment_vel = mvnx_file.get_segment_vel(idx)
        df_temp = pd.DataFrame(segment_vel)
        df_temp = df_temp.add_prefix("{}_{}_".format(segment, "vel"))
        # df.reset_index(inplace=True, drop=True)
        # df_temp.reset_index(inplace=True, drop=True)
        df = pd.concat([df, df_temp], axis=1)

        # add angualar velocity
        segment_ang_vel = mvnx_file.get_segment_angular_vel(idx)
        df_temp = pd.DataFrame(segment_ang_vel)
        df_temp = df_temp.add_prefix("{}_{}_".format(segment, "angular_vel"))
        df = pd.concat([df, df_temp], axis=1)

        # add orientation
        segment_ori = mvnx_file.get_segment_ori(idx)
        df_temp = pd.DataFrame(segment_ori)
        df_temp = df_temp.add_prefix("{}_{}_".format(segment, "ori"))
        df = pd.concat([df, df_temp], axis=1)

        # add position
        get_segment_pos = mvnx_file.get_segment_pos(idx)
        df_temp = pd.DataFrame(get_segment_pos)
        df_temp = df_temp.add_prefix("{}_{}_".format(segment, "pos"))
        df = pd.concat([df, df_temp], axis=1)

    # add time in milliseconds for each frame
    times = mvnx_file.file_data['frames'].get("time")
    df_temp = pd.DataFrame(times)
    df_temp = df_temp.rename(columns={0: "time"})
    df = pd.concat([df, df_temp], axis=1)

    return df


def read_mvnx_metadata(mvnx_file, file_name):
    # get year
    # recording_date = mvnx_file.recording_date
    # recording_date_formatted = datetime.strptime(recording_date, "%a %b %d %H:%M:%S.%f %Y")
    # year = recording_date_formatted.year

    # get name from filepath
    file_name = file_name.split("\\")[-1]

    # get sample number
    regex = r"(?:\(*(\d)\)*)(?:.mvnx)"
    result = re.search(regex, file_name)
    sample = result.group(1)

    # get user_id
    id = file_name.replace('.mvnx', '').split('_')[1]

    # get year
    fn = file_name.replace('.mvnx', '').split('_')[2]
    fn = re.split('-| ', fn)[0]
    fn = fn[-2:]
    year = '20{}'.format(fn)

    # get user_gender
    # original_file_name = mvnx_file.original_file_name
    # if 'mannen' in original_file_name:
    #     gender = 'M'
    # elif 'vrowen' in original_file_name:
    #     gender = 'F'
    # else:
    #     print("Cannot find gender: {}".format(original_file_name))
    #     gender = np.NaN
    gender = np.NaN

    return year, id, sample, gender
